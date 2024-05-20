import gc
import json
import os
import re
import sys

from common import (
    pre_edge_with_defaults,
    read_all_groups,
    read_group,
    xftf_with_defaults,
)

from larch.io import (
    create_athena,
    h5group,
    merge_groups,
    read_ascii,
    set_array_labels,
)
from larch.symboltable import Group
from larch.xafs import rebin_xafs

import matplotlib
import matplotlib.pyplot as plt

import numpy as np


class Reader:
    def __init__(
        self,
        energy_column: str,
        mu_column: str,
        data_format: str,
        extract_group: "dict[str, str]" = None,
    ):
        self.energy_column = energy_column
        self.mu_column = mu_column
        self.data_format = data_format
        self.extract_group = extract_group

    def load_data(
        self,
        dat_file: str,
        merge_inputs: bool,
        is_zipped: bool,
    ) -> "dict[str, Group]":
        if merge_inputs:
            out_group = self.merge_files(
                dat_files=dat_file,
                is_zipped=is_zipped,
            )
            return {"out": out_group}
        else:
            return self.load_single_file(
                filepath=dat_file,
                is_zipped=is_zipped,
            )

    def merge_files(
        self,
        dat_files: str,
        is_zipped: bool,
    ) -> Group:
        if is_zipped:
            all_groups = list(self.load_zipped_files().values())
        else:
            all_groups = []
            for filepath in dat_files.split(","):
                for group in self.load_single_file(filepath).values():
                    all_groups.append(group)

        merged_group = merge_groups(all_groups, xarray="energy", yarray="mu")
        pre_edge_with_defaults(merged_group)
        return merged_group

    def load_single_file(
        self,
        filepath: str,
        is_zipped: bool = False,
    ) -> dict:
        if is_zipped:
            return self.load_zipped_files()

        print(f"Attempting to read from {filepath}")
        if self.data_format == "athena":
            if self.extract_group["extract_group"] == "single":
                group = read_group(filepath, self.extract_group["group_name"])
                return {"out": group}
            elif self.extract_group["extract_group"] == "multiple":
                groups = {}
                for repeat in self.extract_group["multiple"]:
                    name = repeat["group_name"]
                    print(f"\nExtracting group {name}")
                    groups[name] = read_group(filepath, name)
                return groups
            else:
                return read_all_groups(filepath)

        else:
            # Try ascii anyway
            try:
                group = self.load_ascii(filepath)
                if not group.array_labels:
                    # In later versions of larch, won't get a type error it
                    # will just fail to load any data
                    group = self.load_h5(filepath)
            except (UnicodeDecodeError, TypeError):
                # Indicates this isn't plaintext, try h5
                group = self.load_h5(filepath)
            pre_edge_with_defaults(group)
            xftf_with_defaults(group)
            return {"out": group}

    def load_ascii(self, dat_file):
        with open(dat_file) as f:
            labels = None
            last_line = None
            line = f.readline()
            while line:
                if not line.startswith("#"):
                    if last_line is not None and last_line.find("\t") > 0:
                        labels = []
                        for label in last_line.split("\t"):
                            labels.append(label.strip())
                    break

                last_line = line
                line = f.readline()

        xas_data = read_ascii(filename=dat_file, labels=labels)
        xas_data = self.rename_cols(xas_data)
        return xas_data

    def load_h5(self, dat_file):
        h5_group = h5group(fname=dat_file, mode="r")
        energy = h5_group.entry1.instrument.qexafs_energy.qexafs_energy
        mu = h5_group.entry1.instrument.qexafs_counterTimer01.lnI0It
        xafs_group = Group(data=np.array([energy[:], mu[:]]))
        set_array_labels(xafs_group, ["energy", "mu"])
        return xafs_group

    def load_zipped_files(self) -> "dict[str, Group]":
        def sorting_key(filename: str) -> str:
            return re.findall(r"\d+", filename)[-1]

        all_paths = list(os.walk("dat_files"))
        all_paths.sort(key=lambda x: x[0])
        file_total = sum([len(f) for _, _, f in all_paths])
        print(f"{file_total} files found")
        keyed_data = {}
        for dirpath, _, filenames in all_paths:
            try:
                filenames.sort(key=sorting_key)
            except IndexError as e:
                print(
                    "WARNING: Unable to sort files numerically, "
                    f"defaulting to sorting alphabetically:\n{e}"
                )
                filenames.sort()

            for filename in filenames:
                if len(all_paths) > 1:
                    key = f"{dirpath.replace('/', '_')}_{filename}"
                else:
                    key = filename
                filepath = os.path.join(dirpath, filename)
                xas_data = self.load_single_file(filepath)
                keyed_data[key] = xas_data["out"]

        return keyed_data

    def rename_cols(self, xafs_group: Group) -> Group:
        labels = [label.lower() for label in xafs_group.array_labels]
        print(f"Read columns: {labels}")

        if "energy" in labels:
            print("'energy' present in column headers")
        elif self.energy_column:
            if self.energy_column.lower() in labels:
                labels[labels.index(self.energy_column.lower())] = "energy"
            else:
                raise ValueError(f"{self.energy_column} not found in {labels}")
        else:
            for i, label in enumerate(labels):
                if label in ("col1", "ef") or label.endswith("energy"):
                    labels[i] = "energy"
                    break

        if "mu" in labels:
            print("'mu' present in column headers")
        elif self.mu_column:
            if self.mu_column.lower() in labels:
                labels[labels.index(self.mu_column.lower())] = "mu"
            else:
                raise ValueError(f"{self.mu_column} not found in {labels}")
        else:
            for i, label in enumerate(labels):
                if label in ["col2", "xmu", "lni0it", "ffi0", "ff/i1"]:
                    labels[i] = "mu"
                    break

        if labels != xafs_group.array_labels:
            print(f"Renaming columns to: {labels}")
            return set_array_labels(xafs_group, labels)
        else:
            return xafs_group


def calibrate_energy(
    xafs_group: Group,
    calibration_e0: float = None,
    energy_min: float = None,
    energy_max: float = None,
):
    if calibration_e0 is not None:
        print(f"Recalibrating edge from {xafs_group.e0} to {calibration_e0}")
        xafs_group.energy = xafs_group.energy + calibration_e0 - xafs_group.e0
        xafs_group.e0 = calibration_e0

    if not (energy_min or energy_max):
        return xafs_group

    if energy_min is not None:
        index_min = np.searchsorted(xafs_group.energy, energy_min)
    else:
        index_min = 0

    if energy_max is not None:
        index_max = np.searchsorted(xafs_group.energy, energy_max)
    else:
        index_max = len(xafs_group.energy)

    print(
        f"Cropping energy range from {energy_min} to {energy_max}, "
        f"index {index_min} to {index_max}"
    )
    try:
        xafs_group.dmude = xafs_group.dmude[index_min:index_max]
        xafs_group.pre_edge = xafs_group.pre_edge[index_min:index_max]
        xafs_group.post_edge = xafs_group.post_edge[index_min:index_max]
        xafs_group.flat = xafs_group.flat[index_min:index_max]
    except AttributeError:
        pass

    xafs_group.energy = xafs_group.energy[index_min:index_max]
    xafs_group.mu = xafs_group.mu[index_min:index_max]

    # Sanity check
    if len(xafs_group.energy) == 0:
        raise ValueError("Energy cropping led to an empty array")

    return xafs_group


def main(
    xas_data: Group,
    do_calibrate: bool,
    calibrate_settings: dict,
    do_rebin: bool,
    do_pre_edge: bool,
    pre_edge_settings: dict,
    ref_channel: str,
    do_xftf: bool,
    xftf_settings: dict,
    plot_graph: list,
    annotation: str,
    path_key: str = "out",
):
    if do_calibrate:
        print(f"Calibrating energy with {calibrate_settings}")
        xas_data = calibrate_energy(xas_data, **calibrate_settings)
        # After re-calibrating, will need to redo pre-edge with new range
        do_pre_edge = True

    if do_rebin:
        print("Re-binning data")
        rebin_xafs(
            energy=xas_data.energy,
            mu=xas_data.mu,
            group=xas_data,
            **pre_edge_settings,
        )
        xas_data = xas_data.rebinned
        # After re-bin, will need to redo pre-edge
        do_pre_edge = True

    if do_pre_edge:
        pre_edge_with_defaults(xas_data, pre_edge_settings, ref_channel)

    if do_xftf:
        xftf_with_defaults(xas_data, xftf_settings)

    if plot_graph:
        plot_graphs(
            plot_path=f"plot/{path_key}.png",
            xas_data=xas_data,
            plot_keys=plot_graph,
        )

    xas_project = create_athena(f"prj/{path_key}.prj")
    xas_project.add_group(xas_data)
    if annotation:
        group = next(iter(xas_project.groups.values()))
        group.args["annotation"] = annotation
    xas_project.save()

    # Ensure that we do not run out of memory when running on large zips
    gc.collect()


def plot_graphs(
    plot_path: str,
    xas_data: Group,
    plot_keys: list,
) -> None:
    nrows = len(plot_keys)
    index = 1
    plt.figure(figsize=(6.4, nrows * 4.8))
    if "edge" in plot_keys:
        plt.subplot(nrows, 1, index)
        plt.plot(xas_data.energy, xas_data.pre_edge, "g", label="pre-edge")
        plt.plot(xas_data.energy, xas_data.post_edge, "r", label="post-edge")
        plt.plot(xas_data.energy, xas_data.mu, "b", label="fit data")
        if hasattr(xas_data, "mu_std"):
            plt.fill_between(
                x=xas_data.energy,
                y1=xas_data.mu - xas_data.mu_std,
                y2=xas_data.mu + xas_data.mu_std,
                alpha=0.2,
                label="standard deviation",
            )
        e0 = xas_data.e0
        plt.axvline(e0, color="m", label=f"edge energy = {e0}")
        plt.grid(color="r", linestyle=":", linewidth=1)
        plt.xlabel("Energy (eV)")
        plt.ylabel("x$\mu$(E)")  # noqa: W605
        plt.title("Pre-edge and post_edge fitting to $\mu$")  # noqa: W605
        plt.legend()
        index += 1

    if "flat" in plot_keys:
        plt.subplot(nrows, 1, index)
        plt.plot(xas_data.energy, xas_data.flat, label="flattened signal")
        if hasattr(xas_data, "mu_std"):
            mu_std_normalised = xas_data.mu_std / xas_data.edge_step
            plt.fill_between(
                x=xas_data.energy,
                y1=xas_data.flat - mu_std_normalised,
                y2=xas_data.flat + mu_std_normalised,
                alpha=0.2,
                label="standard deviation",
            )
            plt.legend()
        plt.grid(color="r", linestyle=":", linewidth=1)
        plt.xlabel("Energy (eV)")
        plt.ylabel("Flattened x$\mu$(E)")  # noqa: W605
        index += 1

    if "dmude" in plot_keys:
        plt.subplot(nrows, 1, index)
        plt.plot(xas_data.energy, xas_data.dmude)
        plt.grid(color="r", linestyle=":", linewidth=1)
        plt.xlabel("Energy (eV)")
        plt.ylabel("Derivative normalised to x$\mu$(E)")  # noqa: W605
        index += 1

    plt.tight_layout(rect=(0, 0, 0.88, 1))
    plt.savefig(plot_path, format="png")
    plt.close("all")


if __name__ == "__main__":
    # larch imports set this to an interactive backend, so need to change it
    matplotlib.use("Agg")

    dat_file = sys.argv[1]
    input_values = json.load(open(sys.argv[2], "r", encoding="utf-8"))
    merge_inputs = input_values["merge_inputs"]["merge_inputs"]
    format_inputs = input_values["merge_inputs"]["format"]
    if "is_zipped" in format_inputs:
        is_zipped = bool(format_inputs["is_zipped"]["is_zipped"])
    else:
        is_zipped = False

    extract_group = None
    if "extract_group" in format_inputs:
        extract_group = format_inputs["extract_group"]

    if "energy_column" not in format_inputs:
        energy_column = None
    else:
        energy_column = format_inputs["energy_column"]["energy_column"]
        if energy_column == "auto":
            energy_column = None
        elif energy_column == "other":
            energy_column_input = format_inputs["energy_column"]
            energy_column = energy_column_input["energy_column_text"]

    if "mu_column" not in format_inputs:
        mu_column = None
    else:
        mu_column = format_inputs["mu_column"]["mu_column"]
        if mu_column == "auto":
            mu_column = None
        elif mu_column == "other":
            mu_column = format_inputs["mu_column"]["mu_column_text"]

    reader = Reader(
        energy_column=energy_column,
        mu_column=mu_column,
        data_format=format_inputs["format"],
        extract_group=extract_group,
    )
    keyed_data = reader.load_data(
        dat_file=dat_file,
        merge_inputs=merge_inputs,
        is_zipped=is_zipped,
    )

    calibrate_items = input_values["processing"]["calibrate"].items()
    calibrate_settings = {k: v for k, v in calibrate_items if v is not None}
    do_calibrate = calibrate_settings.pop("calibrate") == "true"

    do_rebin = input_values["processing"].pop("rebin")

    pre_edge_items = input_values["processing"]["pre_edge"].items()
    pre_edge_settings = {k: v for k, v in pre_edge_items if v is not None}
    do_pre_edge = bool(pre_edge_settings.pop("pre_edge"))

    ref_channel = None
    if "ref_channel" in pre_edge_settings:
        ref_channel = pre_edge_settings.pop("ref_channel")

    xftf_items = input_values["processing"]["xftf"].items()
    xftf_settings = {k: v for k, v in xftf_items if v is not None}
    do_xftf = xftf_settings.pop("xftf") == "true"

    plot_graph = input_values["plot_graph"]
    annotation = input_values["annotation"]

    for key, group in keyed_data.items():
        main(
            group,
            do_calibrate=do_calibrate,
            calibrate_settings=calibrate_settings,
            do_rebin=do_rebin,
            do_pre_edge=do_pre_edge,
            pre_edge_settings=pre_edge_settings,
            ref_channel=ref_channel,
            do_xftf=do_xftf,
            xftf_settings=xftf_settings,
            plot_graph=plot_graph,
            annotation=annotation,
            path_key=key,
        )
