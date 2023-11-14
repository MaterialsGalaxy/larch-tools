import gc
import json
import os
import re
import sys

from common import read_group

from larch.io import (
    create_athena,
    h5group,
    merge_groups,
    read_ascii,
    set_array_labels,
)
from larch.symboltable import Group
from larch.xafs import autobk, pre_edge, rebin_xafs, xftf

import matplotlib
import matplotlib.pyplot as plt

import numpy as np


class Reader:
    def __init__(
        self,
        energy_column: str,
        mu_column: str,
        xftf_params: dict,
        data_format: str,
        extract_group: str = None,
    ):
        self.energy_column = energy_column
        self.mu_column = mu_column
        self.xftf_params = xftf_params
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
                dat_files=dat_file, is_zipped=is_zipped
            )
            return {"out": out_group}
        else:
            return self.load_single_file(
                filepath=dat_file, is_zipped=is_zipped
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
                group = self.load_single_file(filepath)["out"]
                all_groups.append(group)

        return merge_groups(all_groups, xarray="energy", yarray="mu")

    def load_single_file(
        self,
        filepath: str,
        is_zipped: bool = False,
    ) -> "dict[str,Group]":
        if is_zipped:
            return self.load_zipped_files()

        print(f"Attempting to read from {filepath}")
        if self.data_format == "athena":
            group = read_group(filepath, self.extract_group, self.xftf_params)
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
        key_length = len(str(file_total))
        i = 0
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
                key = str(i).zfill(key_length)
                filepath = os.path.join(dirpath, filename)
                xas_data = self.load_single_file(filepath)
                keyed_data[key] = xas_data["out"]
                i += 1

        return keyed_data

    def rename_cols(self, xafs_group: Group) -> Group:
        labels = [label.lower() for label in xafs_group.array_labels]
        print(f"Read columns: {labels}")

        if "energy" in labels:
            print("'energy' present in column headers")
        elif self.energy_column is not None:
            if self.energy_column.lower() in labels:
                labels[labels.index(self.energy_column.lower())] = "energy"
            else:
                raise ValueError(f"{self.energy_column} not found in {labels}")
        else:
            for i, label in enumerate(labels):
                if label == "col1" or label.endswith("energy"):
                    labels[i] = "energy"
                    break

        if "mu" in labels:
            print("'mu' present in column headers")
        elif self.mu_column is not None:
            if self.mu_column.lower() in labels:
                labels[labels.index(self.mu_column.lower())] = "mu"
            else:
                raise ValueError(f"{self.mu_column} not found in {labels}")
        else:
            for i, label in enumerate(labels):
                if label in ["col2", "xmu", "lni0it", "ffi0"]:
                    labels[i] = "mu"
                    break

        if labels != xafs_group.array_labels:
            print(f"Renaming columns to: {labels}")
            return set_array_labels(xafs_group, labels)
        else:
            return xafs_group


def calibrate_energy(
    xafs_group: Group,
    energy_0: float,
    energy_min: float,
    energy_max: float,
    energy_format: str,
):
    if energy_0 is not None:
        print(f"Recalibrating energy edge from {xafs_group.e0} to {energy_0}")
        xafs_group.energy = xafs_group.energy + energy_0 - xafs_group.e0
        xafs_group.e0 = energy_0

    if not (energy_min or energy_max):
        return xafs_group

    if energy_min:
        if energy_format == "relative":
            energy_min += xafs_group.e0
        index_min = np.searchsorted(xafs_group.energy, energy_min)
    else:
        index_min = 0

    if energy_max:
        if energy_format == "relative":
            energy_max += xafs_group.e0
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
    input_values: dict,
    path_key: str = "out",
):
    energy_0 = input_values["variables"]["energy_0"]
    if energy_0 is None and hasattr(xas_data, "e0"):
        energy_0 = xas_data.e0

    energy_format = input_values["variables"]["energy_format"]
    pre1 = input_values["variables"]["pre1"]
    pre2 = input_values["variables"]["pre2"]
    pre1 = validate_pre(pre1, energy_0, energy_format)
    pre2 = validate_pre(pre2, energy_0, energy_format)

    pre_edge(
        energy=xas_data.energy,
        mu=xas_data.mu,
        group=xas_data,
        e0=energy_0,
        pre1=pre1,
        pre2=pre2,
    )

    energy_min = input_values["variables"]["energy_min"]
    energy_max = input_values["variables"]["energy_max"]
    xas_data = calibrate_energy(
        xas_data,
        energy_0,
        energy_min,
        energy_max,
        energy_format=energy_format,
    )

    if input_values["rebin"]:
        print(xas_data.energy, xas_data.mu)
        rebin_xafs(energy=xas_data.energy, mu=xas_data.mu, group=xas_data)
        xas_data = xas_data.rebinned
        pre_edge(energy=xas_data.energy, mu=xas_data.mu, group=xas_data)

    try:
        autobk(xas_data)
    except ValueError as e:
        raise ValueError(
            f"autobk failed with energy={xas_data.energy}, mu={xas_data.mu}.\n"
            "This may occur if the edge is not included in the above ranges."
        ) from e
    xftf(xas_data, **xftf_params)

    if input_values["plot_graph"]:
        plot_edge_fits(f"edge/{path_key}.png", xas_data)
        plot_flattened(f"flat/{path_key}.png", xas_data)
        plot_derivative(f"derivative/{path_key}.png", xas_data)

    xas_project = create_athena(f"prj/{path_key}.prj")
    xas_project.add_group(xas_data)
    if input_values["annotation"]:
        group = next(iter(xas_project.groups.values()))
        group.args["annotation"] = input_values["annotation"]
    xas_project.save()

    # Ensure that we do not run out of memory when running on large zips
    gc.collect()


def validate_pre(pre, energy_0, energy_format):
    if pre is not None and energy_format == "absolute":
        if energy_0 is None:
            raise ValueError(
                "Edge energy must be set manually or be present in the "
                "existing Athena project if using absolute format."
            )
        pre -= energy_0

    return pre


def plot_derivative(plot_path: str, xafs_group: Group):
    plt.figure()
    plt.plot(xafs_group.energy, xafs_group.dmude)
    plt.grid(color="r", linestyle=":", linewidth=1)
    plt.xlabel("Energy (eV)")
    plt.ylabel("Derivative normalised to x$\mu$(E)")  # noqa: W605
    plt.savefig(plot_path, format="png")
    plt.close("all")


def plot_edge_fits(plot_path: str, xafs_group: Group):
    plt.figure()
    plt.plot(xafs_group.energy, xafs_group.pre_edge, "g", label="pre-edge")
    plt.plot(xafs_group.energy, xafs_group.post_edge, "r", label="post-edge")
    plt.plot(xafs_group.energy, xafs_group.mu, "b", label="fit data")
    plt.grid(color="r", linestyle=":", linewidth=1)
    plt.xlabel("Energy (eV)")
    plt.ylabel("x$\mu$(E)")  # noqa: W605
    plt.title("pre-edge and post_edge fitting to $\mu$")  # noqa: W605
    plt.legend()
    plt.savefig(plot_path, format="png")
    plt.close("all")


def plot_flattened(plot_path: str, xafs_group: Group):
    plt.figure()
    plt.plot(xafs_group.energy, xafs_group.flat)
    plt.grid(color="r", linestyle=":", linewidth=1)
    plt.xlabel("Energy (eV)")
    plt.ylabel("normalised x$\mu$(E)")  # noqa: W605
    plt.savefig(plot_path, format="png")
    plt.close("all")


if __name__ == "__main__":
    # larch imports set this to an interactive backend, so need to change it
    matplotlib.use("Agg")

    dat_file = sys.argv[1]
    input_values = json.load(open(sys.argv[2], "r", encoding="utf-8"))
    merge_inputs = input_values["merge_inputs"]["merge_inputs"]
    data_format = input_values["merge_inputs"]["format"]["format"]
    if "is_zipped" in input_values["merge_inputs"]["format"]:
        is_zipped = bool(
            input_values["merge_inputs"]["format"]["is_zipped"]["is_zipped"]
        )
    else:
        is_zipped = False
    xftf_params = input_values["variables"]["xftf"]
    extract_group = None

    if "extract_group" in input_values["merge_inputs"]["format"]:
        extract_group = input_values["merge_inputs"]["format"]["extract_group"]

    energy_column = None
    mu_column = None
    if "energy_column" in input_values["merge_inputs"]["format"]:
        energy_column = input_values["merge_inputs"]["format"]["energy_column"]
    if "mu_column" in input_values["merge_inputs"]["format"]:
        mu_column = input_values["merge_inputs"]["format"]["mu_column"]

    reader = Reader(
        energy_column=energy_column,
        mu_column=mu_column,
        xftf_params=xftf_params,
        data_format=data_format,
        extract_group=extract_group,
    )
    keyed_data = reader.load_data(
        dat_file=dat_file,
        merge_inputs=merge_inputs,
        is_zipped=is_zipped,
    )
    for key, group in keyed_data.items():
        main(
            group,
            input_values=input_values,
            path_key=key,
        )
