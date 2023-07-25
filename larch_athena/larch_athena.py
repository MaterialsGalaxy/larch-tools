import gc
import json
import os
import re
import shutil
import sys
from zipfile import ZipFile

from larch.io import create_athena, h5group, merge_groups, read_ascii, set_array_labels
from larch.symboltable import Group
from larch.xafs import pre_edge, rebin_xafs

import matplotlib
import matplotlib.pyplot as plt

import numpy as np


def calibrate_energy(
    xafs_group: Group, energy_0: float, energy_min: float, energy_max: float
):
    if energy_0 is not None:
        xafs_group.energy = xafs_group.energy + energy_0 - xafs_group.e0
        xafs_group.e0 = energy_0

    if not (energy_min or energy_max):
        return xafs_group

    if energy_min:
        index_min = np.searchsorted(xafs_group.energy, energy_min)
    else:
        index_min = 0

    if energy_max:
        index_max = np.searchsorted(xafs_group.energy, energy_max)
    else:
        index_max = len(xafs_group.energy)

    xafs_group.dmude = xafs_group.dmude[index_min:index_max]
    xafs_group.pre_edge = xafs_group.pre_edge[index_min:index_max]
    xafs_group.post_edge = xafs_group.post_edge[index_min:index_max]
    xafs_group.flat = xafs_group.flat[index_min:index_max]
    xafs_group.energy = xafs_group.energy[index_min:index_max]
    xafs_group.mu = xafs_group.mu[index_min:index_max]
    return xafs_group


def load_ascii(dat_file):
    xas_data = read_ascii(dat_file)
    xas_data = rename_cols(xas_data)
    return xas_data


def load_h5(dat_file):
    h5_group = h5group(dat_file)
    energy = h5_group.entry1.instrument.qexafs_energy.qexafs_energy
    mu = h5_group.entry1.instrument.qexafs_counterTimer01.lnI0It
    xafs_group = Group(data=np.array([energy[:], mu[:]]))
    set_array_labels(xafs_group, ["energy", "mu"])
    return xafs_group


def load_single_file(filepath: str) -> Group:
    try:
        group = load_ascii(filepath)
    except TypeError:
        # Indicates this isn't plaintext, try h5
        group = load_h5(filepath)
    return group


def load_zipped_files() -> "dict[str, Group]":
    all_paths = list(os.walk("dat_files"))
    all_paths.sort(key=lambda x: x[0])
    file_total = sum([len(f) for _, _, f in all_paths])
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
            xas_data = load_single_file(filepath)
            keyed_data[key] = xas_data
            i += 1

    return keyed_data


def main(
    xas_data: Group,
    plot_graph: bool,
    rebin: bool,
    energy_0: float,
    energy_min: float,
    energy_max: float,
    prj_path: str = "prj/athena.prj",
    edge_plot_path: str = "edge/edge.png",
    flat_plot_path: str = "flat/flat.png",
    derivative_plot_path: str = "derivative/derivative.png",
):
    pre_edge(energy=xas_data.energy, mu=xas_data.mu, group=xas_data)
    xas_data = calibrate_energy(xas_data, energy_0, energy_min, energy_max)

    if rebin:
        rebin_xafs(energy=xas_data.energy, mu=xas_data.mu, group=xas_data)
        # xas_data = rename_cols(xas_data.rebinned, xas_data.array_labels)
        xas_data = xas_data.rebinned
        pre_edge(energy=xas_data.energy, mu=xas_data.mu, group=xas_data)

    if plot_graph:
        plot_edge_fits(edge_plot_path, xas_data)
        plot_flattened(flat_plot_path, xas_data)
        plot_derivative(derivative_plot_path, xas_data)

    xas_project = create_athena(prj_path)
    xas_project.add_group(xas_data)
    xas_project.save()

    # Ensure that we do not run out of memory when running on large zips
    gc.collect()


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


def rename_cols(xafs_group: Group, array_labels: "list[str]" = None) -> Group:
    labels = array_labels or xafs_group.array_labels
    new_labels = []
    for label in labels:
        if label == "col1":
            new_labels.append("energy")
        elif label == "col2" or label == "xmu":
            new_labels.append("mu")
        else:
            new_labels.append(label)

    if new_labels != labels:
        return set_array_labels(xafs_group, new_labels)
    else:
        return xafs_group


def sorting_key(filename: str) -> str:
    return re.findall(r"\d+", filename)[-1]


def merge_files(dat_files: str) -> Group:
    all_groups = []
    for filepath in dat_files.split(","):
        try:
            group = load_single_file(filepath)
            all_groups.append(group)
        except OSError:
            # Indicates it is actually a zip, so unzip it
            os.mkdir("dat_files")
            with ZipFile(filepath) as z:
                z.extractall("dat_files")
            keyed_groups = load_zipped_files()
            all_groups.extend(keyed_groups.values())
            shutil.rmtree("dat_files")

    return merge_groups(all_groups, xarray="energy", yarray="mu")


if __name__ == "__main__":
    # larch imports set this to an interactive backend, so need to change it
    matplotlib.use("Agg")

    dat_file = sys.argv[1]
    extension = sys.argv[2]
    input_values = json.load(open(sys.argv[3], "r", encoding="utf-8"))
    rebin = input_values["rebin"]
    energy_0 = input_values["energy_0"]
    energy_min = input_values["energy_min"]
    energy_max = input_values["energy_max"]
    plot_graph = input_values["plot_graph"]
    zip_outputs = input_values["zip_outputs"]
    merge_inputs = input_values["merge_inputs"]["merge_inputs"]

    if merge_inputs:
        merged_group = merge_files(dat_file)
        main(merged_group, plot_graph, rebin, energy_0, energy_min, energy_max)

    else:
        if extension == "zip":
            keyed_data = load_zipped_files()
            for key, group in keyed_data.items():
                main(
                    group,
                    plot_graph,
                    rebin,
                    energy_0,
                    energy_min,
                    energy_max,
                    f"prj/{key}.prj",
                    f"edge/{key}.png",
                    f"flat/{key}.png",
                    f"derivative/{key}.png",
                )

        elif extension == "h5":
            xas_data = load_h5(dat_file)
            main(xas_data, plot_graph, rebin, energy_0, energy_min, energy_max)

        else:
            xas_data = load_ascii(dat_file)
            main(xas_data, plot_graph, rebin, energy_0, energy_min, energy_max)
