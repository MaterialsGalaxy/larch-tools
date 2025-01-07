import gc
import json
import os
import sys

from common import (
    pre_edge_with_defaults,
    read_group,
    xftf_with_defaults,
)

from larch.io import create_athena
from larch.symboltable import Group
from larch.xafs import rebin_xafs

import matplotlib
import matplotlib.pyplot as plt

import numpy as np


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
            plot_path=f"plot/out.png",
            xas_data=xas_data,
            plot_keys=plot_graph,
        )

    xas_project = create_athena(f"prj/out.prj")
    xas_project.add_group(xas_data)
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

    group = read_group(dat_file)
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
    )
