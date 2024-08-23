import csv
import faulthandler
import gc
import json
import os
import sys

from common import read_group, sorting_key

from larch.fitting import guess, param, param_group
from larch.symboltable import Group
from larch.xafs import (
    FeffPathGroup,
    FeffitDataSet,
    TransformGroup,
    feffit,
    feffit_report,
)

import matplotlib
import matplotlib.pyplot as plt

import numpy as np


def read_csv_data(input_file, id_field="id"):
    csv_data = {}
    try:
        with open(input_file, encoding="utf8") as csvfile:
            reader = csv.DictReader(csvfile, skipinitialspace=True)
            for row in reader:
                csv_data[int(row[id_field])] = row
    except FileNotFoundError:
        print("The specified file does not exist")
    return csv_data


def dict_to_gds(data_dict):
    dgs_group = param_group()
    for par_idx in data_dict:
        # gds file structure:
        gds_name = data_dict[par_idx]["name"]
        gds_val = 0.0
        gds_expr = ""
        try:
            gds_val = float(data_dict[par_idx]["value"])
        except ValueError:
            continue
        gds_expr = data_dict[par_idx]["expr"]
        gds_vary = (
            True
            if str(data_dict[par_idx]["vary"]).strip().capitalize() == "True"
            else False
        )
        one_par = None
        if gds_vary:
            # equivalent to a guess parameter in Demeter
            one_par = guess(
                name=gds_name, value=gds_val, vary=gds_vary, expr=gds_expr
            )
        else:
            # equivalent to a defined parameter in Demeter
            one_par = param(
                name=gds_name, value=gds_val, vary=gds_vary, expr=gds_expr
            )
        if one_par is not None:
            dgs_group.__setattr__(gds_name, one_par)
    return dgs_group


def plot_rmr(path: str, datasets: list, rmin, rmax):
    plt.figure()
    for i, dataset in enumerate(datasets):
        plt.subplot(len(datasets), 1, i + 1)
        data = dataset.data
        model = dataset.model
        plt.plot(data.r, data.chir_mag, color="b")
        plt.plot(data.r, data.chir_re, color="b", label="expt.")
        plt.plot(model.r, model.chir_mag, color="r")
        plt.plot(model.r, model.chir_re, color="r", label="fit")
        plt.ylabel(
            "Magnitude of Fourier Transform of "
            r"$k^2 \cdot \chi$/$\mathrm{\AA}^{-3}$"
        )
        plt.xlabel(r"Radial distance/$\mathrm{\AA}$")
        plt.axvspan(xmin=rmin, xmax=rmax, color="g", alpha=0.1)
        plt.legend()

    plt.savefig(path, format="png")
    plt.close("all")


def plot_chikr(path: str, datasets, rmin, rmax, kmin, kmax):
    fig = plt.figure(figsize=(16, 4))
    for i, dataset in enumerate(datasets):
        data = dataset.data
        model = dataset.model
        ax1 = fig.add_subplot(len(datasets), 2, 2*i + 1)
        ax2 = fig.add_subplot(len(datasets), 2, 2*i + 2)
        ax1.plot(data.k, data.chi * data.k**2, color="b", label="expt.")
        ax1.plot(model.k, model.chi * data.k**2, color="r", label="fit")
        ax1.set_xlabel(r"$k (\mathrm{\AA})^{-1}$")
        ax1.set_ylabel(r"$k^2$ $\chi (k)(\mathrm{\AA})^{-2}$")
        ax1.axvspan(xmin=kmin, xmax=kmax, color="g", alpha=0.1)
        ax1.legend()

        ax2.plot(data.r, data.chir_mag, color="b", label="expt.")
        ax2.plot(model.r, model.chir_mag, color="r", label="fit")
        ax2.set_xlim(0, 5)
        ax2.set_xlabel(r"$R(\mathrm{\AA})$")
        ax2.set_ylabel(r"$|\chi(R)|(\mathrm{\AA}^{-3})$")
        ax2.legend(loc="upper right")
        ax2.axvspan(xmin=rmin, xmax=rmax, color="g", alpha=0.1)

    fig.savefig(path, format="png")
    plt.close("all")


def read_gds(gds_file):
    gds_pars = read_csv_data(gds_file)
    dgs_group = dict_to_gds(gds_pars)
    return dgs_group


def read_selected_paths_list(file_name):
    sp_dict = read_csv_data(file_name)
    sp_list = []
    for path_dict in sp_dict.values():
        filename = path_dict["filename"]
        if not os.path.isfile(filename):
            raise FileNotFoundError(
                f"{filename} not found, check paths in the Selected Paths "
                "table match those in the zipped directory structure."
            )

        print(f"Reading selected path for file {filename}")
        new_path = FeffPathGroup(
            filename=filename,
            label=path_dict["label"],
            degen=path_dict["degen"] if path_dict["degen"] != "" else None,
            s02=path_dict["s02"],
            e0=path_dict["e0"],
            sigma2=path_dict["sigma2"],
            deltar=path_dict["deltar"],
        )
        sp_list.append(new_path)
    return sp_list


def run_fit(
        data_groups: list, gds, pathlist, fv, selected_path_ids: list = None
):
    # create the transform group (prepare the fit space).
    trans = TransformGroup(
        fitspace=fv["fitspace"],
        kmin=fv["kmin"],
        kmax=fv["kmax"],
        kweight=fv["kweight"],
        dk=fv["dk"],
        window=fv["window"],
        rmin=fv["rmin"],
        rmax=fv["rmax"],
    )

    datasets = []
    for i, data_group in enumerate(data_groups):
        if selected_path_ids:
            selected_paths = []
            for path_id in selected_path_ids[i]:
                selected_paths.append(pathlist[path_id - 1])

            dataset = FeffitDataSet(
                data=data_group, pathlist=selected_paths, transform=trans
            )

        else:
            dataset = FeffitDataSet(
                data=data_group, pathlist=pathlist, transform=trans
            )

        datasets.append(dataset)

    out = feffit(gds, datasets)
    return datasets, out


def main(
    prj_file: list,
    gds_file: str,
    sp_file: str,
    fit_vars: dict,
    plot_graph: bool,
    series_id: str = "",
) -> Group:
    report_path = f"report/fit_report{series_id}.txt"
    rmr_path = f"rmr/rmr{series_id}.png"
    chikr_path = f"chikr/chikr{series_id}.png"

    gds = read_gds(gds_file)
    pathlist = read_selected_paths_list(sp_file)

    # calc_with_defaults will hang indefinitely (>6 hours recorded) if the
    # data contains any NaNs - consider adding an early error here if this is
    # not fixed in Larch?
    selected_path_ids = []
    if isinstance(prj_file[0], dict):
        data_groups = []
        for dataset in prj_file:
            data_groups.append(read_group(dataset["prj_file"]))
            selected_path_ids.append([p["path_id"] for p in dataset["paths"]])
    else:
        data_groups = [read_group(p) for p in prj_file]

    print(f"Fitting project from file {[d.filename for d in data_groups]}")

    datasets, out = run_fit(
        data_groups,
        gds,
        pathlist,
        fit_vars,
        selected_path_ids=selected_path_ids,
    )

    fit_report = feffit_report(out)
    with open(report_path, "w") as fit_report_file:
        fit_report_file.write(fit_report)

    if plot_graph:
        plot_rmr(rmr_path, datasets, fit_vars["rmin"], fit_vars["rmax"])
        plot_chikr(
            chikr_path,
            datasets,
            fit_vars["rmin"],
            fit_vars["rmax"],
            fit_vars["kmin"],
            fit_vars["kmax"],
        )
    return out


def check_threshold(
    series_id: str,
    threshold: float,
    variable: str,
    value: float,
    early_stopping: bool = False,
):
    if abs(value) > threshold:
        if early_stopping:
            message = (
                "ERROR: Stopping series fit after project "
                f"{series_id} as {variable} > {threshold}"
            )
        else:
            message = (
                f"WARNING: Project {series_id} has {variable} > {threshold}"
            )

        print(message)
        return early_stopping

    return False


def series_execution(
    filepaths: "list[str]",
    gds_file: str,
    sp_file: str,
    fit_vars: dict,
    plot_graph: bool,
    report_criteria: "list[dict]",
    stop_on_error: bool,
) -> "list[list[str]]":
    report_criteria = input_values["execution"]["report_criteria"]
    id_length = len(str(len(filepaths)))
    stop = False
    rows = [[f"{c['variable']:>12s}" for c in report_criteria]]
    for series_index, series_file in enumerate(filepaths):
        series_id = str(series_index).zfill(id_length)
        try:
            out = main(
                [series_file],
                gds_file,
                sp_file,
                fit_vars,
                plot_graph,
                f"_{series_id}",
            )
        except ValueError as e:
            rows.append([np.NaN for _ in report_criteria])
            if stop_on_error:
                print(
                    f"ERROR: fitting failed for {series_id}"
                    f" due to following error, stopping:\n{e}"
                )
                break
            else:
                print(
                    f"WARNING: fitting failed for {series_id} due to following"
                    f" error, continuing to next project:\n{e}"
                )
                continue

        row = []
        for criterium in report_criteria:
            stop = parse_row(series_id, out, row, criterium) or stop
        rows.append(row)

        gc.collect()

        if stop:
            break

    return rows


def parse_row(series_id: str, group: Group, row: "list[str]", criterium: dict):
    action = criterium["action"]["action"]
    variable = criterium["variable"]
    try:
        value = group.__getattribute__(variable)
    except AttributeError:
        value = group.params[variable].value

    row.append(f"{value:>12f}")
    if action == "stop":
        return check_threshold(
            series_id,
            criterium["action"]["threshold"],
            variable,
            value,
            True,
        )
    elif action == "warn":
        return check_threshold(
            series_id,
            criterium["action"]["threshold"],
            variable,
            value,
            False,
        )

    return False


if __name__ == "__main__":
    faulthandler.enable()
    # larch imports set this to an interactive backend, so need to change it
    matplotlib.use("Agg")

    prj_file = sys.argv[1]
    gds_file = sys.argv[2]
    sp_file = sys.argv[3]
    input_values = json.load(open(sys.argv[4], "r", encoding="utf-8"))
    fit_vars = input_values["fit_vars"]
    plot_graph = input_values["plot_graph"]

    if input_values["execution"]["execution"] == "parallel":
        main([prj_file], gds_file, sp_file, fit_vars, plot_graph)
    elif input_values["execution"]["execution"] == "simultaneous":
        dataset_dicts = input_values["execution"]["simultaneous"]
        main(dataset_dicts, gds_file, sp_file, fit_vars, plot_graph)
    else:
        if os.path.isdir(prj_file):
            # Sort the unzipped directory, all filenames should be zero-padded
            filepaths = [
                os.path.join(prj_file, p) for p in os.listdir(prj_file)
            ]
            filepaths.sort(key=sorting_key)
        else:
            # DO NOT sort if we have multiple Galaxy datasets - the filenames
            # are arbitrary but should be in order
            filepaths = prj_file.split(",")

        rows = series_execution(
            filepaths,
            gds_file,
            sp_file,
            fit_vars,
            plot_graph,
            input_values["execution"]["report_criteria"],
            input_values["execution"]["stop_on_error"],
        )
        if len(rows[0]) > 0:
            with open("criteria_report.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerows(rows)
