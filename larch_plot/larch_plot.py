import json
import sys

from common import read_groups

import matplotlib
import matplotlib.pyplot as plt

import numpy as np


AXIS_LABELS = {
    "norm": r"x$\mu$(E), normalised",
    "dmude": r"d(x$\mu$(E))/dE, normalised",
    "chir_mag": r"|$\chi$(r)|",
    "energy": "Energy (eV)",
    "distance": "r (ang)",
}


def main(dat_files: "list[str]", plot_settings: "list[dict]"):
    groups = list(read_groups(dat_files))

    for i, settings in enumerate(plot_settings):
        data_list = []
        e0_min = None
        e0_max = None
        x_variable = "energy"
        y_variable = settings["variable"]["variable"]
        x_min = settings["variable"]["energy_min"]
        x_max = settings["variable"]["energy_max"]
        plot_path = f"plots/{i}_{y_variable}.png"
        plt.figure()

        for group in groups:
            label = group.athena_params.annotation or group.athena_params.id
            if y_variable == "chir_mag":
                x_variable = "distance"
                x = group.r
                energy_format = None
            else:
                x = group.energy
                energy_format = settings["variable"]["energy_format"]
                if energy_format == "relative":
                    e0 = group.athena_params.bkg.e0
                    e0_min = find_relative_limit(e0_min, e0, min)
                    e0_max = find_relative_limit(e0_max, e0, max)

            y = getattr(group, y_variable)
            if x_min is None and x_max is None:
                plt.plot(x, y, label=label)
            else:
                data_list.append({"x": x, "y": y, "label": label})

        if y_variable != "chir_mag" and energy_format == "relative":
            if x_min is not None:
                x_min += e0_min
            if x_max is not None:
                x_max += e0_max

        if x_min is not None or x_max is not None:
            for data in data_list:
                index_min = None
                index_max = None
                x = data["x"]
                if x_min is not None:
                    index_min = max(np.searchsorted(x, x_min) - 1, 0)
                if x_max is not None:
                    index_max = min(np.searchsorted(x, x_max) + 1, len(x))
                plt.plot(
                    x[index_min:index_max],
                    data["y"][index_min:index_max],
                    label=data["label"],
                )

        plt.xlim(x_min, x_max)

        save_plot(x_variable, y_variable, plot_path)


def find_relative_limit(e0_min: "float|None", e0: float, function: callable):
    if e0_min is None:
        e0_min = e0
    else:
        e0_min = function(e0_min, e0)
    return e0_min


def save_plot(x_type: str, y_type: str, plot_path: str):
    plt.grid(color="r", linestyle=":", linewidth=1)
    plt.xlabel(AXIS_LABELS[x_type])
    plt.ylabel(AXIS_LABELS[y_type])
    plt.legend()
    plt.savefig(plot_path, format="png")
    plt.close("all")


if __name__ == "__main__":
    # larch imports set this to an interactive backend, so need to change it
    matplotlib.use("Agg")

    dat_files = sys.argv[1]
    input_values = json.load(open(sys.argv[2], "r", encoding="utf-8"))

    main(dat_files.split(","), input_values["plots"])
