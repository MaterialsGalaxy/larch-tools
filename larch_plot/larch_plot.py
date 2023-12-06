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
        x_variable = "energy"
        y_variable = settings["variable"]["variable"]
        x_min = settings["variable"]["x_limit_min"]
        x_max = settings["variable"]["x_limit_max"]
        y_min = settings["variable"]["y_limit_min"]
        y_max = settings["variable"]["y_limit_max"]
        plot_path = f"plots/{i}_{y_variable}.png"
        plt.figure()

        for group in groups:
            params = group.athena_params
            label = params.annotation or params.file or params.id
            if y_variable == "chir_mag":
                x_variable = "distance"
                x = group.r
            else:
                x = group.energy

            y = getattr(group, y_variable)
            if x_min is None and x_max is None:
                plt.plot(x, y, label=label)
            else:
                data_list.append({"x": x, "y": y, "label": label})

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
        plt.ylim(y_min, y_max)

        save_plot(x_variable, y_variable, plot_path)


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
