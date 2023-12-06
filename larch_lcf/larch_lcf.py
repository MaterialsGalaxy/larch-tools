import json
import sys

from common import read_group

from larch.math.lincombo_fitting import get_label, lincombo_fit
from larch.symboltable import Group

import matplotlib
import matplotlib.pyplot as plt


def plot(
    group_to_fit: Group,
    fit_group: Group,
    x_limit_min: float,
    x_limit_max: float,
):
    formatted_label = ""
    for label, weight in fit_group.weights.items():
        formatted_label += f"{label}: {weight:.3%}\n"

    plt.figure()
    plt.plot(
        group_to_fit.energy,
        group_to_fit.norm,
        label=group_to_fit.filename,
        linewidth=4,
        color="blue",
    )
    plt.plot(
        fit_group.xdata,
        fit_group.ydata,
        label=formatted_label[:-1],
        linewidth=2,
        color="orange",
        linestyle="--",
    )
    plt.grid(color="black", linestyle=":", linewidth=1)  # show and format grid
    plt.xlim(x_limit_min, x_limit_max)
    plt.xlabel("Energy (eV)")
    plt.ylabel("normalised x$\mu$(E)")  # noqa: W605
    plt.legend()
    plt.savefig("plot.png", format="png")
    plt.close("all")


def set_label(component_group, label):
    if label is not None:
        component_group.filename = label
    else:
        component_group.filename = get_label(component_group)


if __name__ == "__main__":
    # larch imports set this to an interactive backend, so need to change it
    matplotlib.use("Agg")
    prj_file = sys.argv[1]
    input_values = json.load(open(sys.argv[2], "r", encoding="utf-8"))

    group_to_fit = read_group(prj_file)
    set_label(group_to_fit, input_values["label"])

    component_groups = []
    for component in input_values["components"]:
        component_group = read_group(component["component_file"])
        set_label(component_group, component["label"])
        component_groups.append(component_group)

    energy_min = input_values["energy_min"]
    energy_max = input_values["energy_max"]
    fit_group = lincombo_fit(
        group=group_to_fit,
        components=component_groups,
        xmin=energy_min,
        xmax=energy_max,
    )
    print(f"Goodness of fit (rfactor): {fit_group.rfactor:.6%}")

    x_limit_min = input_values["x_limit_min"]
    x_limit_max = input_values["x_limit_max"]
    plot(group_to_fit, fit_group, x_limit_min, x_limit_max)
