import json
import sys

from larch.io import read_athena
from larch.math.lincombo_fitting import get_label, lincombo_fit
from larch.symboltable import Group
from larch.xafs import pre_edge

import matplotlib
import matplotlib.pyplot as plt


def plot(group_to_fit: Group, fit_group: Group):
    formatted_label = ""
    for label, weight in fit_group.weights.items():
        formatted_label += f"{label}: {weight * 100}%\n"

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


def extract_group(filepath: str, extract_label: str, plot_label: str) -> Group:
    component_group = read_athena(filepath)
    if extract_label:
        extracted_group = component_group._athena_groups[extract_label]
    else:
        if len(component_group._athena_groups) > 1:
            print(
                "WARNING: Extracting first Athena group by default, "
                "but project contains the following groups: "
                f"{component_group._athena_groups.keys()}.\n"
                "Consider specifying the appropriate label to ensure "
                "the correct group is used."
            )
        extracted_group = list(component_group._athena_groups.values())[0]

    if not hasattr(extracted_group, "norm"):
        pre_edge(
            energy=extracted_group.energy,
            mu=extracted_group.mu,
            group=extracted_group,
        )

    set_label(extracted_group, plot_label)

    return extracted_group


if __name__ == "__main__":
    # larch imports set this to an interactive backend, so need to change it
    matplotlib.use("Agg")
    prj_file = sys.argv[1]
    input_values = json.load(open(sys.argv[2], "r", encoding="utf-8"))

    group_to_fit = extract_group(prj_file, None, input_values["label"])

    component_groups = []
    for component in input_values["components"]:
        component_group = extract_group(
            component["component_file"], component["extract_group"], component["label"]
        )
        component_groups.append(component_group)

    fit_group = lincombo_fit(group_to_fit, component_groups)
    plot(group_to_fit, fit_group)
