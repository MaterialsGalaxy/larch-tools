import json
import sys

from common import read_groups

import matplotlib
import matplotlib.pyplot as plt


Y_LABELS = {
    "norm": r"x$\mu$(E), normalised",
    "dmude": r"d(x$\mu$(E))/dE, normalised",
}


def main(dat_files: "list[str]", plot_settings: "list[dict]"):
    groups = list(read_groups(dat_files))

    for i, settings in enumerate(plot_settings):
        e0_min = None
        e0_max = None
        energy_min = settings["energy_min"]
        energy_max = settings["energy_max"]
        plot_path = f"plots/{i}_{settings['variable']}.png"
        plt.figure()

        for group in groups:
            label = group.athena_params.annotation or group.athena_params.id
            x = group.energy
            y = getattr(group, settings["variable"])
            plt.plot(x, y, label=label)

            if settings["limits"] == "relative":
                e0 = group.athena_params.bkg.e0

                if e0_min is None:
                    e0_min = e0
                else:
                    e0_min = min(e0_min, e0)

                if e0_max is None:
                    e0_max = e0
                else:
                    e0_max = min(e0_max, e0)

        if settings["limits"] == "relative":
            if energy_min is not None:
                energy_min += e0_min
            if energy_max is not None:
                energy_max += e0_max

        plt.xlim(energy_min, energy_max)

        save_plot(settings["variable"], plot_path)


def save_plot(y_type: str, plot_path: str):
    plt.grid(color="r", linestyle=":", linewidth=1)
    plt.xlabel("Energy (eV)")
    plt.ylabel(Y_LABELS[y_type])
    plt.legend()
    plt.savefig(plot_path, format="png")
    plt.close("all")


if __name__ == "__main__":
    # larch imports set this to an interactive backend, so need to change it
    matplotlib.use("Agg")

    dat_files = sys.argv[1]
    input_values = json.load(open(sys.argv[2], "r", encoding="utf-8"))

    main(dat_files.split(","), input_values["plots"])
