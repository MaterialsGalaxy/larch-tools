import csv
import json
import os
import re
import sys
from zipfile import ZIP_DEFLATED, ZipFile


SP_DATA = [
    f"{'id':>4s}, {'filename':>24s}, {'label':>24s}, {'s02':>3s}, {'e0':>4s}, {'sigma2':>24s}, {'deltar':>10s}\n"
]
GDS_DATA = [f"{'id':>4s}, {'name':>24s}, {'value':>5s}, {'expr':>4s}, {'vary':>4s}\n"]
SP_ROW_ID = 1
GDS_ROW_ID = 1
AMP = "amp"
ENOT = "enot"
ALPHA = "alpha"
ALPHA_REFF = "alpha*reff"


def write_selected_path(
    row: "list[str]",
    s02: str = AMP,
    e0: str = ENOT,
    sigma2: str = "",
    deltar: str = ALPHA_REFF,
    directory_label: str = "",
):
    global SP_ROW_ID
    filename = row[0].strip()
    label = row[-2].strip()

    if directory_label:
        filename = os.path.join(directory_label, filename)
        label = f"{directory_label}.{label}"
    else:
        filename = os.path.join("feff", filename)

    if not sigma2:
        sigma2 = "s" + label.replace(".", "").lower()
        write_gds(sigma2)

    # If using the defaults, these are shared across all FEFF outputs and do
    # not need to be labelled. If they were modified, they will need to be
    # identified with this FEFF output.
    if s02 != AMP:
        s02 = label + s02
    if e0 != ENOT:
        e0 = label + e0
    if deltar != ALPHA_REFF:
        deltar = label + deltar

    SP_DATA.append(
        f"{SP_ROW_ID:>4d}, {filename:>24s}, {label:>24s}, {s02:>3s}, {e0:>4s}, {sigma2:>24s}, {deltar:>10s}\n"
    )
    SP_ROW_ID += 1


def write_gds(
    name: str,
    value: float = 0.003,
    expr: str = None,
    vary: bool = True,
    label: str = "",
):
    global GDS_ROW_ID
    if not expr:
        expr = "    "

    formatted_name = name if (label is None) else label + name

    GDS_DATA.append(
        f"{GDS_ROW_ID:4d}, {formatted_name:>24s}, {str(value):>5s}, {expr:>4s}, {str(vary):>4s}\n"
    )
    GDS_ROW_ID += 1


def write_feff_output(
    paths_file: str,
    select_all: bool,
    paths: list,
    gds: list,
    label: str = "",
    **kwargs,
):
    path_values_ids = [path_value["id"] for path_value in paths]
    gds_names = [gds_value["name"] for gds_value in gds]

    for gds_value in gds:
        write_gds(label=label, **gds_value)

    with open(paths_file) as file:
        reader = csv.reader(file)
        for row in reader:
            id_match = re.search(r"\d+", row[0])
            if id_match:
                path_id = int(id_match.group())
                if path_id in path_values_ids:
                    path_value = paths[path_values_ids.index(path_id)]
                    s02 = path_value["s02"]
                    e0 = path_value["e0"]
                    sigma2 = path_value["sigma2"]
                    if sigma2 and sigma2 not in gds_names:
                        write_gds(sigma2, label=label)
                    deltar = path_value["deltar"]
                    write_selected_path(
                        row, s02, e0, sigma2, deltar, directory_label=label
                    )
                elif select_all or int(row[-1]):
                    write_selected_path(row, directory_label=label)


def main(input_values):
    write_gds(name=AMP, **input_values["amp"])
    write_gds(name=ENOT, **input_values["enot"])
    write_gds(name=ALPHA, **input_values["alpha"])

    if len(input_values["feff_outputs"]) == 1:
        feff_output = input_values["feff_outputs"][0]
        write_feff_output(**feff_output)
    else:
        zfill_length = len(str(len(input_values["feff_outputs"])))
        labels = set()
        with ZipFile("merged.zip", "x", ZIP_DEFLATED) as zipfile_out:
            for i, feff_output in enumerate(input_values["feff_outputs"]):
                label = feff_output.pop("label") or str(i + 1).zfill(zfill_length)
                if label in labels:
                    raise ValueError(f"Label '{label}' is not unique")
                labels.add(label)

                write_feff_output(label=label, **feff_output)

                with ZipFile(feff_output["paths_zip"]) as z:
                    for zipinfo in z.infolist():
                        if zipinfo.filename != "feff/":
                            zipinfo.filename = zipinfo.filename[5:]
                            z.extract(member=zipinfo, path=label)
                            zipfile_out.write(os.path.join(label, zipinfo.filename))

    with open("sp.csv", "w") as out:
        out.writelines(SP_DATA)

    with open("gds.csv", "w") as out:
        out.writelines(GDS_DATA)


if __name__ == "__main__":
    input_values = json.load(open(sys.argv[1], "r", encoding="utf-8"))
    main(input_values)
