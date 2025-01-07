import json
import os
import re
import sys

from common import (
    pre_edge_with_defaults,
    read_all_groups,
    read_group,
    sorting_key,
    xftf_with_defaults,
)

from larch.io import (
    create_athena,
    h5group,
    merge_groups,
    read_ascii,
    set_array_labels,
)
from larch.symboltable import Group

import numpy as np


class Reader:
    def __init__(
        self,
        energy_column: str,
        mu_column: str,
        data_format: str,
        annotation: str = None,
        extract_group: "dict[str, str]" = None,
    ):
        self.energy_column = energy_column
        self.mu_column = mu_column
        self.data_format = data_format
        self.annotation = annotation
        self.extract_group = extract_group

    def load_data(
        self,
        dat_file: str,
        merge_inputs: bool,
        is_zipped: bool,
    ) -> None:
        if merge_inputs:
            self.merge_files(
                dat_files=dat_file,
                is_zipped=is_zipped,
            )
        else:
            self.load_single_file(
                filepath=dat_file,
                is_zipped=is_zipped,
                save=True,
            )

    def merge_files(
        self,
        dat_files: str,
        is_zipped: bool,
    ) -> None:
        if is_zipped:
            all_groups = list(self.load_zipped_files().values())
        else:
            all_groups = []
            for filepath in dat_files.split(","):
                keyed_data = self.load_single_file(filepath, save=False)
                for group in keyed_data.values():
                    all_groups.append(group)

        merged_group = merge_groups(all_groups, xarray="energy", yarray="mu")
        pre_edge_with_defaults(merged_group)
        xas_project = create_athena("prj/out.prj")
        xas_project.add_group(merged_group)
        if self.annotation is not None:
            merged_group.args["annotation"] = self.annotation
        xas_project.save()

    def load_single_file(
        self,
        filepath: str,
        is_zipped: bool = False,
        save: bool = True,
    ) -> dict:
        if is_zipped:
            keyed_data = self.load_zipped_files()
            if save:
                for name, group in keyed_data.items():
                    xas_project = create_athena(f"prj/{name}.prj")
                    xas_project.add_group(group)
                    group.args["annotation"] = name
                    xas_project.save()
            return keyed_data

        print(f"Attempting to read from {filepath}")
        if self.data_format == "athena":
            if self.extract_group["extract_group"] == "single":
                group = read_group(filepath, self.extract_group["group_name"])
                if save:
                    xas_project = create_athena("prj/out.prj")
                    xas_project.add_group(group)
                    if self.annotation is not None:
                        group.args["annotation"] = self.annotation
                    elif self.extract_group["annotation"]:
                        group.args["annotation"] = (
                            self.extract_group["annotation"]
                        )
                    xas_project.save()
                return {"out": group}

            elif self.extract_group["extract_group"] == "multiple":
                groups = {}
                for repeat in self.extract_group["multiple"]:
                    name = repeat["group_name"]
                    print(f"\nExtracting group {name}")
                    group = read_group(filepath, name)
                    groups[name] = group
                    if save:
                        xas_project = create_athena(f"prj/{name}.prj")
                        xas_project.add_group(group)
                        if self.annotation is not None:
                            group.args["annotation"] = self.annotation
                        elif repeat["annotation"]:
                            group.args["annotation"] = repeat["annotation"]
                        xas_project.save()
                return groups

            else:
                all_groups = read_all_groups(filepath)
                if save:
                    for name, group in all_groups.items():
                        if self.annotation is not None:
                            name = self.annotation
                        elif self.extract_group["regex_find"]:
                            pattern = self.extract_group["regex_find"]
                            repl = self.extract_group["regex_replace"]
                            name = re.sub(
                                pattern=pattern,
                                repl=repl or "",
                                string=name,
                            )
                        xas_project = create_athena(f"prj/{name}.prj")
                        xas_project.add_group(group)
                        group.args["annotation"] = name
                        xas_project.save()
                return all_groups

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
            pre_edge_with_defaults(group)
            xftf_with_defaults(group)
            if save:
                xas_project = create_athena("prj/out.prj")
                xas_project.add_group(group)
                if self.annotation is not None:
                    group.args["annotation"] = self.annotation
                xas_project.save()
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
        all_paths = list(os.walk("dat_files"))
        all_paths.sort(key=lambda x: x[0])
        file_total = sum([len(f) for _, _, f in all_paths])
        print(f"{file_total} files found")
        keyed_data = {}
        for dirpath, _, filenames in all_paths:
            if dirpath.endswith("__MACOSX"):
                print(f"Skipping {dirpath}")
                continue

            try:
                filenames.sort(key=sorting_key)
            except IndexError as e:
                print(
                    "WARNING: Unable to sort files numerically, "
                    f"defaulting to sorting alphabetically:\n{e}"
                )
                filenames.sort()

            for filename in filenames:
                if len(all_paths) > 1:
                    key = f"{dirpath.replace('/', '_')}_{filename}"
                else:
                    key = filename
                filepath = os.path.join(dirpath, filename)
                xas_data = self.load_single_file(filepath, save=False)
                keyed_data[key] = xas_data["out"]

        return keyed_data

    def rename_cols(self, xafs_group: Group) -> Group:
        labels = [label.lower() for label in xafs_group.array_labels]
        print(f"Read columns: {labels}")

        if "energy" in labels:
            print("'energy' present in column headers")
        elif self.energy_column:
            if self.energy_column.lower() in labels:
                labels[labels.index(self.energy_column.lower())] = "energy"
            else:
                raise ValueError(f"{self.energy_column} not found in {labels}")
        else:
            for i, label in enumerate(labels):
                if label in ("col1", "ef") or label.endswith("energy"):
                    labels[i] = "energy"
                    break

        if "mu" in labels:
            print("'mu' present in column headers")
        elif self.mu_column:
            if self.mu_column.lower() in labels:
                labels[labels.index(self.mu_column.lower())] = "mu"
            else:
                raise ValueError(f"{self.mu_column} not found in {labels}")
        else:
            for i, label in enumerate(labels):
                if label in ["col2", "xmu", "lni0it", "ffi0", "ff/i1"]:
                    labels[i] = "mu"
                    break

        if labels != xafs_group.array_labels:
            print(f"Renaming columns to: {labels}")
            return set_array_labels(xafs_group, labels)
        else:
            return xafs_group


if __name__ == "__main__":
    dat_file = sys.argv[1]
    input_values = json.load(open(sys.argv[2], "r", encoding="utf-8"))

    annotation = None
    is_zipped = False
    merge_inputs = input_values["merge_inputs"]["merge_inputs"]
    format_inputs = input_values["merge_inputs"]["format"]
    if "annotation" in format_inputs:
        annotation = format_inputs["annotation"]
    if "is_zipped" in format_inputs:
        is_zipped = bool(format_inputs["is_zipped"]["is_zipped"])
        if "annotation" in format_inputs["is_zipped"]:
            annotation = format_inputs["is_zipped"]["annotation"]

    extract_group = None
    if "extract_group" in format_inputs:
        extract_group = format_inputs["extract_group"]

    energy_column = None
    if "energy_column" in format_inputs:
        energy_column_dict = format_inputs["energy_column"]
        if energy_column_dict["energy_column"] == "other":
            energy_column = energy_column_dict["energy_column_text"]
        elif energy_column_dict["energy_column"] != "auto":
            energy_column = energy_column_dict["energy_column"]

    mu_column = None
    if "mu_column" in format_inputs:
        mu_column_dict = format_inputs["mu_column"]
        if mu_column_dict["mu_column"] == "other":
            mu_column = mu_column_dict["mu_column_text"]
        elif mu_column_dict["mu_column"] != "auto":
            mu_column = mu_column_dict["mu_column"]

    reader = Reader(
        energy_column=energy_column,
        mu_column=mu_column,
        data_format=format_inputs["format"],
        annotation=annotation,
        extract_group=extract_group,
    )
    reader.load_data(
        dat_file=dat_file,
        merge_inputs=merge_inputs,
        is_zipped=is_zipped,
    )
