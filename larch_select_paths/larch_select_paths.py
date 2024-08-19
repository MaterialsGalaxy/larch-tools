import csv
import json
import os
import re
import sys
from itertools import combinations
from zipfile import ZIP_DEFLATED, ZipFile


class CriteriaSelector:
    def __init__(self, criteria: "dict[str, int|float]"):
        self.max_number = criteria["max_number"]
        self.max_path_length = criteria["max_path_length"]
        self.min_amp_ratio = criteria["min_amplitude_ratio"]
        self.max_degeneracy = criteria["max_degeneracy"]
        self.path_count = 0

    def evaluate(self, path_id: int, row: "list[str]") -> (bool, None):
        if self.max_number and self.path_count >= self.max_number:
            print(f"Reject path: {self.max_number} paths already reached")
            return (False, None)

        r_effective = float(row[5].strip())
        if self.max_path_length and r_effective > self.max_path_length:
            print(f"Reject path: {r_effective} > {self.max_path_length}")
            return (False, None)

        amplitude_ratio = float(row[2].strip())
        if self.min_amp_ratio and (amplitude_ratio < self.min_amp_ratio):
            print(f"Reject path: {amplitude_ratio} < {self.min_amp_ratio}")
            return (False, None)

        degeneracy = float(row[3].strip())
        if self.max_degeneracy and degeneracy > self.max_degeneracy:
            print(f"Reject path: {degeneracy} > {self.max_degeneracy}")
            return (False, None)

        self.path_count += 1
        return (True, None)


class ManualSelector:
    def __init__(self, selection: dict):
        self.select_all = selection["selection"] == "all"
        self.paths = selection["paths"]
        self.path_values_ids = [path_value["id"] for path_value in self.paths]

    def evaluate(self, path_id: int, row: "list[str]") -> (bool, "None|dict"):
        if path_id in self.path_values_ids:
            return (True, self.paths[self.path_values_ids.index(path_id)])

        if self.select_all or int(row[-1]):
            return (True, None)

        return (False, None)


class GDSWriter:
    def __init__(self, default_variables: "dict[str, dict]"):
        self.default_properties = {
            "degen": {"name": "degen"},
            "s02": {"name": "s02"},
            "e0": {"name": "e0"},
            "deltar": {"name": "alpha*reff"},
            "sigma2": {"name": "sigma2"},
        }
        self.rows = [
            f"{'id':>4s}, {'name':>24s}, {'value':>5s}, {'expr':>4s}, "
            f"{'vary':>4s}\n"
        ]
        self.names = set()

        for property in self.default_properties:
            name = self.default_properties[property]["name"]
            value = default_variables[property]["value"]
            vary = default_variables[property]["vary"]
            is_common = default_variables[property]["is_common"]

            self.default_properties[property]["value"] = value
            self.default_properties[property]["vary"] = vary
            self.default_properties[property]["is_common"] = is_common

            if is_common:
                self.append_gds(name=name, value=value, vary=vary)

    def append_gds(
        self,
        name: str,
        value: float = 0.0,
        expr: str = None,
        vary: bool = True,
        label: str = "",
    ):
        """Append a single GDS variable to the list of rows, later to be
        written to file.

        Args:
            name (str): Name of the GDS variable.
            value (float, optional): Starting value for variable.
                Defaults to 0.
            expr (str, optional): Expression for setting the variable.
                Defaults to None.
            vary (bool, optional): Whether the variable is optimised during the
                fit. Defaults to True.
            label (str, optional): Label to keep variables for different FEFF
                directories distinct. Defaults to "".
        """
        formatted_name = name if (label is None) else label + name
        formatted_name = formatted_name.replace("*reff", "")
        if not expr:
            expr = "    "

        if formatted_name in self.names:
            raise ValueError(f"{formatted_name} already used as variable name")
        self.names.add(formatted_name)

        if value is not None:
            formatted_value = str(value)
        else:
            formatted_value = ""
        
        self.rows.append(
            f"{len(self.rows):4d}, {formatted_name:>24s}, "
            f"{formatted_value:>5s}, {expr:>4s}, {str(vary):>4s}\n"
        )

    def parse_gds(
        self,
        property_name: str,
        variable_name: str = None,
        path_variable: dict = None,
        directory_label: str = None,
        path_label: str = None,
    ) -> str:
        """Parse and append a row defining a GDS variable for a particular
        path.

        Args:
            property_name (str): The property to which the variable
                corresponds. Should be a key in `self.default_properties`.
            variable_name (str, optional): Custom name for this variable.
                Defaults to None.
            path_variable (dict, optional): Dictionary defining the GDS
                settings for this path's variable. Defaults to None.
            directory_label (str, optional): Label to indicate paths from a
                separate directory. Defaults to None.
            path_label (str, optional): Label indicating the atoms involved in
                this path. Defaults to None.

        Returns:
            str: Either `variable_name`, the name used as a default globally
                for this `property_name`, or an automatically generated unique
                name.
        """
        if variable_name:
            self.append_gds(
                name=variable_name,
                value=path_variable["value"],
                expr=path_variable["expr"],
                vary=path_variable["vary"],
            )
            return variable_name
        elif self.default_properties[property_name]["is_common"]:
            return self.default_properties[property_name]["name"]
        else:
            auto_name = self.default_properties[property_name]["name"]
            if directory_label:
                auto_name += f"_{directory_label}"
            if path_label:
                auto_name += f"_{path_label.lower().replace('.', '')}"

            self.append_gds(
                name=auto_name,
                value=self.default_properties[property_name]["value"],
                vary=self.default_properties[property_name]["vary"],
            )
            return auto_name

    def write(self):
        """Write GDS rows to file."""
        with open("gds.csv", "w") as out:
            out.writelines(self.rows)


class PathsWriter:
    def __init__(self, default_variables: "dict[str, dict]"):
        self.rows = [
            f"{'id':>4s}, {'filename':>24s}, {'label':>24s}, {'degen':>5s}, "
            f"{'s02':>3s}, {'e0':>4s}, {'sigma2':>24s}, {'deltar':>10s}\n"
        ]
        self.gds_writer = GDSWriter(default_variables=default_variables)
        self.all_combinations = [[0]]  # 0 corresponds to the header row

    def parse_feff_output(
        self,
        paths_file: str,
        selection: "dict[str, str|list]",
        directory_label: str = "",
    ):
        """Parse selected paths from CSV summary and define GDS variables.

        Args:
            paths_file (str): CSV summary filename.
            selection (dict[str, str|list]): Dictionary indicating which paths
                to select, and how to define their variables.
            directory_label (str, optional): Label to indicate paths from a
                separate directory. Defaults to "".
        """
        combinations_list = []
        if selection["selection"] in {"criteria", "combinations"}:
            selector = CriteriaSelector(selection)
        else:
            selector = ManualSelector(selection)

        selected_ids = self.select_rows(paths_file, directory_label, selector)

        if selection["selection"] == "combinations":
            min_number = selection["min_combination_size"]
            min_number = min(min_number, len(selected_ids))
            max_number = selection["max_combination_size"]
            if not max_number or max_number > len(selected_ids):
                max_number = len(selected_ids)

            for number_of_paths in range(min_number, max_number + 1):
                for combination in combinations(selected_ids, number_of_paths):
                    combinations_list.append(combination)

            new_combinations = len(combinations_list)
            print(
                f"{new_combinations} combinations for {directory_label}:\n"
                f"{combinations_list}"
            )
            old_combinations_len = len(self.all_combinations)
            self.all_combinations *= new_combinations
            for i, combination in enumerate(self.all_combinations):
                new_combinations = combinations_list[i // old_combinations_len]
                self.all_combinations[i] = combination + list(new_combinations)
        else:
            for combination in self.all_combinations:
                combination.extend(selected_ids)

    def select_rows(
        self,
        paths_file: str,
        directory_label: str,
        selector: "CriteriaSelector|ManualSelector",
    ) -> "list[int]":
        """Evaluate each row in turn to decide whether or not it should be
        included in the final output. Does not account for combinations.

        Args:
            paths_file (str): CSV summary filename.
            directory_label (str): Label to indicate paths from a separate
                directory.
            selector (CriteriaSelector|ManualSelector): Object to evaluate
                whether to select each path or not.

        Returns:
            list[int]: The ids of the selected rows.
        """
        row_ids = []
        with open(paths_file) as file:
            reader = csv.reader(file)
            for row in reader:
                id_match = re.search(r"\d+", row[0])
                if id_match:
                    path_id = int(id_match.group())
                    selected, path_value = selector.evaluate(
                        path_id=path_id,
                        row=row,
                    )
                    if selected:
                        filename = row[0].strip()
                        path_label = row[-2].strip()
                        row_id = self.parse_row(
                            directory_label, filename, path_label, path_value
                        )
                        row_ids.append(row_id)

        return row_ids

    def parse_row(
        self,
        directory_label: str,
        filename: str,
        path_label: str,
        path_value: "None|dict",
    ) -> int:
        """Parse row for GDS and path information.

        Args:
            directory_label (str): Label to indicate paths from a separate
                directory.
            filename (str): Filename for the FEFF path, extracted from row.
            path_label (str): Label for the FEFF path, extracted from row.
            path_value (None|dict): The values associated with the selected
                FEFF path. May be None in which case defaults are used.

        Returns:
            int: The id of the added row.
        """
        variables = {}
        if path_value is not None:
            for property in self.gds_writer.default_properties:
                variables[property] = self.gds_writer.parse_gds(
                    property_name=property,
                    variable_name=path_value[property]["name"],
                    path_variable=path_value[property],
                    directory_label=directory_label,
                    path_label=path_label,
                )
        else:
            for property in self.gds_writer.default_properties:
                variables[property] = self.gds_writer.parse_gds(
                    property_name=property,
                    directory_label=directory_label,
                    path_label=path_label,
                )

        return self.parse_selected_path(
            filename=filename,
            path_label=path_label,
            directory_label=directory_label,
            **variables,
        )

    def parse_selected_path(
        self,
        filename: str,
        path_label: str,
        directory_label: str = "",
        degen: str = "degen",
        s02: str = "s02",
        e0: str = "e0",
        sigma2: str = "sigma2",
        deltar: str = "alpha*reff",
    ) -> int:
        """Format and append row representing a selected FEFF path.

        Args:
            filename (str): Name of the underlying FEFF path file, without
                parent directory.
            path_label (str): Label indicating the atoms involved in this path.
            directory_label (str, optional): Label to indicate paths from a
                separate directory. Defaults to "".
            degen (str, optional): Path degeneracy variable name.
                Defaults to "degen".
            s02 (str, optional): Electron screening factor variable name.
                Defaults to "s02".
            e0 (str, optional): Energy shift variable name. Defaults to "e0".
            sigma2 (str, optional): Mean squared displacement variable name.
                Defaults to "sigma2".
            deltar (str, optional): Change in path length variable.
                Defaults to "alpha*reff".

        Returns:
            int: The id of the added row.
        """
        if directory_label:
            filename = os.path.join(directory_label, filename)
            label = f"{directory_label}.{path_label}"
        else:
            filename = os.path.join("feff", filename)
            label = path_label

        row_id = len(self.rows)
        self.rows.append(
            f"{row_id:>4d}, {filename:>24s}, {label:>24s}, {degen:>5s}, "
            f"{s02:>3s}, {e0:>4s}, {sigma2:>24s}, {deltar:>10s}\n"
        )

        return row_id

    def write(self):
        """Write selected path and GDS rows to file."""
        self.gds_writer.write()

        if len(self.all_combinations) == 1:
            with open("sp.csv", "w") as out:
                out.writelines(self.rows)
        else:
            for combination in self.all_combinations:
                filename = "_".join([str(c) for c in combination[1:]])
                print(f"Writing combination {filename}")
                with open(f"sp/{filename}.csv", "w") as out:
                    for row_id, row in enumerate(self.rows):
                        if row_id in combination:
                            out.write(row)


def main(input_values: dict):
    """Select paths and define GDS parameters.

    Args:
        input_values (dict): All input values from the Galaxy tool UI.

    Raises:
        ValueError: If a FEFF label is not unique.
    """
    default_variables = input_values["variables"]

    writer = PathsWriter(default_variables=default_variables)

    if len(input_values["feff_outputs"]) == 1:
        feff_output = input_values["feff_outputs"][0]
        writer.parse_feff_output(
            paths_file=feff_output["paths_file"],
            selection=feff_output["selection"],
        )
    else:
        zfill_length = len(str(len(input_values["feff_outputs"])))
        labels = set()
        with ZipFile("merged.zip", "x", ZIP_DEFLATED) as zipfile_out:
            for i, feff_output in enumerate(input_values["feff_outputs"]):
                label = feff_output["label"]
                if not label:
                    label = str(i + 1).zfill(zfill_length)
                if label in labels:
                    raise ValueError(f"Label '{label}' is not unique")
                labels.add(label)

                writer.parse_feff_output(
                    directory_label=label,
                    paths_file=feff_output["paths_file"],
                    selection=feff_output["selection"],
                )

                with ZipFile(feff_output["paths_zip"]) as z:
                    for zipinfo in z.infolist():
                        if zipinfo.filename != "feff/":
                            zipinfo.filename = zipinfo.filename[5:]
                            z.extract(member=zipinfo, path=label)
                            filename = os.path.join(label, zipinfo.filename)
                            zipfile_out.write(filename)

    writer.write()


if __name__ == "__main__":
    input_values = json.load(open(sys.argv[1], "r", encoding="utf-8"))
    main(input_values)
