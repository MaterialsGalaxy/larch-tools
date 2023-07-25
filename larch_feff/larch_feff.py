import json
import re
import shutil
import sys
from pathlib import Path

from larch.xafs.feffrunner import feff6l
from pymatgen.io.cif import CifParser
from pymatgen.io.feff import Atoms, Header, Potential


def get_path_labels(paths_file):
    is_meta = True
    count = 0
    a_path = {}
    all_paths = {}
    with open(paths_file) as datfile:
        dat_lines = datfile.readlines()
        for a_line in dat_lines:
            count += 1
            if re.match("-{15}", a_line.strip()) is not None:
                is_meta = False
            elif not is_meta:
                if re.match("\s*\d*\s{4}\d*\s{3}", a_line) is not None:
                    if a_path:
                        all_paths[a_path["index"]] = a_path
                    line_data = a_line.split()
                    a_path = {
                        "index": line_data[0],
                        "nleg": line_data[1],
                        "degeneracy": line_data[2],
                    }
                elif (
                    re.match("\s{6}x\s{11}y\s{5}", a_line) is None
                ):  # ignore the intermediate headings
                    line_data = a_line.split()
                    if "label" not in a_path:
                        a_path["label"] = line_data[4].replace("'", "")
                    else:
                        a_path["label"] += "." + line_data[4].replace("'", "")
    if a_path and "index" in a_path:
        all_paths[a_path["index"]] = a_path
    return all_paths


def main(structure_file: str, file_format: dict):
    crystal_f = Path(structure_file)
    feff_dir = "feff"
    feff_inp = "feff.inp"
    header_inp = "header"
    atoms_inp = "atoms"
    potential_inp = "potential"
    path = Path(feff_dir, feff_inp)
    path.parent.mkdir(parents=True, exist_ok=True)

    if file_format["format"] == "cif":
        print(f"Parsing {crystal_f.name} and saving to {path}")
        cif_parser = CifParser(crystal_f)
        structures = cif_parser.get_structures()
        length = len(structures)
        if length != 1:
            raise ValueError(
                f"Execpted single structure in cif file but found {length}"
            )
        try:
            aborsbing_atom = int(file_format["absorbing_atom"])
        except ValueError:
            # aborsbing_atom can be int or chemical symbol
            aborsbing_atom = file_format["absorbing_atom"]

        feff_header = Header(structures[0])
        potential = Potential(structures[0], aborsbing_atom)
        atoms = Atoms(structures[0], aborsbing_atom, file_format["radius"])
        # if len(atoms.struct.sites) < len(potential.):

        # print(atoms.as_dict())
        feff_header.write_file(header_inp)
        potential.write_file(potential_inp)
        atoms.write_file(atoms_inp)
        with open(path, "w") as feff_inp_file:
            with open(header_inp) as f:
                header_text = f.read()
                print(header_text)
                feff_inp_file.write(header_text + "\n")
            with open(potential_inp) as f:
                potential_text = f.readlines()
                print(*potential_text)
                feff_inp_file.writelines(potential_text + ["\n"])
            with open(atoms_inp) as f:
                atoms_text = f.readlines()
                print(*atoms_text)
                feff_inp_file.writelines(atoms_text + ["\n"])
        if len(atoms_text) <= len(potential_text):
            print(
                "WARNING: Every potential in the structure must be represented"
                " by at least one atom, consider increasing the radius"
            )
    else:
        print(f"Copying {crystal_f.name} to {path}")
        shutil.copy(crystal_f, path)

    feff6l(folder=feff_dir, feffinp=feff_inp)

    feff_files = "files.dat"
    input_file = Path(feff_dir, feff_files)
    # the .dat data is stored in fixed width strings
    comma_positions = [13, 21, 32, 41, 48, 61]
    paths_data = []
    # get the list of paths info to assign labels to paths
    try:
        paths_info = get_path_labels(Path(feff_dir, "paths.dat"))
    except FileNotFoundError as err:
        raise FileNotFoundError(
            "paths.dat does not exist, which implies FEFF failed to run"
        ) from err

    print("Reading from: " + str(input_file))
    with open(input_file) as datfile:
        # Read until we find the line at the end of the header
        line = datfile.readline()
        while not re.match("-+", line.strip()):
            line = datfile.readline()

        # Build headers
        line = datfile.readline()
        header = ""
        start = 0
        for end in comma_positions:
            header += line[start:end] + ","
            start = end
        header += f" {'label':30s}, {'select':6s}\n"
        paths_data.append(header)

        # Read data
        line = datfile.readline()
        while line:
            data = ""
            start = 0
            for end in comma_positions[:-1]:
                data += line[start:end] + ","
                start = end

            # Last column needs padding to align
            data += line[start:-1] + "     ,"

            # Add label and select column
            path_id = int(data[5:9])
            try:
                label = paths_info[str(path_id)]["label"] + f".{path_id}"
            except KeyError as err:
                msg = f"{path_id} not in {paths_info.keys()}"
                raise KeyError(msg) from err
            data += f" {label:30s}, {0:6d}\n"
            paths_data.append(data)
            line = datfile.readline()

    with open("out.csv", "w") as out:
        out.writelines(paths_data)


if __name__ == "__main__":
    structure_file = sys.argv[1]
    input_values = json.load(open(sys.argv[2], "r", encoding="utf-8"))
    main(structure_file, input_values["format"])
