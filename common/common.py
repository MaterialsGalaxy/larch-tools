import re
from typing import Iterable

from larch.io import extract_athenagroup, read_athena
from larch.io.athena_project import AthenaGroup
from larch.symboltable import Group
from larch.xafs import autobk, pre_edge, xftf


def get_group(athena_group: AthenaGroup, key: str = None) -> Group:
    group_keys = list(athena_group.keys())
    if key is None:
        key = group_keys[0]

    try:
        return extract_athenagroup(athena_group.groups[key])
    except KeyError as e:
        raise KeyError(f"{key} not in {group_keys}") from e


def read_all_groups(dat_file: str) -> "dict[str, Group]":
    # Cannot rely on do_ABC as _larch is None
    athena_group = read_athena(
        dat_file,
        do_preedge=False,
        do_bkg=False,
        do_fft=False,
    )
    all_groups = {}
    for key in athena_group.keys():
        print(f"\nExtracting group {key}")
        group = get_group(athena_group, key)
        pre_edge_with_defaults(group=group)
        xftf_with_defaults(group=group)
        all_groups[key] = group

    return all_groups


def read_group(dat_file: str, key: str = None):
    if key:
        match_ = key.replace(" ", "_").replace("-", "_").replace(".", "_")
    else:
        match_ = None

    # Cannot rely on do_ABC as _larch is None
    athena_group = read_athena(
        dat_file,
        match=match_,
        do_preedge=False,
        do_bkg=False,
        do_fft=False,
    )
    group = get_group(athena_group, match_)
    pre_edge_with_defaults(group=group)
    xftf_with_defaults(group=group)
    return group


def pre_edge_with_defaults(
    group: Group, settings: dict = None, ref_channel: str = None
):
    merged_settings = {}
    if ref_channel is not None:
        print(f"Performing pre-edge with reference channel {ref_channel}")
        ref = getattr(group, ref_channel.lower())
        group.e0 = None
        pre_edge(energy=group.energy, mu=ref, group=group)
        bkg_parameters = group.pre_edge_details
    else:
        try:
            bkg_parameters = group.athena_params.bkg
        except AttributeError as e:
            print(f"Cannot load group.athena_params.bkg from group:\n{e}")
            bkg_parameters = None

    keys = (
        ("e0", ("e0"), None),
        ("pre1", ("pre1"), None),
        ("pre2", ("pre2"), None),
        ("norm1", ("nor1"), None),
        ("norm2", ("nor2"), None),
        ("nnorm", ("nnorm"), None),
        ("make_flat", ("flatten"), None),
        ("step", ("step"), None),
        ("nvict", ("nvict"), None),
    )
    for key, parameter_keys, default in keys:
        extract_attribute(
            merged_settings=merged_settings,
            key=key,
            parameters_group=bkg_parameters,
            parameter_keys=parameter_keys,
            default=default,
        )

    if settings:
        for k, v in settings.items():
            if k == "nvict":
                print(
                    "WARNING: `nvict` can be used for pre-edge but is not "
                    "saved to file, so value used will not be accessible in "
                    "future operations using this Athena .prj"
                )
            merged_settings[k] = v

    print(f"Pre-edge normalization with {merged_settings}")
    try:
        pre_edge(group, **merged_settings)
    except Warning as e:
        raise Warning(
            "Unable to perform pre-edge fitting with:\n\n"
            f"energy:\n{group.energy}\n\nmu:{group.mu}\n\n"
            "Consider checking the correct columns have been extracted"
        ) from e
    autobk(group, pre_edge_kws=merged_settings)


def xftf_with_defaults(group: Group, settings: dict = None):
    merged_settings = {}
    try:
        fft_parameters = group.athena_params.fft
    except AttributeError as e:
        print(f"Cannot load group.athena_params.fft from group:\n{e}")
        fft_parameters = None

    keys = (
        ("kmin", ("kmin",), 0),
        ("kmax", ("kmax",), 20),
        ("dk", ("dk",), 1),
        ("kweight", ("kw", "kweight"), 2),
        ("window", ("kwindow",), "kaiser"),
    )
    for key, parameter_keys, default in keys:
        extract_attribute(
            merged_settings=merged_settings,
            key=key,
            parameters_group=fft_parameters,
            parameter_keys=parameter_keys,
            default=default,
        )

    if settings:
        for k, v in settings.items():
            merged_settings[k] = v

    print(f"XFTF with {merged_settings}")
    xftf(group, **merged_settings)
    xftf_details = Group()
    setattr(xftf_details, "call_args", merged_settings)
    group.xftf_details = xftf_details


def extract_attribute(
    merged_settings: dict,
    key: str,
    parameters_group: Group,
    parameter_keys: "tuple[str]",
    default: "str|int" = None,
):
    if parameters_group is not None:
        values = []
        for parameter_key in parameter_keys:
            try:
                values.append(getattr(parameters_group, parameter_key))
            except AttributeError:
                pass

        if len(values) > 1:
            print(
                f"WARNING: values {values} for for keys {parameter_keys}, "
                "using first entry"
            )

        if len(values) > 0:
            merged_settings[key] = values[0]
            return

    if default is not None:
        merged_settings[key] = default


def read_groups(dat_files: "list[str]", key: str = None) -> Iterable[Group]:
    for dat_file in dat_files:
        yield read_group(dat_file=dat_file, key=key)


def sorting_key(filename: str) -> str:
    return re.findall(r"\d+", filename)[-1]
