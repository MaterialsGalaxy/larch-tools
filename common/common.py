from typing import Iterable

from larch.io import extract_athenagroup, read_athena
from larch.io.athena_project import AthenaGroup
from larch.symboltable import Group
from larch.xafs import autobk, pre_edge, xftf


def get_group(athena_group: AthenaGroup, key: str = None) -> Group:
    group_keys = list(athena_group._athena_groups.keys())
    if key is None:
        key = group_keys[0]
    else:
        key = key.replace("-", "_")

    try:
        return extract_athenagroup(athena_group._athena_groups[key])
    except KeyError as e:
        raise KeyError(f"{key} not in {group_keys}") from e


def read_all_groups(dat_file: str, key: str = None) -> "dict[str, Group]":
    # Cannot rely on do_ABC as _larch is None
    athena_group = read_athena(
        dat_file,
        do_preedge=False,
        do_bkg=False,
        do_fft=False,
    )
    all_groups = {}
    for key in athena_group._athena_groups.keys():
        print(f"\nExtracting group {key}")
        group = get_group(athena_group, key)
        pre_edge_with_defaults(group=group)
        xftf_with_defaults(group=group)
        all_groups[key] = group

    return all_groups


def read_group(dat_file: str, key: str = None):
    # Cannot rely on do_ABC as _larch is None
    athena_group = read_athena(
        dat_file,
        do_preedge=False,
        do_bkg=False,
        do_fft=False,
    )
    group = get_group(athena_group, key)
    pre_edge_with_defaults(group=group)
    xftf_with_defaults(group=group)
    return group


def pre_edge_with_defaults(group: Group, settings: dict = None):
    merged_settings = {}
    try:
        bkg_parameters = group.athena_params.bkg
    except AttributeError as e:
        print(f"Cannot load group.athena_params.bkg from group:\n{e}")
        bkg_parameters = None

    keys = (
        ("e0", "e0", None),
        ("pre1", "pre1", None),
        ("pre2", "pre2", None),
        ("norm1", "nor1", None),
        ("norm2", "nor2", None),
        ("nnorm", "nnorm", None),
        ("make_flat", "flatten", None),
        ("step", "step", None),
        # This cannot be read from file as it is not stored by Larch (0.9.71)
        # ("nvict", "nvict", None),
    )
    for key, parameters_key, default in keys:
        extract_attribute(
            merged_settings, key, bkg_parameters, parameters_key, default
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
        ("kmin", "kmin", 0),
        ("kmax", "kmax", 20),
        ("dk", "dk", 1),
        ("kweight", "kw", 2),
        ("kweight", "kweight", 2),
        ("window", "kwindow", "kaiser"),
    )
    for key, parameters_key, default in keys:
        extract_attribute(
            merged_settings, key, fft_parameters, parameters_key, default
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
    parameters_key: str,
    default: "str|int" = None,
):
    if parameters_group is not None:
        try:
            merged_settings[key] = getattr(parameters_group, parameters_key)
            return
        except AttributeError:
            pass

    if default is not None:
        merged_settings[key] = default


def read_groups(dat_files: "list[str]", key: str = None) -> Iterable[Group]:
    for dat_file in dat_files:
        yield read_group(dat_file=dat_file, key=key)
