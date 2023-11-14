from typing import Iterable

from larch.io import extract_athenagroup, read_athena
from larch.io.athena_project import AthenaGroup
from larch.symboltable import Group
from larch.xafs import autobk, pre_edge, xftf


def get_group(athena_group: AthenaGroup, key: str = None) -> Group:
    if key is None:
        group_keys = list(athena_group._athena_groups.keys())
        key = group_keys[0]
    return extract_athenagroup(athena_group._athena_groups[key])


def read_group(dat_file: str, key: str = None, xftf_params: dict = None):
    athena_group = read_athena(dat_file)
    group = get_group(athena_group, key)
    bkg_parameters = group.athena_params.bkg
    print(group.athena_params.fft)
    print(group.athena_params.fft.__dict__)
    pre_edge(
        group,
        e0=bkg_parameters.e0,
        pre1=bkg_parameters.pre1,
        pre2=bkg_parameters.pre2,
        norm1=bkg_parameters.nor1,
        norm2=bkg_parameters.nor2,
        nnorm=bkg_parameters.nnorm,
        make_flat=bkg_parameters.flatten,
    )
    autobk(group)
    if xftf_params is None:
        xftf(group)
    else:
        print(xftf_params)
        xftf(group, **xftf_params)
        xftf_details = Group()
        setattr(xftf_details, "call_args", xftf_params)
        group.xftf_details = xftf_details
    return group


def read_groups(dat_files: "list[str]", key: str = None) -> Iterable[Group]:
    for dat_file in dat_files:
        yield read_group(dat_file=dat_file, key=key)
