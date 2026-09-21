#!/usr/bin/env python3

import argparse
import glob
import json
import os

import numpy as np
import dpdata

from dpgen.generator.run import data_system_fmt
from dpgen.util import expand_sys_str


def collect_data(
    target_folder,
    param_file,
    output,
    verbose=True,
    shuffle=True,
    merge=True,
    include_init_data=True,
    iter_output_prefix="sys.",
    discover_existing_iters=False,
):
    """Collect initial and iterative datasets from a DP-GEN job.

    ``include_init_data``, ``iter_output_prefix``, and
    ``discover_existing_iters`` preserve the historical behavior used by
    :mod:`dpgen.tools.collect_data` without duplicating the data loading and
    serialization implementation.
    """
    target_folder = os.path.abspath(target_folder)
    output = os.path.abspath(output)
    # goto input
    cwd = os.getcwd()
    os.chdir(target_folder)
    with open(param_file) as fp:
        jdata = json.load(fp)
    sys_configs_prefix = jdata.get("sys_configs_prefix", "")
    sys_configs = jdata.get("sys_configs", [])
    if verbose:
        max_str_len = max([len(str(ii)) for ii in sys_configs])
        max_form_len = 16
        ptr_fmt = "%%%ds %%%ds natoms %%6d nframes %%6d" % (  # noqa: UP031
            max_str_len + 5,
            max_form_len,
        )
    # init systems
    init_data = []
    init_data_prefix = jdata.get("init_data_prefix", "")
    init_data_sys = jdata.get("init_data_sys", []) if include_init_data else []
    for ii in init_data_sys:
        init_data.append(
            dpdata.LabeledSystem(os.path.join(init_data_prefix, ii), fmt="deepmd/npy")
        )
    # collect systems from iter dirs
    coll_data = {}
    numb_sys = len(sys_configs)
    if discover_existing_iters:
        # The deprecated collector operated on completed directories rather
        # than assuming one iteration per current model_devi_jobs entry.
        iters = sorted(glob.glob("iter.[0-9]*[0-9]"))
    else:
        model_devi_jobs = jdata.get("model_devi_jobs", {})
        numb_jobs = len(model_devi_jobs)
        iters = ["iter.%06d" % ii for ii in range(numb_jobs)]  # noqa: UP031
    # loop over iters to collect data
    for ii in range(len(iters)):
        iter_data = glob.glob(os.path.join(iters[ii], "02.fp", "data.[0-9]*[0-9]"))
        iter_data.sort()
        iter_data = sum([expand_sys_str(ii) for ii in iter_data], [])
        for jj in iter_data:
            sys = dpdata.LabeledSystem(jj, fmt="deepmd/npy")
            if merge:
                sys_str = sys.formula
            else:
                sys_str = os.path.basename(jj).split(".")[-1]
            if sys_str in coll_data.keys():
                coll_data[sys_str].append(sys)
            else:
                coll_data[sys_str] = sys
    # print information
    if verbose:
        for ii in range(len(init_data)):
            print(
                ptr_fmt
                % (
                    str(init_data_sys[ii]),
                    init_data[ii].formula,
                    init_data[ii].get_natoms(),
                    init_data[ii].get_nframes(),
                )
            )
        keys = list(coll_data.keys())
        keys.sort()
        for ii in keys:
            if merge:
                sys_str = ii
            else:
                sys_str = str(sys_configs[int(ii)])
            print(
                ptr_fmt
                % (
                    sys_str,
                    coll_data[ii].formula,
                    coll_data[ii].get_natoms(),
                    coll_data[ii].get_nframes(),
                )
            )
    # shuffle system data
    if shuffle:
        for kk in coll_data.keys():
            coll_data[kk].shuffle()
    # create output dir
    os.chdir(cwd)
    os.makedirs(output, exist_ok=True)
    # dump init data
    if include_init_data:
        for idx, ii in enumerate(init_data):
            out_dir = "init." + (data_system_fmt % idx)
            ii.to("deepmd/npy", os.path.join(output, out_dir))
    # dump iter data
    for kk in coll_data.keys():
        out_dir = f"{iter_output_prefix}{kk}"
        nframes = coll_data[kk].get_nframes()
        coll_data[kk].to("deepmd/npy", os.path.join(output, out_dir), set_size=nframes)
        # coll_data[kk].to('deepmd/npy', os.path.join(output, out_dir))


def write_mlu_system(system, out_dir, set_size=5000, shuffle=None):
    """Write a loaded system dict as DeepMD npy preserving f/aparam/uparam.

    Frames are chunked into set.000, set.001, ... (set_size frames each);
    optional deterministic shuffle. type.raw/type_map.raw come from the
    first merged contributor. Raises when uparam frame/atomic modes are
    mixed across contributors.
    """
    from dpgen.generator.lib.mlu_select import frame_fingerprint

    os.makedirs(out_dir, exist_ok=True)
    n = system["coord"].shape[0]
    order = np.arange(n)
    if shuffle is not None:
        rng = np.random.default_rng(shuffle)
        rng.shuffle(order)
    with open(os.path.join(out_dir, "type.raw"), "w") as fp:
        fp.write("".join(f"{t}\n" for t in system["atom_types"]))
    type_map = system.get("type_map") or ["Type%d" % t for t in system["atom_types"][:1]]
    with open(os.path.join(out_dir, "type_map.raw"), "w") as fp:
        fp.write("".join(f"{t}\n" for t in type_map))

    uparam = system["uparam"]
    if uparam is not None:
        uparam = np.asarray(uparam)

    nsets = max(1, int(np.ceil(n / set_size)))
    for si in range(nsets):
        sel = order[si * set_size : (si + 1) * set_size]
        set_dir = os.path.join(out_dir, "set.%03d" % si)
        os.makedirs(set_dir, exist_ok=True)
        np.save(os.path.join(set_dir, "box.npy"), system["box"][sel])
        np.save(os.path.join(set_dir, "coord.npy"), system["coord"][sel])
        for key in ("energy", "force", "virial", "fparam", "aparam"):
            if system[key] is not None:
                np.save(os.path.join(set_dir, f"{key}.npy"), np.asarray(system[key])[sel])
        if uparam is not None:
            np.save(os.path.join(set_dir, "uparam.npy"), uparam[sel])


def _concat_systems(systems, dedup=True):
    """Concatenate loaded system dicts (same natoms/type layout expected)."""
    from dpgen.generator.lib.mlu_select import frame_fingerprint

    out = None
    seen = set()
    n_dup = 0
    for sysd in systems:
        if sysd is None:
            continue
        if out is not None and sysd["natoms"] != out["natoms"]:
            raise RuntimeError(
                f"cannot merge {sysd['name']} into {out['name']}: natoms "
                f"{sysd['natoms']} != {out['natoms']}"
            )
        if out is None:
            out = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                   for k, v in sysd.items()}
            out["type_map"] = sysd.get("type_map")
            keep = np.arange(sysd["coord"].shape[0])
            if dedup:
                keep = np.array(
                    [
                        i
                        for i in range(sysd["coord"].shape[0])
                        if not (
                            frame_fingerprint(sysd["box"][i], sysd["coord"][i])
                            in seen
                            or seen.add(
                                frame_fingerprint(sysd["box"][i], sysd["coord"][i])
                            )
                        )
                    ],
                    dtype=int,
                )
                n_dup += sysd["coord"].shape[0] - len(keep)
            for k in ("box", "coord", "energy", "force", "virial", "fparam",
                      "aparam", "uparam"):
                if out[k] is not None:
                    out[k] = out[k][keep]
            continue
        n_add = sysd["coord"].shape[0]
        keep = np.arange(n_add)
        if dedup:
            keep = np.array(
                [
                    i
                    for i in range(n_add)
                    if not (
                        frame_fingerprint(sysd["box"][i], sysd["coord"][i])
                        in seen
                        or seen.add(
                            frame_fingerprint(sysd["box"][i], sysd["coord"][i])
                        )
                    )
                ],
                dtype=int,
            )
            n_dup += n_add - len(keep)
        for k in ("box", "coord", "energy", "force", "virial", "fparam",
                  "aparam", "uparam"):
            if sysd[k] is None:
                if out[k] is not None:
                    raise RuntimeError(
                        f"cannot merge {sysd['name']} into {out['name']}: "
                        f"{k} present in one contributor only"
                    )
                continue
            if out[k] is None:
                raise RuntimeError(
                    f"cannot merge {sysd['name']} into {out['name']}: "
                    f"{k} present in one contributor only"
                )
            out[k] = np.concatenate([out[k], sysd[k][keep]], axis=0)
    return out, n_dup


def collect_data_mlu(
    target_folder,
    param_file,
    output,
    verbose=True,
    shuffle=True,
    dedup=True,
    include_init_data=True,
    set_size=5000,
):
    """MLU-aware collection: preserves fparam/aparam/uparam end to end.

    Systems are grouped by source system index (not chemical formula) so the
    phase/magnetic grouping survives; frames are optionally shuffled and
    de-duplicated by structure fingerprint.
    """
    from dpgen.generator.lib.mlu_select import load_system

    target_folder = os.path.abspath(target_folder)
    output = os.path.abspath(output)
    cwd = os.getcwd()
    os.chdir(target_folder)
    with open(param_file) as fp:
        jdata = json.load(fp)

    contributors = {}  # out_name -> list of system dicts
    order = []
    if include_init_data:
        init_data_prefix = jdata.get("init_data_prefix", "")
        for idx, ii in enumerate(jdata.get("init_data_sys", [])):
            sysd = load_system(
                os.path.join(init_data_prefix, ii), name=ii, source="init"
            )
            if sysd is not None:
                name = "init." + (data_system_fmt % idx)
                contributors.setdefault(name, []).append(sysd)
                order.append(name)
    model_devi_jobs = jdata.get("model_devi_jobs", {})
    iters = sorted(glob.glob("iter.[0-9]*[0-9]"))
    for it in iters:
        for data_dir in sorted(glob.glob(os.path.join(it, "02.fp", "data.[0-9]*[0-9]"))):
            sys_idx = os.path.basename(data_dir).split(".")[-1]
            sysd = load_system(data_dir, name=data_dir, source=it)
            if sysd is not None:
                contributors.setdefault(f"sys.{sys_idx}", []).append(sysd)
    for name in contributors:
        if name not in order:
            order.append(name)

    report = {"systems": [], "n_duplicate_frames": 0}
    os.chdir(cwd)
    os.makedirs(output, exist_ok=True)
    for name in order:
        merged, n_dup = _concat_systems(contributors[name], dedup=dedup)
        report["n_duplicate_frames"] += n_dup
        if merged is None or merged["coord"].shape[0] == 0:
            continue
        write_mlu_system(merged, os.path.join(output, name), set_size=set_size,
                         shuffle=(42 if shuffle else None))
        entry = {
            "system": name,
            "nframes": int(merged["coord"].shape[0]),
            "natoms": int(merged["natoms"]),
            "has": {
                k: merged[k] is not None
                for k in ("energy", "force", "virial", "fparam", "aparam", "uparam")
            },
        }
        if merged["uparam"] is not None:
            up = np.asarray(merged["uparam"])
            entry["uparam_shape"] = list(up.shape[1:])
        report["systems"].append(entry)
        if verbose:
            print(
                f"{name:40s} natoms {entry['natoms']:6d} nframes {entry['nframes']:6d} "
                + " ".join(k for k, v in entry["has"].items() if v)
            )
    with open(os.path.join(output, "collect_report.json"), "w") as fp:
        json.dump(report, fp, indent=2)
    if verbose:
        print(f"dropped {report['n_duplicate_frames']} duplicate frames")
    return report


def gen_collect(args):
    if getattr(args, "mlu", False):
        collect_data_mlu(
            args.JOB_DIR,
            args.parameter,
            args.OUTPUT,
            verbose=args.verbose,
            shuffle=args.shuffle,
            dedup=not args.no_dedup,
        )
        return
    collect_data(
        args.JOB_DIR,
        args.parameter,
        args.OUTPUT,
        verbose=args.verbose,
        shuffle=args.shuffle,
        merge=args.merge,
    )


def _main():
    parser = argparse.ArgumentParser(description="Collect data from DP-GEN iterations")
    parser.add_argument("JOB_DIR", type=str, help="the directory of the DP-GEN job")
    parser.add_argument("OUTPUT", type=str, help="the output directory of data")
    parser.add_argument(
        "-p",
        "--parameter",
        type=str,
        default="param.json",
        help="the json file provides DP-GEN paramters, should be located in JOB_DIR",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print number of data in each system",
    )
    parser.add_argument(
        "-m",
        "--merge",
        action="store_true",
        help="merge the systems with the same chemical formula",
    )
    parser.add_argument(
        "-s", "--shuffle", action="store_true", help="shuffle the data systems"
    )
    parser.add_argument(
        "--mlu",
        action="store_true",
        help="MLU collection: preserve fparam/aparam/uparam, group by system index",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="(mlu) keep duplicate frames instead of fingerprint-deduplicating",
    )
    args = parser.parse_args()
    gen_collect(args)


if __name__ == "__main__":
    _main()
