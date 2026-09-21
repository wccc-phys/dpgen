"""CLI for the MLU init pipeline: U bootstrap, AIMD sampling, MSD picking.

    dpgen mlu uinit PARAM [MACHINE]   # generate self-consistent U(T,P) tasks
    dpgen mlu uinit-collect PARAM     # gather finished tasks into u_tp.json
    dpgen mlu aimd PARAM [MACHINE]    # generate pw.x MD coarse-sampling tasks
    dpgen mlu pick PARAM              # MSD analysis -> unlabeled init dataset

`uinit`/`aimd` only generate task directories plus run.sh scripts; submit
them through your scheduler (they mirror the fp stage conventions:
`bash run.sh` with output/ as the log). The actual submission can also be
driven by dpdispatcher machine files in a follow-up.
"""

import argparse
import json
import os

from dpgen import dlog
from dpgen.mlu.aimd import gen_aimd_tasks
from dpgen.mlu.msd import pick_aimd_frames
from dpgen.mlu.uinit import collect_uinit_results, gen_uinit_tasks
from dpgen.remote.decide_machine import convert_mdata


def _load(path):
    with open(path) as fp:
        return json.load(fp)


def _mdata(machine, task_types=("fp",)):
    if machine is None:
        return {}
    mdata = convert_mdata(_load(machine), list(task_types))
    return mdata


def gen_uinit(args):
    jdata = _load(args.PARAM)
    mdata = _mdata(args.MACHINE)
    tasks = gen_uinit_tasks(jdata, mdata)
    dlog.info(f"generated {len(tasks)} self-consistent U tasks under uinit/")


def gen_uinit_collect(args):
    jdata = _load(args.PARAM)
    table = collect_uinit_results(jdata)
    dlog.info(f"collected {len(table)} U(T,P) points -> uinit/u_tp.json")


def gen_aimd(args):
    jdata = _load(args.PARAM)
    mdata = _mdata(args.MACHINE)
    tasks = gen_aimd_tasks(jdata, mdata)
    dlog.info(f"generated {len(tasks)} AIMD tasks under aimd/")


def gen_pick(args):
    jdata = _load(args.PARAM)
    manifest = pick_aimd_frames(jdata)
    dlog.info(f"picked initial frames for {len(manifest)} trajectories")


def mlu_parser(subparsers):
    parser = subparsers.add_parser(
        "mlu", help="MLU (DP+U) init pipeline: U bootstrap, AIMD sampling, MSD picking."
    )
    mlusub = parser.add_subparsers(dest="mlu_command")
    p = mlusub.add_parser("uinit", help="generate self-consistent U(T,P) tasks")
    p.add_argument("PARAM", type=str, help="parameter file, json/yaml format")
    p.add_argument("MACHINE", type=str, default=None, nargs="?", help="machine file")
    p.set_defaults(func=gen_uinit)
    p = mlusub.add_parser("uinit-collect", help="gather u_result.json -> u_tp.json")
    p.add_argument("PARAM", type=str, help="parameter file")
    p.set_defaults(func=gen_uinit_collect)
    p = mlusub.add_parser("aimd", help="generate pw.x MD coarse-sampling tasks")
    p.add_argument("PARAM", type=str, help="parameter file, json/yaml format")
    p.add_argument("MACHINE", type=str, default=None, nargs="?", help="machine file")
    p.set_defaults(func=gen_aimd)
    p = mlusub.add_parser("pick", help="MSD analysis -> unlabeled init dataset")
    p.add_argument("PARAM", type=str, help="parameter file")
    p.set_defaults(func=gen_pick)
    return parser
