#!/usr/bin/env python3
"""Remote-side helper for the MLU QE labeling chain (self-contained).

Forwarded alongside each fp task and invoked from ``run.sh``. Only requires
python3 + numpy on the compute node — dpgen/deepmd are NOT needed there.

Subcommands
-----------
check-scf OUT
    Exit 0 when the pw.x output converged, 1 otherwise.
extract-u DAT [--element Fe] [--manifold 3d]
    Print the per-site U values (eV) from hp.x Hubbard_parameters.dat.
gen-scf-new DIR [--prefix pw] [--element Fe] [--manifold 3d]
    [--u-mode frame|atomic] [--uniform-fallback] [--starting-mag M]
    [--mass AMU] [--pseudo FILE]
    Read ``DIR/{prefix}.scf.in`` and ``DIR/{prefix}.Hubbard_parameters.dat``,
    then write ``DIR/{prefix}.scf_new.in`` with the computed U (frame mode:
    averaged single-species card; atomic mode: per-atom pseudo-species layout).

Deliberately mirrors dpgen.generator.lib.qe_hp (validated against the
mlu_model cluster runs) without importing dpgen.
"""

import argparse
import os
import re
import sys

import numpy as np

_SCF_FAIL_MARKERS = (
    "Maximum number of iterations",
    "convergence NOT achieved",
    "too many iterations",
    "electronic convergence not achieved",
)
_SCF_OK_MARKERS = (
    "End of self-consistent calculation",
    "convergence has been achieved",
)


def check_scf_convergence(path):
    if not os.path.isfile(path):
        return False, "scf_out_missing"
    with open(path, errors="ignore") as fp:
        text = fp.read()
    maxstep = any(m in text for m in _SCF_FAIL_MARKERS)
    converged = (not maxstep) and any(m in text for m in _SCF_OK_MARKERS)
    if not converged and re.search(r"^!.*total energy\s*=", text, re.MULTILINE):
        converged, maxstep = True, False
    if maxstep:
        return False, "scf_maxstep_not_converged"
    if not converged:
        return False, "scf_not_converged"
    if "JOB DONE." not in text:
        return False, "scf_no_job_done"
    return True, "ok"


def extract_u(dat, element, manifold):
    with open(dat, errors="ignore") as fp:
        text = fp.read().splitlines()
    values = []
    for line in text:
        parts = line.split()
        if len(parts) >= 8:
            if parts[2] == element and parts[5] == element and parts[6] == manifold:
                try:
                    values.append(float(parts[7]))
                except ValueError:
                    pass
    return np.array(values, dtype=float)


_QE_CARDS = (
    "CELL_PARAMETERS",
    "ATOMIC_SPECIES",
    "ATOMIC_POSITIONS",
    "K_POINTS",
    "HUBBARD",
    "CONSTRAINTS",
    "OCCUPATIONS",
    "ATOMIC_FORCES",
    "ATOMIC_VELOCITIES",
    "SOLVENTS",
)


def _is_card_header_or_namelist(line):
    s = line.strip().upper()
    if s == "" or s.startswith("/"):
        return True
    if s.startswith("&"):
        return True
    return s.split()[0] in _QE_CARDS


def _parse_atomic_positions(text):
    lines = text.splitlines()
    coords = []
    in_block = False
    for line in lines:
        s = line.strip()
        if not in_block:
            if s.upper().startswith("ATOMIC_POSITIONS"):
                in_block = True
            continue
        if s == "" or s.startswith("/") or s.upper().startswith(_QE_CARDS) or s.startswith("&"):
            break
        parts = s.split()
        if len(parts) >= 4:
            try:
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                break
        else:
            break
    if not coords:
        return None, None
    return len(coords), np.array(coords, dtype=float)


def _hubbard_card(u_arr, element, manifold, atom_labels=None):
    if atom_labels is None:
        return ["HUBBARD atomic", f"  U {element}-{manifold} {float(u_arr.mean()):.6f}"]
    lines = ["HUBBARD atomic"]
    for u, label in zip(u_arr, atom_labels):
        lines.append(f"  U {label}-{manifold} {float(u):.6f}")
    return lines


def _replace_hubbard(text, new_lines):
    lines = text.splitlines()
    out = []
    i = 0
    replaced = False
    while i < len(lines):
        s = lines[i].strip()
        if not replaced and s.upper().startswith("HUBBARD"):
            i += 1
            while i < len(lines):
                if lines[i].strip() == "" or _is_card_header_or_namelist(lines[i]):
                    break
                i += 1
            out.extend(new_lines)
            replaced = True
            continue
        out.append(lines[i])
        i += 1
    if not replaced:
        out.extend(new_lines)
    return "\n".join(out) + "\n"


def gen_scf_new(args):
    d = args.dir
    scf_in = os.path.join(d, f"{args.prefix}.scf.in")
    dat = os.path.join(d, f"{args.prefix}.Hubbard_parameters.dat")
    scf_new = os.path.join(d, f"{args.prefix}.scf_new.in")
    if not os.path.isfile(scf_in):
        print(f"[gen-scf-new] missing {scf_in}", file=sys.stderr)
        return 1
    if not os.path.isfile(dat):
        print(f"[gen-scf-new] missing {dat}", file=sys.stderr)
        return 1
    u = extract_u(dat, args.element, args.manifold)
    if u.size == 0:
        print(f"[gen-scf-new] no U rows for {args.element}-{args.manifold} in {dat}",
              file=sys.stderr)
        return 1

    with open(scf_in, errors="ignore") as fp:
        text = fp.read()
    natoms, coords = _parse_atomic_positions(text)
    if natoms is None:
        print("[gen-scf-new] cannot parse ATOMIC_POSITIONS", file=sys.stderr)
        return 1

    if args.u_mode == "atomic" and not args.uniform_fallback and u.size == natoms:
        # per-atom pseudo-species layout Fe1..FeN
        labels = [f"{args.element}{i + 1}" for i in range(natoms)]
        lines = text.splitlines()
        out = []
        i = 0
        in_system = False
        mag_injected = False
        while i < len(lines):
            s = lines[i].strip()
            up = s.upper()
            if up.startswith("&SYSTEM"):
                in_system = True
            elif up.startswith("&") and not up.startswith("&SYSTEM"):
                in_system = False
            if up.startswith("NTYP"):
                out.append(f"  ntyp = {natoms},")
                i += 1
                continue
            if in_system and re.match(r"STARTING_MAGNETIZATION\s*\(", up):
                i += 1
                continue
            if in_system and s == "/" and not mag_injected:
                for k in range(1, natoms + 1):
                    out.append(f"  starting_magnetization({k}) = {args.starting_mag},")
                mag_injected = True
                in_system = False
                out.append(lines[i])
                i += 1
                continue
            if up.startswith("ATOMIC_SPECIES"):
                i += 1
                while i < len(lines) and not _is_card_header_or_namelist(lines[i]):
                    i += 1
                out.append("ATOMIC_SPECIES")
                for label in labels:
                    out.append(f"  {label}  {args.mass}  {args.pseudo}")
                continue
            if up.startswith("ATOMIC_POSITIONS"):
                i += 1
                while i < len(lines) and not _is_card_header_or_namelist(lines[i]):
                    i += 1
                out.append("ATOMIC_POSITIONS crystal")
                for idx in range(natoms):
                    out.append(
                        f"  {labels[idx]}  {coords[idx, 0]:.8f}  "
                        f"{coords[idx, 1]:.8f}  {coords[idx, 2]:.8f}"
                    )
                continue
            if up.startswith("HUBBARD"):
                i += 1
                while i < len(lines):
                    if lines[i].strip() == "" or _is_card_header_or_namelist(lines[i]):
                        break
                    i += 1
                out.extend(_hubbard_card(u, args.element, args.manifold, labels))
                continue
            out.append(lines[i])
            i += 1
        new_text = "\n".join(out) + "\n"
    else:
        # frame mode: average U, single-species card replaced in place
        new_text = _replace_hubbard(
            text, _hubbard_card(u, args.element, args.manifold)
        )

    with open(scf_new, "w") as fp:
        fp.write(new_text)
    print(f"[gen-scf-new] wrote {scf_new} (mode={args.u_mode}, "
          f"U mean={u.mean():.4f} eV, min={u.min():.4f}, max={u.max():.4f})")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check-scf")
    p.add_argument("out")

    p = sub.add_parser("extract-u")
    p.add_argument("dat")
    p.add_argument("--element", default="Fe")
    p.add_argument("--manifold", default="3d")

    p = sub.add_parser("gen-scf-new")
    p.add_argument("dir")
    p.add_argument("--prefix", default="pw")
    p.add_argument("--element", default="Fe")
    p.add_argument("--manifold", default="3d")
    p.add_argument("--u-mode", choices=["frame", "atomic"], default="frame")
    p.add_argument("--uniform-fallback", action="store_true")
    p.add_argument("--starting-mag", type=float, default=2.2)
    p.add_argument("--mass", type=float, default=55.845)
    p.add_argument("--pseudo", default=None)

    args = parser.parse_args()
    if args.cmd == "check-scf":
        ok, reason = check_scf_convergence(args.out)
        print(f"[check-scf] {args.out}: {'OK' if ok else 'FAIL'} ({reason})")
        sys.exit(0 if ok else 1)
    if args.cmd == "extract-u":
        u = extract_u(args.dat, args.element, args.manifold)
        print(" ".join(f"{v:.6f}" for v in u))
        sys.exit(0)
    if args.cmd == "gen-scf-new":
        if args.pseudo is None:
            args.pseudo = f"{args.element}.UPF"
        sys.exit(gen_scf_new(args))


if __name__ == "__main__":
    main()
