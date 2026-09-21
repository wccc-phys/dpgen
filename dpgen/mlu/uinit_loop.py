#!/usr/bin/env python3
"""Remote driver for the self-consistent Hubbard U loop at fixed (T, P).

Self-contained (stdlib + numpy). Usage inside a task dir:

    python3 uinit_loop.py . 'pw.x command' 'hp.x command'

Iterates: pw.x scf(U_k) -> hp.x -> U_{k+1} until |U_{k+1}-U_k| < tol or
max_iter. Writes u_result.json ({T, P_bar, U, n_iter, converged}).
Only frame-mode U (averaged HUBBARD card) is used here — this loop produces
the region-level U_init bootstrap, not per-atom labels.
"""

import json
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlu_frame import check_scf_convergence, extract_u  # noqa: E402


def replace_hubbard_u(text, element, manifold, u):
    """Replace the numeric U of the single-species HUBBARD card."""
    pat = re.compile(
        r"(U\s+%s-%s\s+)(-?[0-9.]+)" % (re.escape(element), re.escape(manifold))
    )
    new_text, n = pat.subn(lambda m: m.group(1) + f"{u:.6f}", text)
    if n == 0:
        raise RuntimeError(f"HUBBARD U {element}-{manifold} line not found")
    return new_text


def inject_npt(text, press_kbar):
    lines = text.splitlines()
    out = []
    injected = False
    for ln in lines:
        out.append(ln)
        if not injected and ln.strip().upper().startswith("&CONTROL"):
            out.append("  cell_dynamics = 'pr',")
            out.append(f"  press = {press_kbar:.6f},")
            out.append("  cell_dofree = 'all',")
            injected = True
    return "\n".join(out) + "\n"


def main():
    task_dir, pw_cmd, hp_cmd = sys.argv[1], sys.argv[2], sys.argv[3]
    os.chdir(task_dir)
    with open("uinit.json") as fp:
        cfg = json.load(fp)
    element = cfg["element"]
    manifold = cfg["manifold"]
    tol = float(cfg["tol"])
    max_iter = int(cfg["max_iter"])
    npt = bool(cfg.get("npt"))
    press_kbar = float(cfg.get("press_kbar", 0.0))

    with open("pw.scf.in.000", errors="ignore") as fp:
        base_text = fp.read()

    u_prev = None
    converged = False
    n_iter = 0
    for it in range(max_iter):
        text = replace_hubbard_u(base_text, element, manifold, cfg["u0"] if it == 0 else u_prev)
        if npt and press_kbar > 0:
            text = inject_npt(text, press_kbar)
        scf_in = f"pw.scf.in.{it:03d}"
        with open(scf_in, "w") as fp:
            fp.write(text)
        subprocess.run(
            f"{pw_cmd} -in {scf_in} > pw.scf.out.{it:03d} 2>&1",
            shell=True,
        )
        ok, reason = check_scf_convergence(f"pw.scf.out.{it:03d}")
        if not ok:
            print(f"[uinit] iter {it}: scf failed ({reason})")
            sys.exit(1)
        if os.path.exists("pw.Hubbard_parameters.dat"):
            os.replace(
                "pw.Hubbard_parameters.dat",
                f"pw.Hubbard_parameters.dat.{it:03d}",
            )
        r = subprocess.run(f"{hp_cmd} -in pw.hp.in > pw.hp.out.{it:03d} 2>&1", shell=True)
        if not os.path.exists("pw.Hubbard_parameters.dat"):
            print(f"[uinit] iter {it}: hp.x produced no Hubbard_parameters.dat (rc={r.returncode})")
            sys.exit(1)
        u_new = float(extract_u("pw.Hubbard_parameters.dat", element, manifold).mean())
        n_iter = it + 1
        print(f"[uinit] iter {n_iter}: U = {u_new:.6f} eV (mean)")
        if u_prev is not None and abs(u_new - u_prev) < tol:
            converged = True
            u_prev = u_new
            break
        u_prev = u_new

    if u_prev is None:
        print("[uinit] no iteration completed")
        sys.exit(1)
    result = {
        "T": cfg["T"],
        "P_bar": cfg.get("P_bar", 0.0),
        "U": u_prev,
        "n_iter": n_iter,
        "converged": converged,
    }
    with open("u_result.json", "w") as fp:
        json.dump(result, fp, indent=2)
    print(f"[uinit] done: U={u_prev:.6f} eV converged={converged} iters={n_iter}")


if __name__ == "__main__":
    main()
