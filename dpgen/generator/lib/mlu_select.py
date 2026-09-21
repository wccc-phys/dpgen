"""Committee descriptor-union selection for the MLU workflow (M5).

Productionized port of the user's validated ``select_configs.py`` pipeline
(test0): each committee model independently computes descriptors over the
full data pool (initial data + all previous iterations' labeled data + the
current exploration frames), a per-system convergence loop finds the minimal
representative fraction for that model, and the union over models — minus
already-labeled frames — becomes the new fp candidate set. Magnetic and
non-magnetic pools are selected independently so the minority phase is not
swallowed. Model deviation is deliberately NOT used for selection (it is
recorded for monitoring only).

Data loading bypasses dpdata so fparam/aparam/uparam survive every hop.
"""

import glob
import hashlib
import json
import os

import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

FP_TOL = 6  # fingerprint rounding digits


# ============================================================
# System loading (multi-set, f/a/uparam aware)
# ============================================================


def _load_npy(path):
    if os.path.isfile(path):
        return np.load(path)
    return None


def load_system(path, name=None, magnetic=False, source="init", origin=None):
    """Load a DeepMD npy system (all set.* dirs merged in order).

    Returns a dict or None when the directory has no sets. Fields are None
    when the corresponding npy is absent (e.g. unlabeled exploration frames
    have no energy/force/virial/uparam).
    """
    set_dirs = sorted(glob.glob(os.path.join(path, "set.*")))
    if not set_dirs:
        return None
    parts = {k: [] for k in ("box", "coord", "energy", "force", "virial",
                              "fparam", "aparam", "uparam")}
    for sd in set_dirs:
        for k in parts:
            arr = _load_npy(os.path.join(sd, f"{k}.npy"))
            if arr is not None:
                parts[k].append(arr)
    box = np.concatenate(parts["box"], axis=0)
    coord = np.concatenate(parts["coord"], axis=0)
    natoms = coord.shape[1] // 3

    def _join(key, validator=None):
        n_have = len(parts[key])
        if n_have == 0:
            return None
        if n_have != len(set_dirs):
            raise RuntimeError(
                f"inconsistent system {path}: {key}.npy present in "
                f"{n_have}/{len(set_dirs)} sets"
            )
        arr = np.concatenate(parts[key], axis=0)
        return arr if validator is None or validator(arr) else arr

    type_raw = os.path.join(path, "type.raw")
    atom_types = (
        np.loadtxt(type_raw, dtype=int)
        if os.path.isfile(type_raw)
        else np.zeros(natoms, dtype=int)
    )
    return {
        "name": name or os.path.basename(os.path.normpath(path)),
        "path": path,
        "natoms": natoms,
        "atom_types": atom_types,
        "box": box,
        "coord": coord,
        "energy": _join("energy"),
        "force": _join("force"),
        "virial": _join("virial"),
        "fparam": _join("fparam"),
        "aparam": _join("aparam"),
        "uparam": _join("uparam"),
        "magnetic": magnetic,
        "source": source,
        "origin": origin,  # for exploration frames: the modd task dir
        "frame_idx": None,
    }


def load_modd_frames(modd_task, type_map, magnetic=False):
    """Load all frames of one model_devi task's trajectory as a system dict.

    source='modd'; origin is the task dir so selected frames can be turned
    into fp candidates (task, frame).
    """
    import dpdata

    merged = glob.glob(os.path.join(modd_task, "all.lammpstrj"))
    traj_files = merged or sorted(
        glob.glob(os.path.join(modd_task, "traj", "*.lammpstrj"))
    )
    if not traj_files:
        return None
    sys_ = dpdata.System(traj_files[0], fmt="lammps/dump", type_map=type_map)
    return {
        "name": os.path.basename(modd_task),
        "path": modd_task,
        "natoms": sys_.get_natoms(),
        "atom_types": np.array(sys_["atom_types"], dtype=int),
        "box": np.array(sys_["cells"]).reshape(len(sys_), 9),
        "coord": np.array(sys_["coords"]).reshape(len(sys_), -1),
        "energy": None,
        "force": None,
        "virial": None,
        "fparam": None,
        "aparam": None,
        "uparam": None,
        "magnetic": magnetic,
        "source": "modd",
        "origin": modd_task,
        "frame_idx": np.arange(len(sys_)),
    }


def frame_fingerprint(box9, coord_flat, ndigits=FP_TOL):
    """Structure fingerprint for already-labeled subtraction / dedup."""
    h = hashlib.md5()
    h.update(np.asarray(box9, dtype=float).round(ndigits).tobytes())
    h.update(np.asarray(coord_flat, dtype=float).round(ndigits).tobytes())
    return h.hexdigest()


def system_fingerprints(system):
    n = system["coord"].shape[0]
    return {
        frame_fingerprint(system["box"][i], system["coord"][i]) for i in range(n)
    }


# ============================================================
# Descriptors (per committee model)
# ============================================================


def compute_descriptors(system, model_path):
    """Frame-mean DP descriptors from one committee model.

    Passes fparam/aparam/uparam when both the model and the data provide
    them (the deepmd-kit fork's DeepPot.eval accepts uparam).
    """
    from deepmd.infer import DeepPot

    dp = DeepPot(model_path)
    nframes = system["coord"].shape[0]
    coord = np.asarray(system["coord"], dtype=np.float64).reshape(nframes, -1, 3)
    box = np.asarray(system["box"], dtype=np.float64).reshape(nframes, 3, 3)
    atypes = np.asarray(system["atom_types"], dtype=int)

    def _prep(arr, dim, atomic=False):
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=np.float64)
        if atomic:
            return arr.reshape(nframes, system["natoms"], -1)
        return arr.reshape(nframes, dim)

    fparam = _prep(system["fparam"], 1)
    aparam = _prep(system["aparam"], 1, atomic=True)
    uparam = system["uparam"]
    if uparam is not None:
        uparam = np.asarray(uparam, dtype=np.float64)
        if uparam.ndim == 2 and uparam.shape[1] == 1:
            uparam = uparam  # frame mode (nframes, 1)
        else:
            uparam = uparam.reshape(nframes, system["natoms"])  # atomic mode
    kwargs = {}
    if fparam is not None and dp.get_dim_fparam() > 0:
        kwargs["fparam"] = fparam
    if aparam is not None and dp.get_dim_aparam() > 0:
        kwargs["aparam"] = aparam
    if getattr(dp, "get_dim_uparam", lambda: 0)() > 0 and uparam is not None:
        kwargs["uparam"] = uparam
    desc = dp.eval_descriptor(coord, box, atypes, **kwargs)
    desc = np.asarray(desc)
    if desc.ndim == 3:  # (nframes, natoms, dim)
        return desc.mean(axis=1)
    if desc.ndim == 2 and desc.shape[0] == nframes * system["natoms"]:
        return desc.reshape(nframes, system["natoms"], -1).mean(axis=1)
    return desc


# ============================================================
# Selection core (ported from select_configs.py, simplified)
# ============================================================


def _fit_reference(features):
    """Global reference on the group's full feature set: scaler+PCA(2)."""
    scaler = StandardScaler().fit(features)
    pca = PCA(n_components=2, random_state=0).fit(scaler.transform(features))
    pcs = pca.transform(scaler.transform(features))
    try:
        hull = ConvexHull(pcs)
    except Exception:  # noqa: BLE001  degenerate (<3 distinct points)
        hull = None
    return {"scaler": scaler, "pca": pca, "pcs": pcs, "hull": hull}


def _project(ref, features):
    return ref["pca"].transform(ref["scaler"].transform(features))


def _hull_contains(hull, pts):
    """Boolean mask: pts inside the convex hull."""
    if hull is None:
        return np.ones(len(pts), dtype=bool)
    eq = hull.equations  # (nf, ndim+1): normals | offset, normals . x + offset <= 0
    return np.all(
        pts @ eq[:, :-1].T + eq[:, -1] <= 1e-9, axis=1
    )


def _grid_coverage(pcs_full, pcs_sel, ngrid=10):
    """Fraction of full-set occupied grid cells also occupied by selection."""
    lo = pcs_full.min(axis=0)
    hi = pcs_full.max(axis=0)
    rng_ = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    cells_full = set(
        map(
            tuple,
            np.floor((pcs_full - lo) / rng_ * ngrid).astype(int).tolist(),
        )
    )
    cells_sel = set(
        map(
            tuple,
            np.floor((pcs_sel - lo) / rng_ * ngrid).astype(int).tolist(),
        )
    )
    if not cells_full:
        return 1.0
    return len(cells_full & cells_sel) / len(cells_full)


def _kde_js(pcs_full, pcs_sel, ngrid=20):
    """JS divergence between 2D histograms of full and selected PC points."""
    lo = pcs_full.min(axis=0)
    hi = pcs_full.max(axis=0)
    rng_ = np.where(hi - lo < 1e-12, 1.0, hi - lo)


    def _hist(pcs):
        idx = np.floor((pcs - lo) / rng_ * ngrid).astype(int)
        idx = np.clip(idx, 0, ngrid - 1)
        flat = idx[:, 0] * ngrid + idx[:, 1]
        h = np.bincount(flat, minlength=ngrid * ngrid).astype(float)
        return h

    p = _hist(pcs_full) + 1e-12
    q = _hist(pcs_sel) + 1e-12
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return 0.5 * (kl_pm + kl_qm)


def evaluate_metrics(sel_idx, ref, features, energy, force_norms, virial_norms):
    """Metrics for a candidate subset against the group reference.

    Descriptor-space metrics always computed; physical metrics only when the
    system carries labels (unlabeled exploration frames are judged by
    diversity alone).
    """
    metrics = {}
    pcs_sel = _project(ref, np.asarray(features, dtype=float)[sel_idx])
    pcs_full = ref["pcs"]
    metrics["hull_coverage"] = (
        float(_hull_contains(ref["hull"], pcs_sel).mean())
        if ref["hull"] is not None
        else 1.0
    )
    metrics["grid_coverage"] = _grid_coverage(pcs_full, pcs_sel)
    metrics["kde_js"] = float(_kde_js(pcs_full, pcs_sel))
    if energy is not None and len(sel_idx) >= 2:
        e_full = np.asarray(energy, dtype=float)
        e_sel = e_full[sel_idx]
        rng_ = e_full.max() - e_full.min()
        metrics["energy_wasserstein"] = (
            wasserstein_distance(e_full, e_sel) / rng_ if rng_ > 1e-12 else 0.0
        )
        if force_norms is not None:
            f_full = np.asarray(force_norms, dtype=float)
            f_sel = f_full[sel_idx]
            frng = f_full.mean() if f_full.mean() > 1e-12 else 1.0
            metrics["force_wasserstein"] = wasserstein_distance(f_full, f_sel) / frng
        if virial_norms is not None:
            v_full = np.asarray(virial_norms, dtype=float)
            v_sel = v_full[sel_idx]
            vrng = v_full.mean() if v_full.mean() > 1e-12 else 1.0
            metrics["virial_wasserstein"] = wasserstein_distance(v_full, v_sel) / vrng
    return metrics


def check_thresholds(metrics, thresholds):
    for key, bound in thresholds.items():
        val = metrics.get(key)
        if val is None:
            continue
        if key in ("kde_js", "energy_wasserstein", "force_wasserstein",
                   "virial_wasserstein"):
            if val > bound:
                return False
        else:  # coverage-type: bigger is better
            if val < bound:
                return False
    return True


def select_frames(features, n_select, method="pca-kmeans", seed=42):
    """Select n_select frame indices from features (n, d)."""
    n = features.shape[0]
    if n_select >= n:
        return np.arange(n)
    if method == "fps":
        scaler = StandardScaler().fit(features)
        feats = PCA(n_components=min(50, features.shape[1]), random_state=seed).fit_transform(
            scaler.transform(features)
        )
        idx = [0]
        dmin = np.linalg.norm(feats - feats[0], axis=1)
        for _ in range(n_select - 1):
            far = int(np.argmax(dmin))
            idx.append(far)
            dmin = np.minimum(dmin, np.linalg.norm(feats - feats[far], axis=1))
        return np.array(sorted(idx))
    # default: pca-kmeans, nearest-to-centroid per cluster
    scaler = StandardScaler().fit(features)
    pcs = PCA(n_components=2, random_state=seed).fit_transform(scaler.transform(features))
    k = min(n_select, len(np.unique(pcs.round(6), axis=0)))
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(pcs)
    chosen = []
    for c in range(k):
        members = np.where(labels == c)[0]
        center = pcs[members].mean(axis=0)
        chosen.append(int(members[np.argmin(np.linalg.norm(pcs[members] - center, axis=1))]))
    while len(chosen) < n_select:  # degenerate clusters
        rest = [i for i in range(n) if i not in chosen]
        if not rest:
            break
        chosen.append(rest[0])
    return np.array(sorted(set(chosen))[:n_select])


def select_minimal_subset(features, ref, thresholds, energy=None, force_norms=None,
                          virial_norms=None, tolerance=1.0, method="pca-kmeans",
                          seed=42, min_select=1):
    """Two-phase convergence (coarse sweep + downward refinement).

    Phase 1 sweeps 20/40/60/80/100% and finds the first passing fraction;
    phase 2 walks that fraction downward in `tolerance`-percent steps while
    the thresholds still pass. Returns (selected_idx sorted, trajectory
    [(fraction, metrics, passed)]).

    Parameters
    ----------
    tolerance : float
        Refinement step in percent (default 1.0 like select_configs).
    """
    n = features.shape[0]
    if n <= min_select:
        idx = np.arange(n)
        m = evaluate_metrics(idx, ref, features, energy, force_norms, virial_norms)
        return idx, [(1.0, m, check_thresholds(m, thresholds))]

    def _try(frac):
        n_sel = min(n, max(min_select, int(np.ceil(frac * n))))
        idx = select_frames(features, n_sel, method=method, seed=seed)
        m = evaluate_metrics(idx, ref, features, energy, force_norms, virial_norms)
        return idx, m, check_thresholds(m, thresholds)

    traj = []
    best = None
    for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
        idx, m, ok = _try(frac)
        traj.append((frac, m, ok))
        if ok:
            best = idx
            first_pass = frac
            break
    else:
        return np.arange(n), traj  # nothing passed: keep everything

    step = max(tolerance / 100.0, 0.01)
    frac = first_pass
    while frac - step >= 0.05:
        cand = round(frac - step, 4)
        idx, m, ok = _try(cand)
        traj.append((cand, m, ok))
        if ok:
            frac, best = cand, idx
        else:
            break
    return best, traj


# ============================================================
# Orchestration: pool assembly, committee union, fp candidates
# ============================================================


def _sys_of_modd_task(modd_task):
    """sys index of a model_devi task dir named task.SYS.NNN."""
    return int(os.path.basename(modd_task).split(".")[1])


def load_pool(jdata, iter_index):
    """Assemble the selection pool.

    init data (init_data_prefix/init_data_sys) + all previous iterations'
    02.fp/data.* + the current iteration's exploration frames. Magnetic flags
    come from magnetic_sys_idx (systems are grouped by their sys index, which
    labels both fp data.SYS dirs and model_devi task.SYS dirs).
    """
    type_map = jdata["type_map"]
    magnetic_sys = set(expand_sys_idx(jdata.get("magnetic_sys_idx", [])))
    pool = []

    prefix = os.path.abspath(jdata["init_data_prefix"])
    for sys_rel in jdata["init_data_sys"]:
        # init systems carry the magnetic flag only through an optional
        # per-entry annotation; the common case maps them by order is not
        # robust, so init systems default to the magnetic flag of their
        # basename appearing in magnetic_init_sys, else False
        mag = (
            os.path.basename(os.path.normpath(sys_rel)) in jdata.get("magnetic_init_sys", [])
            or _infer_magnetic_from_name(sys_rel)
        )
        sysd = load_system(
            os.path.join(prefix, sys_rel), name=sys_rel, magnetic=mag, source="init"
        )
        if sysd is not None:
            pool.append(sysd)

    for ii in range(iter_index):
        fp_path = os.path.join(f"iter.{ii:06d}", "02.fp")
        for data_dir in sorted(glob.glob(os.path.join(fp_path, "data.*"))):
            sys_idx = int(os.path.basename(data_dir).split(".")[-1])
            sysd = load_system(
                data_dir,
                name=f"iter{ii}/{os.path.basename(data_dir)}",
                magnetic=sys_idx in magnetic_sys,
                source=f"iter{ii}",
            )
            if sysd is not None:
                pool.append(sysd)

    modd_path = os.path.join(f"iter.{iter_index:06d}", "01.model_devi")
    for task in sorted(glob.glob(os.path.join(modd_path, "task.*"))):
        mag = _sys_of_modd_task(task) in magnetic_sys
        sysd = load_modd_frames(task, type_map, magnetic=mag)
        if sysd is not None:
            pool.append(sysd)
    return pool


def _infer_magnetic_from_name(sys_rel):
    """Best-effort magnetic annotation for init systems from directory names."""
    name = os.path.basename(os.path.normpath(sys_rel)).lower()
    return any(tag in name for tag in ("mag", "bcc", "hcp", "fcc"))


def expand_sys_idx(sys_idx):
    out = []
    for ii in sys_idx if isinstance(sys_idx, (list, tuple)) else [sys_idx]:
        if isinstance(ii, str) and ".." in ii:
            a, b = ii.split("..")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(ii))
    return out


def _labels_for(system):
    """(energy, force_norms, virial_norms) or Nones for unlabeled systems."""
    if system["energy"] is None:
        return None, None, None
    n = system["coord"].shape[0]
    natoms = system["natoms"]
    force_norms = (
        np.linalg.norm(
            np.asarray(system["force"], dtype=float).reshape(n, natoms, 3), axis=2
        ).mean(axis=1)
        if system["force"] is not None
        else None
    )
    virial_norms = (
        np.abs(
            np.asarray(system["virial"], dtype=float).reshape(n, 3, 3).trace(axis1=1, axis2=2)
        )
        if system["virial"] is not None
        else None
    )
    return np.asarray(system["energy"], dtype=float).reshape(n), force_norms, virial_norms


def mlu_select_fp_candidates(iter_index, jdata, mdata):
    """Run the committee descriptor-union selection for one iteration.

    Returns (candidates, report): candidates is a list of
    (modd_task_dir, frame_idx, sys_idx) for exploration frames that entered
    the union but are not yet labeled; the report dict mirrors what
    select_configs wrote (per-model per-system fractions and metrics).
    """
    cfg = jdata["mlu_select"]
    method = cfg.get("method", "pca-kmeans")
    seed = cfg.get("seed", 42)
    tolerance = cfg.get("convergence_tolerance", 1.0)
    min_select = cfg.get("min_select", 1)
    thresholds = {
        k: cfg[k]
        for k in (
            "hull_coverage",
            "grid_coverage",
            "kde_js",
            "energy_wasserstein",
            "force_wasserstein",
            "virial_wasserstein",
        )
        if k in cfg
    }

    pool = load_pool(jdata, iter_index)
    if not pool:
        return [], {"error": "empty pool"}

    # fingerprints of already-labeled frames for the union subtraction
    labeled_fps = set()
    for sysd in pool:
        if sysd["source"] != "modd":
            labeled_fps |= system_fingerprints(sysd)

    # committee models of THIS iteration
    suffix = mdata.get("model_suffix", ".pth")
    model_files = sorted(
        glob.glob(os.path.join(f"iter.{iter_index:06d}", "00.train", f"graph*{suffix}"))
    )
    if not model_files:
        model_files = sorted(
            glob.glob(os.path.join(f"iter.{iter_index:06d}", "00.train", "graph*"))
        )
    if not model_files:
        return [], {"error": "no committee models found in 00.train"}

    report = {"models": [], "groups": {}}
    union = {}  # (pool_sys_name, frame_global_idx) -> set of model names

    for group_name, magnetic in (("magnetic", True), ("non_magnetic", False)):
        group = [s for s in pool if s["magnetic"] == magnetic]
        if not group:
            continue
        # one reference per (model, group): PCA fit on the model's descriptors
        for model_path in model_files:
            model_name = os.path.basename(model_path)
            descs = []
            for sysd in group:
                descs.append(compute_descriptors(sysd, model_path))
            feats = np.concatenate(descs, axis=0)
            ref = _fit_reference(feats)
            offset = 0
            model_rep = {"model": model_name, "systems": []}
            for sysd, desc in zip(group, descs):  # noqa: B905
                n = desc.shape[0]
                energy, force_norms, virial_norms = _labels_for(sysd)
                idx, traj = select_minimal_subset(
                    desc,
                    ref,
                    thresholds,
                    energy=energy,
                    force_norms=force_norms,
                    virial_norms=virial_norms,
                    tolerance=tolerance,
                    method=method,
                    seed=seed,
                    min_select=min_select,
                )
                for gi in idx:
                    union.setdefault((sysd["name"], int(gi)), set()).add(model_name)
                model_rep["systems"].append(
                    {
                        "system": sysd["name"],
                        "n_frames": n,
                        "selected": len(idx),
                        "fraction": round(len(idx) / max(n, 1), 4),
                        "trajectory": [
                            {
                                "fraction": round(f, 4),
                                "metrics": {k: round(v, 6) for k, v in m.items()},
                                "passed": ok,
                            }
                            for f, m, ok in traj
                        ],
                    }
                )
                offset += n
            report["models"].append(model_rep)
        report["groups"][group_name] = len(group)

    # union -> candidates: exploration frames not already labeled
    name_to_sys = {s["name"]: s for s in pool}
    candidates = []
    seen_fps = set()
    for (sys_name, gidx), models in sorted(union.items()):
        sysd = name_to_sys[sys_name]
        if sysd["source"] != "modd":
            continue
        fp = frame_fingerprint(sysd["box"][gidx], sysd["coord"][gidx])
        if fp in labeled_fps or fp in seen_fps:
            continue
        seen_fps.add(fp)
        candidates.append(
            {
                "task": sysd["origin"],
                "frame": int(gidx),
                "sys_idx": _sys_of_modd_task(sysd["origin"]),
                "models": sorted(models),
                "magnetic": sysd["magnetic"],
            }
        )
    report["n_pool_systems"] = len(pool)
    report["n_union_frames"] = len(union)
    report["n_candidates"] = len(candidates)
    return candidates, report
