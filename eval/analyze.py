#!/usr/bin/env python3
"""Metrics, statistics and comparisons over eval/results/*/*/runs.npz. numpy + scipy + yaml only.

    python3 eval/analyze.py                         # everything under eval/results
    python3 eval/analyze.py --only T0_walk          # sanity gate
    py eval/analyze.py --results eval/results --out eval/analysis   (local, after download)

Writes in --out: per_run.csv, summary.csv, comparisons.csv, relations.csv, sanity.csv,
twin_check.csv and <policy>/<condition_dir>/plot_data.npz. Angles are in degrees;
phase differences (dphi_*) are in [0, 360).
"""

import argparse
import csv
import glob
import json
import multiprocessing as mp
import os
import sys
import warnings

import numpy as np
from scipy import signal, stats
from scipy.ndimage import uniform_filter1d

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)
import common as C  # noqa: E402

DISTANCES = ((9.0, "d9"), (27.5, "d27"), (55.0, "d55"))
TERM_TYPES = ("finished", "fell_height", "fell_speed", "timeout_moved", "did_not_start")
CONDITIONS_MAIN = ("nominal", "perturbed")
BOOT_B = 2000
BOOT_SEED = 20261021
VIABLE_S = 0.95
TWIN_TOL = 1.0e-4
MOVED_M = 1.0
PATH_GATE_M = 9.0
REGIME_MIN_S = 2.0
MIN_STRIDES = 3
X95_HEADING_MIN_M = 1.0
V_MIN_CURV = 0.05
CURV_EPS = 1.0e-4
SIGMA_MIN_RUNS = 50
CONTACT_ON = 0.05
CONTACT_OFF = 0.025
# A later touchdown of the same foot closer than this fraction of T_cmd is the same stance.
CONTACT_REFRACTORY = 0.6
PHASE_POINTS = 50
GRID_STEP = 0.5

GAIT_BASE = ("stance_t", "swing_t", "stride_t", "duty", "step_t", "step_len")
SI_BASE = (("si_stance", "stance_t"), ("si_swing", "swing_t"), ("si_duty", "duty"), ("si_step_t", "step_t"), ("si_step_len", "step_len"))
ARM_COLS = (
    "dphi_arm_leg_L",
    "dphi_arm_leg_R",
    "R_arm_leg_L",
    "R_arm_leg_R",
    "dphi_arms",
    "R_arms",
    "rho_freq_L",
    "rho_freq_R",
    "amp_shoulder_L",
    "amp_shoulder_R",
)
ID_COLS = ("policy", "condition", "vcmd", "distance_target", "run_idx")
KEEP_FOR_NONVIABLE = (
    "term_type", "d_max", "t_end", "path_len_at_end", "path_len",
    "fell", "reach_path_9", "yaw_rate_ss", "yaw_rate_ss_deg",
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


# --------------------------------------------------------------------------- columns


def per_run_columns(joint_pairs):
    cols = list(ID_COLS) + [
        "term_type", "d_max", "t_end", "path_len_at_end", "path_len",
        "fell", "reach_path_9",
    ]
    for _, s in DISTANCES:
        cols += ["y_" + s, "theta_" + s, "theta_fit0_" + s, "theta_fit2_" + s]
    cols += ["y_x95", "theta_x95", "theta_fit0_x95", "theta_fit2_x95"]
    cols += [
        "psi", "heading_change", "v_mean_ss", "e_v", "cv_v", "t90", "d90",
        "overshoot", "reached90", "yaw_rate_ss", "yaw_rate_ss_deg",
        "curvature_ss", "turn_radius",
        "insufficient_speed", "insufficient_heading",
        "insufficient_gait", "insufficient_symmetry",
    ]
    for b in GAIT_BASE:
        cols += [b + "_L", b + "_R"]
    cols += ["flight_frac", "is_running", "n_strides", "n_strides_L", "n_strides_R", "stride_mismatch"]
    cols += ["period_detected", "extra_contacts_per_stride"]
    cols += [c for c, _ in SI_BASE] + ["si_peak_" + j for j in joint_pairs]
    for j in joint_pairs:
        cols += ["amp_" + j + "_L", "amp_" + j + "_R"]
    cols += [
        "em_all", "em_legs", "em_arms", "et_all",
        "twin_lat_d9", "twin_lat_d27", "twin_lat_d55",
        "y_twin_x95", "twin_lat_x95",
    ]
    cols += list(ARM_COLS)
    return cols


def numeric_cols(cols):
    return [c for c in cols if c not in ID_COLS and c != "term_type"]


# --------------------------------------------------------------------------- pure metrics (tested)


def to_yaw0_frame(p_world, p0, yaw0):
    """R(-yaw0)(p - p0); p_world (T,2)."""
    d = np.asarray(p_world, dtype=np.float64) - np.asarray(p0, dtype=np.float64)
    c, s = np.cos(yaw0), np.sin(yaw0)
    return np.stack((c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]), axis=-1)


def si(left, right):
    return 100.0 * (right - left) / (0.5 * (right + left))


def first_crossing(xs, ys, targets):
    """y at the first time the running max of x reaches each target (linear interpolation)."""
    targets = np.atleast_1d(np.asarray(targets, dtype=np.float64))
    if len(xs) == 0:
        return np.full(targets.shape, np.nan)
    runmax = np.maximum.accumulate(xs)
    new = np.r_[True, xs[1:] > runmax[:-1]]
    xm, ym = xs[new], ys[new]
    out = np.interp(targets, xm, ym, left=np.nan, right=np.nan)
    out[targets > xm[-1]] = np.nan
    return out


def trajectory_metrics(xs, ys, d):
    """y(d), theta(d), theta_fit0(d), theta_fit2(d) in degrees. xs, ys start at the origin."""
    yd = first_crossing(xs, ys, [d])[0]
    if not np.isfinite(yd):
        return np.nan, np.nan, np.nan, np.nan
    j = int(np.argmax(xs >= d))
    xx, yy = xs[: j + 1], ys[: j + 1]
    fits = []
    for lo in (0.0, 2.0):
        sel = (xx >= lo) & (xx <= d)
        if sel.sum() >= 3 and np.ptp(xx[sel]) > 1e-6:
            fits.append(np.degrees(np.arctan(np.polyfit(xx[sel], yy[sel], 1)[0])))
        else:
            fits.append(np.nan)
    return yd, np.degrees(np.arctan2(yd, d)), fits[0], fits[1]


def straightness(xs, ys):
    path = np.hypot(np.diff(xs), np.diff(ys)).sum()
    if path < 1e-6:
        return np.nan
    return float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]) / path)


def hysteresis(F, on, off):
    s = np.zeros(F.shape, dtype=np.int8)
    s[F > on] = 1
    s[F < off] = -1
    idx = np.where(s != 0, np.arange(len(F)), 0)
    np.maximum.accumulate(idx, out=idx)
    return s[idx] == 1


def contact_events(F, ts, dt_sub, on, off):
    """Touchdown and liftoff times (interpolated at the on/off thresholds) and the contact state."""
    st = hysteresis(F, on, off)
    d = np.diff(st.astype(np.int8))
    rise = np.nonzero(d == 1)[0] + 1
    fall = np.nonzero(d == -1)[0] + 1

    def at(k, th):
        if len(k) == 0:
            return np.zeros(0)
        f0, f1 = F[k - 1], F[k]
        den = np.where(f1 != f0, f1 - f0, 1.0)
        return ts[k - 1] + np.clip((th - f0) / den, 0.0, 1.0) * dt_sub

    return at(rise, on), at(fall, off), st


def fuse_contacts(td, lo, min_gap):
    """Keep a touchdown only when it is at least min_gap after the previous one kept.

    Touchdowns inside that gap are dropped and their liftoff extends the current stance,
    so a short second contact stays inside the stride it interrupted.
    """
    td = np.asarray(td, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    if len(td) == 0:
        return td, lo
    j = np.searchsorted(lo, td, side="right")
    paired = np.full(len(td), np.nan)
    ok = j < len(lo)
    paired[ok] = lo[j[ok]]
    kept_td = [td[0]]
    kept_lo = [paired[0]]
    for t, liftoff in zip(td[1:], paired[1:]):
        if t - kept_td[-1] >= min_gap:
            kept_td.append(t)
            kept_lo.append(liftoff)
        elif np.isfinite(liftoff) and (not np.isfinite(kept_lo[-1]) or liftoff > kept_lo[-1]):
            kept_lo[-1] = liftoff
    liftoffs = np.asarray(kept_lo, dtype=np.float64)
    return np.asarray(kept_td, dtype=np.float64), liftoffs[np.isfinite(liftoffs)]


def foot_strides(td, lo, tw0, tw1):
    """Strides of one foot inside [tw0, tw1], first stride dropped. Returns starts, ends, stance."""
    tdw = td[(td >= tw0) & (td <= tw1)]
    if len(tdw) < 3:
        e = np.zeros(0)
        return e, e, e
    a, b = tdw[1:-1], tdw[2:]
    j = np.searchsorted(lo, a, side="right")
    lo_next = np.where(j < len(lo), lo[np.minimum(j, len(lo) - 1)] if len(lo) else np.inf, np.inf)
    stance = np.where(lo_next < b, lo_next - a, np.nan)
    return a, b, stance


def mirror_error(t, q, perm, sign, period, sel):
    """E_M = sum ||q(t) - M(q(t+T/2))||^2 / sum ||q(t) - qbar||^2 over the joints in sel."""
    if not np.isfinite(period) or len(t) < 3 or len(sel) == 0:
        return np.nan
    tq = t + 0.5 * period
    ok = tq <= t[-1]
    if ok.sum() < 3:
        return np.nan
    qs = np.column_stack([np.interp(tq[ok], t, q[:, j]) for j in range(q.shape[1])])
    mqs = qs[:, perm] * sign
    q_ = q[ok]
    num = np.square(q_[:, sel] - mqs[:, sel]).sum()
    den = np.square(q_[:, sel] - q_[:, sel].mean(axis=0)).sum()
    return float(num / den) if den > 1e-12 else np.nan


def phase_lock(a, b, fs, f):
    """(dphi in [0,360) degrees, R) between two joint signals, band-pass [0.5f, 2f] + Hilbert."""
    nyq = 0.5 * fs
    hi = min(2.0 * f, 0.95 * nyq)
    bb, aa = signal.butter(2, [0.5 * f / nyq, hi / nyq], btype="band")
    if len(a) <= 3 * max(len(aa), len(bb)):
        return np.nan, np.nan
    xa = signal.filtfilt(bb, aa, a - a.mean())
    xb = signal.filtfilt(bb, aa, b - b.mean())
    z = np.mean(np.exp(1j * (np.angle(signal.hilbert(xa)) - np.angle(signal.hilbert(xb)))))
    return float(np.degrees(np.angle(z)) % 360.0), float(np.abs(z))


def peak_freq(x, fs):
    f, p = signal.welch(x - x.mean(), fs=fs, nperseg=min(256, len(x)))
    if len(p) < 2:
        return np.nan
    return float(f[1:][np.argmax(p[1:])])


def mirror_obs_np(obs, perm, sign, prefix_sign):
    obs = np.asarray(obs, dtype=np.float64)
    n = len(perm)
    pl = len(prefix_sign)
    nb = (obs.shape[-1] - pl) // n
    if pl + nb * n != obs.shape[-1]:
        raise ValueError("obs dim {} does not fit prefix {} + k*{}".format(obs.shape[-1], pl, n))
    out = [obs[..., :pl] * np.asarray(prefix_sign)]
    for k in range(nb):
        blk = obs[..., pl + k * n : pl + (k + 1) * n]
        out.append(blk[..., perm] * sign)
    return np.concatenate(out, axis=-1)


def twin_init_errors(meta_n, zn, zt, symmetry):
    """Max errors of the twin initial state against the mirror of the nominal one."""
    perm = np.asarray(meta_n["perm"])
    sign = np.asarray(meta_n["sign"], dtype=np.float64)
    default = np.asarray(meta_n["default_dof_pos"], dtype=np.float64)
    q0, q0t = zn["q0"].astype(np.float64), zt["q0"].astype(np.float64)
    err_q = np.abs(q0t - (default + (q0 - default)[:, perm] * sign)).max()
    v0, v0t = zn["v0"].astype(np.float64), zt["v0"].astype(np.float64)
    err_v = max(np.abs(v0t[:, 0] - v0[:, 0]).max(), np.abs(v0t[:, 1] + v0[:, 1]).max())
    err_p = max(np.abs(zt["p0"] - zn["p0"]).max(), np.abs(zt["yaw0"] - zn["yaw0"]).max())
    o, ot = zn["obs0"].astype(np.float64), zt["obs0"].astype(np.float64)
    if symmetry == "phase":
        err_o = np.abs(ot - o).max()
        expected = "identity"
    else:
        err_o = np.abs(ot - mirror_obs_np(o, perm, sign, meta_n["obs_prefix_sign"])).max()
        expected = "mirror"
    return {"err_q0": float(err_q), "err_v0": float(err_v), "err_p0_yaw0": float(err_p), "err_obs0": float(err_o), "obs_expected": expected}


def cliffs_delta(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.sort(np.asarray(b, dtype=np.float64))
    if len(a) == 0 or len(b) == 0:
        return np.nan
    less = np.searchsorted(b, a, side="left")
    greater = len(b) - np.searchsorted(b, a, side="right")
    return float((less.sum() - greater.sum()) / (len(a) * len(b)))


def boot_idx(n, seed_offset=0):
    rng = np.random.default_rng(BOOT_SEED + seed_offset)
    return rng.integers(0, n, size=(BOOT_B, n))


def mean_ci(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, np.nan
    if len(x) == 1:
        return x[0], x[0], x[0], x[0]
    m = x[boot_idx(len(x))].mean(axis=1)
    return float(x.mean()), float(np.median(x)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# --------------------------------------------------------------------------- loading


HEAVY = ("pos", "vel", "fz", "feet_xy", "q")


def load_conditions(results, only=None):
    out = []
    for npz in sorted(glob.glob(os.path.join(results, "*", "*", "runs.npz"))):
        d = os.path.dirname(npz)
        with open(os.path.join(d, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        name = meta["policy"]["name"]
        if only and name not in only:
            continue
        with np.load(npz) as zf:
            z = {k: zf[k] for k in zf.files if k not in HEAVY}
        out.append({"dir": d, "meta": meta, "z": z, "npz": npz, "name": name})
    return out


def analyze_condition_file(job):
    """Worker entry point: reload one condition to avoid pickling large arrays."""
    npz, cols = job
    d = os.path.dirname(npz)
    with open(os.path.join(d, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    with np.load(npz) as zf:
        z = {k: zf[k] for k in zf.files if k not in HEAVY}
    cond = {"dir": d, "meta": meta, "z": z, "npz": npz, "name": meta["policy"]["name"]}
    return analyze_condition(cond, cols)


def load_heavy(cond, keys=HEAVY):
    """Each npz key decompresses the whole array on access: read every key exactly once."""
    with np.load(cond["npz"]) as zf:
        return {k: zf[k] for k in keys}


def joint_pairs_of(dof_names):
    pairs = []
    for n in dof_names:
        if n.startswith("Left_") and ("Right_" + n[5:]) in dof_names:
            pairs.append(n[5:])
    return pairs


# --------------------------------------------------------------------------- per condition


def term_type_of(code, d_max):
    if code == C.TERM_FINISHED:
        return "finished"
    if code in (C.TERM_HEIGHT, C.TERM_CONTACT):
        return "fell_height"
    if code == C.TERM_SPEED:
        return "fell_speed"
    return "timeout_moved" if d_max >= MOVED_M else "did_not_start"


def condition_x95(d_max, distance):
    """Largest forward x reached by at least 95% of runs, capped at the target."""
    a = np.sort(np.asarray(d_max, dtype=np.float64))
    if len(a) == 0:
        return np.nan
    k = max(0, len(a) - int(np.ceil(VIABLE_S * len(a))))
    return float(min(float(distance), a[k]))


def analyze_condition(cond, cols):
    meta, z = cond["meta"], cond["z"]
    N = int(meta["num_runs"])
    dist = float(meta["distance"])
    v = float(meta["vcmd"])
    g = float(meta["gait_freq"])
    dt = float(meta["dt"])
    sim_dt = float(meta["sim_dt"])
    dec = int(meta["decimation"])
    names = list(meta["dof_names"])
    perm = np.asarray(meta["perm"])
    sign = np.asarray(meta["sign"], dtype=np.float64)
    legs = [i for i, n in enumerate(names) if any(k in n for k in ("Hip", "Knee", "Ankle"))]
    arms = [i for i, n in enumerate(names) if any(k in n for k in ("Shoulder", "Elbow"))]
    free = meta["policy"].get("arms") == "free"
    ix = {n: i for i, n in enumerate(names)}

    t_all = z["t"].astype(np.float64)
    h = load_heavy(cond)
    pos = h["pos"]
    valid = z["valid"]
    nvalid = valid.sum(axis=1)
    term = z["term_code"]
    t_end = z["t_end"]
    weight = z["robot_weight"].astype(np.float64)

    d_max = np.array([float(np.nanmax(np.r_[0.0, pos[i, : nvalid[i], 0]])) for i in range(N)])
    path_len = np.zeros(N, dtype=np.float64)
    for i in range(N):
        xy = np.vstack((np.zeros((1, 2)), pos[i, : nvalid[i], :2].astype(np.float64)))
        path_len[i] = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])).sum()
    reach_path_9 = path_len >= PATH_GATE_M - 1e-9
    s_path_9 = float(np.mean(reach_path_9))
    reached = {s: (d_max >= d - 1e-9) if d <= dist + 1e-9 else np.full(N, np.nan) for d, s in DISTANCES}
    s_d9 = float(np.mean(reached["d9"]))
    viable = s_path_9 >= VIABLE_S
    x95 = condition_x95(d_max, dist)

    rows = {c: np.full(N, np.nan, dtype=object if c in ("policy", "condition", "term_type") else np.float64) for c in cols}
    rows["policy"][:] = cond["name"]
    rows["condition"][:] = meta["condition"]
    rows["vcmd"][:] = v
    rows["distance_target"][:] = dist
    rows["run_idx"][:] = np.arange(N)
    rows["d_max"][:] = d_max
    rows["t_end"][:] = t_end
    rows["term_type"][:] = [term_type_of(int(term[i]), d_max[i]) for i in range(N)]
    rows["path_len_at_end"][:] = path_len
    rows["path_len"][:] = path_len
    rows["reach_path_9"][:] = reach_path_9.astype(np.float64)
    rows["fell"][:] = np.isin(rows["term_type"], ("fell_height", "fell_speed")).astype(np.float64)

    grid = np.arange(0.0, dist + 1e-9, GRID_STEP)
    extras = {
        "x_grid": grid,
        "y_on_grid": np.full((N, len(grid)), np.nan),
        "v_on_grid": np.full((N, len(grid)), np.nan),
        "ang": np.full((N, len(names), PHASE_POINTS), np.nan),
        "reached": reached,
        "d_max": d_max,
        "s_d9": s_d9,
        "s_path_9": s_path_9,
        "viable": viable,
        "x95": x95,
    }

    win = max(1, int(round(1.0 / (g * dt))))
    phase = np.linspace(0.0, 1.0, PHASE_POINTS, endpoint=False)
    for i in range(N):
        n = int(nvalid[i])
        if n < 2:
            continue
        t = t_all[:n]
        xs = np.r_[0.0, pos[i, :n, 0].astype(np.float64)]
        ys = np.r_[0.0, pos[i, :n, 1].astype(np.float64)]
        yaw = np.unwrap(z["yaw"][i, :n].astype(np.float64))
        vx = h["vel"][i, :n, 0].astype(np.float64)
        vy = h["vel"][i, :n, 1].astype(np.float64)
        v_head = vx * np.cos(yaw) + vy * np.sin(yaw)
        vs = uniform_filter1d(v_head, win, mode="nearest")
        extras["v_on_grid"][i] = first_crossing(xs, np.r_[vs[0], vs], grid)
        hit = np.nonzero(vs >= 0.9 * v)[0]
        if len(hit):
            w0 = int(hit[0])
            rows["t90"][i] = t[w0]
            rows["d90"][i] = np.hypot(np.diff(xs[: w0 + 2]), np.diff(ys[: w0 + 2])).sum()
            rows["reached90"][i] = 1.0
        else:
            w0 = n // 2
            rows["reached90"][i] = 0.0
        regime_ok = n - w0 >= 2 and t[n - 1] - t[w0] >= REGIME_MIN_S - 1e-9
        rows["insufficient_speed"][i] = float(not regime_ok)
        rows["insufficient_heading"][i] = float(not regime_ok or x95 < X95_HEADING_MIN_M)
        rows["insufficient_gait"][i] = 1.0
        rows["insufficient_symmetry"][i] = 1.0
        vw = vs[w0:]
        if regime_ok:
            m = vw.mean()
            rows["v_mean_ss"][i] = m
            rows["e_v"][i] = m - v
            rows["cv_v"][i] = vw.std() / m if abs(m) > 1e-9 else np.nan
            rows["overshoot"][i] = (vw.max() - v) / v
            rate = (yaw[-1] - yaw[w0]) / (t[-1] - t[w0])
            rows["yaw_rate_ss"][i] = rate
            rows["yaw_rate_ss_deg"][i] = np.degrees(rate)
            if abs(m) >= V_MIN_CURV:
                curv = rate / m
                rows["curvature_ss"][i] = curv
                rows["turn_radius"][i] = 1.0 / abs(curv) if abs(curv) >= CURV_EPS else np.nan
        tw0, tw1 = t[w0], t[n - 1]

        if not viable:
            continue
        for d, s in DISTANCES:
            if d <= dist + 1e-9:
                yd, th, f0, f2 = trajectory_metrics(xs, ys, d)
                rows["y_" + s][i], rows["theta_" + s][i] = yd, th
                rows["theta_fit0_" + s][i], rows["theta_fit2_" + s][i] = f0, f2
        if x95 >= X95_HEADING_MIN_M:
            yd, th, f0, f2 = trajectory_metrics(xs, ys, x95)
            rows["y_x95"][i], rows["theta_x95"][i], rows["theta_fit0_x95"][i] = yd, th, f0
            rows["theta_fit2_x95"][i] = f2 if x95 > 3.0 else np.nan
        rows["psi"][i] = straightness(xs, ys)
        rows["heading_change"][i] = np.degrees(yaw[-1])
        extras["y_on_grid"][i] = first_crossing(xs, ys, grid)

        F = h["fz"][i, :n].astype(np.float64).reshape(n * dec, 2)
        ts = (np.arange(n * dec) + 1) * sim_dt
        on, off = CONTACT_ON * weight[i], CONTACT_OFF * weight[i]
        t_cmd = 1.0 / g
        min_gap = CONTACT_REFRACTORY * t_cmd
        td, lo, st, raw_td = [], [], [], []
        for foot in range(2):
            a, b, s_ = contact_events(F[:, foot], ts, sim_dt, on, off)
            raw_td.append(a)
            af, bf = fuse_contacts(a, b, min_gap)
            td.append(af)
            lo.append(bf)
            st.append(s_)
        feet_xy = h["feet_xy"][i, :n].astype(np.float64)
        strides = []
        per = {}
        for foot, side in ((0, "L"), (1, "R")):
            a, b, stance = foot_strides(td[foot], lo[foot], tw0, tw1)
            strides.append((a, b))
            other = 1 - foot
            j = np.searchsorted(td[other], a, side="left") - 1
            okj = j >= 0
            prev = np.where(okj, td[other][np.maximum(j, 0)] if len(td[other]) else np.nan, np.nan)
            step_t = np.where(okj, a - prev, np.nan)
            px = np.nan_to_num(prev)
            dx = np.interp(a, t, feet_xy[:, foot, 0]) - np.interp(px, t, feet_xy[:, other, 0])
            dy = np.interp(a, t, feet_xy[:, foot, 1]) - np.interp(px, t, feet_xy[:, other, 1])
            ya = np.interp(a, t, yaw)
            step_len = np.where(okj, dx * np.cos(ya) + dy * np.sin(ya), np.nan)
            stride = b - a
            per[side] = {
                "stance_t": stance,
                "swing_t": stride - stance,
                "stride_t": stride,
                "duty": stance / stride,
                "step_t": step_t,
                "step_len": step_len,
            }
        nL, nR = len(strides[0][0]), len(strides[1][0])
        rows["n_strides_L"][i], rows["n_strides_R"][i] = nL, nR
        rows["n_strides"][i] = min(nL, nR)
        rows["stride_mismatch"][i] = float(abs(nL - nR) > 1)
        gait_ok = nL >= MIN_STRIDES and nR >= MIN_STRIDES
        rows["insufficient_gait"][i] = float(not gait_ok)
        rows["insufficient_symmetry"][i] = float(not gait_ok)
        if gait_ok:
            for side in ("L", "R"):
                for k_, arr in per[side].items():
                    rows["{}_{}".format(k_, side)][i] = np.nanmean(arr) if len(arr) else np.nan
            ratios = []
            for foot in range(2):
                a_s, b_s = strides[foot]
                if len(a_s) == 0:
                    continue
                raw = raw_td[foot]
                extra = sum(int(np.sum((raw > a_) & (raw < b_))) for a_, b_ in zip(a_s, b_s))
                ratios.append(extra / len(a_s))
            if ratios:
                rows["extra_contacts_per_stride"][i] = float(np.mean(ratios))
        mask = (ts >= tw0) & (ts <= tw1)
        if gait_ok and mask.any():
            rows["flight_frac"][i] = float(np.mean(~(st[0][mask] | st[1][mask])))
        duty_mean = np.nanmean([rows["duty_L"][i], rows["duty_R"][i]])
        if np.isfinite(duty_mean) and np.isfinite(rows["flight_frac"][i]):
            rows["is_running"][i] = float(rows["flight_frac"][i] > 0.0 and duty_mean < 0.5)
        if gait_ok:
            for col, base in SI_BASE:
                rows[col][i] = si(rows[base + "_L"][i], rows[base + "_R"][i])

        q = h["q"][i, :n].astype(np.float64)
        qw, tw = q[w0:], t[w0:]
        if gait_ok and len(qw) >= 3:
            amp = np.percentile(qw, 95, axis=0) - np.percentile(qw, 5, axis=0)
            for jn in joint_pairs_of(names):
                rows["si_peak_" + jn][i] = si(amp[ix["Left_" + jn]], amp[ix["Right_" + jn]])
                rows["amp_" + jn + "_L"][i] = amp[ix["Left_" + jn]]
                rows["amp_" + jn + "_R"][i] = amp[ix["Right_" + jn]]
            all_strides = np.r_[strides[0][1] - strides[0][0], strides[1][1] - strides[1][0]]
            period = float(np.median(all_strides)) if len(all_strides) else np.nan
            rows["period_detected"][i] = period
            rows["em_all"][i] = mirror_error(tw, qw, perm, sign, t_cmd, list(range(len(names))))
            rows["em_legs"][i] = mirror_error(tw, qw, perm, sign, t_cmd, legs)
            if free and arms:
                rows["em_arms"][i] = mirror_error(tw, qw, perm, sign, t_cmd, arms)
            if free and "Left_Shoulder_Pitch" in ix and len(qw) >= 3 * int(round(1.0 / (g * dt))):
                fs = 1.0 / dt
                sl, sr = qw[:, ix["Left_Shoulder_Pitch"]], qw[:, ix["Right_Shoulder_Pitch"]]
                hl, hr = qw[:, ix["Left_Hip_Pitch"]], qw[:, ix["Right_Hip_Pitch"]]
                rows["dphi_arm_leg_L"][i], rows["R_arm_leg_L"][i] = phase_lock(sl, hl, fs, g)
                rows["dphi_arm_leg_R"][i], rows["R_arm_leg_R"][i] = phase_lock(sr, hr, fs, g)
                rows["dphi_arms"][i], rows["R_arms"][i] = phase_lock(sl, sr, fs, g)
                rows["rho_freq_L"][i] = peak_freq(sl, fs) / peak_freq(hl, fs)
                rows["rho_freq_R"][i] = peak_freq(sr, fs) / peak_freq(hr, fs)
                rows["amp_shoulder_L"][i] = np.percentile(sl, 95) - np.percentile(sl, 5)
                rows["amp_shoulder_R"][i] = np.percentile(sr, 95) - np.percentile(sr, 5)
            a, b = strides[0]
            if len(a):
                curves = [np.column_stack([np.interp(a_ + phase * (b_ - a_), t, q[:, j]) for j in range(len(names))]) for a_, b_ in zip(a, b)]
                extras["ang"][i] = np.mean(curves, axis=0).T
    return rows, extras


def add_twin_columns(rows_n, cond_n, cond_t, rows_t):
    """et_all, fixed-distance twin errors, and x95 twin lateral error."""
    zn, zt = cond_n["z"], cond_t["z"]
    perm = np.asarray(cond_n["meta"]["perm"])
    sign = np.asarray(cond_n["meta"]["sign"], dtype=np.float64)
    vn, vt = zn["valid"].sum(axis=1), zt["valid"].sum(axis=1)
    N = min(len(vn), len(vt))
    qn_all, qt_all = load_heavy(cond_n, ("q",))["q"], load_heavy(cond_t, ("q",))["q"]
    pt_all = load_heavy(cond_t, ("pos",))["pos"]
    x95 = float(cond_n["extras"]["x95"])
    for i in range(N):
        if not np.isfinite(rows_n["em_all"][i]) and not np.isfinite(rows_n["v_mean_ss"][i]):
            continue
        m = int(min(vn[i], vt[i]))
        if m >= 3:
            q = qn_all[i, :m].astype(np.float64)
            qt = qt_all[i, :m].astype(np.float64)
            den = np.square(q - q.mean(axis=0)).sum()
            rows_n["et_all"][i] = np.square(qt - q[:, perm] * sign).sum() / den if den > 1e-12 else np.nan
        for _, s in DISTANCES:
            if "y_" + s in rows_t:
                rows_n["twin_lat_" + s][i] = abs(rows_t["y_" + s][i] + rows_n["y_" + s][i])
        if x95 >= X95_HEADING_MIN_M and np.isfinite(rows_n["y_x95"][i]):
            mt = int(vt[i])
            xt = np.r_[0.0, pt_all[i, :mt, 0].astype(np.float64)]
            yt = np.r_[0.0, pt_all[i, :mt, 1].astype(np.float64)]
            y_twin = first_crossing(xt, yt, [x95])[0]
            rows_n["y_twin_x95"][i] = y_twin
            rows_n["twin_lat_x95"][i] = abs(y_twin + rows_n["y_x95"][i])


# --------------------------------------------------------------------------- summary


def agg_from(theta, y, reached, idx=None):
    """Aggregate stats (vectorized over bootstrap rows when idx is (B,n))."""
    if idx is None:
        idx = np.arange(len(theta))[None, :]
    th, yy, rr = theta[idx], y[idx], reached[idx]
    beta = np.nanmean(th, axis=1)
    out = {
        "S": rr.mean(axis=1),
        "beta": beta,
        "sigma_theta": np.nanstd(th, axis=1, ddof=1),
        "alpha95": np.nanpercentile(np.abs(th - beta[:, None]), 95, axis=1),
        "w95": np.nanpercentile(np.abs(yy - np.nanmedian(yy, axis=1)[:, None]), 95, axis=1),
    }
    return out


def _rank_three(ids, score):
    ids, score = np.asarray(ids), np.asarray(score, dtype=np.float64)
    return (
        int(ids[np.argmin(np.abs(score - np.median(score)))]),
        int(ids[np.argmin(score)]),
        int(ids[np.argmax(score)]),
    )


def representative_indices(rows, viable, x95):
    out = {
        "idx_median": np.nan, "idx_best": np.nan, "idx_worst": np.nan, "idx_mark": "",
        "idx_median_disp": np.nan, "idx_best_disp": np.nan, "idx_worst_disp": np.nan,
        "idx_median_abs": np.nan, "idx_best_abs": np.nan, "idx_worst_abs": np.nan,
        "idx_rank_basis": "",
    }
    if not viable:
        return out
    th55 = rows["theta_d55"].astype(np.float64)
    ok55 = np.isfinite(th55)
    if ok55.any():
        ids = np.nonzero(ok55)[0]
        out["idx_median"], out["idx_best"], out["idx_worst"] = _rank_three(ids, np.abs(th55[ids]))
        out["idx_mark"] = "d55"
    if x95 >= X95_HEADING_MIN_M:
        a = rows["theta_x95"].astype(np.float64)
        ok = np.isfinite(a)
        if ok.any():
            ids, vals = np.nonzero(ok)[0], a[ok]
            beta = vals.mean()
            out["idx_median_disp"], out["idx_best_disp"], out["idx_worst_disp"] = _rank_three(ids, np.abs(vals - beta))
            out["idx_median_abs"], out["idx_best_abs"], out["idx_worst_abs"] = _rank_three(ids, np.abs(vals))
            out["idx_rank_basis"] = "theta_x95"
    else:
        a = rows["yaw_rate_ss"].astype(np.float64)
        ok = np.isfinite(a)
        if ok.any():
            ids, vals = np.nonzero(ok)[0], a[ok]
            out["idx_median_disp"], out["idx_best_disp"], out["idx_worst_disp"] = _rank_three(ids, np.abs(vals - np.median(vals)))
            out["idx_median_abs"], out["idx_best_abs"], out["idx_worst_abs"] = _rank_three(ids, np.abs(vals))
            out["idx_rank_basis"] = "yaw_rate"
    return out


def summarize(rows, extras, meta, cols):
    pol = meta["policy"]
    N = len(rows["run_idx"])
    out = {
        "policy": pol["name"],
        "condition": meta["condition"],
        "vcmd": float(meta["vcmd"]),
        "distance_target": float(meta["distance"]),
        "condition_dir": meta["condition_dir"],
        "experiments": ";".join(pol.get("experiments", [])),
        "arms": pol.get("arms"),
        "symmetry": pol.get("symmetry"),
        "tool": pol.get("tool"),
        "gait_range": "{}-{}".format(*pol["gait_range"]),
        "n_runs": N,
        "training_issue": np.nan,
    }
    for d, s in DISTANCES:
        r = extras["reached"][s]
        out["S_" + s] = float(np.mean(r)) if np.all(np.isfinite(r)) else np.nan
    tt = rows["term_type"]
    for k in TERM_TYPES:
        n = int(np.sum(tt == k))
        out["n_" + k] = n
        out["frac_" + k] = n / N
    out["viable"] = int(extras["viable"])
    out["S_path_9"] = float(extras["s_path_9"])
    out["x95"] = float(extras["x95"])
    out["S_x95"] = float(np.mean(extras["d_max"] >= extras["x95"] - 1e-9))
    out["frac_fell"] = out["frac_fell_height"] + out["frac_fell_speed"]
    for d, s in DISTANCES:
        for base, th_col in (("", "theta_"), ("_fit2", "theta_fit2_")):
            th = rows[th_col + s].astype(np.float64)
            ok = np.isfinite(th)
            if extras["viable"] and ok.sum() >= 2:
                beta = th[ok].mean()
                out["beta{}_{}".format(base, s)] = beta
                out[("sigma_theta_{}" if base == "" else "sigma_fit2_{}").format(s)] = th[ok].std(ddof=1)
                out["alpha95{}_{}".format(base, s)] = np.percentile(np.abs(th[ok] - beta), 95)
            else:
                out["beta{}_{}".format(base, s)] = np.nan
                out[("sigma_theta_{}" if base == "" else "sigma_fit2_{}").format(s)] = np.nan
                out["alpha95{}_{}".format(base, s)] = np.nan
        y = rows["y_" + s].astype(np.float64)
        y = y[np.isfinite(y)]
        out["w95_" + s] = np.percentile(np.abs(y - np.median(y)), 95) if extras["viable"] and len(y) >= 2 else np.nan
    for base, th_col in (("", "theta_x95"), ("_fit2", "theta_fit2_x95")):
        th = rows[th_col].astype(np.float64)
        ok = np.isfinite(th)
        if extras["viable"] and extras["x95"] >= X95_HEADING_MIN_M and ok.sum() >= 2:
            beta = th[ok].mean()
            out["beta{}_x95".format(base)] = beta
            out["sigma_{}_x95".format("theta" if base == "" else "fit2")] = th[ok].std(ddof=1)
            out["alpha95{}_x95".format(base)] = np.percentile(np.abs(th[ok] - beta), 95)
        else:
            out["beta{}_x95".format(base)] = np.nan
            out["sigma_{}_x95".format("theta" if base == "" else "fit2")] = np.nan
            out["alpha95{}_x95".format(base)] = np.nan
    y95 = rows["y_x95"].astype(np.float64)
    y95 = y95[np.isfinite(y95)]
    out["w95_x95"] = np.percentile(np.abs(y95 - np.median(y95)), 95) if extras["viable"] and len(y95) >= 2 else np.nan
    out["sigma_y_x95"] = np.std(y95, ddof=1) if extras["viable"] and len(y95) >= 2 else np.nan
    yn = rows["y_x95"].astype(np.float64)
    yt = rows["y_twin_x95"].astype(np.float64)
    ok_twin = np.isfinite(yn) & np.isfinite(yt)
    out["twin_turn_same_side"] = (
        int(np.sign(np.median(yn[ok_twin])) == np.sign(np.median(yt[ok_twin])))
        if extras["viable"] and ok_twin.any() else np.nan
    )
    out.update(representative_indices(rows, extras["viable"], extras["x95"]))
    for c in numeric_cols(cols):
        vals = rows[c].astype(np.float64)
        if not extras["viable"] and c not in KEEP_FOR_NONVIABLE:
            m = md = lo = hi = np.nan
        else:
            m, md, lo, hi = mean_ci(vals)
        out[c + "_mean"], out[c + "_median"], out[c + "_ci_lo"], out[c + "_ci_hi"] = m, md, lo, hi
    return out


# --------------------------------------------------------------------------- comparisons


def experiment_pairs(policies):
    """[(experiment, policy_a, policy_b, ref_kind)] built only inside each experiment."""
    pairs = []

    def members(exp):
        return [n for n, p in policies.items() if exp in p.get("experiments", [])]

    for grp in ("E1_locked", "E1_free"):
        mem = members(grp)
        base = [n for n in mem if policies[n]["symmetry"] == "none"]
        if len(base) == 1:
            pairs += [(grp, n, base[0], "base_same_arms") for n in mem if n != base[0]]
    locked = {policies[n]["symmetry"]: n for n in members("E1_locked")}
    free = {policies[n]["symmetry"]: n for n in members("E1_free")}
    for m in ("none", "dup", "loss", "phase"):
        if m in locked and m in free and "E2" in policies[locked[m]].get("experiments", []):
            pairs.append(("E2", locked[m], free[m], "arms_effect"))
    e3 = members("E3")
    refs = {policies[n]["symmetry"]: n for n in e3 if policies[n].get("tool", "none") == "none"}
    for n in e3:
        if policies[n].get("tool", "none") != "none" and policies[n]["symmetry"] in refs:
            pairs.append(("E3", n, refs[policies[n]["symmetry"]], "tau_same_method"))
    for m, r in refs.items():
        if m in free:
            pairs.append(("E3_obs", r, free[m], "obs_effect"))
    return [p for p in pairs if policies[p[1]]["gait_range"] == policies[p[2]]["gait_range"]]


def same_robot(ca, cb):
    if ca["meta"]["dof_names"] != cb["meta"]["dof_names"]:
        return False
    za, zb = ca["z"], cb["z"]
    if za["q0"].shape != zb["q0"].shape:
        return False
    return bool(np.allclose(za["q0"], zb["q0"], atol=1e-6) and np.allclose(za["p0"], zb["p0"], atol=1e-5) and np.allclose(za["yaw0"], zb["yaw0"], atol=1e-6))


def compare_metric(a, b, paired):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if paired:
        ok = np.isfinite(a) & np.isfinite(b)
        d = a[ok] - b[ok]
        if len(d) == 0:
            return np.nan, np.nan, np.nan, cliffs_delta(a[np.isfinite(a)], b[np.isfinite(b)]), 0
        bs = d[boot_idx(len(d))].mean(axis=1)
        return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), cliffs_delta(a[np.isfinite(a)], b[np.isfinite(b)]), int(len(d))
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    bs = a[boot_idx(len(a), 1)].mean(axis=1) - b[boot_idx(len(b), 2)].mean(axis=1)
    return float(a.mean() - b.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), cliffs_delta(a, b), 0


def compare_aggregates(ra, rb, ea, eb, paired, s):
    """Diff of (S, beta, sigma_theta, alpha95, w95) at distance s, bootstrap over runs."""
    def arrays(r, e):
        return r["theta_" + s].astype(np.float64), r["y_" + s].astype(np.float64), np.asarray(e["reached"][s], dtype=np.float64)

    ta, ya, sa = arrays(ra, ea)
    tb, yb, sb = arrays(rb, eb)
    if not (np.all(np.isfinite(sa)) and np.all(np.isfinite(sb))):
        return {}
    pa, pb = agg_from(ta, ya, sa), agg_from(tb, yb, sb)
    if paired and len(ta) == len(tb):
        idx = boot_idx(len(ta), 3)
        ba, bb = agg_from(ta, ya, sa, idx), agg_from(tb, yb, sb, idx)
        n = len(ta)
    else:
        ba, bb = agg_from(ta, ya, sa, boot_idx(len(ta), 4)), agg_from(tb, yb, sb, boot_idx(len(tb), 5))
        n = 0
    out = {}
    for k in ("S", "beta", "sigma_theta", "alpha95", "w95"):
        diff = ba[k] - bb[k]
        diff = diff[np.isfinite(diff)]
        point = float(pa[k][0] - pb[k][0])
        lo, hi = (float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))) if len(diff) else (np.nan, np.nan)
        name = "S_" + s if k == "S" else "{}_{}".format(k, s)
        out[name] = (point, lo, hi, n)
    return out


def arrays_at_x(cond, x):
    """theta, y and reached arrays at a common x, interpolated from trajectory grid data."""
    key = round(float(x), 9)
    cache = cond.setdefault("_arrays_at_x", {})
    if key in cache:
        return cache[key]
    grid = cond["extras"]["x_grid"]
    yg = cond["extras"]["y_on_grid"]
    y = np.full(len(yg), np.nan)
    for i in range(len(yg)):
        ok = np.isfinite(yg[i])
        if ok.any() and x <= grid[ok][-1] + 1e-9:
            y[i] = np.interp(x, grid[ok], yg[i, ok])
    theta = np.degrees(np.arctan2(y, x))
    cache[key] = (theta, y, cond["extras"]["d_max"] >= x - 1e-9)
    return cache[key]


def compare_at_x(ca, cb, paired, x):
    ta, ya, sa = arrays_at_x(ca, x)
    tb, yb, sb = arrays_at_x(cb, x)
    pa, pb = agg_from(ta, ya, sa), agg_from(tb, yb, sb)
    if paired and len(ta) == len(tb):
        idx = boot_idx(len(ta), 13)
        ba, bb = agg_from(ta, ya, sa, idx), agg_from(tb, yb, sb, idx)
        n = len(ta)
    else:
        ba = agg_from(ta, ya, sa, boot_idx(len(ta), 14))
        bb = agg_from(tb, yb, sb, boot_idx(len(tb), 15))
        n = 0
    out = {}
    for k in ("S", "beta", "sigma_theta", "alpha95", "w95"):
        diff = ba[k] - bb[k]
        diff = diff[np.isfinite(diff)]
        out["{}_xcmp".format(k)] = (
            float(pa[k][0] - pb[k][0]),
            float(np.percentile(diff, 2.5)) if len(diff) else np.nan,
            float(np.percentile(diff, 97.5)) if len(diff) else np.nan,
            n,
        )
    return out


def comparison_metrics(both_free, joint_pairs):
    m = ["e_v", "cv_v", "t90", "yaw_rate_ss", "curvature_ss", "flight_frac"]
    m += ["|{}|".format(c) for c, _ in SI_BASE] + ["|si_peak_{}|".format(j) for j in joint_pairs]
    m += ["em_all", "em_legs", "em_arms"]
    if both_free:
        m += list(ARM_COLS)
    return m


def metric_values(rows, name):
    if name.startswith("|") and name.endswith("|"):
        return np.abs(rows[name[1:-1]].astype(np.float64))
    return rows[name].astype(np.float64)


# --------------------------------------------------------------------------- relations (E4)

REL_X = ("|si_duty|", "|si_stance|", "em_all")


def rel_y_names():
    out = []
    for _, s in DISTANCES:
        out.append("|theta_{}-beta_{}|".format(s, s))
    for _, s in DISTANCES:
        out.append("|y_{}|".format(s))
    return out + ["cv_v"]


def rel_values(rows, name):
    if name.startswith("|theta_"):
        s = name.split("-")[0][len("|theta_") :]
        th = rows["theta_" + s].astype(np.float64)
        return np.abs(th - np.nanmean(th))
    return metric_values(rows, name)


def spearman_ci(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 5:
        return np.nan, np.nan, np.nan, n
    rho = stats.spearmanr(x, y)[0]
    idx = boot_idx(n, 6)
    rx = stats.rankdata(x[idx], axis=1)
    ry = stats.rankdata(y[idx], axis=1)
    rx -= rx.mean(axis=1, keepdims=True)
    ry -= ry.mean(axis=1, keepdims=True)
    den = np.sqrt((rx**2).sum(axis=1) * (ry**2).sum(axis=1))
    r = np.where(den > 0, (rx * ry).sum(axis=1) / np.where(den > 0, den, 1), np.nan)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return float(rho), np.nan, np.nan, n
    return float(rho), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)), n


# --------------------------------------------------------------------------- csv


def fmt(v):
    if v is None:
        return "nan"
    if isinstance(v, (float, np.floating)):
        return "nan" if not np.isfinite(v) else "{:.7g}".format(float(v))
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return str(v)


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([fmt(r.get(h, np.nan)) for h in header])


def read_csv(path):
    """List of dicts; numbers as float, 'nan' as NaN. Used by tests and the report."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = {}
            for k, v in r.items():
                try:
                    d[k] = float(v)
                except (TypeError, ValueError):
                    d[k] = v
            out.append(d)
    return out


# --------------------------------------------------------------------------- main


def analyze(results, out, only=None, workers=1):
    conds = load_conditions(results, only)
    if not conds:
        raise SystemExit("no runs.npz under {}".format(results))
    os.makedirs(out, exist_ok=True)
    jp = []
    for c in conds:
        for j in joint_pairs_of(c["meta"]["dof_names"]):
            if j not in jp:
                jp.append(j)
    cols = per_run_columns(jp)

    for c in conds:
        print("[analyze] {} / {}".format(c["name"], os.path.basename(c["dir"])), flush=True)
    if workers > 1:
        with mp.get_context("spawn").Pool(workers) as pool:
            results = pool.map(analyze_condition_file, [(c["npz"], cols) for c in conds])
        for c, (rows, extras) in zip(conds, results):
            c["rows"], c["extras"] = rows, extras
    else:
        for c in conds:
            c["rows"], c["extras"] = analyze_condition(c, cols)
    key = {(c["name"], os.path.basename(c["dir"])): c for c in conds}
    for c in conds:
        if c["meta"]["condition"] == "twin":
            nom = key.get((c["name"], os.path.basename(c["dir"])[: -len("_twin")]))
            if nom is not None:
                add_twin_columns(nom["rows"], nom, c, c["rows"])

    # per_run.csv
    with open(os.path.join(out, "per_run.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for c in conds:
            r = c["rows"]
            n = len(r["run_idx"])
            for i in range(n):
                row = []
                for col in cols:
                    val = r[col][i]
                    if not c["extras"]["viable"] and col not in ID_COLS and col not in KEEP_FOR_NONVIABLE:
                        val = np.nan
                    row.append(fmt(int(val) if col == "run_idx" else val))
                w.writerow(row)

    # summary.csv (nominal and perturbed only)
    summ = []
    for c in conds:
        if c["meta"]["condition"] in CONDITIONS_MAIN:
            summ.append(summarize(c["rows"], c["extras"], c["meta"], cols))
    training_issue = {}
    for name in {r["policy"] for r in summ}:
        pol_rows = [r for r in summ if r["policy"] == name]
        sanity = any("SANITY" in str(r.get("experiments", "")).split(";") for r in pol_rows)
        base = [
            r for r in pol_rows
            if r["condition"] == "nominal"
            and abs(float(r["vcmd"]) - C.V_MAIN) < 1e-9
            and abs(float(r["distance_target"]) - C.D_MAIN) < 1e-9
        ]
        issue = np.nan if sanity or not base else int(float(base[0]["viable"]) != 1.0)
        training_issue[name] = issue
        for r in pol_rows:
            r["training_issue"] = issue
    for c in conds:
        c["training_issue"] = training_issue.get(c["name"], np.nan)
    header = list(summ[0].keys()) if summ else ["policy", "condition", "vcmd"]
    write_csv(os.path.join(out, "summary.csv"), header, summ)

    # plot_data.npz
    for c in conds:
        if c["meta"]["condition"] not in CONDITIONS_MAIN:
            continue
        write_plot_data(c, out)

    # twin_check.csv
    twin_rows = []
    for c in conds:
        if c["meta"]["condition"] != "twin":
            continue
        nom = key.get((c["name"], os.path.basename(c["dir"])[: -len("_twin")]))
        if nom is None:
            continue
        e = twin_init_errors(nom["meta"], nom["z"], c["z"], c["meta"]["policy"]["symmetry"])
        e.update({"policy": c["name"], "vcmd": c["meta"]["vcmd"], "distance_target": c["meta"]["distance"], "symmetry": c["meta"]["policy"]["symmetry"]})
        e["max_err"] = max(e["err_q0"], e["err_v0"], e["err_p0_yaw0"], e["err_obs0"])
        e["result"] = "PASSA" if e["max_err"] <= TWIN_TOL else "FALHA"
        twin_rows.append(e)
    write_csv(
        os.path.join(out, "twin_check.csv"),
        ["policy", "vcmd", "distance_target", "symmetry", "obs_expected", "err_q0", "err_v0", "err_p0_yaw0", "err_obs0", "max_err", "result"],
        twin_rows,
    )

    # comparisons.csv
    policies = {}
    for c in conds:
        policies.setdefault(c["name"], c["meta"]["policy"])
    comp_rows = []
    for exp, pa, pb, kind in experiment_pairs(policies):
        for ca in conds:
            if ca["name"] != pa or ca["meta"]["condition"] not in CONDITIONS_MAIN:
                continue
            cb = key.get((pb, os.path.basename(ca["dir"])))
            if cb is None:
                continue
            paired = ca["meta"]["condition"] == "nominal" and same_robot(ca, cb)
            base = {
                "experiment": exp, "policy_a": pa, "policy_b": pb, "ref_kind": kind,
                "condition": ca["meta"]["condition"], "vcmd": ca["meta"]["vcmd"],
                "distance_target": ca["meta"]["distance"], "paired": int(paired),
                "x_cmp": np.nan, "reason": "",
            }
            bad_a = not ca["extras"]["viable"] or ca.get("training_issue") == 1
            bad_b = not cb["extras"]["viable"] or cb.get("training_issue") == 1
            if bad_a or bad_b:
                reason = "reference_not_viable" if bad_b and not bad_a else "policy_not_viable"
                comp_rows.append(dict(base, metric="", diff_mean=np.nan, ci_lo=np.nan, ci_hi=np.nan, cliff_delta=np.nan, n_pairs=0, reason=reason))
                continue
            for _, s in DISTANCES:
                for name, (point, lo, hi, n) in compare_aggregates(ca["rows"], cb["rows"], ca["extras"], cb["extras"], paired, s).items():
                    comp_rows.append(dict(base, metric=name, diff_mean=point, ci_lo=lo, ci_hi=hi, cliff_delta=np.nan, n_pairs=n))
            xcmp = min(float(ca["extras"]["x95"]), float(cb["extras"]["x95"]))
            if xcmp < X95_HEADING_MIN_M:
                for name in ("S_xcmp", "beta_xcmp", "sigma_theta_xcmp", "alpha95_xcmp", "w95_xcmp"):
                    comp_rows.append(dict(base, metric=name, x_cmp=xcmp, reason="x_cmp_too_short", diff_mean=np.nan, ci_lo=np.nan, ci_hi=np.nan, cliff_delta=np.nan, n_pairs=0))
            else:
                for name, (point, lo, hi, n) in compare_at_x(ca, cb, paired, xcmp).items():
                    comp_rows.append(dict(base, metric=name, x_cmp=xcmp, diff_mean=point, ci_lo=lo, ci_hi=hi, cliff_delta=np.nan, n_pairs=n))
            both_free = policies[pa].get("arms") == "free" and policies[pb].get("arms") == "free"
            for name in comparison_metrics(both_free, jp):
                a, b = metric_values(ca["rows"], name), metric_values(cb["rows"], name)
                dm, lo, hi, cd, n = compare_metric(a, b, paired)
                comp_rows.append(dict(base, metric=name, diff_mean=dm, ci_lo=lo, ci_hi=hi, cliff_delta=cd, n_pairs=n))
    write_csv(
        os.path.join(out, "comparisons.csv"),
        ["experiment", "policy_a", "policy_b", "ref_kind", "metric", "condition", "vcmd", "paired", "x_cmp", "reason", "diff_mean", "ci_lo", "ci_hi", "cliff_delta", "n_pairs", "distance_target"],
        comp_rows,
    )

    # relations.csv (E4): within each viable nominal condition, sprint policies only
    rel_rows = []
    for c in conds:
        pol = c["meta"]["policy"]
        if c["meta"]["condition"] != "nominal" or not c["extras"]["viable"] or c.get("training_issue") == 1 or "SANITY" in pol.get("experiments", []):
            continue
        for xn in REL_X:
            for yn in rel_y_names():
                rho, lo, hi, n = spearman_ci(metric_values(c["rows"], xn), rel_values(c["rows"], yn))
                rel_rows.append({"policy": c["name"], "x_metric": xn, "y_metric": yn, "spearman_rho": rho, "ci_lo": lo, "ci_hi": hi, "n": n, "condition": "nominal", "vcmd": c["meta"]["vcmd"], "distance_target": c["meta"]["distance"]})
    write_csv(os.path.join(out, "relations.csv"), ["policy", "x_metric", "y_metric", "spearman_rho", "ci_lo", "ci_hi", "n", "condition", "vcmd", "distance_target"], rel_rows)

    # sanity.csv (T0_walk)
    write_csv(os.path.join(out, "sanity.csv"), ["check", "condition", "vcmd", "distance_target", "value", "threshold", "result", "note"], sanity_rows(conds, twin_rows))
    sweep = t0_speed_sweep_rows(conds, summ, jp)
    sweep_header = list(sweep[0].keys()) if sweep else ["policy", "vcmd", "distance_target", "viable", "S_path_9"]
    write_csv(os.path.join(out, "t0_speed_sweep.csv"), sweep_header, sweep)
    print("[analyze] wrote {}".format(out))


def write_plot_data(c, out):
    rows, ex, meta = c["rows"], c["extras"], c["meta"]
    d = os.path.join(out, c["name"], os.path.basename(c["dir"]))
    os.makedirs(d, exist_ok=True)
    yg = ex["y_on_grid"]
    vg = ex["v_on_grid"] if ex["viable"] else np.full_like(ex["v_on_grid"], np.nan)
    grid = ex["x_grid"]
    s_x = np.array([np.mean(ex["d_max"] >= x - 1e-9) for x in grid])
    n_runs_x = np.sum(np.isfinite(yg), axis=0)
    sigma_y_x = np.nanstd(yg, axis=0)
    sigma_y_x[n_runs_x < SIGMA_MIN_RUNS] = np.nan
    ang = ex["ang"]
    half = PHASE_POINTS // 2
    sym_names = list(REL_X)
    ds = [s for dd, s in DISTANCES if dd <= float(meta["distance"]) + 1e-9]
    last = ds[-1]
    lat_names = ["|theta_{}-beta_{}|".format(last, last), "|y_{}|".format(last)]
    np.savez_compressed(
        os.path.join(d, "plot_data.npz"),
        x_grid=grid,
        y_on_grid=yg,
        y_median=np.nanmedian(yg, axis=0),
        y_p05=np.nanpercentile(yg, 5, axis=0),
        y_p95=np.nanpercentile(yg, 95, axis=0),
        S_x=s_x,
        sigma_y_x=sigma_y_x,
        n_runs_x=n_runs_x,
        v_mean_x=np.nanmean(vg, axis=0),
        v_std_x=np.nanstd(vg, axis=0),
        phase_grid=np.linspace(0.0, 1.0, PHASE_POINTS, endpoint=False),
        ang_L_mean=np.nanmean(ang, axis=0),
        ang_L_std=np.nanstd(ang, axis=0),
        ang_R_shift_mean=np.roll(np.nanmean(ang, axis=0), -half, axis=-1),
        ang_R_shift_std=np.roll(np.nanstd(ang, axis=0), -half, axis=-1),
        joint_names=np.array(meta["dof_names"]),
        dphi_arm_leg_L=rows["dphi_arm_leg_L"].astype(np.float64),
        dphi_arm_leg_R=rows["dphi_arm_leg_R"].astype(np.float64),
        dphi_arms=rows["dphi_arms"].astype(np.float64),
        sym_x=np.column_stack([metric_values(rows, n) for n in sym_names]),
        sym_x_names=np.array(sym_names),
        lat_y=np.column_stack([rel_values(rows, n) for n in lat_names]),
        lat_y_names=np.array(lat_names),
        term_type=np.array(rows["term_type"], dtype=str),
        viable=np.array(ex["viable"]),
        vcmd=np.array(float(meta["vcmd"])),
        distance=np.array(float(meta["distance"])),
    )


def yaw_drift_note(c):
    """Median heading drift (deg/s) of a T0 condition; the T0 walker turns with yaw command 0."""
    if c is None:
        return ""
    rates = c["rows"]["yaw_rate_ss_deg"].astype(np.float64)
    rates = rates[np.isfinite(rates)]
    if len(rates) == 0:
        return ""
    return "propriedade conhecida do T0 (aceita): deriva de yaw mediana {:.1f} graus/s com comando de giro 0".format(float(np.median(rates)))


def sanity_rows(conds, twin_rows):
    t0 = [c for c in conds if "SANITY" in c["meta"]["policy"].get("experiments", [])]
    if not t0:
        return []
    name = t0[0]["name"]

    def find(cond, v, dist):
        for c in t0:
            m = c["meta"]
            if c["name"] == name and m["condition"] == cond and abs(m["vcmd"] - v) < 1e-9 and abs(m["distance"] - dist) < 1e-9:
                return c
        return None

    out = []

    def add(check, cond, v, dist, value, thr, ok=True, note="", result=None):
        missing = value is None or (isinstance(value, float) and not np.isfinite(value))
        out.append({
            "check": check, "condition": cond, "vcmd": v, "distance_target": dist,
            "value": np.nan if value is None else value, "threshold": thr,
            "result": result if result is not None else ("PASSA" if (ok and not missing) else "FALHA"),
            "note": "sem dados" if missing else note,
        })

    def condition_checks(label, v, dist, c):
        viable = bool(c["extras"]["viable"])
        s_path = float(c["extras"]["s_path_9"])
        drift = yaw_drift_note(c)
        bad_note = "" if viable else "não viável (S_path_9 < 0,95)"
        add("S_path_9 >= 0.95", label, v, dist, s_path, ">= 0.95", viable, bad_note)
        if label == "nominal" and abs(v - 0.7) < 1e-9 and abs(dist - 9.0) < 1e-9:
            sd9 = float(np.mean(c["extras"]["reached"]["d9"]))
            add("S_d9 >= 0.95 (informativo)", label, v, dist, sd9, "informativo", result="INFO")
        frac_fell = float(np.mean(c["rows"]["fell"].astype(np.float64)))
        add("frac_fell <= 0.05", label, v, dist, frac_fell, "<= 0.05", frac_fell <= 0.05)
        n = float(np.sum(c["rows"]["term_type"] == "did_not_start"))
        add("n_did_not_start == 0", label, v, dist, n, "== 0", n == 0)
        yr = float(np.nanmedian(c["rows"]["yaw_rate_ss_deg"].astype(np.float64)))
        rad = float(np.nanmedian(c["rows"]["turn_radius"].astype(np.float64)))
        add("yaw_rate_ss_deg (informativo)", label, v, dist, yr, "informativo", result="INFO")
        add("turn_radius (informativo)", label, v, dist, rad, "informativo", result="INFO")
        if label != "nominal":
            return
        ir = c["rows"]["is_running"].astype(np.float64)
        ir = ir[np.isfinite(ir)]
        frac = float(np.mean(ir == 0.0)) if len(ir) else None
        duty = float(np.nanmean(np.r_[c["rows"]["duty_L"].astype(np.float64), c["rows"]["duty_R"].astype(np.float64)]))
        ev = float(np.nanmean(c["rows"]["e_v"].astype(np.float64)))
        add("frac(is_running = False) >= 0.95", label, v, dist, frac, ">= 0.95", viable and frac is not None and frac >= 0.95, bad_note)
        add("duty medio > 0.5", label, v, dist, duty, "> 0.5", viable and duty is not None and np.isfinite(duty) and duty > 0.5, bad_note)
        okv = np.isfinite(ev) and abs(ev) <= 0.1
        add("|mean(e_v)| <= 0.1 m/s", label, v, dist, abs(ev), "<= 0.1", viable and okv, bad_note or ("" if okv else drift))

    for v, dist in ((0.7, 9.0), (1.0, 55.0), (0.7, 55.0), (0.4, 55.0)):
        c = find("nominal", v, dist)
        if c is not None:
            condition_checks("nominal", v, dist, c)
    for v, dist in ((1.0, 55.0), (0.7, 55.0), (0.4, 55.0)):
        c = find("twin", v, dist)
        if c is None:
            continue
        condition_checks("twin", v, dist, c)
        tw = [r for r in twin_rows if r["policy"] == name and abs(float(r["vcmd"]) - v) < 1e-9 and abs(float(r["distance_target"]) - dist) < 1e-9]
        err = tw[0]["max_err"] if tw else None
        add("gemeo: estado inicial espelhado (q0, v0, p0/yaw0, obs0)", "twin", v, dist, err, "<= 1e-4", err is not None and err <= TWIN_TOL, "erro maximo")
        nom = find("nominal", v, dist)
        same = None
        if nom is not None:
            yn = nom["rows"]["y_x95"].astype(np.float64)
            yt = nom["rows"]["y_twin_x95"].astype(np.float64)
            ok = np.isfinite(yn) & np.isfinite(yt)
            same = float(np.sign(np.median(yn[ok])) == np.sign(np.median(yt[ok]))) if ok.any() else None
        add("twin_turn_same_side (informativo)", "twin", v, dist, same, "informativo", result="INFO")
    return out


def t0_speed_sweep_rows(conds, summary_rows, joint_pairs):
    """One nominal d55 row per requested T0 speed, with mean/bootstrap CI columns."""
    metrics = [
        "yaw_rate_ss_deg", "curvature_ss", "turn_radius", "v_mean_ss", "e_v", "cv_v",
        "psi", "path_len", "duty_L", "duty_R", "flight_frac",
        "si_stance", "si_duty", "si_step_len", "em_all",
    ] + ["si_peak_" + j for j in joint_pairs]
    out = []
    for v in (0.4, 0.7, 1.0):
        c = next((
            c for c in conds
            if "SANITY" in c["meta"]["policy"].get("experiments", [])
            and c["meta"]["condition"] == "nominal"
            and abs(float(c["meta"]["vcmd"]) - v) < 1e-9
            and abs(float(c["meta"]["distance"]) - C.D_MAIN) < 1e-9
        ), None)
        if c is None:
            continue
        sr = next(r for r in summary_rows if r["policy"] == c["name"] and r["condition"] == "nominal" and abs(float(r["vcmd"]) - v) < 1e-9 and abs(float(r["distance_target"]) - C.D_MAIN) < 1e-9)
        row = {
            "policy": c["name"], "vcmd": v, "distance_target": C.D_MAIN,
            "viable": sr["viable"], "S_path_9": sr["S_path_9"],
        }
        for name in metrics:
            m, _, lo, hi = mean_ci(c["rows"][name].astype(np.float64))
            row[name + "_mean"], row[name + "_ci_lo"], row[name + "_ci_hi"] = m, lo, hi
        # Aggregate termination statistic.
        fell = c["rows"]["fell"].astype(np.float64)
        m, _, lo, hi = mean_ci(fell)
        row["frac_fell_mean"], row["frac_fell_ci_lo"], row["frac_fell_ci_hi"] = m, lo, hi

        n = len(fell)
        idx = boot_idx(n, 31)
        dmax_bs = c["extras"]["d_max"][idx]
        k = max(0, n - int(np.ceil(VIABLE_S * n)))
        xbs = np.minimum(float(c["meta"]["distance"]), np.partition(dmax_bs, k, axis=1)[:, k])
        row["x95_mean"] = sr["x95"]
        row["x95_ci_lo"], row["x95_ci_hi"] = np.percentile(xbs, (2.5, 97.5))

        th = c["rows"]["theta_x95"].astype(np.float64)
        yy = c["rows"]["y_x95"].astype(np.float64)
        reached = np.isfinite(th).astype(np.float64)
        point = agg_from(th, yy, reached)
        bs = agg_from(th, yy, reached, idx)
        for src, dst in (("beta", "beta_x95"), ("sigma_theta", "sigma_theta_x95"), ("alpha95", "alpha95_x95"), ("w95", "w95_x95")):
            vals = bs[src]
            vals = vals[np.isfinite(vals)]
            row[dst + "_mean"] = point[src][0] if np.isfinite(point[src][0]) else np.nan
            row[dst + "_ci_lo"] = np.percentile(vals, 2.5) if len(vals) else np.nan
            row[dst + "_ci_hi"] = np.percentile(vals, 97.5) if len(vals) else np.nan
        yn = c["rows"]["y_x95"].astype(np.float64)
        yt = c["rows"]["y_twin_x95"].astype(np.float64)
        ok = np.isfinite(yn) & np.isfinite(yt)
        if ok.any():
            yn, yt = yn[ok], yt[ok]
            tidx = boot_idx(len(yn), 32)
            same_bs = (np.sign(np.median(yn[tidx], axis=1)) == np.sign(np.median(yt[tidx], axis=1))).astype(np.float64)
            row["twin_turn_same_side_mean"] = sr.get("twin_turn_same_side", np.nan)
            row["twin_turn_same_side_ci_lo"], row["twin_turn_same_side_ci_hi"] = np.percentile(same_bs, (2.5, 97.5))
        else:
            row["twin_turn_same_side_mean"] = np.nan
            row["twin_turn_same_side_ci_lo"] = np.nan
            row["twin_turn_same_side_ci_hi"] = np.nan
        out.append(row)
    return out


def merge_summary(src, dst):
    """Append summary rows from src into dst, replacing the same policy/condition/vcmd/distance."""
    with open(src, "r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        src_fields, new = rd.fieldnames, list(rd)
    with open(dst, "r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        dst_fields, old = rd.fieldnames, list(rd)

    def key(r):
        return (r["policy"], r["condition"], r["vcmd"], r["distance_target"])

    seen = {key(r) for r in new}
    fields = list(dst_fields)
    for col in src_fields:
        if col not in fields:
            fields.append(col)
    rows = [r for r in old if key(r) not in seen] + new
    with open(dst, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({col: r.get(col, "") for col in fields})
    print("[analyze] merged {} rows into {} ({} total)".format(len(new), dst, len(rows)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=os.path.join(EVAL_DIR, "results"))
    p.add_argument("--out", default=os.path.join(EVAL_DIR, "analysis"))
    p.add_argument("--only", nargs="+", default=None)
    p.add_argument("--workers", type=int, default=1, help="conditions analyzed in parallel (local CPU only)")
    p.add_argument("--merge-summary", nargs=2, metavar=("SRC", "DST"), help="append SRC summary rows into DST and exit")
    a = p.parse_args()
    if a.merge_summary:
        merge_summary(a.merge_summary[0], a.merge_summary[1])
        return
    analyze(os.path.abspath(a.results), os.path.abspath(a.out), a.only, max(1, a.workers))


if __name__ == "__main__":
    main()
