"""Synthetic runs.npz + meta.json in the exact run_eval.py format (numpy only).

Gait: mirrored sinusoids with a half-cycle offset, alternating foot contacts, straight-line
trajectory with a small per-run heading error. Used by test_schema.py and test_analyze_synthetic.py.
"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common as C  # noqa: E402

DOF21 = [
    "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Waist",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]
DOF12 = [n for n in DOF21 if any(k in n for k in ("Hip", "Knee", "Ankle"))]
PREFIX_SIGN = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0]
DT, DEC, SIM_DT = 0.02, 10, 0.002
WEIGHT = 500.0


def mirror_maps(names):
    idx = {n: i for i, n in enumerate(names)}
    perm, sign = [], []
    for n in names:
        pair = "Right_" + n[5:] if n.startswith("Left_") else ("Left_" + n[6:] if n.startswith("Right_") else n)
        perm.append(idx[pair])
        low = n.lower()
        sign.append(-1.0 if n == "Waist" or "roll" in low or "yaw" in low else 1.0)
    return np.array(perm), np.array(sign)


def mirror_obs(obs, perm, sign):
    n = len(perm)
    pl = len(PREFIX_SIGN)
    out = [obs[..., :pl] * np.array(PREFIX_SIGN)]
    for k in range((obs.shape[-1] - pl) // n):
        b = obs[..., pl + k * n : pl + (k + 1) * n]
        out.append(b[..., perm] * sign)
    return np.concatenate(out, axis=-1)


def joint_signals(names, t, f):
    """q - default (T, D): left joints A sin(wt + c), right = sign * left(t + T/2)."""
    w = 2 * math.pi * f
    perm, sign = mirror_maps(names)
    phases = {}
    for j, n in enumerate(names):
        if n.startswith("Left_"):
            phases[n] = 0.3 * j
    if "Left_Shoulder_Pitch" in phases:
        phases["Left_Shoulder_Pitch"] = phases["Left_Hip_Pitch"] + math.pi
    q = np.zeros((len(t), len(names)))
    for j, n in enumerate(names):
        if n.startswith("Left_"):
            q[:, j] = 0.3 * np.sin(w * t + phases[n])
        elif n.startswith("Right_"):
            left = "Left_" + n[6:]
            q[:, j] = sign[j] * 0.3 * np.sin(w * (t + 0.5 / f) + phases[left])
        else:
            q[:, j] = 0.1 * np.sin(w * t)
    return q


def make_condition(
    results, entry, vcmd, distance, condition, n_runs=24, seed=0,
    viable=True, duty=0.6, shape="line", radius=None,
    stopped=False, stopped_runs=0,
):
    """Write results/<policy>/<condition_dir>/{runs.npz,meta.json}. condition in nominal|perturbed|twin."""
    rng = np.random.default_rng(seed)
    names = DOF21 if entry["arms"] == "free" else DOF12
    D = len(names)
    perm, sign = mirror_maps(names)
    f = float(entry["gait_freq"])
    tl = C.time_limit(distance, vcmd)
    T = int(math.ceil(tl / DT))
    t = (np.arange(T) + 1) * DT
    twin = condition == "twin"
    default = np.where(np.array(["Knee" in n for n in names]), 0.4, 0.0)

    pos = np.full((n_runs, T, 3), np.nan, np.float32)
    feet = np.full((n_runs, T, 2, 2), np.nan, np.float32)
    yaw = np.full((n_runs, T), np.nan, np.float16)
    vel = np.full((n_runs, T, 2), np.nan, np.float16)
    q = np.full((n_runs, T, D), np.nan, np.float16)
    fz = np.full((n_runs, T, DEC, 2), np.nan, np.float16)
    valid = np.zeros((n_runs, T), bool)
    term = np.zeros(n_runs, np.int8)
    t_end = np.zeros(n_runs, np.float32)
    p0 = rng.uniform(-1, 1, (n_runs, 2)).astype(np.float32)
    yaw0 = rng.uniform(-math.pi, math.pi, n_runs).astype(np.float32)
    dq0 = rng.normal(0, 0.05, (n_runs, D))
    v0 = np.column_stack([rng.normal(0, 0.1, n_runs), rng.normal(0, 0.1, n_runs)]).astype(np.float32)
    obs0 = np.concatenate([rng.normal(0, 0.3, (n_runs, 11)), dq0, np.zeros((n_runs, 2 * D))], axis=1)
    slope = np.tan(np.radians(rng.normal(0, 1.0, n_runs)))

    speed = vcmd * np.minimum(1.0, t / 1.0)
    path = np.cumsum(speed) * DT
    if stopped:
        path = np.minimum(path, 0.5)
        speed = np.gradient(path, DT)
    if shape == "circle":
        if radius is None:
            raise ValueError("circle requires radius")
        yaw_path = path / float(radius)
        x = radius * np.sin(yaw_path)
        y_base = radius * (1.0 - np.cos(yaw_path))
        vx_path = speed * np.cos(yaw_path)
        vy_path = speed * np.sin(yaw_path)
    else:
        yaw_path = np.zeros_like(path)
        x, y_base = path, np.zeros_like(path)
        vx_path, vy_path = speed, np.zeros_like(path)
    qs = joint_signals(names, t, f)
    ts = ((np.arange(T)[:, None] * DT) + (np.arange(DEC)[None, :] + 1) * SIM_DT).reshape(-1)
    ph = np.mod(f * ts, 1.0)
    cL = ph < duty
    cR = np.mod(ph - 0.5, 1.0) < duty
    F = np.stack([cL, cR], axis=-1).astype(np.float64) * WEIGHT
    w = 2 * math.pi * f
    for i in range(n_runs):
        stopped_i = stopped or i < stopped_runs
        if stopped_i:
            n = T
            term[i] = C.TERM_TIME_LIMIT
        elif viable and shape == "line":
            hit = np.nonzero(x >= distance)[0]
            n = int(hit[0]) + 1 if len(hit) else T
            term[i] = C.TERM_FINISHED if len(hit) else C.TERM_TIME_LIMIT
        elif viable:
            n = T
            term[i] = C.TERM_TIME_LIMIT
        else:
            n = int(3.0 / DT)
            term[i] = C.TERM_HEIGHT
        t_end[i] = n * DT
        pi = np.minimum(path, 0.5) if stopped_i else path
        if shape == "circle" and not stopped_i:
            yi = y_base[:n]
            yawi = yaw_path[:n]
            vxi, vyi = vx_path[:n], vy_path[:n]
            xi = x[:n]
        else:
            xi = pi[:n]
            yi = slope[i] * xi + 0.01 * np.sin(w * t[:n])
            yawi = np.zeros(n)
            vxi = np.gradient(xi, DT)
            vyi = np.gradient(yi, DT)
        qi = qs[:n]
        Fi = F[: n * DEC].reshape(n, DEC, 2)
        osc = 0.1 * np.sin(w * t[:n])
        fx = np.stack([xi + osc, xi - osc], axis=-1)
        fy = np.stack([yi + 0.1, yi - 0.1], axis=-1)
        if shape == "circle" and not stopped_i:
            # Rotate the nominal foot offsets with the base heading.
            ox = np.stack([osc, -osc], axis=-1)
            oy = np.stack([np.full(n, 0.1), np.full(n, -0.1)], axis=-1)
            fx = xi[:, None] + ox * np.cos(yawi[:, None]) - oy * np.sin(yawi[:, None])
            fy = yi[:, None] + ox * np.sin(yawi[:, None]) + oy * np.cos(yawi[:, None])
        if twin:
            yi, vyi, yawi = -yi, -vyi, -yawi
            qi = qi[:, perm] * sign
            Fi = Fi[..., ::-1]
            fx = fx[:, ::-1]
            fy = -fy[:, ::-1]
        pos[i, :n, 0], pos[i, :n, 1], pos[i, :n, 2] = xi, yi, 0.68
        feet[i, :n, :, 0], feet[i, :n, :, 1] = fx, fy
        yaw[i, :n] = yawi
        vel[i, :n, 0], vel[i, :n, 1] = vxi, vyi
        q[i, :n] = qi
        fz[i, :n] = Fi
        valid[i, :n] = True

    q0 = default + dq0
    if twin:
        q0 = default + dq0[:, perm] * sign
        v0 = v0 * np.array([1.0, -1.0], np.float32)
        obs0 = obs0 if entry["symmetry"] == "phase" else mirror_obs(obs0, perm, sign)

    cond_dir = C.result_dir(results, entry["name"], vcmd, distance, condition == "perturbed", twin)
    os.makedirs(cond_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(cond_dir, "runs.npz"),
        t=t.astype(np.float32), pos=pos, feet_xy=feet, yaw=yaw, vel=vel, q=q, fz=fz, valid=valid,
        term_code=term, t_end=t_end, p0=p0, yaw0=yaw0, q0=q0.astype(np.float32), v0=v0,
        obs0=obs0.astype(np.float32), robot_weight=np.full(n_runs, WEIGHT, np.float32),
    )
    meta = {
        "policy": entry,
        "condition": condition,
        "condition_dir": os.path.basename(cond_dir),
        "vcmd": float(vcmd),
        "distance": float(distance),
        "time_limit": float(tl),
        "num_runs": n_runs,
        "seed": 2026,
        "twin": twin,
        "perturbed": condition == "perturbed",
        "dt": DT,
        "sim_dt": SIM_DT,
        "decimation": DEC,
        "gait_range": entry["gait_range"],
        "gait_freq": f,
        "dof_names": names,
        "perm": perm.tolist(),
        "sign": sign.tolist(),
        "default_dof_pos": default.tolist(),
        "obs_prefix_len": 11,
        "obs_prefix_sign": PREFIX_SIGN,
        "num_obs": int(obs0.shape[1]),
        "robot_weight_N": WEIGHT,
        "elapsed_s": 1.0,
        "n_steps": T,
        "num_envs": n_runs,
    }
    with open(os.path.join(cond_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    return cond_dir


def entry(name, arms, symmetry, experiments, gait_range=(1.5, 3.5), tool="none"):
    return {
        "name": name,
        "task": "synthetic",
        "code": ".",
        "checkpoint": "synthetic/nn/model_10000.pth",
        "symmetry": symmetry,
        "gait_range": list(gait_range),
        "gait_freq": 0.5 * (gait_range[0] + gait_range[1]),
        "arms": arms,
        "tool": tool,
        "start": "T0",
        "experiments": list(experiments),
    }


def build_dataset(results, with_twin=True, n_runs=24):
    """Viable + non-viable arms-free sprint policies (E1_free), twin of the viable one, and a T0_walk."""
    good = entry("SYN_free_BASE", "free", "none", ["E1_free", "E2"])
    bad = entry("SYN_free_dup", "free", "dup", ["E1_free", "E2"])
    short_cmp = entry("SYN_free_phase", "free", "phase", ["E1_free", "E2"])
    t0 = entry("T0_walk", "locked", "none", ["SANITY"], gait_range=(1.0, 2.0))
    circle15 = entry("SYN_circle15", "locked", "none", ["CIRCLE"], gait_range=(1.0, 2.0))
    circle5 = entry("T0_walk_circle5", "locked", "none", ["SANITY"], gait_range=(1.0, 2.0))
    circle05 = entry("T0_walk_circle05", "locked", "none", ["SANITY"], gait_range=(1.0, 2.0))
    stopped = entry("SYN_stopped", "locked", "none", ["STOPPED"], gait_range=(1.0, 2.0))
    bad_ref = entry("SYN_E3_bad_ref_none", "free", "none", ["E3"], tool="none")
    candidate = entry("SYN_E3_candidate_none", "free", "none", ["E3"], tool="ciclo")
    make_condition(results, good, 2.0, 55, "nominal", n_runs, seed=1, viable=True, duty=0.4)
    make_condition(results, good, 0.7, 9, "nominal", n_runs, seed=2, viable=True, duty=0.4)
    make_condition(results, bad, 2.0, 55, "nominal", n_runs, seed=1, viable=False, duty=0.4)
    make_condition(results, bad, 0.7, 9, "nominal", n_runs, seed=2, viable=False, duty=0.4)
    make_condition(results, short_cmp, 2.0, 55, "nominal", n_runs, seed=1, viable=True, duty=0.4, shape="circle", radius=0.5)
    make_condition(results, t0, 0.7, 9, "nominal", n_runs, seed=3, viable=True, duty=0.6)
    make_condition(results, t0, 1.0, 55, "nominal", n_runs, seed=4, viable=True, duty=0.6)
    make_condition(results, t0, 0.7, 55, "nominal", n_runs, seed=5, viable=True, duty=0.6)
    make_condition(results, t0, 0.4, 55, "nominal", n_runs, seed=6, viable=True, duty=0.6)
    make_condition(results, circle15, 1.0, 55, "nominal", 60, seed=7, viable=True, duty=0.6, shape="circle", radius=15.0, stopped_runs=3)
    make_condition(results, circle5, 1.0, 55, "nominal", 60, seed=8, viable=True, duty=0.6, shape="circle", radius=5.0)
    make_condition(results, circle05, 1.0, 55, "nominal", 60, seed=9, viable=True, duty=0.6, shape="circle", radius=0.5)
    make_condition(results, stopped, 2.0, 55, "nominal", n_runs, seed=10, viable=True, duty=0.6, stopped=True)
    make_condition(results, bad_ref, 2.0, 55, "nominal", n_runs, seed=11, viable=False, duty=0.4)
    make_condition(results, candidate, 2.0, 55, "nominal", n_runs, seed=12, viable=True, duty=0.4)
    if with_twin:
        make_condition(results, good, 2.0, 55, "twin", n_runs, seed=1, viable=True, duty=0.4)
        make_condition(results, t0, 1.0, 55, "twin", n_runs, seed=4, viable=True, duty=0.6)
        make_condition(results, t0, 0.7, 55, "twin", n_runs, seed=5, viable=True, duty=0.6)
        make_condition(results, t0, 0.4, 55, "twin", n_runs, seed=6, viable=True, duty=0.6)
    return good, bad, t0
