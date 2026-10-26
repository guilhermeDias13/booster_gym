"""Synthetic checks of the metric functions in analyze.py.

Mirrored sinusoids with a half-cycle offset -> SI ~ 0, E_M ~ 0, dphi ~ 180 deg.
Straight line with any heading -> theta = 0 and psi = 1.

    py eval/tests/test_analyze_synthetic.py
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import analyze as A  # noqa: E402
import synth  # noqa: E402

F = 2.5
DT = 0.02


def test_mirror_error_and_phase():
    names = synth.DOF21
    perm, sign = synth.mirror_maps(names)
    t = np.arange(0, 8, DT)
    q = synth.joint_signals(names, t, F)
    em = A.mirror_error(t, q, perm, sign, 1.0 / F, list(range(len(names))))
    assert em < 1e-3, em
    ix = {n: i for i, n in enumerate(names)}
    dphi, r = A.phase_lock(q[:, ix["Left_Shoulder_Pitch"]], q[:, ix["Left_Hip_Pitch"]], 1 / DT, F)
    assert abs(dphi - 180.0) < 2.0 and r > 0.99, (dphi, r)
    dphi, r = A.phase_lock(q[:, ix["Left_Shoulder_Pitch"]], q[:, ix["Right_Shoulder_Pitch"]], 1 / DT, F)
    assert abs(dphi - 180.0) < 2.0, dphi
    assert abs(A.peak_freq(q[:, ix["Left_Hip_Pitch"]], 1 / DT) - F) < 0.25


def test_si_from_contacts():
    sim_dt = 0.002
    ts = np.arange(1, 10 / sim_dt + 1) * sim_dt
    ph = np.mod(F * ts, 1.0)
    w = 500.0
    feet = [(ph < 0.6) * w, (np.mod(ph - 0.5, 1.0) < 0.6) * w]
    res = []
    for Fz in feet:
        td, lo, _ = A.contact_events(Fz.astype(float), ts, sim_dt, 0.05 * w, 0.025 * w)
        a, b, stance = A.foot_strides(td, lo, 1.0, 10.0)
        res.append((np.mean(stance), np.mean(b - a)))
    (sl, pl), (sr, pr) = res
    assert abs(A.si(sl, sr)) < 0.5 and abs(A.si(pl, pr)) < 0.5, (sl, sr, pl, pr)
    assert abs(sl / pl - 0.6) < 0.01


def test_short_second_contact_does_not_add_strides():
    """A 10 ms contact starting 30 ms after each touchdown must not create strides."""
    sim_dt = 0.002
    T = 1.0 / F
    dur = 6.0
    ts = (np.arange(int(round(dur / sim_dt))) + 1) * sim_dt
    w = 500.0
    on, off = 0.05 * w, 0.025 * w

    def signal(extra):
        Fz = np.zeros_like(ts)
        t0 = 0.2
        while t0 < dur - 0.05:
            Fz[(ts >= t0) & (ts < t0 + 0.06)] = w
            if extra:
                Fz[(ts >= t0 + 0.020) & (ts < t0 + 0.030)] = 0.0
                Fz[(ts >= t0 + 0.030) & (ts < t0 + 0.040)] = w
            t0 += T
        return Fz

    def n_strides(Fz, fuse):
        td, lo, _ = A.contact_events(Fz, ts, sim_dt, on, off)
        if fuse:
            td, lo = A.fuse_contacts(td, lo, 0.6 * T)
        a, _, _ = A.foot_strides(td, lo, 0.5, dur - 0.3)
        return len(a)

    plain, split = signal(False), signal(True)
    assert n_strides(split, fuse=False) > n_strides(plain, fuse=False)
    assert n_strides(plain, fuse=True) == n_strides(split, fuse=True)
    assert n_strides(plain, fuse=True) > 3


def test_straight_line_any_yaw():
    for yaw0 in (0.0, 0.7, -2.1, math.pi):
        p0 = np.array([3.0, -1.0])
        s = np.linspace(0, 60, 400)
        world = p0 + np.column_stack([np.cos(yaw0) * s, np.sin(yaw0) * s])
        local = A.to_yaw0_frame(world, p0, yaw0)
        xs, ys = local[:, 0], local[:, 1]
        for d in (9.0, 27.5, 55.0):
            yd, th, f0, f2 = A.trajectory_metrics(xs, ys, d)
            assert abs(yd) < 1e-9 and abs(th) < 1e-6 and abs(f0) < 1e-6 and abs(f2) < 1e-6, (yaw0, d, yd, th, f0, f2)
        assert abs(A.straightness(xs, ys) - 1.0) < 1e-9


def test_cliffs_delta():
    assert A.cliffs_delta([2, 3, 4], [0, 1]) == 1.0
    assert A.cliffs_delta([0, 1], [2, 3, 4]) == -1.0
    assert A.cliffs_delta([1, 2], [1, 2]) == 0.0


def test_twin_init_mirror():
    names = synth.DOF21
    perm, sign = synth.mirror_maps(names)
    rng = np.random.default_rng(0)
    default = np.zeros(len(names))
    q0 = rng.normal(0, 0.1, (5, len(names)))
    obs = np.concatenate([rng.normal(0, 1, (5, 11)), q0, np.zeros((5, 42))], axis=1)
    meta = {"perm": perm.tolist(), "sign": sign.tolist(), "default_dof_pos": default.tolist(), "obs_prefix_sign": synth.PREFIX_SIGN}
    zn = {"q0": q0, "v0": np.array([[0.1, 0.2]] * 5), "p0": np.zeros((5, 2)), "yaw0": np.zeros(5), "obs0": obs}
    zt = {"q0": q0[:, perm] * sign, "v0": np.array([[0.1, -0.2]] * 5), "p0": np.zeros((5, 2)), "yaw0": np.zeros(5), "obs0": synth.mirror_obs(obs, perm, sign)}
    e = A.twin_init_errors(meta, zn, zt, "none")
    assert max(e["err_q0"], e["err_v0"], e["err_obs0"]) < 1e-12, e
    zt["obs0"] = obs
    e = A.twin_init_errors(meta, zn, zt, "phase")
    assert e["err_obs0"] < 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(name, "OK")
