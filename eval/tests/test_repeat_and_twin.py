"""Deterministic repetition and twin initial state (checks are numpy only, read meta.json).

On the Vast (from /app/eval_ws), 8 envs and 2 s per run:
    python3 eval/tests/test_repeat_and_twin.py --run
Only the checks, on results that already exist:
    python3 eval/tests/test_repeat_and_twin.py

Repetition: the same policy twice with different --out; prints the max error of every array.
Twin: for every policy, q0_twin = default + M(q0 - default), vy0_twin = -vy0, p0/yaw0 equal,
obs0_twin = mirror_obs(obs0) (none/dup/loss) or obs0 (phase). Tolerance 1e-4.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(HERE)
sys.path.insert(0, EVAL)
import analyze as A  # noqa: E402
import common as C  # noqa: E402

OUT = os.path.join(EVAL, "tests_out")
TOL = A.TWIN_TOL


def run_eval(policy, vcmd, out, extra=()):
    cmd = [sys.executable, os.path.join(EVAL, "run_eval.py"), "--policy", policy, "--vcmd", str(vcmd), "--num_runs", "8", "--time_limit", "2", "--out", out] + list(extra)
    print("[test] " + " ".join(cmd), flush=True)
    if subprocess.call(cmd) != 0:
        raise SystemExit("run_eval failed: {}".format(" ".join(cmd)))


def vcmd_of(entry):
    return 1.0 if "SANITY" in entry.get("experiments", []) else C.V_MAIN


def launch(policies, repeat_policy):
    for sub in ("repeat_a", "repeat_b"):
        run_eval(repeat_policy, C.V_MAIN, os.path.join(OUT, sub))
    for e in policies:
        v = vcmd_of(e)
        run_eval(e["name"], v, os.path.join(OUT, "twin"))
        run_eval(e["name"], v, os.path.join(OUT, "twin"), ["--twin"])


def max_err(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return np.inf
    if a.dtype.kind not in "fc":
        return float(np.max(a != b)) if a.size else 0.0
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    na, nb = np.isnan(a64), np.isnan(b64)
    if np.any(na != nb):
        return np.inf
    d = np.abs(a64[~na] - b64[~na])
    return float(d.max()) if d.size else 0.0


def check_repeat(repeat_policy):
    da = C.result_dir(os.path.join(OUT, "repeat_a"), repeat_policy, C.V_MAIN, C.D_MAIN)
    db = C.result_dir(os.path.join(OUT, "repeat_b"), repeat_policy, C.V_MAIN, C.D_MAIN)
    za, zb = np.load(os.path.join(da, "runs.npz")), np.load(os.path.join(db, "runs.npz"))
    worst = 0.0
    print("\n== repetition: {} ({} vs {})".format(repeat_policy, da, db))
    for k in za.files:
        e = max_err(za[k], zb[k])
        worst = max(worst, e)
        print("  {:<22s} max err {:.3g}".format(k, e))
    print("  max error over all arrays: {:.3g}".format(worst))
    return worst


def check_twin(policies):
    rows, ok = [], True
    for e in policies:
        v = vcmd_of(e)
        dn = C.result_dir(os.path.join(OUT, "twin"), e["name"], v, C.D_MAIN)
        dt = C.result_dir(os.path.join(OUT, "twin"), e["name"], v, C.D_MAIN, twin=True)
        if not (os.path.isfile(os.path.join(dn, "runs.npz")) and os.path.isfile(os.path.join(dt, "runs.npz"))):
            rows.append((e["name"], e["symmetry"], "-", np.nan, "SEM DADOS"))
            ok = False
            continue
        with open(os.path.join(dn, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        err = A.twin_init_errors(meta, np.load(os.path.join(dn, "runs.npz")), np.load(os.path.join(dt, "runs.npz")), e["symmetry"])
        m = max(err["err_q0"], err["err_v0"], err["err_p0_yaw0"], err["err_obs0"])
        res = "PASSA" if m <= TOL else "FALHA"
        ok &= res == "PASSA"
        rows.append((e["name"], e["symmetry"], err["obs_expected"], m, res))
        if res == "FALHA":
            print("  {} detail: {}".format(e["name"], err))
    print("\n== twin initial state (tol {:g})".format(TOL))
    print("  {:<22s} {:<6s} {:<9s} {:>10s}  {}".format("policy", "sym", "obs0", "max_err", "result"))
    for r in rows:
        print("  {:<22s} {:<6s} {:<9s} {:>10.3g}  {}".format(*r))
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="store_true", help="launch run_eval.py first (needs isaacgym)")
    p.add_argument("--repeat_policy", default="E1_free_BASE")
    p.add_argument("--only", nargs="+", default=None, help="twin check only for these policies")
    a = p.parse_args()
    policies = [e for e in C.load_policies() if not C.is_todo(e) and (a.only is None or e["name"] in a.only)]
    if a.run:
        launch(policies, a.repeat_policy)
    check_repeat(a.repeat_policy)
    if not check_twin(policies):
        raise SystemExit("twin check: FALHA")
    print("\ntest_repeat_and_twin: OK")


if __name__ == "__main__":
    main()
