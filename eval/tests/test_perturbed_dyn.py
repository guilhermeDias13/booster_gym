"""Perturbed condition: two policies of the same robot must draw the same dynamics (dyn_* equal).

On the Vast (from /app/eval_ws), 8 envs and 2 s per run:
    python3 eval/tests/test_perturbed_dyn.py --run
Only the check, on results that already exist:
    python3 eval/tests/test_perturbed_dyn.py
"""

import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(HERE)
sys.path.insert(0, EVAL)
import common as C  # noqa: E402

OUT = os.path.join(EVAL, "tests_out", "perturbed")
PAIRS = (("E1_free_BASE", "E1_free_dup"), ("E1_free_BASE", "E3_ciclo_none"), ("E1_locked_BASE", "E1_locked_dup"))


def run_eval(policy):
    cmd = [sys.executable, os.path.join(EVAL, "run_eval.py"), "--policy", policy, "--vcmd", str(C.V_MAIN), "--num_runs", "8", "--time_limit", "2", "--perturbed", "--out", OUT]
    print("[test] " + " ".join(cmd), flush=True)
    if subprocess.call(cmd) != 0:
        raise SystemExit("run_eval failed: {}".format(" ".join(cmd)))


def load(policy):
    path = os.path.join(C.result_dir(OUT, policy, C.V_MAIN, C.D_MAIN, perturbed=True), "runs.npz")
    z = np.load(path)
    return {k: z[k] for k in z.files if k.startswith("dyn_")}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="store_true", help="launch run_eval.py --perturbed first (needs isaacgym)")
    a = p.parse_args()
    if a.run:
        for name in sorted({n for pair in PAIRS for n in pair}):
            run_eval(name)
    ok = True
    for pa, pb in PAIRS:
        da, db = load(pa), load(pb)
        print("\n== {} vs {}".format(pa, pb))
        if not da or set(da) != set(db):
            print("  dyn_* keys differ: {} vs {}".format(sorted(da), sorted(db)))
            ok = False
            continue
        for k in sorted(da):
            same = da[k].shape == db[k].shape and np.array_equal(da[k], db[k])
            err = np.abs(da[k].astype(np.float64) - db[k].astype(np.float64)).max() if da[k].shape == db[k].shape and da[k].size else np.inf
            print("  {:<24s} {:<6s} max diff {:.3g}".format(k, "igual" if same else "DIFERE", err))
            ok &= same
    if not ok:
        raise SystemExit("test_perturbed_dyn: FALHA (aplicar o reseed antes de criar o ambiente no EvalRunner e repetir)")
    print("\ntest_perturbed_dyn: OK")


if __name__ == "__main__":
    main()
