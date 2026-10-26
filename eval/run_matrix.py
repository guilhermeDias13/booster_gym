"""Evaluation matrix on the Vast: one run_eval.py subprocess per missing combination.

    python3 eval/run_matrix.py --stage 1 [--only NAME ...] [--dry_run]
    python3 eval/run_matrix.py --stage 2 [--only NAME ...] [--dry_run]   (needs eval/analysis/summary.csv)
    python3 eval/run_matrix.py --estimate --machines K                   (time from the existing meta.json)
    python3 eval/run_matrix.py --videos [--vcmd 2.0] [--only NAME ...]   (idx_median/best/worst; every nominal condition unless --vcmd)

Stage 1: every policy, nominal 2.0 (d55) and nominal 0.7 (d9); T0_walk uses its own stage1 list.
Stage 2: sprint policies with viable = 1 at nominal 2.0: nominal 3.0, nominal 1.0, perturbed 2.0, twin 2.0.
Combinations that already have runs.npz are skipped; TODO checkpoints are skipped with a warning.
"""

import argparse
import csv
import glob
import json
import math
import os
import subprocess
import sys

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL_DIR)
import common as C  # noqa: E402

DT = 0.02


def read_summary(path):
    if not os.path.isfile(path):
        sys.exit(
            "[run_matrix] {} not found. Stage 2 and --videos need the stage-1 analysis:\n"
            "    python3 eval/run_matrix.py --stage 1\n"
            "    python3 eval/analyze.py".format(path)
        )
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summary_row(rows, policy, vcmd, distance=C.D_MAIN, condition="nominal"):
    for r in rows:
        if r["policy"] == policy and r["condition"] == condition and abs(float(r["vcmd"]) - vcmd) < 1e-9 and abs(float(r["distance_target"]) - distance) < 1e-9:
            return r
    return None


def is_viable(rows, policy):
    r = summary_row(rows, policy, C.V_MAIN)
    return r is not None and r.get("viable") not in (None, "", "nan") and float(r["viable"]) == 1.0


def selected(args):
    out = []
    for e in C.load_policies(args.policies):
        if args.only and e["name"] not in args.only:
            continue
        if C.is_todo(e):
            print("[run_matrix] AVISO: {} has checkpoint TODO in policies.yaml, skipped".format(e["name"]))
            continue
        out.append(e)
    if args.only:
        unknown = set(args.only) - {e["name"] for e in C.load_policies(args.policies)}
        if unknown:
            sys.exit("[run_matrix] unknown policies in --only: {}".format(sorted(unknown)))
    return out


def plan(args, stage, summary=None, unknown_viable=False):
    jobs = []
    for e in selected(args):
        if stage == 2 and "SANITY" not in e.get("experiments", []):
            known = summary is not None and summary_row(summary, e["name"], C.V_MAIN) is not None
            if summary is not None and not is_viable(summary, e["name"]) and (known or not unknown_viable):
                print("[run_matrix] {}: not viable at nominal 2.0, no stage 2".format(e["name"]))
                continue
        for c in C.stage_combos(e, stage):
            d = C.result_dir(args.out, e["name"], c["vcmd"], c["distance"], c["perturbed"], c["twin"])
            jobs.append((e, c, os.path.isfile(os.path.join(d, "runs.npz"))))
    return jobs


def eval_cmd(e, c, args, extra=()):
    cmd = [sys.executable, os.path.join(EVAL_DIR, "run_eval.py"), "--policy", e["name"], "--vcmd", str(c["vcmd"]), "--distance", C.fmt_num(c["distance"]), "--out", args.out, "--policies", args.policies]
    if c["perturbed"]:
        cmd.append("--perturbed")
    if c["twin"]:
        cmd.append("--twin")
    cmd += ["--sim_device", args.sim_device, "--rl_device", args.rl_device]
    return cmd + list(extra)


def check_configs(entries):
    errs = []
    for e in entries:
        try:
            run_cfg, _ = C.load_run_config(e)
        except OSError as exc:
            errs.append("{}: {}".format(e["name"], exc))
            continue
        errs += C.check_entry(e, run_cfg)
    if errs:
        sys.exit("[run_matrix] config check failed, nothing launched:\n  " + "\n  ".join(errs))


def run_stage(args):
    summary = read_summary(args.summary) if args.stage == 2 else None
    jobs = plan(args, args.stage, summary)
    check_configs({e["name"]: e for e, _, _ in jobs}.values())
    todo = [(e, c) for e, c, done in jobs if not done]
    for e, c, done in jobs:
        if done:
            print("[run_matrix] skip {} {}: runs.npz exists".format(e["name"], C.condition_name(c["vcmd"], c["distance"], c["perturbed"], c["twin"])))
    print("[run_matrix] stage {}: {} to run, {} already done".format(args.stage, len(todo), len(jobs) - len(todo)))
    failed = []
    for e, c in todo:
        cmd = eval_cmd(e, c, args)
        print("[run_matrix] " + " ".join(cmd), flush=True)
        if args.dry_run:
            continue
        if subprocess.call(cmd) != 0:
            failed.append("{} {}".format(e["name"], C.condition_name(c["vcmd"], c["distance"], c["perturbed"], c["twin"])))
    if failed:
        sys.exit("[run_matrix] failed: " + "; ".join(failed))
    print("[run_matrix] stage {} done".format(args.stage))


# --------------------------------------------------------------------------- estimate


def run_estimate(args):
    metas = []
    for path in glob.glob(os.path.join(args.out, "*", "*", "meta.json")):
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        if m.get("elapsed_s") and m.get("n_steps") and m.get("num_envs"):
            metas.append((path, m))
    if not metas:
        sys.exit("[run_matrix] no meta.json with elapsed_s in {}; run the first full runs (step 5a) first".format(args.out))
    print("[run_matrix] measured runs:")
    costs = []
    for path, m in sorted(metas, key=lambda x: x[0]):
        per_step = m["elapsed_s"] / m["n_steps"]
        per_step_1000 = per_step * C.NUM_RUNS / m["num_envs"]
        costs.append((m["num_envs"], per_step_1000))
        print("  {:<22s} {:<24s} {:5d} envs {:6d} steps {:8.1f} s  ({:.4f} s/step, {:.4f} s/step for 1000 envs)".format(
            m["policy"]["name"], m["condition_dir"], m["num_envs"], m["n_steps"], m["elapsed_s"], per_step, per_step_1000))
    full = [c for n, c in costs if n >= C.NUM_RUNS]
    use = full if full else [c for _, c in costs]
    cost = sorted(use)[len(use) // 2]
    note = "" if full else " (scaled linearly from runs with fewer envs, rough)"
    print("[run_matrix] cost per simulated step with 1000 envs: {:.4f} s{}".format(cost, note))

    summary = read_summary(args.summary) if os.path.isfile(args.summary) else None
    if summary is None:
        print("[run_matrix] no summary.csv yet: stage 2 assumes every sprint policy is viable (upper bound)")
    else:
        print("[run_matrix] stage 2: policies without a summary.csv row are counted as viable (upper bound)")
    k = max(1, args.machines)
    total = 0.0
    for stage in (1, 2):
        jobs = plan(args, stage, summary, unknown_viable=True)
        per_policy = {}
        for e, c, done in jobs:
            if done:
                continue
            steps = math.ceil(C.time_limit(c["distance"], c["vcmd"]) / DT)
            per_policy[e["name"]] = per_policy.get(e["name"], 0.0) + steps * cost
        t = sum(per_policy.values())
        total += t
        machines = [[0.0, []] for _ in range(k)]
        for name, sec in sorted(per_policy.items(), key=lambda x: -x[1]):
            m = min(machines, key=lambda x: x[0])
            m[0] += sec
            m[1].append(name)
        print("\n[run_matrix] stage {}: {} combinations left, {:.1f} h total (worst case: every run to the time limit)".format(stage, sum(1 for j in jobs if not j[2]), t / 3600))
        for i, (sec, names) in enumerate(machines):
            if names:
                print("  machine {}: {:.1f} h  --only {}".format(i + 1, sec / 3600, " ".join(names)))
    print("\n[run_matrix] stages 1+2: {:.1f} h total, {:.1f} h per machine with {} machine(s)".format(total / 3600, total / 3600 / k, k))


# --------------------------------------------------------------------------- videos


def index_triple(row):
    for keys in (
        ("idx_median", "idx_best", "idx_worst"),
        ("idx_median_disp", "idx_best_disp", "idx_worst_disp"),
    ):
        idx = []
        for k in keys:
            v = row.get(k, "")
            if v in ("", "nan", "NaN"):
                idx = []
                break
            idx.append(int(float(v)))
        if idx:
            return sorted(set(idx))
    return None


def run_videos(args):
    """Record idx_median, idx_best and idx_worst for every nominal summary row that has them.

    A run that never reaches 55 m still has those indices, taken at the farthest
    mark it did cross. T0_walk is included. Rows without indices are skipped.
    """
    summary = read_summary(args.summary)
    failed = []
    for e in selected(args):
        rows = [r for r in summary if r["policy"] == e["name"] and r.get("condition") == "nominal"]
        if args.vcmd is not None:
            rows = [r for r in rows if abs(float(r["vcmd"]) - args.vcmd) < 1e-9]
        if not rows:
            print("[run_matrix] {}: no nominal summary row".format(e["name"]))
            continue
        for r in rows:
            vcmd, dist = float(r["vcmd"]), float(r["distance_target"])
            label = "{} {}".format(e["name"], C.condition_name(vcmd, dist))
            idx = index_triple(r)
            issue = r.get("training_issue", "")
            if float(r.get("viable") or 0) != 1.0 or issue in ("1", "1.0") or not idx:
                print("[run_matrix] {}: no indices, no video".format(label))
                continue
            vdir = os.path.join(C.result_dir(args.out, e["name"], vcmd, dist), "videos")
            if all(os.path.isfile(os.path.join(vdir, "env{}.mp4".format(i))) for i in idx):
                print("[run_matrix] skip {}: videos exist".format(label))
                continue
            c = {"vcmd": vcmd, "distance": dist, "perturbed": False, "twin": False}
            cmd = ["xvfb-run", "-a"] + eval_cmd(e, c, args, ["--record_env"] + [str(i) for i in idx])
            print("[run_matrix] " + " ".join(cmd), flush=True)
            if not args.dry_run and subprocess.call(cmd) != 0:
                failed.append(label)
    if failed:
        sys.exit("[run_matrix] video failed: " + ", ".join(failed))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--stage", type=int, choices=(1, 2))
    g.add_argument("--estimate", action="store_true")
    g.add_argument("--videos", action="store_true")
    p.add_argument("--only", nargs="+", default=None)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--machines", type=int, default=1)
    p.add_argument("--vcmd", type=float, default=None, help="--videos: only this forward speed; default records every nominal condition that has indices")
    p.add_argument("--out", default=os.path.join(EVAL_DIR, "results"))
    p.add_argument("--summary", default=os.path.join(EVAL_DIR, "analysis", "summary.csv"))
    p.add_argument("--policies", default=C.POLICIES_FILE)
    p.add_argument("--sim_device", default="cuda:0")
    p.add_argument("--rl_device", default="cuda:0")
    args = p.parse_args()
    args.out = os.path.abspath(args.out)
    if args.estimate:
        run_estimate(args)
    elif args.videos:
        run_videos(args)
    else:
        run_stage(args)


if __name__ == "__main__":
    main()
