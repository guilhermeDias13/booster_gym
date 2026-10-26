"""Training-log summary per policy, from the tfevents in logs/<run>/summaries/ (local).

    py -m pip install tensorboard pyyaml
    py eval/train_logs.py                      -> eval/analysis/train_summary.csv

Per policy: mean of the last 50 logged iterations of `reward`, `steps` and every `episode/<term>`,
the mean episode duration (steps x 0.02 s) and the last `curriculum/max_lin_vel_level`.
Policies with checkpoint TODO get one row with status = TODO.
"""

import argparse
import csv
import glob
import os
import sys

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL_DIR)
import common as C  # noqa: E402

LAST = 50
POLICY_DT = 0.02
CURR_TAG = "curriculum/max_lin_vel_level"


def summarize_run(summaries):
    from tensorboard.backend.event_processing.event_accumulator import SCALARS, EventAccumulator

    files = sorted(glob.glob(os.path.join(summaries, "events.out.tfevents.*")))
    if not files:
        return None, "no tfevents in {}".format(summaries)
    ea = EventAccumulator(summaries, size_guidance={SCALARS: 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    row = {}
    for tag in tags:
        if tag in ("reward", "steps") or tag.startswith("episode/"):
            vals = [s.value for s in ea.Scalars(tag)]
            row[tag + "_last50"] = sum(vals[-LAST:]) / len(vals[-LAST:]) if vals else float("nan")
    if "steps" in tags:
        sc = ea.Scalars("steps")
        row["n_iters"] = sc[-1].step + 1
        row["episode_duration_s"] = row["steps_last50"] * POLICY_DT
    if CURR_TAG in tags:
        sc = ea.Scalars(CURR_TAG)
        row["curr_max_lin_vel_last"] = sc[-1].value
        row["curr_max_lin_vel_peak"] = max(s.value for s in sc)
    return row, "ok" if len(files) == 1 else "ok ({} tfevents files)".format(len(files))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policies", default=C.POLICIES_FILE)
    p.add_argument("--out", default=os.path.join(EVAL_DIR, "analysis", "train_summary.csv"))
    a = p.parse_args()

    rows = []
    for e in C.load_policies(a.policies):
        base = {"policy": e["name"], "experiments": "+".join(e.get("experiments", [])), "symmetry": e["symmetry"], "arms": e["arms"], "tool": e.get("tool", "none")}
        if C.is_todo(e):
            rows.append(dict(base, status="TODO"))
            print("[train_logs] {}: TODO".format(e["name"]))
            continue
        rd = C.run_dir(e)
        base["run_dir"] = os.path.relpath(rd, C.REPO).replace("\\", "/")
        row, status = summarize_run(os.path.join(rd, "summaries"))
        rows.append(dict(base, status=status, **(row or {})))
        if row:
            print("[train_logs] {:<20s} reward {:8.2f}  ep {:6.2f} s  curr_max {}".format(
                e["name"], row.get("reward_last50", float("nan")), row.get("episode_duration_s", float("nan")), row.get("curr_max_lin_vel_last", "-")))
        else:
            print("[train_logs] {}: {}".format(e["name"], status))

    head = ["policy", "status", "experiments", "symmetry", "arms", "tool", "run_dir", "n_iters", "reward_last50", "steps_last50", "episode_duration_s", "curr_max_lin_vel_last", "curr_max_lin_vel_peak"]
    extra = sorted({k for r in rows for k in r if k not in head})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=head + extra, restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("{:.6g}".format(v) if isinstance(v, float) else v) for k, v in r.items()})
    print("[train_logs] wrote {}".format(a.out))


if __name__ == "__main__":
    main()
