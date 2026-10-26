"""Shared helpers for eval/ scripts. Only stdlib + yaml (no torch, no isaacgym)."""

import copy
import math
import os

import yaml

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(EVAL_DIR)
POLICIES_FILE = os.path.join(EVAL_DIR, "policies.yaml")

V_MAIN = 2.0
V_HIGH = 3.0
V_LOW = 1.0
V_DIAG = 0.7
D_MAIN = 55.0
D_DIAG = 9.0
NUM_RUNS = 1000
SEED = 2026
CHECKPOINT_SUFFIX = "nn/model_10000.pth"
T0_CHECKPOINT_SUFFIX = "T0_model_10000.pth"

# Termination codes written by run_eval.py
TERM_ALIVE = 0
TERM_FINISHED = 1
TERM_HEIGHT = 2
TERM_SPEED = 3
TERM_CONTACT = 4
TERM_TIME_LIMIT = 5
TERM_ENV_TIMEOUT = 6
TERM_NAMES = {
    TERM_ALIVE: "alive",
    TERM_FINISHED: "finished",
    TERM_HEIGHT: "height",
    TERM_SPEED: "speed",
    TERM_CONTACT: "contact",
    TERM_TIME_LIMIT: "time_limit",
    TERM_ENV_TIMEOUT: "env_timeout",
}


def load_policies(path=POLICIES_FILE):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    defaults = data.get("defaults", {})
    out = []
    for raw in data["policies"]:
        entry = copy.deepcopy(defaults)
        entry.update(raw)
        out.append(entry)
    names = [e["name"] for e in out]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate policy names in {}".format(path))
    return out


def get_policy(name, path=POLICIES_FILE):
    for entry in load_policies(path):
        if entry["name"] == name:
            return entry
    raise KeyError("Policy '{}' not in {}".format(name, path))


def is_todo(entry):
    ck = entry.get("checkpoint")
    return ck is None or str(ck).strip().upper().startswith("TODO")


def abs_path(rel, root=REPO):
    return os.path.normpath(os.path.join(root, rel))


def run_dir(entry, root=REPO):
    """Directory holding config.yaml (parent of nn/)."""
    return os.path.dirname(os.path.dirname(abs_path(entry["checkpoint"], root)))


def load_run_config(entry, root=REPO):
    path = os.path.join(run_dir(entry, root), "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader), path


def time_limit(distance, vcmd):
    return math.ceil(1.5 * float(distance) / float(vcmd)) + 5


def fmt_num(x):
    x = float(x)
    return str(int(x)) if x.is_integer() else "{:g}".format(x)


def condition_name(vcmd, distance, perturbed=False, twin=False):
    name = "v{:.1f}_d{}_{}".format(float(vcmd), fmt_num(distance), "perturbed" if perturbed else "nominal")
    if twin:
        name += "_twin"
    return name


def result_dir(out_root, policy, vcmd, distance, perturbed=False, twin=False):
    return os.path.join(out_root, policy, condition_name(vcmd, distance, perturbed, twin))


def _get(cfg, keys, default=None):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def check_entry(entry, run_cfg):
    """Return a list of human-readable problems; empty list means OK."""
    errs = []
    name = entry["name"]
    ck = str(entry["checkpoint"]).replace("\\", "/")
    if not ck.endswith(CHECKPOINT_SUFFIX):
        errs.append("{}: checkpoint must end with {} (got {})".format(name, CHECKPOINT_SUFFIX, ck))

    task_cfg = _get(run_cfg, ["basic", "task"])
    if task_cfg is not None and task_cfg != entry["task"]:
        errs.append("{}: config.yaml basic.task={} != policies.yaml task={}".format(name, task_cfg, entry["task"]))

    method = _get(run_cfg, ["algorithm", "symmetry", "method"], "none") or "none"
    if str(method).lower() != entry["symmetry"]:
        errs.append("{}: config.yaml symmetry={} != policies.yaml symmetry={}".format(name, method, entry["symmetry"]))

    gait = _get(run_cfg, ["commands", "gait_frequency"])
    gr = entry["gait_range"]
    if gait is None or len(gait) != 2 or not (_close(gait[0], gr[0]) and _close(gait[1], gr[1])):
        errs.append("{}: config.yaml gait_frequency={} != gait_range={}".format(name, gait, gr))
    if not _close(entry["gait_freq"], 0.5 * (gr[0] + gr[1])):
        errs.append("{}: gait_freq={} is not the center of {}".format(name, entry["gait_freq"], gr))

    sanity = "SANITY" in entry.get("experiments", [])
    jd = _get(run_cfg, ["rewards", "scales", "joint_deviation"], 0.0)
    if entry["arms"] == "free" and not sanity and not _close(jd, -1.0):
        errs.append("{}: arms=free requires joint_deviation == -1 (config.yaml has {})".format(name, jd))

    if entry.get("start") == "T0" and not sanity:
        base_ck = str(_get(run_cfg, ["basic", "checkpoint"], "") or "")
        if not base_ck.replace("\\", "/").endswith(T0_CHECKPOINT_SUFFIX):
            errs.append("{}: start=T0 requires basic.checkpoint ending in {} (got {})".format(name, T0_CHECKPOINT_SUFFIX, base_ck))

    cyc = float(_get(run_cfg, ["algorithm", "torque_cycle", "loss_weight"], 0.0) or 0.0)
    spin = float(_get(run_cfg, ["rewards", "scales", "yaw_spin"], 0.0) or 0.0)
    tool = entry.get("tool", "none")
    want = {"none": (False, False), "ciclo": (True, False), "inercia": (False, True), "ambos": (True, True)}
    if tool not in want:
        errs.append("{}: unknown tool {}".format(name, tool))
    else:
        w_cyc, w_spin = want[tool]
        if (cyc != 0.0) != w_cyc or (spin != 0.0) != w_spin:
            errs.append("{}: tool={} but config.yaml has torque_cycle.loss_weight={} yaw_spin={}".format(name, tool, cyc, spin))

    if "E3" in entry.get("experiments", []):
        if not name.endswith("_" + entry["symmetry"]):
            errs.append("{}: E3 name suffix must equal symmetry ({})".format(name, entry["symmetry"]))
    return errs


def stage_combos(entry, stage):
    """List of dicts {vcmd, distance, perturbed, twin} for the given stage."""
    if stage == 1:
        if entry.get("stage1"):
            return [
                {"vcmd": float(c["vcmd"]), "distance": float(c["distance"]), "perturbed": bool(c.get("perturbed", False)), "twin": bool(c.get("twin", False))}
                for c in entry["stage1"]
            ]
        return [
            {"vcmd": V_MAIN, "distance": D_MAIN, "perturbed": False, "twin": False},
            {"vcmd": V_DIAG, "distance": D_DIAG, "perturbed": False, "twin": False},
        ]
    if stage == 2:
        if "SANITY" in entry.get("experiments", []):
            return []
        return [
            {"vcmd": V_HIGH, "distance": D_MAIN, "perturbed": False, "twin": False},
            {"vcmd": V_LOW, "distance": D_MAIN, "perturbed": False, "twin": False},
            {"vcmd": V_MAIN, "distance": D_MAIN, "perturbed": True, "twin": False},
            {"vcmd": V_MAIN, "distance": D_MAIN, "perturbed": False, "twin": True},
        ]
    raise ValueError("stage must be 1 or 2")
