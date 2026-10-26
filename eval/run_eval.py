#!/usr/bin/env python3
"""Roll out one trained policy (or all, via subprocesses) in Isaac Gym and save runs.npz + meta.json.

Same path as Runner(test=True).play(): same Runner class, same cfg (the run's config.yaml),
same checkpoint loading, action = dist.loc. Overrides only touch the in-memory cfg.

    python3 eval/run_eval.py --policy E1_free_BASE --vcmd 2.0
    python3 eval/run_eval.py --policy T0_walk --vcmd 0.7 --distance 9
    python3 eval/run_eval.py --policy E1_free_BASE --vcmd 2.0 --twin
    python3 eval/run_eval.py --all --vcmd 2.0
    xvfb-run -a python3 eval/run_eval.py --policy E1_free_BASE --vcmd 2.0 --record_env 12 401 77
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)
import common as C  # noqa: E402

import yaml  # noqa: E402

DYN_KEYS = (
    "dof_stiffness",
    "dof_damping",
    "dof_friction",
    "friction",
    "compliance",
    "restitution",
    "base_com",
    "base_mass",
    "other_com",
    "other_mass",
)
CHECK_EVERY = 50
TOL = 1.0e-5
FALLBACK_SYMMETRY = os.path.join("T1_sprint_arms_locked", "code", "utils", "symmetry.py")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--policy", type=str)
    g.add_argument("--all", action="store_true", help="every policy in policies.yaml, one subprocess each")
    p.add_argument("--vcmd", type=float, required=True)
    p.add_argument("--distance", type=float, default=C.D_MAIN)
    p.add_argument("--num_runs", type=int, default=C.NUM_RUNS)
    p.add_argument("--time_limit", type=float, default=None, help="default ceil(1.5*distance/vcmd)+5")
    p.add_argument("--perturbed", action="store_true", help="keep the training noise/kicks/pushes/dynamics/delay")
    p.add_argument("--twin", action="store_true", help="mirrored twin of each run (nominal only)")
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--out", type=str, default=os.path.join(EVAL_DIR, "results"))
    p.add_argument("--record_env", type=int, nargs="+", default=None, help="write one mp4 per env index; no runs.npz")
    p.add_argument("--policies", type=str, default=C.POLICIES_FILE)
    p.add_argument("--root", type=str, default=C.REPO, help="folder that holds the code trees and checkpoints")
    p.add_argument("--sim_device", type=str, default="cuda:0")
    p.add_argument("--rl_device", type=str, default="cuda:0")
    args = p.parse_args(argv)
    if args.twin and args.perturbed:
        p.error("--twin is nominal only")
    args.out = os.path.abspath(args.out)
    args.root = os.path.abspath(args.root)
    args.policies = os.path.abspath(args.policies)
    return args


def die(msg):
    sys.stderr.write("\n[run_eval] ABORT: {}\n".format(msg))
    sys.exit(2)


# --------------------------------------------------------------------------- cfg


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def flat(d, prefix=""):
    o = {}
    for k, v in d.items():
        if isinstance(v, dict):
            o.update(flat(v, prefix + k + "."))
        else:
            o[prefix + k] = v
    return o


def build_cfg(entry, args, tl, record):
    """Run config.yaml (exact training cfg) filled with task-YAML keys it lacks, plus eval overrides."""
    run_cfg, run_cfg_path = C.load_run_config(entry, args.root)
    errs = C.check_entry(entry, run_cfg)
    if errs:
        die("config.yaml check failed:\n  " + "\n  ".join(errs))
    code_dir = C.abs_path(entry["code"], args.root)
    with open(os.path.join(code_dir, "envs", "{}.yaml".format(entry["task"])), "r", encoding="utf-8") as f:
        task_cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    cfg = deep_merge(task_cfg, run_cfg)
    a, b = flat(run_cfg), flat(task_cfg)
    cfg_diff = {k: {"task_yaml": b.get(k), "run_config": a.get(k)} for k in sorted(set(a) | set(b)) if a.get(k, "<none>") != b.get(k, "<none>")}

    ov = {}

    def setv(path, val):
        cur = cfg
        for k in path[:-1]:
            cur = cur.setdefault(k, {})
        ov[".".join(path)] = {"train": copy.deepcopy(cur.get(path[-1])), "eval": copy.deepcopy(val)}
        cur[path[-1]] = val

    v = float(args.vcmd)
    g = float(entry["gait_freq"])
    ckpt = C.abs_path(entry["checkpoint"], args.root)
    setv(["basic", "checkpoint"], ckpt)
    setv(["basic", "headless"], True)
    setv(["basic", "seed"], int(args.seed))
    setv(["basic", "sim_device"], args.sim_device)
    setv(["basic", "rl_device"], args.rl_device)
    setv(["basic", "play_lin_vel_x"], v)
    setv(["env", "num_envs"], int(args.num_runs))
    setv(["viewer", "record_video"], bool(record))
    if record:
        setv(["viewer", "record_env_idx"], int(args.record_env[0]))
    setv(["commands", "curriculum"], False)
    setv(["commands", "lin_vel_x"], [v, v])
    setv(["commands", "lin_vel_y"], [0.0, 0.0])
    setv(["commands", "ang_vel_yaw"], [0.0, 0.0])
    setv(["commands", "gait_frequency"], [g, g])
    setv(["commands", "still_proportion"], 0.0)
    setv(["commands", "resampling_time_s"], [tl + 2.0, tl + 3.0])
    setv(["rewards", "episode_length_s"], tl + 1.0)
    if entry["task"] == "T0":
        setv(["terrain", "type"], "plane")
    if not args.perturbed:
        for key, params in cfg.get("noise", {}).items():
            if isinstance(params, dict) and "range" in params:
                setv(["noise", key, "range"], [0.0, 0.0])
        setv(["randomization", "kick_interval_s"], tl + 10.0)
        setv(["randomization", "push_interval_s"], tl + 10.0)
        for key in DYN_KEYS:
            params = cfg["randomization"].get(key)
            if params is None:
                continue
            lo, hi = params["range"]
            if params["distribution"] == "uniform":
                c = 0.5 * (float(lo) + float(hi))
                setv(["randomization", key, "range"], [c, c])
            else:
                setv(["randomization", key, "range"], [float(lo), 0.0])
        ov["control.delay_steps"] = {"train": "randint(0, decimation) at reset", "eval": int(cfg["control"]["decimation"]) // 2}
    return cfg, ov, cfg_diff, run_cfg_path, code_dir, ckpt


# --------------------------------------------------------------------------- helpers


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def eval_commit():
    path = os.path.join(EVAL_DIR, "EVAL_COMMIT")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except ValueError:
                return {"raw": f.read()}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=C.REPO, stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "eval"], cwd=C.REPO, stderr=subprocess.DEVNULL).strip())
        return {"commit": commit, "dirty_eval": dirty, "source": "git"}
    except Exception:
        return {"commit": "unknown"}


def load_symmetry(code_dir, root):
    path = os.path.join(code_dir, "utils", "symmetry.py")
    if not os.path.isfile(path):
        path = os.path.join(root, FALLBACK_SYMMETRY)
    spec = importlib.util.spec_from_file_location("eval_symmetry", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


def yaw_of(quat, torch):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rot(xy, yaw, torch):
    """Rotate world xy into the frame with heading yaw: R(-yaw) xy. xy (N,...,2), yaw (N,)."""
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    shape = [-1] + [1] * (xy.dim() - 2)
    c = c.view(shape)
    s = s.view(shape)
    x = xy[..., 0]
    y = xy[..., 1]
    return torch.stack((c * x + s * y, -s * x + c * y), dim=-1)


# --------------------------------------------------------------------------- --all


def run_all(args):
    combo = {"vcmd": float(args.vcmd), "distance": float(args.distance), "perturbed": bool(args.perturbed), "twin": bool(args.twin)}
    failed = []
    for entry in C.load_policies(args.policies):
        name = entry["name"]
        if C.is_todo(entry):
            print("[run_eval --all] skip {}: checkpoint TODO".format(name))
            continue
        if entry.get("stage1") and combo not in C.stage_combos(entry, 1):
            print("[run_eval --all] skip {}: combination not in its stage1 list".format(name))
            continue
        out = C.result_dir(args.out, name, args.vcmd, args.distance, args.perturbed, args.twin)
        if args.record_env is None and os.path.isfile(os.path.join(out, "runs.npz")):
            print("[run_eval --all] skip {}: {} exists".format(name, os.path.join(out, "runs.npz")))
            continue
        cmd = [sys.executable, os.path.abspath(__file__), "--policy", name] + strip_all(sys.argv[1:])
        print("[run_eval --all] " + " ".join(cmd), flush=True)
        if subprocess.call(cmd) != 0:
            failed.append(name)
    if failed:
        die("failed policies: {}".format(", ".join(failed)))


def strip_all(argv):
    return [a for a in argv if a != "--all"]


# --------------------------------------------------------------------------- one policy


def run_one(args):
    entry = C.get_policy(args.policy, args.policies)
    if C.is_todo(entry):
        die("{} has checkpoint TODO in policies.yaml".format(entry["name"]))
    record = args.record_env is not None
    tl = float(args.time_limit) if args.time_limit is not None else float(C.time_limit(args.distance, args.vcmd))
    out_dir = C.result_dir(args.out, entry["name"], args.vcmd, args.distance, args.perturbed, args.twin)
    if not record and os.path.isfile(os.path.join(out_dir, "runs.npz")):
        print("[run_eval] {} exists, skipping".format(os.path.join(out_dir, "runs.npz")))
        return
    cfg, ov, cfg_diff, run_cfg_path, code_dir, ckpt = build_cfg(entry, args, tl, record)
    if not os.path.isfile(ckpt):
        die("checkpoint not found: {}".format(ckpt))
    if record:
        bad = [i for i in args.record_env if not 0 <= i < args.num_runs]
        if bad:
            die("--record_env indices out of range [0, {}): {}".format(args.num_runs, bad))

    os.chdir(code_dir)
    sys.path.insert(0, code_dir)
    from isaacgym import gymapi, gymtorch  # noqa: F401  (must precede torch)
    from isaacgym.torch_utils import quat_rotate_inverse
    import numpy as np
    import torch
    from utils.runner import Runner

    sym, sym_path = load_symmetry(code_dir, args.root)

    class EvalRunner(Runner):
        def __init__(self, eval_cfg, task):
            self._eval_cfg = eval_cfg
            self._eval_task = task
            super().__init__(test=True)

        def _get_args(self):
            self.args = argparse.Namespace(task=self._eval_task)

        def _update_cfg_from_args(self):
            self.cfg = self._eval_cfg

    t_start = time.time()
    runner = EvalRunner(cfg, entry["task"])
    env = runner.env
    model = runner.model
    rl_device = runner.device

    # The Runner loads with strict=False; make sure every tensor of the checkpoint landed.
    try:
        ck = torch.load(ckpt, map_location=rl_device, weights_only=False)["model"]
    except TypeError:
        ck = torch.load(ckpt, map_location=rl_device)["model"]
    sd = model.state_dict()
    bad = [k for k, v in ck.items() if k not in sd or sd[k].shape != v.shape or not torch.equal(sd[k], v.to(sd[k].device))]
    if bad or set(sd) - set(ck):
        die("checkpoint does not match the model exactly (mismatched: {}, missing: {})".format(bad[:5], sorted(set(sd) - set(ck))[:5]))

    if env.terrain.type != "plane":
        die("terrain.type is {}, expected plane".format(env.terrain.type))

    N = env.num_envs
    D = env.num_dofs
    dec = int(cfg["control"]["decimation"])
    dt = float(env.dt)
    v = float(args.vcmd)
    g = float(entry["gait_freq"])
    nominal = not args.perturbed
    perm, sign = sym.build_dof_mirror(env.dof_names, device=env.device)
    default = env.default_dof_pos  # (1, D)

    env.eval_mode = True

    reason = torch.zeros(N, dtype=torch.long, device=env.device)
    orig_check = env._check_termination
    term_idx = env.termination_contact_indices

    def check_termination():
        orig_check()
        r = torch.zeros(N, dtype=torch.long, device=env.device)
        r[env.time_out_buf & env.reset_buf] = C.TERM_ENV_TIMEOUT
        r[env.root_states[:, 7:13].square().sum(dim=-1) > env.cfg["rewards"]["terminate_vel"]] = C.TERM_SPEED
        r[env.base_pos[:, 2] - env.terrain.terrain_heights(env.base_pos) < env.cfg["rewards"]["terminate_height"]] = C.TERM_HEIGHT
        if len(term_idx) > 0:
            r[torch.any(torch.norm(env.contact_forces[:, term_idx, :], dim=-1) > 1.0, dim=1)] = C.TERM_CONTACT
        r[~env.reset_buf] = 0
        reason[:] = r

    env._check_termination = check_termination

    orig_teleport = env._teleport_robot

    def teleport_guard():
        before = env.root_states[:, :2].clone()
        orig_teleport()
        if not torch.equal(before, env.root_states[:, :2]):
            die("_teleport_robot moved a robot during evaluation")

    env._teleport_robot = teleport_guard

    video = None
    if record:
        video = VideoRecorder(env, args.record_env, os.path.join(out_dir, "videos"), gymapi)
        env.render = video.render

    # Per-env robot weight and (perturbed) the sampled dynamics, read back from the simulator.
    gz = abs(float(cfg["sim"]["gravity"][2]))
    masses = np.zeros((N, env.num_bodies), dtype=np.float32)
    coms = np.zeros((N, env.num_bodies, 3), dtype=np.float32)
    foot_shape = np.zeros((N, len(env.foot_shape_indices), 3), dtype=np.float32)
    for i in range(N):
        props = env.gym.get_actor_rigid_body_properties(env.envs[i], env.actor_handles[i])
        for b, pr in enumerate(props):
            masses[i, b] = pr.mass
            coms[i, b] = (pr.com.x, pr.com.y, pr.com.z)
        if args.perturbed:
            sp = env.gym.get_actor_rigid_shape_properties(env.envs[i], env.actor_handles[i])
            for j, s in enumerate(env.foot_shape_indices):
                foot_shape[i, j] = (sp[s].friction, sp[s].compliance, sp[s].restitution)
    robot_weight = masses.sum(axis=1) * gz

    # Same robot + same seed -> same initial states for every policy.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    env.reset()
    delay_after_reset = env.delay_steps.clone()

    env.commands[:, 0] = v
    env.commands[:, 1] = 0.0
    env.commands[:, 2] = 0.0
    env.gait_frequency[:] = g
    env.gait_process[:] = 0.5 if args.twin else 0.0
    if nominal:
        env.delay_steps[:] = dec // 2
    yaw0 = yaw_of(env.base_quat, torch).clone()
    if args.twin:
        env.dof_pos[:] = default + sym.mirror_action(env.dof_pos - default, perm, sign)
        env.dof_vel[:] = sym.mirror_action(env.dof_vel, perm, sign)
        vb = rot(env.root_states[:, 7:9], yaw0, torch)
        vb[:, 1] = -vb[:, 1]
        vw = rot(vb, -yaw0, torch)
        env.root_states[:, 7:9] = vw
        wb = rot(env.root_states[:, 10:12], yaw0, torch)
        wb[:, 0] = -wb[:, 0]
        env.root_states[:, 10:12] = rot(wb, -yaw0, torch)
        env.root_states[:, 12] = -env.root_states[:, 12]
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
    # T1.reset() leaves these buffers from before the reset; the first observation must see the new state.
    env.base_lin_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 7:10])
    env.base_ang_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 10:13])
    env.projected_gravity[:] = quat_rotate_inverse(env.base_quat, env.gravity_vec)
    env.last_dof_targets[:] = env.dof_pos
    env._compute_observations()
    obs = env.obs_buf.to(rl_device)

    p0 = env.base_pos[:, :2].clone()
    q0 = env.dof_pos.clone()
    v0 = rot(env.root_states[:, 7:9], yaw0, torch)
    obs0 = env.obs_buf.clone()

    noise_ranges = {k: p["range"] for k, p in env.cfg.get("noise", {}).items() if isinstance(p, dict) and "range" in p}

    def run_checks(step, alive):
        ids = alive.nonzero(as_tuple=False).flatten()
        if len(ids) == 0:
            return

        def fail(what, values, expected):
            err = (values - expected).abs()
            j = int(torch.argmax(err))
            die("check '{}' failed at step {} env {}: found {} expected {}".format(what, step, int(ids[j]), float(values[j]), float(expected)))

        for col, want, what in ((0, v, "commands vx == vcmd"), (1, 0.0, "commands vy == 0"), (2, 0.0, "commands yaw == 0")):
            vals = env.commands[ids, col]
            if (vals - want).abs().max() > TOL:
                fail(what, vals, want)
        vals = env.gait_frequency[ids]
        if (vals - g).abs().max() > TOL:
            fail("gait_frequency == gait_freq", vals, g)
        if nominal:
            vals = env.delay_steps[ids].float()
            if (vals - dec // 2).abs().max() > 0:
                fail("delay_steps == {}".format(dec // 2), vals, dec // 2)
            for k, r in noise_ranges.items():
                if any(float(x) != 0.0 for x in r):
                    die("check 'observation noise == 0' failed at step {}: noise.{}.range = {}".format(step, k, r))

    T = int(math.ceil(tl / dt))
    dev = env.device
    pos = torch.zeros(N, T, 3, dtype=torch.float32, device=dev)
    feet = torch.zeros(N, T, 2, 2, dtype=torch.float32, device=dev)
    yaw = torch.zeros(N, T, dtype=torch.float16, device=dev)
    vel = torch.zeros(N, T, 2, dtype=torch.float16, device=dev)
    qrel = torch.zeros(N, T, D, dtype=torch.float16, device=dev)
    fz = torch.zeros(N, T, dec, 2, dtype=torch.float16, device=dev)
    valid = torch.zeros(N, T, dtype=torch.bool, device=dev)
    alive = torch.ones(N, dtype=torch.bool, device=dev)
    term = torch.zeros(N, dtype=torch.long, device=dev)
    t_end = torch.full((N,), float("nan"), dtype=torch.float32, device=dev)
    dist = float(args.distance)

    print("[run_eval] {} | {} | N={} TL={:.0f}s T={} steps".format(entry["name"], os.path.basename(out_dir), N, tl, T), flush=True)
    k = -1
    for k in range(T):
        if k % CHECK_EVERY == 0:
            run_checks(k, alive)
        with torch.no_grad():
            act = model.act(obs).loc
            obs, _, done, _ = env.step(act)
        obs = obs.to(rl_device)
        done = done.to(dev).bool()
        rel = rot(env.base_pos[:, :2] - p0, yaw0, torch)
        pos[:, k, :2] = rel
        pos[:, k, 2] = env.base_pos[:, 2]
        yaw[:, k] = torch.remainder(yaw_of(env.base_quat, torch) - yaw0 + math.pi, 2 * math.pi) - math.pi
        vel[:, k] = rot(env.root_states[:, 7:9], yaw0, torch).half()
        feet[:, k] = rot(env.feet_pos[:, :, :2] - p0.unsqueeze(1), yaw0, torch)
        qrel[:, k] = (env.dof_pos - default).half()
        fz[:, k] = env.eval_feet_fz.half()
        new_done = alive & done
        ok = alive & ~done
        valid[:, k] = ok
        finished = ok & (rel[:, 0] >= dist)
        codes = reason.clone()
        codes[codes == 0] = C.TERM_ENV_TIMEOUT
        term[new_done] = codes[new_done]
        t_end[new_done] = k * dt
        term[finished] = C.TERM_FINISHED
        t_end[finished] = (k + 1) * dt
        alive &= ~(new_done | finished)
        if video is not None:
            video.update(k, alive)
        if (k + 1) % 500 == 0:
            print("[run_eval] step {}/{} alive {}".format(k + 1, T, int(alive.sum())), flush=True)
        if not alive.any():
            break
    n_steps = k + 1
    term[alive] = C.TERM_TIME_LIMIT
    t_end[alive] = n_steps * dt
    elapsed = time.time() - t_start

    if video is not None:
        video.close()
        print("[run_eval] videos in {}".format(video.out_dir))
        return

    def npy(x):
        return x[:, :n_steps].cpu().numpy()

    valid_np = npy(valid)
    arrays = {
        "t": ((np.arange(n_steps) + 1) * dt).astype(np.float32),
        "pos": npy(pos),
        "feet_xy": npy(feet),
        "yaw": npy(yaw),
        "vel": npy(vel),
        "q": npy(qrel),
        "fz": npy(fz),
    }
    for key, arr in arrays.items():
        if key != "t":
            arr[~valid_np] = np.nan
    arrays.update(
        {
            "valid": valid_np,
            "term_code": term.cpu().numpy().astype(np.int8),
            "t_end": t_end.cpu().numpy(),
            "p0": p0.cpu().numpy().astype(np.float32),
            "yaw0": yaw0.cpu().numpy().astype(np.float32),
            "q0": q0.cpu().numpy().astype(np.float32),
            "v0": v0.cpu().numpy().astype(np.float32),
            "obs0": obs0.cpu().numpy().astype(np.float32),
            "robot_weight": robot_weight.astype(np.float32),
        }
    )
    if args.perturbed:
        arrays.update(
            {
                "dyn_dof_stiffness": env.dof_stiffness.cpu().numpy(),
                "dyn_dof_damping": env.dof_damping.cpu().numpy(),
                "dyn_dof_friction": env.dof_friction.cpu().numpy(),
                "dyn_foot_friction": foot_shape[..., 0],
                "dyn_foot_compliance": foot_shape[..., 1],
                "dyn_foot_restitution": foot_shape[..., 2],
                "dyn_body_mass": masses,
                "dyn_body_com": coms,
                "dyn_base_mass_scaled": env.base_mass_scaled.cpu().numpy(),
                "dyn_delay_steps": delay_after_reset.cpu().numpy(),
            }
        )

    meta = {
        "policy": entry,
        "condition": "twin" if args.twin else ("perturbed" if args.perturbed else "nominal"),
        "condition_dir": os.path.basename(out_dir),
        "vcmd": v,
        "distance": dist,
        "time_limit": tl,
        "num_runs": N,
        "seed": int(args.seed),
        "twin": bool(args.twin),
        "perturbed": bool(args.perturbed),
        "checkpoint": entry["checkpoint"],
        "checkpoint_sha256": sha256(ckpt),
        "run_config": os.path.relpath(run_cfg_path, args.root),
        "eval_commit": eval_commit(),
        "overrides": ov,
        "cfg_diff_vs_task_yaml": cfg_diff,
        "dt": dt,
        "sim_dt": float(cfg["sim"]["dt"]),
        "decimation": dec,
        "gait_range": entry["gait_range"],
        "gait_freq": g,
        "dof_names": list(env.dof_names),
        "perm": perm.cpu().tolist(),
        "sign": sign.cpu().tolist(),
        "default_dof_pos": default[0].cpu().tolist(),
        "obs_prefix_len": int(sym.OBS_PREFIX),
        "obs_prefix_sign": [float(x) for x in sym._OBS_PREFIX_SIGN.tolist()],
        "num_obs": int(env.num_obs),
        "symmetry_module": os.path.relpath(sym_path, args.root),
        "robot_weight_N": float(np.median(robot_weight)),
        "foot_names": list(cfg["asset"]["foot_names"]),
        "terminate_height": float(cfg["rewards"]["terminate_height"]),
        "terminate_vel": float(cfg["rewards"]["terminate_vel"]),
        "curriculum_ceiling": None,
        "term_codes": {str(k_): v_ for k_, v_ in C.TERM_NAMES.items()},
        "elapsed_s": elapsed,
        "n_steps": n_steps,
        "num_envs": N,
        "sim_device": args.sim_device,
    }

    os.makedirs(out_dir, exist_ok=True)
    npz_tmp = os.path.join(out_dir, "runs.npz.tmp")
    meta_tmp = os.path.join(out_dir, "meta.json.tmp")
    with open(npz_tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    with open(meta_tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    os.replace(meta_tmp, os.path.join(out_dir, "meta.json"))
    os.replace(npz_tmp, os.path.join(out_dir, "runs.npz"))
    counts = {C.TERM_NAMES[c]: int((arrays["term_code"] == c).sum()) for c in C.TERM_NAMES}
    print("[run_eval] wrote {} ({} steps, {:.0f} s) terminations: {}".format(out_dir, n_steps, elapsed, counts), flush=True)


class VideoRecorder:
    """One 1280x720 camera per requested env; replaces env.render on the instance only."""

    TAIL_STEPS = 50

    def __init__(self, env, idxs, out_dir, gymapi):
        import imageio

        self.env = env
        self.idxs = list(dict.fromkeys(idxs))
        self.out_dir = out_dir
        self.gymapi = gymapi
        os.makedirs(out_dir, exist_ok=True)
        fps = int(round(1.0 / env.dt))
        self.writers = {i: imageio.get_writer(os.path.join(out_dir, "env{}.mp4".format(i)), fps=fps) for i in self.idxs}
        self.cams = {}
        self.active = set(self.idxs)
        self.ended_at = {}

    def render(self):
        env = self.env
        gym = env.gym
        if not self.active:
            return
        if env.device != "cpu":
            gym.fetch_results(env.sim, True)
        gym.step_graphics(env.sim)
        for i in self.active:
            if i not in self.cams:
                props = self.gymapi.CameraProperties()
                props.width = 1280
                props.height = 720
                props.use_collision_geometry = False
                self.cams[i] = gym.create_camera_sensor(env.envs[i], props)
            base = env.root_states[i, 0:3].tolist()
            cam_pos = self.gymapi.Vec3(*(x + y for x, y in zip(base, env.cfg["viewer"]["pos"])))
            gym.set_camera_location(self.cams[i], env.envs[i], cam_pos, self.gymapi.Vec3(*base))
        gym.render_all_camera_sensors(env.sim)
        for i in self.active:
            img = gym.get_camera_image(env.sim, env.envs[i], self.cams[i], self.gymapi.IMAGE_COLOR)
            self.writers[i].append_data(img.reshape(img.shape[0], -1, 4)[..., :3])

    def update(self, step, alive):
        for i in list(self.active):
            if not bool(alive[i]):
                self.ended_at.setdefault(i, step)
                if step - self.ended_at[i] >= self.TAIL_STEPS:
                    self.active.discard(i)

    def close(self):
        for w in self.writers.values():
            w.close()


def main():
    args = parse_args()
    if args.all:
        run_all(args)
    else:
        run_one(args)


if __name__ == "__main__":
    main()
