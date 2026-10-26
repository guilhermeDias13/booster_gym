import os
import glob
import yaml
import argparse
import numpy as np
import random
import time
import signal
import imageio
import torch
import torch.nn.functional as F
from utils.model import *
from utils.buffer import ExperienceBuffer
from utils.utils import discount_values, surrogate_loss
from utils.recorder import Recorder
from utils.symmetry import (
    build_dof_mirror,
    get_symmetry_loss_weight,
    get_symmetry_method,
    maybe_symmetry_loss,
    mirror_action,
    mirror_obs,
)
from envs import *


class Runner:

    def __init__(self, test=False):
        self.test = test
        # prepare the environment
        self._get_args()
        self._update_cfg_from_args()
        self._set_seed()
        task_class = eval(self.cfg["basic"]["task"])
        self.env = task_class(self.cfg)

        self.device = self.cfg["basic"]["rl_device"]
        self.sym_method = get_symmetry_method(self.cfg)
        self.sym_loss_weight = get_symmetry_loss_weight(self.cfg)
        self.dof_perm, self.dof_sign = build_dof_mirror(self.env.dof_names, device=self.device)
        self.learning_rate = self.cfg["algorithm"]["learning_rate"]
        self.model = ActorCritic(self.env.num_actions, self.env.num_obs, self.env.num_privileged_obs).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self._load()
        print("T1_sprint_arms_locked symmetry method: {}".format(self.sym_method))

        self.buffer = ExperienceBuffer(self.cfg["runner"]["horizon_length"], self.env.num_envs, self.device)
        self.buffer.add_buffer("actions", (self.env.num_actions,))
        self.buffer.add_buffer("obses", (self.env.num_obs,))
        self.buffer.add_buffer("privileged_obses", (self.env.num_privileged_obs,))
        self.buffer.add_buffer("rewards", ())
        self.buffer.add_buffer("dones", (), dtype=bool)
        self.buffer.add_buffer("time_outs", (), dtype=bool)

    def _get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
        parser.add_argument("--checkpoint", type=str, help="Path of the model checkpoint to load. Overrides config file if provided.")
        parser.add_argument("--num_envs", type=int, help="Number of environments to create. Overrides config file if provided.")
        parser.add_argument("--headless", type=bool, help="Run headless without creating a viewer window. Overrides config file if provided.")
        parser.add_argument("--sim_device", type=str, help="Device for physics simulation. Overrides config file if provided.")
        parser.add_argument("--rl_device", type=str, help="Device for the RL algorithm. Overrides config file if provided.")
        parser.add_argument("--seed", type=int, help="Random seed. Overrides config file if provided.")
        parser.add_argument("--max_iterations", type=int, help="Maximum number of training iterations. Overrides config file if provided.")
        parser.add_argument(
            "--play_lin_vel_x",
            type=float,
            help="Fixed forward/tangential speed for play/eval only (e.g. 2.0). Ignored during training.",
        )
        parser.add_argument(
            "--play_radius",
            type=float,
            help="Circle radius R for T3_sprint play (yaw=±v/R). Ignored during training.",
        )
        parser.add_argument(
            "--play_yaw_sign",
            type=float,
            help="Turn direction for T3/T4 play: +1 or -1. Ignored during training.",
        )
        parser.add_argument(
            "--play_straight_distance",
            type=float,
            help="Straight-phase distance D (m) before curve for T4_sprint play. Ignored during training.",
        )
        parser.add_argument(
            "--symmetry",
            type=str,
            choices=["none", "dup", "loss", "phase"],
            help="Symmetry method. Overrides algorithm.symmetry.method.",
        )
        self.args = parser.parse_args()

    # Override config file with args if needed
    def _update_cfg_from_args(self):
        cfg_file = os.path.join("envs", "{}.yaml".format(self.args.task))
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        play_keys = ("play_lin_vel_x", "play_radius", "play_yaw_sign", "play_straight_distance")
        for arg in vars(self.args):
            if getattr(self.args, arg) is not None:
                if arg == "num_envs":
                    self.cfg["env"][arg] = getattr(self.args, arg)
                elif arg == "symmetry":
                    self.cfg.setdefault("algorithm", {}).setdefault("symmetry", {})
                    self.cfg["algorithm"]["symmetry"]["method"] = getattr(self.args, arg)
                elif arg in play_keys:
                    if self.test:
                        self.cfg["basic"][arg] = getattr(self.args, arg)
                else:
                    self.cfg["basic"][arg] = getattr(self.args, arg)
        if not self.test:
            self.cfg["viewer"]["record_video"] = False
            for key in play_keys:
                self.cfg["basic"].pop(key, None)

    def _set_seed(self):
        if self.cfg["basic"]["seed"] == -1:
            self.cfg["basic"]["seed"] = np.random.randint(0, 10000)
        print("Setting seed: {}".format(self.cfg["basic"]["seed"]))

        random.seed(self.cfg["basic"]["seed"])
        np.random.seed(self.cfg["basic"]["seed"])
        torch.manual_seed(self.cfg["basic"]["seed"])
        os.environ["PYTHONHASHSEED"] = str(self.cfg["basic"]["seed"])
        torch.cuda.manual_seed(self.cfg["basic"]["seed"])
        torch.cuda.manual_seed_all(self.cfg["basic"]["seed"])

    def _load(self):
        if not self.cfg["basic"]["checkpoint"]:
            return
        if (self.cfg["basic"]["checkpoint"] == "-1") or (self.cfg["basic"]["checkpoint"] == -1):
            self.cfg["basic"]["checkpoint"] = sorted(glob.glob(os.path.join("logs", "**/*.pth"), recursive=True), key=os.path.getmtime)[-1]
        print("Loading model from {}".format(self.cfg["basic"]["checkpoint"]))
        model_dict = torch.load(self.cfg["basic"]["checkpoint"], map_location=self.device, weights_only=True)
        ckpt_sd = model_dict["model"]
        if is_locomotion_to_serial_ckpt(ckpt_sd, self.model.state_dict()):
            expanded = expand_locomotion_state_dict(ckpt_sd, self.model.state_dict())
            self.model.load_state_dict(expanded, strict=True)
            print("Expanded T1 12-DoF locomotion checkpoint into 23-DoF serial policy (legs copied, upper body zeroed)")
            return
        self.model.load_state_dict(ckpt_sd, strict=False)
        try:
            ckpt_curriculum = model_dict["curriculum"]
            if ckpt_curriculum.shape == self.env.curriculum_prob.shape:
                self.env.curriculum_prob = ckpt_curriculum
            else:
                print(
                    "Skipping curriculum load: checkpoint shape {} != env shape {}".format(
                        tuple(ckpt_curriculum.shape), tuple(self.env.curriculum_prob.shape)
                    )
                )
        except Exception as e:
            print(f"Failed to load curriculum: {e}")
        try:
            self.optimizer.load_state_dict(model_dict["optimizer"])
        except Exception as e:
            print(f"Failed to load optimizer: {e}")

    def train(self):
        self.recorder = Recorder(self.cfg)
        obs, infos = self.env.reset()
        obs = obs.to(self.device)
        privileged_obs = infos["privileged_obs"].to(self.device)
        for it in range(self.cfg["basic"]["max_iterations"]):
            # within horizon_length, env.step() is called with same act
            tracking_cmd_vx_sum = 0.0
            tracking_vel_vx_sum = 0.0
            tracking_err_vx_sum = 0.0
            tracking_cmd_vy_sum = 0.0
            tracking_vel_vy_sum = 0.0
            tracking_err_vy_sum = 0.0
            tracking_cmd_yaw_sum = 0.0
            tracking_vel_yaw_sum = 0.0
            tracking_err_yaw_sum = 0.0
            tracking_err_speed_sum = 0.0
            tracking_cmd_radius_sum = 0.0
            tracking_vel_radius_sum = 0.0
            tracking_err_radius_sum = 0.0
            tracking_count = 0
            for n in range(self.cfg["runner"]["horizon_length"]):
                self.buffer.update_data("obses", n, obs)
                self.buffer.update_data("privileged_obses", n, privileged_obs)
                with torch.no_grad():
                    dist = self.model.act(obs)
                    act = dist.sample()
                obs, rew, done, infos = self.env.step(act)
                obs, rew, done = obs.to(self.device), rew.to(self.device), done.to(self.device)
                privileged_obs = infos["privileged_obs"].to(self.device)
                self.buffer.update_data("actions", n, act)
                self.buffer.update_data("rewards", n, rew)
                self.buffer.update_data("dones", n, done)
                self.buffer.update_data("time_outs", n, infos["time_outs"].to(self.device))
                ep_info = {"reward": rew}
                ep_info.update(infos["rew_terms"])
                self.recorder.record_episode_statistics(done, ep_info, it, n == (self.cfg["runner"]["horizon_length"] - 1))
                with torch.no_grad():
                    cmd_vx = self.env.commands[:, 0]
                    vel_vx = self.env.filtered_lin_vel[:, 0]
                    cmd_vy = self.env.commands[:, 1]
                    vel_vy = self.env.filtered_lin_vel[:, 1]
                    cmd_yaw = self.env.commands[:, 2]
                    vel_yaw = self.env.filtered_ang_vel[:, 2]
                    cmd_speed = torch.sqrt(cmd_vx * cmd_vx + cmd_vy * cmd_vy)
                    vel_speed = torch.sqrt(vel_vx * vel_vx + vel_vy * vel_vy)
                    if hasattr(self.env, "circle_radius"):
                        cmd_radius = self.env.circle_radius
                    else:
                        cmd_radius = cmd_speed / torch.abs(cmd_yaw).clamp(min=1.0e-3)
                    vel_radius = vel_speed / torch.abs(vel_yaw).clamp(min=1.0e-3)
                    tracking_cmd_vx_sum += cmd_vx.mean().item()
                    tracking_vel_vx_sum += vel_vx.mean().item()
                    tracking_err_vx_sum += torch.abs(cmd_vx - vel_vx).mean().item()
                    tracking_cmd_vy_sum += cmd_vy.mean().item()
                    tracking_vel_vy_sum += vel_vy.mean().item()
                    tracking_err_vy_sum += torch.abs(cmd_vy - vel_vy).mean().item()
                    tracking_cmd_yaw_sum += cmd_yaw.mean().item()
                    tracking_vel_yaw_sum += vel_yaw.mean().item()
                    tracking_err_yaw_sum += torch.abs(cmd_yaw - vel_yaw).mean().item()
                    tracking_err_speed_sum += torch.abs(cmd_speed - vel_speed).mean().item()
                    tracking_cmd_radius_sum += cmd_radius.mean().item()
                    tracking_vel_radius_sum += vel_radius.mean().item()
                    tracking_err_radius_sum += torch.abs(cmd_radius - vel_radius).mean().item()
                    tracking_count += 1

            with torch.no_grad():
                old_dist = self.model.act(self.buffer["obses"])
                old_actions_log_prob = old_dist.log_prob(self.buffer["actions"]).sum(dim=-1)

            dup_old_log_prob_mir = None
            dup_mir_obs = None
            dup_mir_act = None
            if self.sym_method == "dup":
                with torch.no_grad():
                    dup_mir_obs = mirror_obs(self.buffer["obses"], self.dof_perm, self.dof_sign)
                    dup_mir_act = mirror_action(self.buffer["actions"], self.dof_perm, self.dof_sign)
                    mu_mir_old = self.model.actor(dup_mir_obs)
                    std_mir_old = torch.exp(self.model.logstd).expand_as(mu_mir_old)
                    if torch.isfinite(mu_mir_old).all() and torch.isfinite(std_mir_old).all():
                        dup_old_log_prob_mir = (
                            torch.distributions.Normal(mu_mir_old, std_mir_old).log_prob(dup_mir_act).sum(dim=-1)
                        )

            mean_value_loss = 0
            mean_actor_loss = 0
            mean_bound_loss = 0
            mean_entropy = 0
            for n in range(self.cfg["runner"]["mini_epochs"]):
                values = self.model.est_value(self.buffer["obses"], self.buffer["privileged_obses"])
                last_values = self.model.est_value(obs, privileged_obs)
                with torch.no_grad():
                    self.buffer["rewards"][self.buffer["time_outs"]] = values[self.buffer["time_outs"]]
                    advantages = discount_values(
                        self.buffer["rewards"],
                        self.buffer["dones"] | self.buffer["time_outs"],
                        values,
                        last_values,
                        self.cfg["algorithm"]["gamma"],
                        self.cfg["algorithm"]["lam"],
                    )
                    returns = values + advantages
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                value_loss = F.mse_loss(values, returns)

                dist = self.model.act(self.buffer["obses"])
                actions_log_prob = dist.log_prob(self.buffer["actions"]).sum(dim=-1)
                actor_loss = surrogate_loss(old_actions_log_prob, actions_log_prob, advantages)

                bound_loss = torch.clip(dist.loc - 1.0, min=0.0).square().mean() + torch.clip(dist.loc + 1.0, max=0.0).square().mean()

                entropy = dist.entropy().sum(dim=-1)
                if self.sym_method == "dup" and dup_old_log_prob_mir is not None:
                    # Policy surrogate on M(s), M(a) with copied advantages (SymLoco method 1).
                    # Value / bound / entropy stay on the real trajectory: at 4096 envs the
                    # expanded T0 policy emits huge loc on mirrored obs, bound_loss inf, and
                    # clip_grad_norm_(inf) writes NaN into the actor (crash on the next act()).
                    mu_mir = self.model.actor(dup_mir_obs)
                    std_mir = torch.exp(self.model.logstd).expand_as(mu_mir)
                    if torch.isfinite(mu_mir).all() and torch.isfinite(std_mir).all():
                        dist_mir = torch.distributions.Normal(mu_mir, std_mir)
                        actor_loss_mir = surrogate_loss(
                            dup_old_log_prob_mir,
                            dist_mir.log_prob(dup_mir_act).sum(dim=-1),
                            advantages,
                        )
                        if torch.isfinite(actor_loss_mir):
                            actor_loss = 0.5 * (actor_loss + actor_loss_mir)

                loss = (
                    value_loss
                    + actor_loss
                    + self.cfg["algorithm"]["bound_coef"] * bound_loss
                    + self.cfg["algorithm"]["entropy_coef"] * entropy.mean()
                )
                if self.sym_method == "loss":
                    mirrored_mu = self.model.act(mirror_obs(self.buffer["obses"], self.dof_perm, self.dof_sign)).loc
                    extra_loss = maybe_symmetry_loss(
                        "loss",
                        dist.loc,
                        mirrored_mu,
                        self.dof_perm,
                        self.dof_sign,
                        self.sym_loss_weight,
                    )
                    loss = loss + extra_loss
                self.optimizer.zero_grad()
                if self.sym_method == "dup" and not torch.isfinite(loss).item():
                    continue
                loss.backward()
                if self.sym_method == "dup":
                    grads_ok = True
                    for param in self.model.parameters():
                        if param.grad is not None and not torch.isfinite(param.grad).all():
                            grads_ok = False
                            break
                    if not grads_ok:
                        self.optimizer.zero_grad()
                        continue
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                with torch.no_grad():
                    kl = torch.sum(
                        torch.log(dist.scale / old_dist.scale)
                        + 0.5 * (torch.square(old_dist.scale) + torch.square(dist.loc - old_dist.loc)) / torch.square(dist.scale)
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.cfg["algorithm"]["desired_kl"] * 2:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.cfg["algorithm"]["desired_kl"] / 2:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

                mean_value_loss += value_loss.item()
                mean_actor_loss += actor_loss.item()
                mean_bound_loss += bound_loss.item()
                mean_entropy += entropy.mean()
            mean_value_loss /= self.cfg["runner"]["mini_epochs"]
            mean_actor_loss /= self.cfg["runner"]["mini_epochs"]
            mean_bound_loss /= self.cfg["runner"]["mini_epochs"]
            mean_entropy /= self.cfg["runner"]["mini_epochs"]
            tracking_stats = {}
            if tracking_count > 0:
                tracking_stats = {
                    "tracking/cmd_vx": tracking_cmd_vx_sum / tracking_count,
                    "tracking/vel_vx": tracking_vel_vx_sum / tracking_count,
                    "tracking/abs_err_vx": tracking_err_vx_sum / tracking_count,
                    "tracking/cmd_vy": tracking_cmd_vy_sum / tracking_count,
                    "tracking/vel_vy": tracking_vel_vy_sum / tracking_count,
                    "tracking/abs_err_vy": tracking_err_vy_sum / tracking_count,
                    "tracking/cmd_yaw": tracking_cmd_yaw_sum / tracking_count,
                    "tracking/vel_yaw": tracking_vel_yaw_sum / tracking_count,
                    "tracking/abs_err_yaw": tracking_err_yaw_sum / tracking_count,
                    "tracking/abs_err_speed": tracking_err_speed_sum / tracking_count,
                    "tracking/cmd_radius": tracking_cmd_radius_sum / tracking_count,
                    "tracking/vel_radius": tracking_vel_radius_sum / tracking_count,
                    "tracking/abs_err_radius": tracking_err_radius_sum / tracking_count,
                }
            self.recorder.record_statistics(
                {
                    "value_loss": mean_value_loss,
                    "actor_loss": mean_actor_loss,
                    "bound_loss": mean_bound_loss,
                    "entropy": mean_entropy,
                    "kl_mean": kl_mean,
                    "lr": self.learning_rate,
                    "curriculum/mean_lin_vel_level": self.env.mean_lin_vel_level,
                    "curriculum/mean_ang_vel_level": self.env.mean_ang_vel_level,
                    "curriculum/max_lin_vel_level": self.env.max_lin_vel_level,
                    "curriculum/max_ang_vel_level": self.env.max_ang_vel_level,
                    **tracking_stats,
                },
                it,
            )

            if (it + 1) % self.cfg["runner"]["save_interval"] == 0:
                self.recorder.save(
                    {
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "curriculum": self.env.curriculum_prob,
                    },
                    it + 1,
                )
            print("epoch: {}/{}".format(it + 1, self.cfg["basic"]["max_iterations"]))

    def play(self):
        obs, infos = self.env.reset()
        obs = obs.to(self.device)
        if self.cfg["viewer"]["record_video"]:
            os.makedirs("videos", exist_ok=True)
            name = time.strftime("%Y-%m-%d-%H-%M-%S.mp4", time.localtime())
            record_time = self.cfg["viewer"]["record_interval"]
        print_every = max(1, int(0.5 / self.env.dt))
        step_i = 0
        while True:
            with torch.no_grad():
                dist = self.model.act(obs)
                act = dist.loc
                obs, rew, done, infos = self.env.step(act)
                obs, rew, done = obs.to(self.device), rew.to(self.device), done.to(self.device)
            step_i += 1
            if step_i % print_every == 0:
                cmd_vx = self.env.commands[0, 0].item()
                vel_vx = self.env.filtered_lin_vel[0, 0].item()
                cmd_vy = self.env.commands[0, 1].item()
                vel_vy = self.env.filtered_lin_vel[0, 1].item()
                cmd_yaw = self.env.commands[0, 2].item()
                vel_yaw = self.env.filtered_ang_vel[0, 2].item()
                cmd_speed = (cmd_vx**2 + cmd_vy**2) ** 0.5
                vel_speed = (vel_vx**2 + vel_vy**2) ** 0.5
                if hasattr(self.env, "circle_radius"):
                    r_cmd = self.env.circle_radius[0].item()
                else:
                    r_cmd = cmd_speed / abs(cmd_yaw) if abs(cmd_yaw) > 1.0e-6 else float("inf")
                r_vel = vel_speed / abs(vel_yaw) if abs(vel_yaw) > 1.0e-6 else float("inf")
                print(
                    "cmd_vx={:.2f} vel_vx={:.2f}  cmd_yaw={:.2f} vel_yaw={:.2f}  "
                    "err_speed={:+.2f}  R_cmd={:.2f} R_vel={:.2f} err_R={:+.2f}".format(
                        cmd_vx,
                        vel_vx,
                        cmd_yaw,
                        vel_yaw,
                        vel_speed - cmd_speed,
                        r_cmd,
                        r_vel,
                        r_vel - r_cmd if r_cmd != float("inf") and r_vel != float("inf") else float("nan"),
                    )
                )
                if hasattr(self.env, "phase") and hasattr(self.env, "distance_traveled"):
                    phase = int(self.env.phase[0].item())
                    dist = self.env.distance_traveled[0].item()
                    dist_tgt = self.env.straight_distance_target[0].item()
                    print("phase={} ({})  dist={:.1f}/{:.1f} m".format(phase, "curve" if phase == 1 else "straight", dist, dist_tgt))
            if self.cfg["viewer"]["record_video"]:
                record_time -= self.env.dt
                if record_time < 0:
                    record_time += self.cfg["viewer"]["record_interval"]
                    self.interrupt = False
                    signal.signal(signal.SIGINT, self.interrupt_handler)
                    with imageio.get_writer(os.path.join("videos", name), fps=int(1.0 / self.env.dt)) as self.writer:
                        for frame in self.env.camera_frames:
                            self.writer.append_data(frame)
                    if self.interrupt:
                        raise KeyboardInterrupt
                    signal.signal(signal.SIGINT, signal.default_int_handler)

    def interrupt_handler(self, signal, frame):
        print("\nInterrupt received, waiting for video to finish...")
        self.interrupt = True
