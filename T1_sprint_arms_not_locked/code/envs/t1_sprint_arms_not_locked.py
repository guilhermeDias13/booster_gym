"""Straight-line run, T1_serial_head_fixed (21 DoF, arms and waist free), 1D speed curriculum.

T1_sprint's 2D grid freezes vx at min_lin_vel_x. This class samples
vx ~ U(curr_v_min, curr_v_max) and raises curr_v_max on tracking success
(same expansion as T3_sprint). Head joints stay fixed.
"""

import numpy as np
import torch
from isaacgym.torch_utils import torch_rand_float

from envs.t1 import T1_sprint
from utils.symmetry import build_dof_mirror, get_symmetry_method, mirror_action, mirror_obs, mirror_privileged


def next_v_max(curr, v_hi, rate):
    """Raise the commanded-speed ceiling toward v_hi. Pure helper for tests."""
    return min(float(v_hi), float(curr) + float(rate) * (float(v_hi) - float(curr)))


def clamp_vx(vx, v_min, v_max):
    """Keep a scalar command inside the live curriculum window."""
    return max(float(v_min), min(float(v_max), float(vx)))


def joint_deviation_weights(dof_names):
    """Per-DoF weights of the joint deviation term (arXiv 2502.16230).

    Hip/knee/ankle pitch 0.5, hip yaw/roll 0.1, everything else 0.2.
    The reward itself is the squared error, not the paper's absolute value.
    """
    weights = []
    for name in dof_names:
        if any(k in name for k in ("Hip_Pitch", "Knee_Pitch", "Ankle_Pitch")):
            weights.append(0.5)
        elif any(k in name for k in ("Hip_Yaw", "Hip_Roll")):
            weights.append(0.1)
        else:
            weights.append(0.2)
    return weights


class T1_sprint_arms_not_locked(T1_sprint):
    """Fine-tune T0 walk to >2 m/s in one pass with arms and waist free (head fixed)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        cmd = cfg["commands"]
        v_lo, v_hi = cmd["lin_vel_x"]
        self.curr_v_min = float(cmd.get("min_lin_vel_x", v_lo))
        self.curr_v_max = float(cmd.get("lin_vel_start", min(self.curr_v_min + 0.4, v_hi)))
        self.symmetry_method = get_symmetry_method(cfg)
        self.dof_perm, self.dof_sign = build_dof_mirror(self.dof_names, device=self.device)
        self.joint_dev_weights = torch.tensor(
            joint_deviation_weights(self.dof_names), dtype=torch.float, device=self.device
        )
        self._sync_curriculum_metrics()

    def _reward_joint_deviation(self):
        return torch.sum(self.joint_dev_weights * torch.square(self.dof_pos - self.default_dof_pos), dim=-1)

    def _phase_enabled(self):
        return self.symmetry_method == "phase"

    def _second_half(self):
        return self.gait_process >= 0.5

    def step(self, actions):
        if self._phase_enabled():
            second = self._second_half()
            if second.any():
                mirrored = mirror_action(actions, self.dof_perm, self.dof_sign)
                actions = torch.where(second.unsqueeze(-1), mirrored, actions)
        return super().step(actions)

    def _compute_observations(self):
        super()._compute_observations()
        if not self._phase_enabled():
            return
        second = self._second_half()
        if not second.any():
            return
        mask = second.unsqueeze(-1)
        self.obs_buf = torch.where(mask, mirror_obs(self.obs_buf, self.dof_perm, self.dof_sign), self.obs_buf)
        self.privileged_obs_buf = torch.where(mask, mirror_privileged(self.privileged_obs_buf), self.privileged_obs_buf)
        self.extras["privileged_obs"] = self.privileged_obs_buf

    def _sync_curriculum_metrics(self):
        self.mean_lin_vel_level = self.curr_v_max
        self.max_lin_vel_level = self.curr_v_max
        self.mean_ang_vel_level = 0.0
        self.max_ang_vel_level = 0.0

    def _apply_sprint_command_constraints(self, env_ids):
        if len(env_ids) == 0:
            return
        play_vx = self.cfg["basic"].get("play_lin_vel_x")
        if play_vx is not None:
            self.commands[env_ids, 0] = float(play_vx)
            gait_lo, gait_hi = self.cfg["commands"]["gait_frequency"]
            self.gait_frequency[env_ids] = 0.5 * (gait_lo + gait_hi)
        else:
            self.commands[env_ids, 0] = torch.clamp(
                self.commands[env_ids, 0], min=self.curr_v_min, max=self.curr_v_max
            )
        self.commands[env_ids, 1] = 0.0
        self.commands[env_ids, 2] = 0.0

    def _resample_curriculum_commands(self, env_ids):
        n = len(env_ids)
        if n == 0:
            return
        self.commands[env_ids, 0] = torch_rand_float(
            self.curr_v_min, self.curr_v_max, (n, 1), device=self.device
        ).squeeze(1)
        self.commands[env_ids, 1] = 0.0
        self.commands[env_ids, 2] = 0.0
        self._sync_curriculum_metrics()
        self._apply_sprint_command_constraints(env_ids)

    def _update_curriculum(self, env_ids):
        if not self.cfg["commands"]["curriculum"]:
            self._sync_curriculum_metrics()
            return
        cmd = self.cfg["commands"]
        success = self.episode_length_buf[env_ids] > np.ceil(self.cfg["rewards"]["episode_length_s"] / self.dt) * (
            1 - cmd["episode_length_toler"]
        )
        success &= torch.abs(self.filtered_lin_vel[env_ids, 0] - self.commands[env_ids, 0]) < cmd["lin_vel_x_toler"]
        success &= torch.abs(self.filtered_lin_vel[env_ids, 1] - self.commands[env_ids, 1]) < cmd["lin_vel_y_toler"]
        success &= torch.abs(self.filtered_ang_vel[env_ids, 2] - self.commands[env_ids, 2]) < cmd["ang_vel_yaw_toler"]
        success &= self.filtered_lin_vel[env_ids, 0] >= self.curr_v_min * 0.8
        if success.any():
            self.curr_v_max = next_v_max(self.curr_v_max, cmd["lin_vel_x"][1], cmd["update_rate"])
        self._sync_curriculum_metrics()
