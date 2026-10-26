"""Sagittal mirror maps for the T1_serial_head_fixed 21-DoF observation and action.

Permutation and signs are derived from joint names (Left_ <-> Right_;
pitch keeps sign; roll/yaw negate; Waist negates in place).
Isaac Gym is not required.
"""

import torch


OBS_PREFIX = 11
PRIVILEGED_DIM = 14
NUM_ACTIONS = 21

EXPECTED_DOF_NAMES = [
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Waist",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]

FIXED_MIRROR_JOINTS = ("Waist",)

# Prefix (11): gravity y; ang_vel x,z; commands vy, yaw; gait cos, sin.
_OBS_PREFIX_SIGN = torch.tensor(
    [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
    dtype=torch.float32,
)

# Privileged (14): com_y; lin_vel_y; force_y; torque_x, torque_z.
_PRIV_SIGN = torch.tensor(
    [1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
    dtype=torch.float32,
)


def get_symmetry_method(cfg):
    symmetry = cfg.get("algorithm", {}).get("symmetry", {}) if cfg else {}
    method = symmetry.get("method", "none")
    if method is None:
        return "none"
    method = str(method).lower()
    if method not in {"none", "dup", "loss", "phase"}:
        raise ValueError("Unknown symmetry method: {}".format(method))
    return method


def get_symmetry_loss_weight(cfg):
    symmetry = cfg.get("algorithm", {}).get("symmetry", {}) if cfg else {}
    return float(symmetry.get("loss_weight", 4.0))


def _paired_name(name):
    if name.startswith("Left_"):
        return "Right_" + name[len("Left_") :]
    if name.startswith("Right_"):
        return "Left_" + name[len("Right_") :]
    return name


def _sign_for_name(name):
    if name == "Waist":
        return -1.0
    lower = name.lower()
    if "pitch" in lower:
        return 1.0
    if "roll" in lower or "yaw" in lower:
        return -1.0
    return 1.0


def build_dof_mirror(dof_names, device=None):
    """Return (perm, sign) Long/Float tensors of length num_dof."""
    name_to_idx = {name: i for i, name in enumerate(dof_names)}
    perm = []
    sign = []
    for name in dof_names:
        pair = _paired_name(name)
        if pair not in name_to_idx:
            raise ValueError("Mirror pair '{}' for joint '{}' is not in dof_names".format(pair, name))
        perm.append(name_to_idx[pair])
        sign.append(_sign_for_name(name))
    perm_t = torch.tensor(perm, dtype=torch.long, device=device)
    sign_t = torch.tensor(sign, dtype=torch.float32, device=device)
    return perm_t, sign_t


def _as_tensor(values, ref):
    if not torch.is_tensor(values):
        values = torch.as_tensor(values, dtype=ref.dtype, device=ref.device)
    return values.to(device=ref.device, dtype=ref.dtype if values.is_floating_point() else values.dtype)


def mirror_action(act, perm, sign):
    """M_a: permute last dim by perm and multiply by sign."""
    perm = _as_tensor(perm, act).long()
    sign = _as_tensor(sign, act).to(dtype=act.dtype)
    return act[..., perm] * sign


def mirror_obs(obs, perm, sign):
    """M_s on the actor observation (prefix + 3 DoF blocks). 21-DoF: 11+63=74."""
    n_dof = perm.numel() if torch.is_tensor(perm) else len(perm)
    expected = OBS_PREFIX + 3 * n_dof
    if obs.shape[-1] != expected:
        raise ValueError("Observation last dim {} != expected {}".format(obs.shape[-1], expected))
    prefix_sign = _OBS_PREFIX_SIGN.to(device=obs.device, dtype=obs.dtype)
    prefix = obs[..., :OBS_PREFIX] * prefix_sign
    blocks = []
    offset = OBS_PREFIX
    for _ in range(3):
        block = obs[..., offset : offset + n_dof]
        blocks.append(mirror_action(block, perm, sign))
        offset += n_dof
    return torch.cat((prefix, *blocks), dim=-1)


def mirror_privileged(priv):
    """M_p on the 14-D privileged observation."""
    if priv.shape[-1] != PRIVILEGED_DIM:
        raise ValueError("Privileged last dim {} != {}".format(priv.shape[-1], PRIVILEGED_DIM))
    priv_sign = _PRIV_SIGN.to(device=priv.device, dtype=priv.dtype)
    return priv * priv_sign


def assert_involution(x, transform, *args, atol=1e-6):
    twice = transform(transform(x, *args), *args)
    if not torch.allclose(twice, x, atol=atol):
        raise AssertionError("Transform is not an involution (max abs err {})".format((twice - x).abs().max().item()))
    return True


def symmetry_policy_loss(mu, mirrored_mu, perm, sign):
    """L_sym = mean_batch sum_actions (mu(s) - M_a(mu(M_s(s))))^2."""
    return (mu - mirror_action(mirrored_mu, perm, sign)).square().sum(dim=-1).mean()


def maybe_symmetry_loss(method, mu, mirrored_mu, perm, sign, weight):
    """Return w * L_sym if method == 'loss', else a zero scalar on mu's device."""
    if method != "loss":
        return mu.new_zeros(())
    return float(weight) * symmetry_policy_loss(mu, mirrored_mu, perm, sign)


def augment_ppo_batch(
    obses,
    privileged,
    actions,
    advantages,
    returns,
    values,
    old_log_prob,
    perm,
    sign,
    values_mirrored,
    old_log_prob_mirrored,
):
    """Concatenate the mirrored half along the env dimension (dim=1).

    GAE/returns must already have been computed on the real trajectory.
    Advantages and returns are copied. values_mirrored is the critic on the
    mirrored (obs, privileged) pair. old_log_prob_mirrored is log π_old of
    the mirrored actions given the mirrored observations.
    """
    mir_obs = mirror_obs(obses, perm, sign)
    mir_priv = mirror_privileged(privileged)
    mir_act = mirror_action(actions, perm, sign)
    return (
        torch.cat([obses, mir_obs], dim=1),
        torch.cat([privileged, mir_priv], dim=1),
        torch.cat([actions, mir_act], dim=1),
        torch.cat([advantages, advantages], dim=1),
        torch.cat([returns, returns], dim=1),
        torch.cat([values, values_mirrored], dim=1),
        torch.cat([old_log_prob, old_log_prob_mirrored], dim=1),
    )


def runner_uses_dup(method):
    return method == "dup"


def runner_uses_loss(method):
    return method == "loss"


def runner_uses_phase(method):
    """PHASE is environment-only; the runner must not branch on it."""
    return False
