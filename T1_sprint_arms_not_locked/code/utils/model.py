import torch
import torch.nn.functional as F

# T1 locomotion (12 DoF) vs T1_serial_head_fixed (head fixed, 21 DoF).
_T1_OBS, _T1_ACT = 47, 12
_SERIAL_OBS, _SERIAL_ACT = 74, 21
_OBS_PREFIX = 11  # gravity + ang_vel + commands + gait sin/cos
_N_UPPER = 9  # arms + waist — first DoFs; head joints are fixed and collapsed
_N_PRIV = 14


def is_locomotion_to_serial_ckpt(ckpt_sd, model_sd):
    """True when checkpoint is 12-DoF T1 walk and the live model is 21-DoF head-fixed."""
    try:
        return (
            tuple(ckpt_sd["actor.0.weight"].shape) == (256, _T1_OBS)
            and tuple(model_sd["actor.0.weight"].shape) == (256, _SERIAL_OBS)
            and tuple(ckpt_sd["actor.6.weight"].shape) == (_T1_ACT, 128)
            and tuple(model_sd["actor.6.weight"].shape) == (_SERIAL_ACT, 128)
        )
    except KeyError:
        return False


def _serial_leg_obs_indices():
    """T1 obs dim i maps to these head-fixed obs indices (legs only; prefix + 9+12 blocks)."""
    dst = list(range(_OBS_PREFIX))
    pos0 = _OBS_PREFIX
    vel0 = _OBS_PREFIX + _SERIAL_ACT
    act0 = _OBS_PREFIX + 2 * _SERIAL_ACT
    dst += list(range(pos0 + _N_UPPER, pos0 + _N_UPPER + _T1_ACT))
    dst += list(range(vel0 + _N_UPPER, vel0 + _N_UPPER + _T1_ACT))
    dst += list(range(act0 + _N_UPPER, act0 + _N_UPPER + _T1_ACT))
    return dst


def expand_locomotion_state_dict(ckpt_sd, model_sd):
    """Copy T1 12-DoF walk weights into a 21-DoF head-fixed policy.

    Legs keep the omni-walk policy. Arms and waist start at action 0 (default pose):
    new input columns and new action rows are zeroed so they do not disturb the walk.
    """
    out = {k: v.clone() for k, v in model_sd.items()}
    for key, value in ckpt_sd.items():
        if key in out and out[key].shape == value.shape:
            out[key] = value.clone()

    dst = torch.tensor(_serial_leg_obs_indices(), dtype=torch.long)
    actor_in = torch.zeros_like(out["actor.0.weight"])
    actor_in[:, dst] = ckpt_sd["actor.0.weight"]
    out["actor.0.weight"] = actor_in
    out["actor.0.bias"] = ckpt_sd["actor.0.bias"].clone()

    actor_out_w = torch.zeros_like(out["actor.6.weight"])
    actor_out_b = torch.zeros_like(out["actor.6.bias"])
    actor_out_w[_N_UPPER:] = ckpt_sd["actor.6.weight"]
    actor_out_b[_N_UPPER:] = ckpt_sd["actor.6.bias"]
    out["actor.6.weight"] = actor_out_w
    out["actor.6.bias"] = actor_out_b

    logstd = out["logstd"].clone()
    logstd[0, _N_UPPER:] = ckpt_sd["logstd"][0]
    out["logstd"] = logstd

    critic_in = torch.zeros_like(out["critic.0.weight"])
    critic_in[:, dst] = ckpt_sd["critic.0.weight"][:, :_T1_OBS]
    critic_in[:, _SERIAL_OBS : _SERIAL_OBS + _N_PRIV] = ckpt_sd["critic.0.weight"][:, _T1_OBS : _T1_OBS + _N_PRIV]
    out["critic.0.weight"] = critic_in
    out["critic.0.bias"] = ckpt_sd["critic.0.bias"].clone()
    return out


class ActorCritic(torch.nn.Module):

    def __init__(self, num_act, num_obs, num_privileged_obs):
        super().__init__()
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(num_obs + num_privileged_obs, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 1),
        )
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(num_obs, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, num_act),
        )
        self.logstd = torch.nn.parameter.Parameter(torch.full((1, num_act), fill_value=-2.0), requires_grad=True)

    def act(self, obs):
        action_mean = self.actor(obs)
        action_std = torch.exp(self.logstd).expand_as(action_mean)
        return torch.distributions.Normal(action_mean, action_std)

    def est_value(self, obs, privileged_obs):
        critic_input = torch.cat((obs, privileged_obs), dim=-1)
        return self.critic(critic_input).squeeze(-1)
