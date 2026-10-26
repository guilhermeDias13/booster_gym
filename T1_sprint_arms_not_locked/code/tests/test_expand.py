"""12-DoF T0 checkpoint expands into the 21-DoF head-fixed policy.

Run from T1_sprint_arms_not_locked/code/:

    python3 tests/test_expand.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from utils.model import (
    _N_PRIV,
    _N_UPPER,
    _OBS_PREFIX,
    _SERIAL_ACT,
    _SERIAL_OBS,
    _T1_ACT,
    _T1_OBS,
    _serial_leg_obs_indices,
    expand_locomotion_state_dict,
    is_locomotion_to_serial_ckpt,
)


def _linear(in_f, out_f, fill):
    weight = torch.full((out_f, in_f), fill)
    bias = torch.full((out_f,), fill)
    return weight, bias


def _sd(n_act, n_obs, n_priv, fill):
    a_w, a_b = _linear(n_obs, 256, fill)
    h_w, h_b = _linear(256, 128, fill)
    o_w, o_b = _linear(128, n_act, fill)
    c_w, c_b = _linear(n_obs + n_priv, 256, fill)
    return {
        "actor.0.weight": a_w,
        "actor.0.bias": a_b,
        "actor.2.weight": h_w,
        "actor.2.bias": h_b,
        "actor.4.weight": h_w.clone(),
        "actor.4.bias": h_b.clone(),
        "actor.6.weight": o_w,
        "actor.6.bias": o_b,
        "logstd": torch.full((1, n_act), fill),
        "critic.0.weight": c_w,
        "critic.0.bias": c_b,
    }


def test_shape_gate_accepts_12_into_21():
    ckpt = _sd(_T1_ACT, _T1_OBS, _N_PRIV, 1.0)
    model = _sd(_SERIAL_ACT, _SERIAL_OBS, _N_PRIV, 0.0)
    assert is_locomotion_to_serial_ckpt(ckpt, model)
    assert not is_locomotion_to_serial_ckpt(model, model)


def test_legs_copied_upper_zero():
    ckpt = _sd(_T1_ACT, _T1_OBS, _N_PRIV, 3.0)
    model = _sd(_SERIAL_ACT, _SERIAL_OBS, _N_PRIV, -2.0)
    out = expand_locomotion_state_dict(ckpt, model)

    assert torch.equal(out["actor.6.weight"][:_N_UPPER], torch.zeros(_N_UPPER, 128))
    assert torch.equal(out["actor.6.bias"][:_N_UPPER], torch.zeros(_N_UPPER))
    assert torch.equal(out["actor.6.weight"][_N_UPPER:], ckpt["actor.6.weight"])
    assert torch.equal(out["actor.6.bias"][_N_UPPER:], ckpt["actor.6.bias"])
    assert torch.equal(out["logstd"][0, :_N_UPPER], torch.full((_N_UPPER,), -2.0))
    assert torch.equal(out["logstd"][0, _N_UPPER:], ckpt["logstd"][0])

    dst = _serial_leg_obs_indices()
    assert len(dst) == _T1_OBS
    assert dst[:_OBS_PREFIX] == list(range(_OBS_PREFIX))
    pos0 = _OBS_PREFIX + _N_UPPER
    assert dst[_OBS_PREFIX : _OBS_PREFIX + _T1_ACT] == list(range(pos0, pos0 + _T1_ACT))
    upper_cols = [i for i in range(_SERIAL_OBS) if i not in set(dst)]
    assert torch.equal(out["actor.0.weight"][:, dst], ckpt["actor.0.weight"])
    assert torch.equal(out["actor.0.weight"][:, upper_cols], torch.zeros(256, len(upper_cols)))
    assert torch.equal(out["actor.0.bias"], ckpt["actor.0.bias"])

    priv0 = _SERIAL_OBS
    assert torch.equal(
        out["critic.0.weight"][:, priv0 : priv0 + _N_PRIV],
        ckpt["critic.0.weight"][:, _T1_OBS : _T1_OBS + _N_PRIV],
    )
    assert torch.equal(out["critic.0.weight"][:, dst], ckpt["critic.0.weight"][:, :_T1_OBS])
    assert torch.equal(out["actor.2.weight"], ckpt["actor.2.weight"])


if __name__ == "__main__":
    tests = [
        test_shape_gate_accepts_12_into_21,
        test_legs_copied_upper_zero,
    ]
    for fn in tests:
        fn()
        print("ok  {}".format(fn.__name__))
    print("all {} tests passed".format(len(tests)))
    sys.exit(0)
