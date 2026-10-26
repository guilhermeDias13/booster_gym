"""Isolation of BASE / DUP / LOSS / PHASE. Run from T1_sprint_arms_not_locked/code/:

    python3 tests/test_method_isolation.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from utils.symmetry import (
    EXPECTED_DOF_NAMES,
    NUM_ACTIONS,
    OBS_PREFIX,
    augment_ppo_batch,
    build_dof_mirror,
    get_symmetry_method,
    maybe_symmetry_loss,
    runner_uses_dup,
    runner_uses_loss,
    runner_uses_phase,
)


def _fake_batch(n_time=3, n_env=4):
    perm, sign = build_dof_mirror(EXPECTED_DOF_NAMES)
    obses = torch.randn(n_time, n_env, OBS_PREFIX + 3 * NUM_ACTIONS)
    privileged = torch.randn(n_time, n_env, 14)
    actions = torch.randn(n_time, n_env, NUM_ACTIONS)
    advantages = torch.randn(n_time, n_env)
    returns = torch.randn(n_time, n_env)
    values = torch.randn(n_time, n_env)
    old_log_prob = torch.randn(n_time, n_env)
    values_mir = torch.randn(n_time, n_env)
    old_lp_mir = torch.randn(n_time, n_env)
    return {
        "perm": perm,
        "sign": sign,
        "obses": obses,
        "privileged": privileged,
        "actions": actions,
        "advantages": advantages,
        "returns": returns,
        "values": values,
        "old_log_prob": old_log_prob,
        "values_mir": values_mir,
        "old_lp_mir": old_lp_mir,
    }


def _select_batch(method, batch):
    if not runner_uses_dup(method):
        return (
            batch["obses"],
            batch["privileged"],
            batch["actions"],
            batch["advantages"],
            batch["returns"],
            batch["values"],
            batch["old_log_prob"],
        )
    return augment_ppo_batch(
        batch["obses"],
        batch["privileged"],
        batch["actions"],
        batch["advantages"],
        batch["returns"],
        batch["values"],
        batch["old_log_prob"],
        batch["perm"],
        batch["sign"],
        batch["values_mir"],
        batch["old_lp_mir"],
    )


def test_none_keeps_original_tensors():
    batch = _fake_batch()
    out = _select_batch("none", batch)
    assert out[0] is batch["obses"]
    assert out[1] is batch["privileged"]
    assert out[2] is batch["actions"]
    assert out[3] is batch["advantages"]
    assert out[6] is batch["old_log_prob"]
    extra = maybe_symmetry_loss("none", batch["actions"][0], batch["actions"][0], batch["perm"], batch["sign"], 4.0)
    assert float(extra) == 0.0
    assert not runner_uses_dup("none")
    assert not runner_uses_loss("none")


def test_loss_does_not_change_batch():
    batch = _fake_batch()
    out = _select_batch("loss", batch)
    assert out[0].shape == batch["obses"].shape
    assert torch.equal(out[0], batch["obses"])
    assert torch.equal(out[2], batch["actions"])
    extra = maybe_symmetry_loss("loss", batch["actions"][0], torch.zeros_like(batch["actions"][0]), batch["perm"], batch["sign"], 4.0)
    assert float(extra) > 0.0
    assert not runner_uses_dup("loss")
    assert runner_uses_loss("loss")


def test_phase_is_not_a_runner_branch():
    assert get_symmetry_method({"algorithm": {"symmetry": {"method": "phase"}}}) == "phase"
    assert not runner_uses_dup("phase")
    assert not runner_uses_loss("phase")
    assert not runner_uses_phase("phase")
    batch = _fake_batch()
    out = _select_batch("phase", batch)
    assert out[0] is batch["obses"]
    extra = maybe_symmetry_loss("phase", batch["actions"][0], batch["actions"][0], batch["perm"], batch["sign"], 4.0)
    assert float(extra) == 0.0


def test_dup_is_the_only_batch_doubler():
    batch = _fake_batch()
    for method in ("none", "loss", "phase"):
        out = _select_batch(method, batch)
        assert out[0].shape[1] == batch["obses"].shape[1]
    out = _select_batch("dup", batch)
    assert runner_uses_dup("dup")
    assert out[0].shape[1] == 2 * batch["obses"].shape[1]
    assert out[2].shape[1] == 2 * batch["actions"].shape[1]
    assert out[3].shape[1] == 2 * batch["advantages"].shape[1]
    n_env = batch["advantages"].shape[1]
    assert torch.equal(out[3][:, n_env:], batch["advantages"])
    assert torch.equal(out[4][:, n_env:], batch["returns"])
    extra = maybe_symmetry_loss("dup", batch["actions"][0], batch["actions"][0], batch["perm"], batch["sign"], 4.0)
    assert float(extra) == 0.0


if __name__ == "__main__":
    tests = [
        test_none_keeps_original_tensors,
        test_loss_does_not_change_batch,
        test_phase_is_not_a_runner_branch,
        test_dup_is_the_only_batch_doubler,
    ]
    for fn in tests:
        fn()
        print("ok  {}".format(fn.__name__))
    print("all {} tests passed".format(len(tests)))
    sys.exit(0)
