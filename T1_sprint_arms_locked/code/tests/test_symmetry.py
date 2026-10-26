"""Unit tests for sagittal mirroring. Run from T1_sprint_arms_locked/code/:

    python3 tests/test_symmetry.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch

from utils.symmetry import (
    EXPECTED_DOF_NAMES,
    FIXED_MIRROR_JOINTS,
    OBS_PREFIX,
    PRIVILEGED_DIM,
    assert_involution,
    build_dof_mirror,
    get_symmetry_method,
    mirror_action,
    mirror_obs,
    mirror_privileged,
    maybe_symmetry_loss,
    symmetry_policy_loss,
)


EXPECTED_PERM = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]
EXPECTED_SIGN = [
    1.0,
    -1.0,
    -1.0,
    1.0,
    1.0,
    -1.0,
    1.0,
    -1.0,
    -1.0,
    1.0,
    1.0,
    -1.0,
]


def _default_pose_from_yaml(dof_names, joint_angles):
    pose = []
    for name in dof_names:
        value = joint_angles.get("default", 0.0)
        for key, angle in joint_angles.items():
            if key != "default" and key in name:
                value = angle
        pose.append(float(value))
    return torch.tensor(pose, dtype=torch.float32)


def test_build_dof_mirror_matches_table():
    perm, sign = build_dof_mirror(EXPECTED_DOF_NAMES)
    assert perm.tolist() == EXPECTED_PERM
    assert torch.allclose(sign, torch.tensor(EXPECTED_SIGN))


def test_permutation_involution_fixed_points():
    perm, _ = build_dof_mirror(EXPECTED_DOF_NAMES)
    twice = perm[perm]
    assert torch.equal(twice, torch.arange(len(EXPECTED_DOF_NAMES)))
    for i, name in enumerate(EXPECTED_DOF_NAMES):
        if perm[i].item() == i:
            assert name in FIXED_MIRROR_JOINTS, name
        else:
            assert name not in FIXED_MIRROR_JOINTS, name


def test_action_and_obs_are_involutions():
    perm, sign = build_dof_mirror(EXPECTED_DOF_NAMES)
    act = torch.randn(4, 12)
    obs = torch.randn(4, OBS_PREFIX + 3 * 12)
    assert_involution(act, mirror_action, perm, sign)
    assert_involution(obs, mirror_obs, perm, sign)
    twice = mirror_obs(mirror_obs(obs, perm, sign), perm, sign)
    assert torch.allclose(twice[..., 9:11], obs[..., 9:11], atol=1e-6)


def test_privileged_involution():
    priv = torch.randn(8, PRIVILEGED_DIM)
    assert_involution(priv, lambda x: mirror_privileged(x))


def test_default_pose_is_symmetric():
    yaml_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "envs", "T1_sprint_arms_locked.yaml")
    with open(yaml_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    angles = cfg["init_state"]["default_joint_angles"]
    pose = _default_pose_from_yaml(EXPECTED_DOF_NAMES, angles)
    perm, sign = build_dof_mirror(EXPECTED_DOF_NAMES)
    mirrored = pose[perm] * sign
    if not torch.allclose(mirrored, pose, atol=1e-6):
        raise AssertionError(
            "Default pose is not sagittally symmetric; M_a would need an offset. "
            "diff={}".format((mirrored - pose).tolist())
        )


def test_phase_shift_negates_gait_channels():
    perm, sign = build_dof_mirror(EXPECTED_DOF_NAMES)
    obs = torch.zeros(1, OBS_PREFIX + 3 * 12)
    obs[0, 9] = 0.6
    obs[0, 10] = -0.3
    mirrored = mirror_obs(obs, perm, sign)
    assert torch.allclose(mirrored[0, 9:11], torch.tensor([-0.6, 0.3]))


def test_loss_uses_mean_not_sample_and_zero_when_symmetric():
    perm, sign = build_dof_mirror(EXPECTED_DOF_NAMES)
    mu = torch.randn(5, 12)
    mirrored_mu = mirror_action(mu, perm, sign)
    loss = symmetry_policy_loss(mu, mirrored_mu, perm, sign)
    assert float(loss) < 1e-6
    extra = maybe_symmetry_loss("none", mu, mirrored_mu, perm, sign, 4.0)
    assert float(extra) == 0.0
    extra_loss = maybe_symmetry_loss("loss", mu, torch.zeros_like(mu), perm, sign, 4.0)
    assert extra_loss.ndim == 0
    assert float(extra_loss) > 0.0


def test_get_symmetry_method():
    assert get_symmetry_method({"algorithm": {"symmetry": {"method": "DUP"}}}) == "dup"
    assert get_symmetry_method({}) == "none"


if __name__ == "__main__":
    tests = [
        test_build_dof_mirror_matches_table,
        test_permutation_involution_fixed_points,
        test_action_and_obs_are_involutions,
        test_privileged_involution,
        test_default_pose_is_symmetric,
        test_phase_shift_negates_gait_channels,
        test_loss_uses_mean_not_sample_and_zero_when_symmetric,
        test_get_symmetry_method,
    ]
    for fn in tests:
        fn()
        print("ok  {}".format(fn.__name__))
    print("all {} tests passed".format(len(tests)))
    sys.exit(0)
