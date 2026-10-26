"""Joint deviation weights (arXiv 2502.16230). Run from T1_sprint_arms_not_locked/code/:

    python3 tests/test_joint_deviation.py
"""

import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from utils.symmetry import EXPECTED_DOF_NAMES, build_dof_mirror

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ENV_FILE = os.path.join(CODE, "envs", "t1_sprint_arms_not_locked.py")

PITCH = ("Hip_Pitch", "Knee_Pitch", "Ankle_Pitch")
HIP_YAW_ROLL = ("Hip_Yaw", "Hip_Roll")


def _load_weights_fn():
    with open(ENV_FILE, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=ENV_FILE)
    namespace = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "joint_deviation_weights":
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, ENV_FILE, "exec"), namespace)
    assert "joint_deviation_weights" in namespace
    return namespace["joint_deviation_weights"]


def _expected_weight(name):
    if any(k in name for k in PITCH):
        return 0.5
    if any(k in name for k in HIP_YAW_ROLL):
        return 0.1
    return 0.2


def test_groups_match_paper():
    weights = _load_weights_fn()(EXPECTED_DOF_NAMES)
    assert len(weights) == len(EXPECTED_DOF_NAMES)
    for name, weight in zip(EXPECTED_DOF_NAMES, weights):
        assert weight == _expected_weight(name), (name, weight)
    assert sum(1 for w in weights if w == 0.5) == 6  # hip, knee, ankle pitch, both sides
    assert sum(1 for w in weights if w == 0.1) == 4  # hip yaw and hip roll, both sides
    assert sum(1 for w in weights if w == 0.2) == len(EXPECTED_DOF_NAMES) - 10


def test_weights_survive_mirror():
    weights = torch.tensor(_load_weights_fn()(EXPECTED_DOF_NAMES), dtype=torch.float)
    perm, _sign = build_dof_mirror(EXPECTED_DOF_NAMES)
    assert torch.equal(weights[perm], weights)


if __name__ == "__main__":
    tests = [
        test_groups_match_paper,
        test_weights_survive_mirror,
    ]
    for fn in tests:
        fn()
        print("ok  {}".format(fn.__name__))
    print("all {} tests passed".format(len(tests)))
    sys.exit(0)
