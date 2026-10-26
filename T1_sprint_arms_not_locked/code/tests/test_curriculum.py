"""1D speed curriculum helpers. Run from T1_sprint_arms_not_locked/code/:

    python3 tests/test_curriculum.py
"""

import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ENV_FILE = os.path.join(CODE, "envs", "t1_sprint_arms_not_locked.py")


def _load_helpers():
    with open(ENV_FILE, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=ENV_FILE)
    namespace = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("next_v_max", "clamp_vx"):
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, ENV_FILE, "exec"), namespace)
    assert "next_v_max" in namespace and "clamp_vx" in namespace
    return namespace["next_v_max"], namespace["clamp_vx"]


def test_success_raises_ceiling():
    next_v_max, _ = _load_helpers()
    first = next_v_max(0.9, 3.5, 0.1)
    assert first > 0.9, first
    assert first < 3.5, first
    assert abs(first - (0.9 + 0.1 * (3.5 - 0.9))) < 1.0e-9


def test_ceiling_caps_at_v_hi():
    next_v_max, _ = _load_helpers()
    assert next_v_max(3.5, 3.5, 0.1) == 3.5
    assert next_v_max(3.49, 3.5, 0.1) <= 3.5


def test_vx_not_stuck_at_minimum():
    _, clamp_vx = _load_helpers()
    assert clamp_vx(0.1, 0.5, 0.9) == 0.5
    assert clamp_vx(2.4, 0.5, 2.2) == 2.2
    mid = clamp_vx(1.4, 0.5, 2.2)
    assert mid == 1.4
    assert mid != 0.5


def test_repeated_success_passes_two_mps():
    next_v_max, _ = _load_helpers()
    curr = 0.9
    for _ in range(20):
        curr = next_v_max(curr, 3.5, 0.1)
    assert curr > 2.0, curr


if __name__ == "__main__":
    tests = [
        test_success_raises_ceiling,
        test_ceiling_caps_at_v_hi,
        test_vx_not_stuck_at_minimum,
        test_repeated_success_passes_two_mps,
    ]
    for fn in tests:
        fn()
        print("ok  {}".format(fn.__name__))
    print("all {} tests passed".format(len(tests)))
    sys.exit(0)
