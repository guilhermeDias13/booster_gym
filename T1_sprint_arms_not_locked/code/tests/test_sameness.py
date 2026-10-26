"""Sameness: shared files stay byte-equal to T1_sprint_arms_locked; robot matches T7 head-fixed.

Run from T1_sprint_arms_not_locked/code/:

    python3 tests/test_sameness.py
"""

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
TASK_ROOT = os.path.dirname(CODE)
REPO_ROOT = os.path.dirname(TASK_ROOT)
LOCKED = os.path.join(REPO_ROOT, "T1_sprint_arms_locked", "code")
T7_T1 = os.path.join(REPO_ROOT, "T7", "code", "resources", "T1")

LOCKED_REL = (
    "train.py",
    "play.py",
    "envs/t1.py",
    "envs/base_task.py",
    "utils/buffer.py",
    "utils/recorder.py",
    "utils/terrain.py",
    "utils/utils.py",
    "utils/__init__.py",
)

T7_REL = {
    "resources/T1/T1_serial_head_fixed.urdf": "T1_serial_head_fixed.urdf",
    "resources/T1/T1_serial_head_fixed.xml": "T1_serial_head_fixed.xml",
}

EDITABLE_REL = {
    "envs/t1_sprint_arms_not_locked.py",
    "envs/T1_sprint_arms_not_locked.yaml",
    "envs/__init__.py",
    "utils/model.py",
    "utils/runner.py",
    "utils/symmetry.py",
}

ALLOWED_NEW_REL = {
    "SOURCE_COMMIT",
    "tests/test_sameness.py",
    "tests/test_curriculum.py",
    "tests/test_symmetry.py",
    "tests/test_method_isolation.py",
    "tests/test_expand.py",
    "tests/test_joint_deviation.py",
}

SKIP_DIR_NAMES = {"__pycache__", "logs", "videos", ".git"}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _list_code_files():
    files = []
    for dirpath, dirnames, filenames in os.walk(CODE):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIR_NAMES]
        for name in filenames:
            rel = os.path.relpath(os.path.join(dirpath, name), CODE).replace("\\", "/")
            files.append(rel)
    return sorted(files)


def test_shared_files_match_locked():
    mismatches = []
    for rel in LOCKED_REL:
        local = os.path.join(CODE, rel.replace("/", os.sep))
        remote = os.path.join(LOCKED, rel.replace("/", os.sep))
        assert os.path.isfile(local), rel
        assert os.path.isfile(remote), rel
        if sha256_file(local) != sha256_file(remote):
            mismatches.append(rel)
    assert not mismatches, "diverged from T1_sprint_arms_locked: {}".format(mismatches)


def test_robot_matches_t7_head_fixed():
    mismatches = []
    for rel, name in T7_REL.items():
        local = os.path.join(CODE, rel.replace("/", os.sep))
        remote = os.path.join(T7_T1, name)
        assert os.path.isfile(local), rel
        assert os.path.isfile(remote), remote
        if sha256_file(local) != sha256_file(remote):
            mismatches.append(rel)
    assert not mismatches, mismatches


def test_meshes_match_locked():
    mesh_dir = os.path.join(CODE, "resources", "T1", "meshes")
    locked_meshes = os.path.join(LOCKED, "resources", "T1", "meshes")
    assert os.path.isdir(mesh_dir), mesh_dir
    names = [name for name in os.listdir(mesh_dir) if name.endswith(".STL")]
    assert names, "no meshes copied"
    mismatches = []
    for name in names:
        src = os.path.join(locked_meshes, name)
        dst = os.path.join(mesh_dir, name)
        assert os.path.isfile(src), src
        if sha256_file(src) != sha256_file(dst):
            mismatches.append(name)
    assert not mismatches, mismatches


def test_no_unexpected_new_files():
    known = set(LOCKED_REL) | set(T7_REL) | EDITABLE_REL | ALLOWED_NEW_REL
    unexpected = []
    for rel in _list_code_files():
        if rel in known:
            continue
        if rel.startswith("resources/T1/meshes/") and rel.endswith(".STL"):
            continue
        unexpected.append(rel)
    assert not unexpected, "unexpected new files: {}".format(unexpected)


if __name__ == "__main__":
    tests = [
        test_shared_files_match_locked,
        test_robot_matches_t7_head_fixed,
        test_meshes_match_locked,
        test_no_unexpected_new_files,
    ]
    for fn in tests:
        fn()
        print("ok  {}".format(fn.__name__))
    print("all {} tests passed".format(len(tests)))
    sys.exit(0)
