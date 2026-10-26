"""Sameness: copied files stay byte-equal to root / T7 meshes.

Run from T1_sprint_arms_locked/code/:

    python3 tests/test_sameness.py
"""

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
TASK_ROOT = os.path.dirname(CODE)
REPO_ROOT = os.path.dirname(TASK_ROOT)
PRECOPY = os.path.join(TASK_ROOT, "_precopy_hashes.txt")
T7_MESHES = os.path.join(REPO_ROOT, "T7", "code", "resources", "T1", "meshes")

IMMUTABLE_REL = (
    "train.py",
    "play.py",
    "envs/t1.py",
    "envs/base_task.py",
    "utils/model.py",
    "utils/buffer.py",
    "utils/recorder.py",
    "utils/terrain.py",
    "utils/utils.py",
    "utils/__init__.py",
    "resources/T1/T1_locomotion.urdf",
    "resources/T1/T1_locomotion.xml",
)

ROOT_REL = {
    "train.py": "train.py",
    "play.py": "play.py",
    "envs/t1.py": "envs/t1.py",
    "envs/base_task.py": "envs/base_task.py",
    "utils/model.py": "utils/model.py",
    "utils/buffer.py": "utils/buffer.py",
    "utils/recorder.py": "utils/recorder.py",
    "utils/terrain.py": "utils/terrain.py",
    "utils/utils.py": "utils/utils.py",
    "utils/__init__.py": "utils/__init__.py",
    "resources/T1/T1_locomotion.urdf": "resources/T1/T1_locomotion.urdf",
    "resources/T1/T1_locomotion.xml": "resources/T1/T1_locomotion.xml",
}

EDITABLE_REL = {
    "envs/t1_sprint_arms_locked.py",
    "envs/T1_sprint_arms_locked.yaml",
    "envs/__init__.py",
    "utils/runner.py",
}

ALLOWED_NEW_REL = {
    "SOURCE_COMMIT",
    "utils/symmetry.py",
    "tests/test_sameness.py",
    "tests/test_curriculum.py",
    "tests/test_symmetry.py",
    "tests/test_method_isolation.py",
}

SKIP_DIR_NAMES = {"__pycache__", "logs", "videos", ".git"}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_precopy():
    assert os.path.isfile(PRECOPY), PRECOPY
    hashes = {}
    with open(PRECOPY, "r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            digest, rel = line.split(None, 1)
            hashes[rel.replace("\\", "/")] = digest.lower()
    return hashes


def _list_code_files():
    files = []
    for dirpath, dirnames, filenames in os.walk(CODE):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIR_NAMES]
        for name in filenames:
            rel = os.path.relpath(os.path.join(dirpath, name), CODE).replace("\\", "/")
            files.append(rel)
    return sorted(files)


def test_precopy_hashes_match_code():
    precopy = _load_precopy()
    mismatches = []
    for rel, digest in precopy.items():
        if rel in EDITABLE_REL:
            continue
        path = os.path.join(CODE, rel.replace("/", os.sep))
        assert os.path.isfile(path), rel
        if sha256_file(path) != digest:
            mismatches.append(rel)
    assert not mismatches, "precopy hash mismatch: {}".format(mismatches)


def test_immutable_files_match_root():
    mismatches = []
    for rel, root_rel in ROOT_REL.items():
        local = os.path.join(CODE, rel.replace("/", os.sep))
        remote = os.path.join(REPO_ROOT, root_rel.replace("/", os.sep))
        assert os.path.isfile(local), rel
        assert os.path.isfile(remote), root_rel
        if sha256_file(local) != sha256_file(remote):
            mismatches.append(rel)
    assert not mismatches, "immutable files diverged from repo root: {}".format(mismatches)


def test_meshes_match_t7():
    mesh_dir = os.path.join(CODE, "resources", "T1", "meshes")
    assert os.path.isdir(mesh_dir), mesh_dir
    names = [name for name in os.listdir(mesh_dir) if name.endswith(".STL")]
    assert names, "no meshes copied"
    mismatches = []
    for name in names:
        src = os.path.join(T7_MESHES, name)
        dst = os.path.join(mesh_dir, name)
        assert os.path.isfile(src), src
        if sha256_file(src) != sha256_file(dst):
            mismatches.append(name)
    assert not mismatches, mismatches


def test_no_unexpected_new_files():
    precopy = _load_precopy()
    known = set(precopy) | set(IMMUTABLE_REL) | EDITABLE_REL | ALLOWED_NEW_REL
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
        test_precopy_hashes_match_code,
        test_immutable_files_match_root,
        test_meshes_match_t7,
        test_no_unexpected_new_files,
    ]
    for fn in tests:
        fn()
        print("ok  {}".format(fn.__name__))
    print("all {} tests passed".format(len(tests)))
    sys.exit(0)
