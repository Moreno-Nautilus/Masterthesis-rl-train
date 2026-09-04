"""Offline tests for the assembly-pkl pose loader (pose-format normalisation).

    JAX_PLATFORMS=cpu python -m pytest iiwa_serl/tests/test_pkl_poses.py -q
"""

from __future__ import annotations

import pickle

import numpy as np
from scipy.spatial.transform import Rotation

from iiwa_serl.envs.pkl_poses import load_insert_poses, get_insert


def _write(tmp_path, obj):
    p = tmp_path / "assembly.pkl"
    with open(p, "wb") as f:
        pickle.dump(obj, f)
    return p


def test_pose6_roundtrip(tmp_path):
    pre = [0.70, -0.32, 0.12, 3.06, 0.0, 1.71]
    goal = [0.70, -0.32, 0.05, 3.06, 0.0, 1.71]
    p = _write(tmp_path, {"asmA_insert1": {"preinsert": pre, "goal": goal}})
    poses = load_insert_poses(p)
    np.testing.assert_allclose(poses["asmA_insert1"].preinsert, pre, atol=1e-9)
    np.testing.assert_allclose(poses["asmA_insert1"].goal, goal, atol=1e-9)


def test_pose7_and_dict_and_matrix_normalise_to_pose6(tmp_path):
    pos = [0.7, -0.3, 0.1]
    euler = [0.1, -0.2, 1.5]
    quat = Rotation.from_euler("xyz", euler).as_quat()  # xyzw
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", euler).as_matrix()
    T[:3, 3] = pos
    p = _write(tmp_path, {
        "as_pose7": {"preinsert": np.concatenate([pos, quat])},
        "as_dict": {"preinsert": {"position": pos, "quaternion": quat}},
        "as_matrix": {"preinsert": T},
    })
    poses = load_insert_poses(p)
    expected = np.concatenate([pos, euler])
    for name in ("as_pose7", "as_dict", "as_matrix"):
        np.testing.assert_allclose(poses[name].preinsert, expected, atol=1e-6)


def test_missing_goal_is_allowed(tmp_path):
    p = _write(tmp_path, {"x": {"preinsert": [0, 0, 0, 0, 0, 0]}})
    assert load_insert_poses(p)["x"].goal is None


def test_terse_form_pose_only(tmp_path):
    p = _write(tmp_path, {"x": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]})
    np.testing.assert_allclose(get_insert(p, "x").preinsert, [0.1, 0.2, 0.3, 0, 0, 0], atol=1e-9)


def test_missing_preinsert_raises(tmp_path):
    p = _write(tmp_path, {"x": {"goal": [0, 0, 0, 0, 0, 0]}})
    try:
        load_insert_poses(p)
    except KeyError as e:
        assert "pre-insert" in str(e)
    else:
        raise AssertionError("expected KeyError for missing pre-insert pose")


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d))
            print(f"ok  {name}")
    print("all pkl_poses tests passed")
