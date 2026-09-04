"""Loader for the assembly-pipeline pickle of annotated insertion poses.

The final assembly pipeline hands us a pickle describing, per insert, the fixed
**pre-insert** pose (used as the policy's hard reset target — the base sits at ONE
fixed table position so this pose is repeatable) and, likely, the **goal**
(seated / inserted) pose. We convert whatever pose convention the pipeline uses
into our internal **pose6 = [x, y, z, rx, ry, rz]** (metres + intrinsic-xyz Euler
radians, the same convention as `config.reset_pose` / `config.target_pose`).

>>> IMPORTANT — the exact pkl schema is NOT yet fixed. This loader accepts the
>>> most likely shapes and normalises them; once the real file arrives, confirm
>>> the schema and trim this to match. The assumptions are called out inline.

Expected (assumed) top-level shape — a dict keyed by insert name::

    {
      "asmA_insert1": {"preinsert": <pose>, "goal": <pose>},
      "asmA_insert2": {"preinsert": <pose>, ...},
      ...
    }

where each <pose> is one of:
  * pose6  : [x, y, z, rx, ry, rz]                (len 6, Euler xyz radians)
  * pose7  : [x, y, z, qx, qy, qz, qw]            (len 7, scipy quat order xyzw)
  * a 4x4 homogeneous transform                   (shape (4, 4))
  * dict   : {"position": [...3], "quaternion"|"orientation": [...4 xyzw]}
             or {"position": [...3], "euler": [...3]}

Alternate pre-insert / goal key spellings accepted:
  preinsert: "preinsert", "pre_insert", "pre-insert", "reset", "reset_pose", "approach"
  goal:      "goal", "goal_pose", "target", "target_pose", "inserted", "seated"
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

_PREINSERT_KEYS = ("preinsert", "pre_insert", "pre-insert", "reset", "reset_pose", "approach")
_GOAL_KEYS = ("goal", "goal_pose", "target", "target_pose", "inserted", "seated")


@dataclass
class InsertPoses:
    """Resolved poses for one insert, in our pose6 convention."""
    name: str
    preinsert: np.ndarray            # pose6 — hard reset target
    goal: Optional[np.ndarray] = None  # pose6 — seated pose (target_pose), if provided


def _to_pose6(pose) -> np.ndarray:
    """Normalise any accepted pose representation to pose6 [x,y,z,rx,ry,rz]."""
    # dict form
    if isinstance(pose, dict):
        pos = np.asarray(pose["position"], dtype=np.float64).reshape(3)
        if "euler" in pose:
            rot = np.asarray(pose["euler"], dtype=np.float64).reshape(3)
        else:
            quat = pose.get("quaternion", pose.get("orientation"))
            if quat is None:
                raise ValueError(f"pose dict has no orientation/quaternion/euler: {list(pose)}")
            rot = Rotation.from_quat(np.asarray(quat, dtype=np.float64).reshape(4)).as_euler("xyz")
        return np.concatenate([pos, rot])

    arr = np.asarray(pose, dtype=np.float64)
    if arr.shape == (4, 4):                       # homogeneous transform
        pos = arr[:3, 3]
        rot = Rotation.from_matrix(arr[:3, :3]).as_euler("xyz")
        return np.concatenate([pos, rot])
    if arr.shape == (6,):                          # already pose6
        return arr
    if arr.shape == (7,):                          # pose7 (xyzw quat)
        return np.concatenate([arr[:3], Rotation.from_quat(arr[3:]).as_euler("xyz")])
    raise ValueError(f"Unrecognised pose shape {arr.shape}; expected 6, 7, or (4,4).")


def _pick(entry: dict, keys) -> Optional[object]:
    for k in keys:
        if k in entry:
            return entry[k]
    return None


def load_insert_poses(pkl_path: str | Path) -> dict[str, InsertPoses]:
    """Load and normalise the assembly pkl into {insert_name: InsertPoses}.

    Raises with a clear message if an entry lacks a pre-insert pose (that one is
    mandatory — it is the reset target). A missing goal is allowed (falls back to
    manual reward, no geometric target).
    """
    pkl_path = Path(pkl_path)
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    if not isinstance(raw, dict):
        raise TypeError(
            f"{pkl_path} did not unpickle to a dict of inserts (got {type(raw).__name__}). "
            "Confirm the assembly-pipeline schema and adapt pkl_poses.load_insert_poses."
        )

    out: dict[str, InsertPoses] = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            # Allow the terse form {name: <preinsert_pose>} with no goal.
            out[str(name)] = InsertPoses(name=str(name), preinsert=_to_pose6(entry))
            continue
        pre = _pick(entry, _PREINSERT_KEYS)
        if pre is None:
            raise KeyError(
                f"insert '{name}' has no pre-insert pose under any of {_PREINSERT_KEYS}. "
                f"Keys present: {list(entry)}."
            )
        goal = _pick(entry, _GOAL_KEYS)
        out[str(name)] = InsertPoses(
            name=str(name),
            preinsert=_to_pose6(pre),
            goal=_to_pose6(goal) if goal is not None else None,
        )
    return out


def get_insert(pkl_path: str | Path, insert_name: str) -> InsertPoses:
    """Convenience: load the pkl and return one insert's poses by name."""
    poses = load_insert_poses(pkl_path)
    if insert_name not in poses:
        raise KeyError(
            f"insert '{insert_name}' not in {pkl_path}. Available: {sorted(poses)}."
        )
    return poses[insert_name]
