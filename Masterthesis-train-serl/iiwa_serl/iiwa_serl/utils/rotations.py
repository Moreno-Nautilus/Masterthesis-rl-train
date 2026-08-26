from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def quat_to_euler(quat: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(np.asarray(quat, dtype=np.float64)).as_euler("xyz")


def euler_to_quat(euler_xyz: np.ndarray) -> np.ndarray:
    return Rotation.from_euler("xyz", np.asarray(euler_xyz, dtype=np.float64)).as_quat()
