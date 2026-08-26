from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def pose6_to_pose7(pose6: np.ndarray) -> np.ndarray:
    pose6 = np.asarray(pose6, dtype=np.float64)
    return np.concatenate([pose6[:3], Rotation.from_euler("xyz", pose6[3:]).as_quat()])


def pose7_to_pose6(pose7: np.ndarray) -> np.ndarray:
    pose7 = np.asarray(pose7, dtype=np.float64)
    return np.concatenate([pose7[:3], Rotation.from_quat(pose7[3:]).as_euler("xyz")])


def clip_pose7(pose7: np.ndarray, low6: np.ndarray, high6: np.ndarray) -> np.ndarray:
    pose6 = pose7_to_pose6(pose7)
    pose6[:3] = np.clip(pose6[:3], np.asarray(low6)[:3], np.asarray(high6)[:3])
    pose6[3:] = np.clip(pose6[3:], np.asarray(low6)[3:], np.asarray(high6)[3:])
    return pose6_to_pose7(pose6)


def compose_delta_pose(current_pose7: np.ndarray, delta_xyz: np.ndarray, delta_rot_xyz: np.ndarray) -> np.ndarray:
    current_pose7 = np.asarray(current_pose7, dtype=np.float64)
    delta_xyz = np.asarray(delta_xyz, dtype=np.float64)
    delta_rot_xyz = np.asarray(delta_rot_xyz, dtype=np.float64)
    current_rot = Rotation.from_quat(current_pose7[3:])
    target_pos = current_pose7[:3] + current_rot.apply(delta_xyz)
    target_rot = current_rot * Rotation.from_euler("xyz", delta_rot_xyz)
    return np.concatenate([target_pos, target_rot.as_quat()])


def goal_delta_in_tcp_frame(current_pose7: np.ndarray, target_pose7: np.ndarray) -> np.ndarray:
    current_pose7 = np.asarray(current_pose7, dtype=np.float64)
    target_pose7 = np.asarray(target_pose7, dtype=np.float64)
    current_rot = Rotation.from_quat(current_pose7[3:])
    target_rot = Rotation.from_quat(target_pose7[3:])
    pos_delta_world = target_pose7[:3] - current_pose7[:3]
    pos_delta_tcp = current_rot.inv().apply(pos_delta_world)
    rot_delta = (target_rot * current_rot.inv()).as_euler("xyz")
    return np.concatenate([pos_delta_tcp, rot_delta])
