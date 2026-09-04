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


def clip_pose7_relative(
    pose7: np.ndarray,
    origin_pose7: np.ndarray,
    low6: np.ndarray,
    high6: np.ndarray,
) -> np.ndarray:
    """Clip position and orientation relative to an origin pose.

    Translation limits are expressed in the base frame. Rotation limits are
    expressed as a rotation vector in the origin/tool frame. This matches the
    tool-frame increments used by ``compose_delta_pose`` and avoids absolute
    Euler-angle wrap/coupling near the insertion posture.
    """
    pose7 = np.asarray(pose7, dtype=np.float64)
    origin_pose7 = np.asarray(origin_pose7, dtype=np.float64)
    low6 = np.asarray(low6, dtype=np.float64)
    high6 = np.asarray(high6, dtype=np.float64)

    clipped = pose7.copy()
    relative_position = pose7[:3] - origin_pose7[:3]
    clipped[:3] = origin_pose7[:3] + np.clip(
        relative_position, low6[:3], high6[:3]
    )

    origin_rotation = Rotation.from_quat(origin_pose7[3:7])
    target_rotation = Rotation.from_quat(pose7[3:7])
    relative_rotvec = (origin_rotation.inv() * target_rotation).as_rotvec()
    clipped_rotvec = np.clip(relative_rotvec, low6[3:6], high6[3:6])
    clipped[3:7] = (origin_rotation * Rotation.from_rotvec(clipped_rotvec)).as_quat()
    return clipped


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
