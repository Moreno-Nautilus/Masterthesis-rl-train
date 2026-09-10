from scipy.spatial.transform import Rotation as R
import numpy as np
from pyquaternion import Quaternion


def quat_2_euler(quat):
    """calculates and returns: yaw, pitch, roll from given quaternion"""
    return R.from_quat(quat).as_euler("xyz")


def euler_2_quat(xyz):
    """Inverse of quat_2_euler: XYZ euler (rad) -> quaternion (x, y, z, w).

    REWRITTEN 2026-09-10. The upstream implementation was NOT the inverse of
    quat_2_euler and disagreed with it by up to 149 degrees:
      * it read the input as (yaw, pitch, roll) while quat_2_euler returns (x, y, z);
      * it applied `yaw = pi - yaw`;
      * it composed Z*Y*X where scipy's "xyz" is intrinsic X*Y*Z;
      * it returned pyquaternion's (w, x, y, z) while the rest of the codebase (and
        scipy) uses (x, y, z, w).
    On the rig this made RESET_POSE's stored pitch of +74.6 deg come back as -74.6 deg,
    so a reset that should have been an 82 deg move swept ~150 deg — the arm made a wild
    turn into a joint torque limit. Insert 0 only escaped because its orientation
    (roll ~ pi, pitch ~ 0) sat near a symmetric point where the errors cancelled.

    Verified round-trip exact against quat_2_euler over random orientations.
    """
    return R.from_euler("xyz", np.asarray(xyz, dtype=float)).as_quat()
