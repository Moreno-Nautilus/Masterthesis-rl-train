from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


def _arr(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def repo_iiwa_reset_joints() -> np.ndarray:
    return _arr(
        [
            0.0,
            105.0 * math.pi / 180.0,
            0.0,
            -60.0 * math.pi / 180.0,
            0.0,
            118.0 * math.pi / 180.0,
            0.0,
        ]
    )


@dataclass
class CameraConfig:
    serial: str | None = None
    image_key: str = "wrist"
    width: int = 128
    height: int = 128
    crop_x: tuple[int, int] | None = None
    crop_y: tuple[int, int] | None = None


@dataclass
class IiwaInsertionConfig:
    server_url: str = "http://127.0.0.1:5000"
    hz: int = 10
    max_episode_length: int = 100
    action_dim: int = 6
    action_scale_xyz_m: float = 0.01
    action_scale_rot_rad: float = 0.10
    target_pose: np.ndarray = field(
        default_factory=lambda: _arr([0.72, 0.0, 0.05, math.pi, 0.0, 0.0])
    )
    reset_pose: np.ndarray = field(
        default_factory=lambda: _arr([0.72, 0.0, 0.10, math.pi, 0.0, 0.0])
    )
    reward_threshold: np.ndarray = field(
        default_factory=lambda: _arr([0.005, 0.005, 0.005, 0.17, 0.17, 0.17])
    )
    abs_pose_limit_low: np.ndarray = field(
        default_factory=lambda: _arr([0.64, -0.08, 0.03, math.pi - 0.35, -0.35, -0.70])
    )
    abs_pose_limit_high: np.ndarray = field(
        default_factory=lambda: _arr([0.80, 0.08, 0.16, math.pi + 0.35, 0.35, 0.70])
    )
    random_reset: bool = True
    random_xy_range: float = 0.01
    random_yaw_range: float = 0.20
    sparse_reward: bool = True
    success_bonus: float = 1.0
    dense_pos_weight: float = 1.0
    dense_rot_weight: float = 0.25

    # --- Success / reward mode ------------------------------------------------
    # "geometric": success = EE pose within `reward_threshold` of `target_pose`.
    #     Only trustworthy if EE->socket registration is sub-mm. NOT recommended
    #     here: the separate vision pipeline gives socket pose at ~1cm + several deg,
    #     far too coarse for a 1mm-clearance insertion. Kept for completeness.
    # "force_depth": success = the EE reached the seated insertion depth (TCP-frame
    #     z below `seat_depth_z_m` relative to target) AND the wrist force magnitude
    #     exceeded `seat_force_thresh_n` (the seating spike). Independent of the noisy
    #     socket estimate and of the camera. THE DEFAULT for the real task.
    # "manual": success is decided externally (e.g. the human success-button during
    #     demo recording, or a wrapper). The env never auto-fires success.
    #     Use for the very first real runs while calibrating thresholds live.
    # "reach": success = EE position within `reach_radius_m` of the target (position
    #     only, ignores orientation and force). This is the TRIVIAL rig hello-world:
    #     prove RLPD learns on hardware in ~10 min before risking the hard insertion.
    reward_mode: str = "force_depth"
    reach_radius_m: float = 0.03         # reach-task success radius (m)
    # force_depth thresholds (CALIBRATE AT THE RIG — these are placeholders):
    seat_depth_z_m: float = 0.008        # TCP-frame goal z-delta must be within this (m) of seated
    seat_force_thresh_n: float = 5.0     # wrist force magnitude (N) that indicates seating contact
    seat_force_max_n: float = 40.0       # safety: force above this = jam/collision, not success
    joint_reset_on_reset: bool = True
    sticky_gripper_closed: bool = True
    reset_joints: np.ndarray = field(default_factory=repo_iiwa_reset_joints)
    cameras: list[CameraConfig] = field(default_factory=lambda: [CameraConfig()])


@dataclass
class IsaacAdapterConfig:
    task: str = "Isaac-Insertion-CoolingPeg-Iiwa-E2E-Vision-Direct-v0"
    num_envs: int = 1
    state_key: str = "state"
    image_key: str = "wrist"
    keep_rgb_only: bool = True
    sparse_reward: bool = False
    success_reward_threshold: float = 0.75
