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
    # If set, frames are read from this ROS2 sensor_msgs/Image topic (via ROSCameraProvider)
    # instead of a direct pyrealsense2 pipeline. Use for the mv_launch RealSense driver.
    ros_topic: str | None = None
    # If True this camera is captured into the obs dict (so the recorder can SAVE it) but is
    # NOT registered in the training observation_space / policy input. Use for the fixed ZED
    # scene camera recorded for backup only.
    record_only: bool = False


@dataclass
class IiwaInsertionConfig:
    server_url: str = "http://127.0.0.1:5000"
    hz: int = 10
    max_episode_length: int = 100
    action_dim: int = 6
    action_scale_xyz_m: float = 0.01
    action_scale_rot_rad: float = 0.10
    # Per-axis multipliers on action_scale_xyz_m (x, y, z). Lets Z be slower/finer than XY,
    # e.g. for careful insertion approach. 1.0 = same as action_scale_xyz_m.
    action_scale_xyz_mult: np.ndarray = field(
        default_factory=lambda: _arr([1.0, 1.0, 1.0])
    )
    target_pose: np.ndarray = field(
        default_factory=lambda: _arr([0.72, 0.0, 0.05, math.pi, 0.0, 0.0])
    )
    # Captured on THIS left arm at the operator's pre-insert pose (2026-09-02), above the
    # real hole. Replaces a foreign-arm pose that drove the arm to the wrong place and
    # dropped FRI on the first auto-reset.
    reset_pose: np.ndarray = field(
        default_factory=lambda: _arr([0.70085, -0.32389, 0.1221, 3.06627, -0.00284, 1.71917])
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
    random_yaw_range: float = 0.05   # small: keep resets gentle/reachable (was 0.20)
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

    # --- Manual pre-insert reset + relative workspace box ----------------------
    # For the "base anywhere on the table" workflow: each episode starts from a
    # pre-insert pose (aligned above the hole) that the operator jogs to (or that an
    # upstream system provides). The safety box is defined RELATIVE to that captured
    # start pose, so it travels with the hole wherever the base is.
    base_frame_actions: bool = False     # if True: env.step translates in BASE frame (matches teleop jog)
    integrate_action_target: bool = False  # stateful target + direct joint hold; enable for manual teleop only
    debug_step: bool = False             # if True: env.step prints per-step cmd/measured pose gap (diagnostic)
    manual_reset: bool = False           # if True: don't auto-move on reset; hold current pose as origin
    relative_pose_limit: bool = False    # if True: clip relative to the episode start pose, not abs box
    reuse_last_camera_frame: bool = False  # tolerate a transient camera read failure after one good frame
    # Force-triggered retract safety net: if measured wrist force magnitude exceeds
    # `force_retract_thresh_n`, env.step OVERRIDES the policy action and commands a small +Z
    # retract (`force_retract_step_m`, base-frame up) instead of pushing further. Keeps contact
    # under the FRI force/velocity guard so the arm backs off before FRI drops. Off by default.
    force_retract_enable: bool = False
    # Threshold is force ABOVE the episode baseline (captured at reset, no contact), because the
    # torque-estimated force has a large constant offset (~90-120 N gravity of peg+gripper). We
    # trigger on the CONTACT INCREMENT, not absolute magnitude.
    force_retract_thresh_n: float = 20.0   # N above baseline; back off above this
    force_retract_step_m: float = 0.003    # m per step to retract (along -insertion_axis)

    # --- Residual-on-nominal insertion (HIL_RESIDUAL_SPEC.md) ------------------
    # residual_enable=False  -> E2E / free-delta: policy learns the whole pre-insert->seat
    #                           delta from scratch (today's working behaviour). FALLBACK.
    # residual_enable=True   -> a slow nominal insertion trajectory auto-plays; the policy
    #                           (and the operator) add a residual ORTHOGONAL to the insertion
    #                           axis (2 lateral trans + all 3 rotations); the nominal owns the
    #                           along-axis forward push via an integer clock.
    residual_enable: bool = False
    # Per-insert straight-line insertion endpoints (EE pose6) + unit world/base insertion axis
    # + nominal waypoint count. Populated from the assembly handoff per experiment config.
    # insertion_axis is NOT always +Z (plumbers k=2 is +Y) — it drives the residual projection
    # AND the force-retract direction.
    insertion_axis: np.ndarray = field(default_factory=lambda: _arr([0.0, 0.0, -1.0]))
    nominal_reset_pose6: np.ndarray = field(default_factory=lambda: _arr([0, 0, 0, 0, 0, 0]))
    nominal_goal_pose6: np.ndarray = field(default_factory=lambda: _arr([0, 0, 0, 0, 0, 0]))
    nominal_n: int = 1
    # Forward pauses when the contact increment (force above episode baseline) exceeds this.
    nominal_pause_force_n: float = 8.0
    # Residual semantics (reworked after supervisor call 2026-09-03):
    #  * The residual's ORTHOGONAL-to-axis translation ACCUMULATES into a PERSISTENT lateral
    #    offset (metres) that permanently shifts the whole insertion line — NOT a one-step nudge.
    #  * The residual's ALONG-axis translation modulates the nominal forward SPEED via a scale
    #    (1 + forward_gain * a_along), clamped to [min,max]. 0->nominal, negative->halt/back off
    #    a little to feel contact. Clamps keep it from lunging forward or yanking back.
    #  * Rotation is OFF by default (locked to nominal); enable for pipe/screw where the
    #    rotational approach error is not negligible.
    # forward_gain is now in NORMALIZED-action units (#16): scale = 1 + gain*action_along_norm,
    # action_along_norm in [-1,1]. gain=1.5 -> a full pull (-1) reaches the -0.5 halt clamp.
    residual_forward_gain: float = 1.5
    residual_min_forward_scale: float = -0.5  # most it can ease back per step (x nominal step)
    residual_max_forward_scale: float = 1.0   # never faster than the nominal rate
    residual_lateral_gain: float = 1.0      # how much orthogonal residual accumulates per step
    residual_lateral_limit: np.ndarray = field(default_factory=lambda: _arr([0.03, 0.03, 0.03]))
    residual_rotation_enable: bool = False
    # #8/#9: append [nominal_progress, force_increment, lateral_offset(3)] to the state vector in
    # residual mode so the compliant-controller state is observable (Markov). ON; +5 state dims.
    residual_obs_augment: bool = True
    # Nominal forward speed (mm/s): sets the EFFECTIVE nominal step count = dist/(speed*dt), so the
    # push runs at a safe fixed contact speed (NOT the handoff's fine 0.18mm/step). Default 4 mm/s.
    nominal_speed_mm_s: float = 4.0
    # extra steps beyond the nominal length for lateral search / contact (per-insert horizon).
    episode_search_margin: int = 150
    # Per-insert JOINT configs from the handoff (frame-independent). goal_joints is FK'd at reset to
    # get the goal pose + insertion axis in the robot's own frame (fixes #2/#3). None -> use config
    # target_pose/insertion_axis directly (E2E / non-handoff).
    reset_joints_target: np.ndarray | None = None
    goal_joints: np.ndarray | None = None
    # relative box (metres / rad) around the captured pre-insert start pose:
    rel_limit_low: np.ndarray = field(
        default_factory=lambda: _arr([-0.03, -0.03, -0.06, -0.20, -0.20, -0.35])
    )
    rel_limit_high: np.ndarray = field(
        default_factory=lambda: _arr([0.03, 0.03, 0.02, 0.20, 0.20, 0.35])
    )
    # Captured on THIS left arm at the pre-insert pose (2026-09-02); matches reset_pose above.
    reset_joints: np.ndarray = field(
        default_factory=lambda: _arr([-0.17289, 1.04601, 0.63629, -1.06604, -0.60515, 1.1218, 1.85074])
    )
    cameras: list[CameraConfig] = field(
        default_factory=lambda: [
            CameraConfig(ros_topic="/realsense_1/camera/color/image_raw")
        ]
    )


@dataclass
class IsaacAdapterConfig:
    task: str = "Isaac-Insertion-CoolingPeg-Iiwa-E2E-Vision-Direct-v0"
    num_envs: int = 1
    state_key: str = "state"
    image_key: str = "wrist"
    keep_rgb_only: bool = True
    sparse_reward: bool = False
    success_reward_threshold: float = 0.75
