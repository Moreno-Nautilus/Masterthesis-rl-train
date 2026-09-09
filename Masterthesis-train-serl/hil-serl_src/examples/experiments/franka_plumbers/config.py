"""franka_plumbers — HIL-SERL configs for the 4 plumbers inserts on the FR3.

Franka pivot (2026-09-08). Copied from ram_insertion/config.py, adapted per the plan:

- Camera      : ONE ZED Mini wrist cam ("wrist"); CAMERA_TYPE="zed"; image_keys=["wrist"].
- Robot comm  : ROBOT_CLIENT="ros2" -> cartesian_impedance_control over franka_ros2.
- Reward      : HUMAN via the PS4 pad (PS4RewardWrapper: X=success/end, Triangle=abort);
                NO classifier. TELEOP_DEVICE=ps4.
- Gripper     : pregrasped, GripperCloseEnv + single-arm-fixed-gripper.
- Proprio     : includes the estimated wrench (tcp_force/tcp_torque) from franka_ros2.

Inserts 0..3 share one base config (FrankaPlumbersConfigBase / FrankaPlumbersTrainBase);
each insert only overrides its per-insert poses + image crop. ALL pose/crop numbers below
are PLACEHOLDERS — capture them at the rig by jogging (see FRANKA_RIG_CHECKLIST.md), then
fill per insert. Compliance/precision params are the FR3-range starting points.
"""

import os
import numpy as np

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    GripperCloseEnv,
    PS4RewardWrapper,
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

from experiments.config import DefaultTrainingConfig
from experiments.franka_plumbers.wrapper import FrankaPlumbersEnv


class FrankaPlumbersConfigBase(DefaultEnvConfig):
    """Shared env config for all plumbers inserts. Per-insert subclasses override poses."""

    SERVER_URL = "http://127.0.0.1:5000/"  # unused when ROBOT_CLIENT="ros2"; kept for parity
    ROBOT_CLIENT = "ros2"                    # cartesian_impedance_control over franka_ros2
    CAMERA_TYPE = "zed"                      # ZED Mini wrist camera

    # ONE wrist camera. The env keys cameras off this dict; ZEDCapture is RSCapture-compatible.
    # TODO(rig): serial can stay None for a single ZED; confirm crop by jogging + viewing.
    REALSENSE_CAMERAS = {
        "wrist": {
            "serial_number": None,
            "dim": (1280, 720),
        },
    }
    IMAGE_CROP = {
        # TODO(rig): set the crop so the socket + part tip fill the frame (per insert).
        "wrist": lambda img: img,
    }

    # --- PLACEHOLDER poses (fill at rig by jogging). xyz(m) + euler(rad). ---
    # RESET_POSE sits roughly ALIGNED with the socket, a few cm along the approach axis:
    # ABOVE the base for a vertical insert, or BESIDE it for the horizontal pipe insert,
    # so the wrist camera already sees where the part must go. Capture per insert.
    TARGET_POSE = np.zeros((6,))                       # seated pose
    GRASP_POSE = np.zeros((6,))                        # unused (human hands part in)
    RESET_POSE = np.zeros((6,))                        # fixed pre-insert pose (aligned + offset)
    REWARD_THRESHOLD = np.array([0.01, 0.01, 0.01, 0.2, 0.2, 0.2])  # unused: human reward

    # Safety box around the reset/target region (fill at rig).
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))

    # Reset noise = TRANSLATIONAL ONLY (~1cm XY). Rotational variation comes for free from
    # the human placing the part differently in the hand each grasp, so RZ noise is OFF.
    # (Fixed-spot base: this small XY jitter models approach variation. For a MOVING base,
    # see set_reset_from_base_pose() below — you'd measure the base and recompute RESET_POSE
    # each episode instead of relying on this jitter.)
    RANDOM_RESET = False    # start OFF (day 1: fixed reset); turn ON once insert 0 learns
    RANDOM_XY_RANGE = 0.01  # ±1cm in X/Y
    RANDOM_RZ_RANGE = 0.0   # no yaw jitter (grasp variation supplies rotation)
    # Reduced from the stock (0.01, 0.06, 1) at the rig 2026-09-09: 1 cm and 0.06 rad
    # (3.4 deg) per unit action was too coarse for a 7 cm screw insert — it drove the arm
    # into JOINT TORQUE LIMITS on contact, because a single step could demand a centimetre
    # of penetration into a seated screw. 4 mm / 0.02 rad (1.1 deg) keeps the per-step
    # commanded displacement below the compliance the controller can absorb.
    # Distance from the TCP (o_t_ee, which already includes the Franka Hand) DOWN to the
    # point the tool should pivot about — i.e. how far the screw tip sticks out past the
    # fingers. Rotation is compensated so THAT point stays fixed while the tool tilts.
    # Without it, a 2.3 deg tilt at a 10 cm lever drags the tip ~4 mm sideways and every
    # alignment correction pushes the part out of the hole (rig 2026-09-09).
    # MEASURE THIS on the real screw+gripper; 0.0 = stock flange-centred rotation.
    # HARD AXIS SEPARATION for teleop/demos: while the operator is purely tilting, x/y/z
    # are PINNED to where the tilt started; while purely translating, the attitude is
    # PINNED. Without this the controller's rotational loop drags translation (and vice
    # versa) and a screw cannot be aligned to its hole by hand.
    # NOTE: this constrains the COMMAND, so it also applies to the policy during training.
    # Set False for stock behaviour.
    LOCK_TRANSLATION_DURING_ROTATION = False   # reverted: did not help on the robot
    # Downward-z motion is refused once |Fz| exceeds this (see FrankaEnv.step). Stops the
    # screw being driven harder into a seated socket, which is what tripped the joint
    # torque limits while recording. Upward motion is always allowed. 0 disables.
    # HARD-LOCK the tool orientation for demos: every commanded pose uses RESET_POSE's
    # orientation (straight down), and rotation actions are IGNORED. The task is a pure
    # vertical descent onto the screw, so orientation is not a needed DoF — and locking
    # it removes the translation/rotation coupling entirely instead of trying to cancel
    # it. Set False to restore 6-DoF teleop.
    LOCK_ORIENTATION = True
    Z_FORCE_LIMIT_N = 15.0
    ROTATION_LEVER_M = 0.0    # reverted: coupling is not a fixed geometric lever
    # 0.01 m/unit at 10 Hz = 10 cm/s at full stick. I had cut this to 0.004-0.005 chasing
    # torque limits, which made the arm so slow the operator could not traverse more than
    # a few cm inside an episode — it felt like a hard -x limit but was just speed.
    # The torque limits are handled properly by Z_FORCE_LIMIT_N instead.
    # 0.01 -> 0.004 m/unit (10 -> 4 cm/s at 10 Hz). The actor was RAMMING the base hard
    # enough to trip the joint-torque reflex, and the reflex fires on a fast transient
    # that a 10 Hz guard cannot catch (measured: filtered tau_j read only 9.6% while the
    # robot was already faulted). The only reliable fix is to not generate the impact:
    # a smaller per-step delta means less commanded penetration per control period, so
    # the compliant controller absorbs contact instead of driving through it.
    ACTION_SCALE = (0.01, 0.04, 1)
    # Cap the DOWNWARD z delta only (m per control step). The policy learned from demos
    # where z was at full stick 41% of the time, so it slams down and trips the joint-
    # torque reflex — but halving ACTION_SCALE globally made everything else uselessly
    # slow. This limits only the descent: full speed in x/y and for lifting, gentle
    # into contact. 0.004 m/step at 10 Hz = 4 cm/s down. 0 disables.
    MAX_Z_DOWN_STEP = 0.004
    # The camera preview uses cv2.imshow, which needs an X display. On a headless / SSH rig
    # shell there is none and Qt ABORTS THE PROCESS ("could not connect to display",
    # core dumped) a few steps into the episode — it killed the env smoke on 2026-09-09.
    # Auto-detect instead of hardcoding: keep the preview when running at the machine,
    # skip it over SSH. Force either way with FRANKA_DISPLAY_IMAGE=0|1.
    DISPLAY_IMAGE = (
        os.environ.get("FRANKA_DISPLAY_IMAGE", "1" if os.environ.get("DISPLAY") else "0")
        not in ("0", "false", "False")
    )
    # 100 steps at 10 Hz = only 10 s per episode — not enough time to reposition and then
    # insert. 300 = 30 s.
    MAX_EPISODE_LENGTH = 300
    # Settle pause after each reset before the policy starts moving (a beat to get ready /
    # hit Circle to regrasp). Set 0 to disable.
    POST_RESET_WAIT_S = 1.5

    # EXTRACTION axis for the pre-reset retract: unit vector in BASE frame pointing OUT of
    # the socket. Default +Z suits a VERTICAL (top-down) insert. A HORIZONTAL insert (the
    # pb_pipe one) MUST override this — retracting +Z on a horizontally-seated part drives
    # it sideways into the socket wall. Set it to the measured pull-out direction, e.g.
    # RETRACT_DIR = (-1.0, 0.0, 0.0) for a part inserted along +X.
    # TODO(rig): confirm per insert when capturing poses. Direction = OUT of the socket:
    #   RETRACT_DIR = normalize(RESET_POSE[:3] - TARGET_POSE[:3])
    # i.e. pre-insert MINUS seated (NOT seated-minus-pre-insert, which points INTO the hole).
    RETRACT_DIR = (0.0, 0.0, 1.0)
    RETRACT_DIST = 0.04         # m, pre-reset retract along RETRACT_DIR
    REGRASP_RETRACT_DIST = 0.08  # m, extra clearance for the human hand-off

    # FR3 impedance ranges: transl 10-3000 N/m, rot 1-300 Nm/rad. Starting points; the
    # Ros2FrankaClient maps these onto cartesian_impedance_control's stiffness params.
    # TODO(rig): tune with the arm in the loop.
    # 2026-09-09: aligned with the gains actually TUNED INTO cartesian_impedance_control
    # (K = 1500 translational, 100/100/40 rotational). The previous 2000/150-250 values
    # were stiffer than what the arm is running and would have fought the controller if
    # they ever took effect.
    #
    # NOTE: on the ROS2 backend these are currently INERT. Ros2FrankaClient.update_param()
    # does `ros2 param set` on the controller node, but cartesian_impedance_control does
    # NOT expose stiffness as ROS parameters — the gains are compile-time constants in the
    # header. To actually change stiffness you must edit the controller and rebuild.
    # Kept here so the HTTP path stays correct and so the intent is recorded.
    COMPLIANCE_PARAM = {
        "translational_stiffness": 1500,
        "translational_damping": 80,
        "rotational_stiffness": 100,
        "rotational_damping": 30,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 1500,
        "translational_damping": 80,
        "rotational_stiffness": 100,
        "rotational_damping": 30,
    }


class FrankaPlumbersTrainBase(DefaultTrainingConfig):
    """Shared training config: single ZED wrist, wrench in proprio, human reward."""

    image_keys = ["wrist"]
    classifier_keys = ["wrist"]  # unused (no classifier); kept for schema parity
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    # The learner PREALLOCATES the whole replay buffer. At the stock 200000 with a
    # 128x128x3 image per transition that is ~9.8 GB per image field (~20 GB counting
    # next_observations) requested in one allocation — the learner was OOM-KILLED at
    # startup on this box (2026-09-09).
    # 20000 transitions = ~2 GB and is ample: the 20 demos are 1471 transitions and an
    # on-robot session adds maybe 10-20k more. Raise if a run ever fills it.
    replay_buffer_capacity = 20000
    # Save often (2026-09-09): the learner runs at ~33 it/s, so 5000 steps was ~2.5 min
    # of policy progress at risk per save and 1000 steps of on-robot interaction — data
    # that costs real operator time to re-collect. 1000/500 keeps losses small.
    buffer_period = 500        # online transitions -> disk
    checkpoint_period = 1000   # policy weights -> disk
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"

    # Per-insert config subclasses set this to their EnvConfig.
    env_config_cls = FrankaPlumbersConfigBase

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = FrankaPlumbersEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=self.env_config_cls(),
        )
        env = GripperCloseEnv(env)
        if not fake_env:
            # FORCE PS4 for this experiment (not setdefault — a stale TELEOP_DEVICE=spacemouse
            # would otherwise silently give a non-event expert and drop the human reward).
            os.environ["TELEOP_DEVICE"] = "ps4"
            env = SpacemouseIntervention(env)  # -> PS4Expert via TELEOP_DEVICE=ps4
            # require_expert: hard-fail if no event-capable expert (no silent pose-reward fallback).
            env = PS4RewardWrapper(env, require_expert=True)  # X=success/end, Triangle=abort
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        # NOTE: no reward-classifier wrapper — reward is human (PS4RewardWrapper). The
        # `classifier` arg is accepted for interface parity but intentionally ignored.
        return env


# --- Per-insert configs (0..3). Poses/crops filled at rig; everything else inherited. ---

class EnvConfigInsert0(FrankaPlumbersConfigBase):
    """Insert 0 — SCREW insertion, VERTICAL (tool pointing down, roll ~ pi).

    Captured at the rig 2026-09-09 by hand-jogging with jog_and_capture.py.
    RESET_POSE is the measured pre-insert pose. TARGET_POSE is that pose lowered by
    INSERT_DEPTH along -z (vertical insert), so RETRACT_DIR comes out as +z.
    """

    # BOTH poses MEASURED at the rig by hand-jogging [x, y, z, roll, pitch, yaw]
    # (m, XYZ-Euler rad). Insertion depth between them = 71.7 mm.
    # RE-CAPTURED after the base was moved (2026-09-09). Constructed to spec: same XY as
    # the goal, tool pointing STRAIGHT DOWN (roll = -pi, pitch = 0), yaw matched to the
    # goal, 102.7 mm above it. So the approach is a pure -z descent onto the screw hole.
    RESET_POSE = np.array([0.46136, -0.09502, 0.21092, -3.14159, 0.00000, -1.52317])
    # seated: screw fully down. Measured with ~9 N of contact force, i.e. really seated.
    # MEASURED seated pose (screw fully down) after the base move.
    TARGET_POSE = np.array([0.46136, -0.09502, 0.10826, -3.11550, -0.06806, -1.52317])

    # normalize(RESET[:3] - TARGET[:3]) = (0.001, -0.014, 0.9999) -> essentially pure +z,
    # as expected for a vertical screw insert. OUT of the socket = straight up.
    RETRACT_DIR = (0.0, 0.0, 1.0)

    # Safety box: covers reset + seated AND the full 4 cm reset / 8 cm regrasp retraction
    # along +z (0.178 + 0.08 = 0.258 < 0.298 high bound), with +/-8 cm of xy room for the
    # +/-1 cm reset noise and human corrections, and 2 cm below seated.
    # VERIFY BY HAND before the first env.reset().
    #
    # SIX values: franka_env slices [:3] for the xyz box and [3:] for the rpy box, so a
    # 3-element array leaves rpy_bounding_box EMPTY and clip_safety_box then throws
    # IndexError on the first step (caught by test_demo_recording.py).
    # Rotation bounds: +/-0.35 rad (~20 deg) around the measured seated orientation, which
    # is roll ~ pi. Roll is near the +/-pi wrap so it gets the full range rather than a
    # window that a sign flip would fall outside of.
    # WIDENED 2026-09-09: the original +/-8 cm box left only 80 mm of -x travel from the
    # reset pose, and a tilt walks the TCP toward that wall — the operator hit the clip and
    # "could not go back any further in x". Widened to +/-15 cm in xy and given more z
    # headroom so teleop corrections are never silently clipped mid-insert.
    # STILL VERIFY BY HAND that the whole box is physically clear before resetting.
    # x LOW pushed out to 0.15 m: the operator wants unrestricted -x travel to back
    # away from the socket. 0.15 is still well inside the FR3's reachable workspace
    # (and the arm's own joint limits stop it long before anything structural).
    # z floor lowered 0.0783 -> 0.0580 and x/y given margin: the arm parked 0.1 MM below
    # the old z floor and reset REFUSED (rig 2026-09-09). Bounds computed exactly from
    # captured poses leave no room for where the arm happens to be sitting beforehand.
    ABS_POSE_LIMIT_LOW = np.array([0.1500, -0.2600, 0.0580, -np.pi, -0.60, -2.3232])
    ABS_POSE_LIMIT_HIGH = np.array([0.6500, 0.0900, 0.3800, np.pi, 0.60, -0.7232])

    # Reset noise ON from the first demos (user decision 2026-09-09): the demos should
    # already cover the +/-1 cm of start variation the policy will face, rather than all
    # starting from one exact pose. Translation only — rotation variation comes from the
    # human placing the screw differently in the hand each grasp.
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.01   # +/-1 cm in X and Y
    RANDOM_RZ_RANGE = 0.0    # no yaw jitter


class EnvConfigInsert1(FrankaPlumbersConfigBase):
    pass


class EnvConfigInsert2(FrankaPlumbersConfigBase):
    pass


class EnvConfigInsert3(FrankaPlumbersConfigBase):
    pass


class TrainConfigInsert0(FrankaPlumbersTrainBase):
    env_config_cls = EnvConfigInsert0


class TrainConfigInsert1(FrankaPlumbersTrainBase):
    env_config_cls = EnvConfigInsert1


class TrainConfigInsert2(FrankaPlumbersTrainBase):
    env_config_cls = EnvConfigInsert2


class TrainConfigInsert3(FrankaPlumbersTrainBase):
    env_config_cls = EnvConfigInsert3
