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
    RANDOM_Z_RANGE = 0.0    # z jitter OFF by default (stock jitters the xy plane only)
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
    """Insert 1 — HORIZONTAL insert along -Y (tool points sideways, pitch ~1.3 rad).

    Captured at the rig 2026-09-10. Unlike insert 0 (vertical descent), the approach here
    is 111 mm along -Y (96% of the motion), so RETRACT_DIR and the Z-specific guards differ.
    RESET takes x and z from the GOAL and only y from the measured start, so the approach
    is a pure -Y translation with no lateral component in the nominal path.
    """

    # MEASURED seated pose [x, y, z, roll, pitch, yaw] (m, XYZ-Euler rad)
    # RE-CAPTURED 2026-09-10 under control_mode 2 (compliant rotation), i.e. the attitude
    # the wrist actually settles at with a soft rotational loop — not the stiff-mode
    # reading, which the arm could not hold during the approach.
    TARGET_POSE = np.array([0.44826, 0.17034, 0.06129, -3.06513, 1.24848, -1.49364])
    # x, z, and ORIENTATION from the goal; y backed off 100 mm along the approach axis.
    RESET_POSE = np.array([0.44826, 0.07034, 0.06129, -3.06513, 1.24848, -1.49364])

    # normalize(RESET[:3] - TARGET[:3]) = (0, -1, 0): OUT of the socket is -Y.
    # The +Z default would drag the part sideways into the socket wall.
    RETRACT_DIR = (0.0, -1.0, 0.0)

    # Safety box: +/-15 cm around the work volume in x/z, y spanning the full approach plus
    # room for the 4 cm reset / 8 cm regrasp retraction along -y, with margin so the arm's
    # parked pose is never marginally outside (that refused a reset on insert 0).
    # Rotation bounds are WIDE on purpose. Built tightly around the captured insert
    # orientation they rejected the very first reset: the arm was parked at pitch -0.19 /
    # yaw -0.67, outside the window, and clip_safety_box would have ROTATED the retract
    # target — which _retract_target refuses (correctly). Orientation is hard-locked by
    # LOCK_ORIENTATION anyway, so these bounds only need to not fight the parked pose.
    # z floor 0.035: a HARD table/fixture guard. Reset and seated both sit at z=0.063,
    # so this leaves 28 mm of downward room for corrections while making it impossible
    # for the policy or a mis-set target to drive the pipe into the table.
    # y low: -0.0494 -> -0.1202 -> -0.0300 (2026-09-10, twice in one day).
    # The first widening fixed a retract that would not fit, but it handed the POLICY 191 mm
    # of empty space BEHIND the reset pose to wander into, when the task only needs the
    # 100 mm between RESET (+0.070) and TARGET (+0.170). The actor then repeatedly drove to
    # the box edge and stalled the reset. A tighter floor is the actual fix: it keeps the
    # policy in the region where the task lives, and 100 mm below RESET is still ample for
    # the 40 mm retract plus margin. Retract shortening + retract-skip (see wrapper.py)
    # handle the edge cases; this stops them arising in the first place.
    ABS_POSE_LIMIT_LOW = np.array([0.2984, -0.0300, 0.0350, -np.pi, -np.pi/2, -np.pi])
    ABS_POSE_LIMIT_HIGH = np.array([0.5984, 0.2506, 0.2100, np.pi, np.pi/2, np.pi])

    # ORIENTATION LOCKED (2026-09-10, after trying the unlocked version on the rig).
    # Unlocked, ONE number (ACTION_SCALE rot) has to serve two masters: the operator's
    # counter-steer needs it LARGE, the untrained policy needs it SMALL. At 0.10 the agent
    # tumbled the wrist into its 12 Nm joint-torque limit on nearly every episode; at 0.05
    # the operator could no longer rescue an episode. There is no value that satisfies both.
    # Locking removes rotation from the ACTION SPACE entirely: the policy cannot tumble the
    # wrist, and control_mode 2's soft wrist lets the BORE do the aligning mechanically —
    # which is the whole point of the compliant controller. This is what insert 0 used.
    # COST: the demos contain the operator's rotation corrections (13.3% of steps at full
    # stick); locked, that part of the demo data is unlearnable and the policy must seat the
    # pipe on translation + mechanical compliance alone. If the bore cannot absorb the
    # misalignment this will fail, and the honest next move is re-fixturing, not more gains.
    #
    # WARMUP LOCK: an INT means "locked for the first N episodes, then released". Locked
    # while the policy is still random (that is when it tumbles the wrist into the torque
    # limit); released once it has some experience, so it can still learn the rotation
    # corrections the demos contain. True = lock forever, False = never lock.
    LOCK_ORIENTATION = 6

    # Rotation authority for this insert: 0.04 (stock) was too weak to counter-steer by
    # hand against the soft wrist of control_mode 2; 0.10 was fine for a HUMAN jogging but
    # far too much for the POLICY — the same number is the agent's per-step rotation budget,
    # and an untrained agent sampling +/-0.10 rad at 10 Hz tumbles the wrist straight into
    # its 12 Nm joint-torque limit on almost every episode (observed 2026-09-10).
    # Back to 0.10 (2026-09-10): 0.05 made the operator's counter-steer too slow to rescue
    # an episode, and an operator who cannot intervene is worse than a twitchy policy —
    # HIL-SERL learns from the interventions. It also MATCHES THE DEMOS, which were recorded
    # at 0.10: at 0.05 every demo action meant 2x the rotation it would at execution.
    ACTION_SCALE = (0.01, 0.10, 1)

    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.01
    RANDOM_RZ_RANGE = 0.0

    # Guards act along the INSERTION AXIS, which FrankaEnv derives from RETRACT_DIR — so
    # for this insert they apply to Y, not Z. Same protection as insert 0, right axis:
    #   - refuse further motion INTO the bore once the y contact force exceeds the limit
    #   - cap the per-step motion INTO the bore (4 mm/step = 4 cm/s at 10 Hz)
    # Motion OUT (+y) is never limited, so the arm can always retreat.
    INSERT_FORCE_LIMIT_N = 15.0
    MAX_INSERT_STEP = 0.004
    # legacy z-named knobs off: superseded by the two above
    Z_FORCE_LIMIT_N = 0.0
    MAX_Z_DOWN_STEP = 0.0


class EnvConfigInsert2(FrankaPlumbersConfigBase):
    """Insert 2 — COOLING BASE, vertical -Z descent (tool points straight down).

    Captured at the rig 2026-09-10. Same geometry family as insert 0 (vertical descent),
    NOT insert 1 (horizontal -Y), so the guards act on Z again.
    """

    # MEASURED seated pose [x, y, z, roll, pitch, yaw] (m, XYZ-Euler rad).
    # Orientation SNAPPED to exactly vertical: the capture read roll -3.1360 / pitch +0.0160,
    # i.e. 0.3 deg and 0.9 deg off straight-down. That is hand-placement noise, not a needed
    # DoF — the operator asked for "completely upright" — so it is pinned to roll = -pi,
    # pitch = 0 and LOCK_ORIENTATION holds it there for the whole episode.
    TARGET_POSE = np.array([0.46300, -0.32180, 0.05840, -np.pi, 0.0, -1.56300])
    # Same x, y and orientation; 65 mm of clearance straight up.
    RESET_POSE = np.array([0.46300, -0.32180, 0.12340, -np.pi, 0.0, -1.56300])

    # OUT of the socket is +Z (a pure vertical extraction), as for insert 0.
    RETRACT_DIR = (0.0, 0.0, 1.0)

    # Safety box: +/-15 cm around the work volume in x/y; z from a hard floor 18 mm below
    # the seated pose up to ~16 cm above it (room for the 4 cm reset / 8 cm regrasp retract
    # plus margin). Rotation bounds WIDE on purpose — orientation is hard-locked, so these
    # only need to not fight the parked pose (a tight window rejected the first reset on
    # inserts 0 and 1).
    # z floor 0.0400 -> 0.0250 (2026-09-10). 0.0400 was a GUESS ("18 mm below seated"), not
    # a measurement, and it was too tight: with +/-1 cm of reset noise on top of approach
    # variation the operator could not descend far enough to seat the part on some runs.
    # 0.0250 leaves 33 mm below the seated pose. The real protection against driving the
    # part into the base is the Z FORCE CAP (INSERT_FORCE_LIMIT_N), which reacts to contact;
    # a hard floor set from a single capture cannot know how deep any given approach needs.
    ABS_POSE_LIMIT_LOW = np.array([0.3130, -0.4718, 0.0250, -np.pi, -np.pi/2, -np.pi])
    ABS_POSE_LIMIT_HIGH = np.array([0.6130, -0.1718, 0.2184, np.pi, np.pi/2, np.pi])

    # Orientation hard-locked to RESET_POSE's (straight down). This is a pure vertical
    # descent, so rotation is not a DoF the task needs — same call as insert 0, which
    # trained to 96%. (Insert 1 needed it unlocked only because the horizontal pipe could
    # not be seated without counter-steering.)
    LOCK_ORIENTATION = True

    RANDOM_RESET = True
    # +/-2 cm (vs 1 cm on inserts 0/1) — requested 2026-09-10 for a wider start
    # distribution. The demos were recorded at the 1 cm setting, so the policy sees start
    # states outside the demo distribution; that is the point (more generalisation) but it
    # makes early episodes harder, so do not read a slow start as a broken run.
    RANDOM_XY_RANGE = 0.02
    RANDOM_Z_RANGE = 0.02      # noise in ALL directions, not just the xy plane
    RANDOM_RZ_RANGE = 0.0

    # Guards act along the INSERTION AXIS, derived from RETRACT_DIR -> Z for this insert:
    #   - refuse further DOWNWARD motion once the z contact force exceeds the limit
    #   - cap the per-step descent (4 mm/step = 4 cm/s at 10 Hz)
    # Motion UP (+z) is never limited, so the arm can always retreat.
    INSERT_FORCE_LIMIT_N = 15.0
    MAX_INSERT_STEP = 0.004
    # legacy z-named knobs kept in sync (this insert IS the z case)
    Z_FORCE_LIMIT_N = 15.0
    MAX_Z_DOWN_STEP = 0.004


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
