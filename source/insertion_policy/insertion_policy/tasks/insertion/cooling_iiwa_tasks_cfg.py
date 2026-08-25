"""iiwa7 + custom-Y-gripper variant of the cooling-screw insertion task (the sim2real hardware).

A NEW env variant that swaps the Franka for the real target hardware (KUKA LBR iiwa7 + the custom
parallel Y-gripper), leaving the Franka cooling tasks (``ForgeTaskCoolingInsert*Cfg``) untouched so the
validated pipeline / fallback stays intact. Only the ROBOT and the geometry that depends on it change;
every task rule (sockets, residual reward, pre-insert tilt, grasp misalignment, appearance DR, camera
machinery) is inherited from the Franka cooling cfgs.

What differs from the Franka:
  * ``robot``   -> the injected iiwa7+gripper USD (7 arm DOFs + a mimic'd 2-finger gripper, + a
                  ``force_sensor`` wrist link so Forge's F/T lookup succeeds). See build_iiwa_gripper_usd.py.
  * ``ctrl``    -> iiwa reset seed + null-space posture (a folded, tool-DOWN pose; the iiwa rest pose
                  is a straight-up candle = a singular IK seed, so we seed the reset from a folded pose).
  * grasp/cam   -> pressing axis is still fingertip-Y (fingers close along Y, same as the Franka), the
                  fingertip control frame is ``gripper_tcp``, and the wrist camera uses the REAL D405
                  mount pose read off the vendor USD. Grasp offset + camera are render-verified/tuned.

Body/DOF layout was probed (scripts/probe_iiwa.py): body[0]=base_link (root), arm DOFs 0:7
(joint1..7), gripper DOFs 7:9 (left/right finger) -> matches every index Forge/Factory hardcode.
"""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.factory.factory_tasks_cfg import RobotCfg
from isaaclab_tasks.direct.forge.forge_env_cfg import ForgeCtrlCfg

from .assets_cfg import ASSET_DIR
from .cooling_tasks_cfg import (
    ForgeCoolingInsert,
    ForgeTaskCoolingInsertCameraCfg,
    ForgeTaskCoolingInsertCfg,
)

ROBOT_USD = os.path.join(ASSET_DIR, "robots", "kuka_iiwa7_pdz_gripper.usd")  # NEW pdz gripper (2026-08-24)

# Folded, tool-DOWN reset seed for the iiwa (deg -> rad), TCP ~ (0.72, 0, 0.15) pointing straight down
# (probe_iiwa.py sweep). Used as the reset IK seed AND the OSC null-space posture, so the arm stays in a
# sane elbow config and the reset IK never has to un-flip the tool from the straight-up candle.
_D2R = math.pi / 180.0
IIWA_RESET_JOINTS = [0.0, 105.0 * _D2R, 0.0, -60.0 * _D2R, 0.0, 118.0 * _D2R, 0.0]


# --- robot articulation (iiwa7 + Y-gripper) ----------------------------------------------------
# Mirrors the Franka Forge robot cfg: arm joints are TORQUE-controlled (stiffness=0, damping=0 -> the
# OSC in factory_control drives them), the 2 finger joints are PhysX-PD position-controlled, gravity is
# disabled on the arm (the OSC does no gravity comp), and the tight-contact solver props are shared.
IIWA_GRIPPER_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ROBOT_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
            fix_root_link=True,  # fixed base, anchored PER-ENV by the cloner (replaces the USD world joint)
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": IIWA_RESET_JOINTS[0],
            "joint2": IIWA_RESET_JOINTS[1],
            "joint3": IIWA_RESET_JOINTS[2],
            "joint4": IIWA_RESET_JOINTS[3],
            "joint5": IIWA_RESET_JOINTS[4],
            "joint6": IIWA_RESET_JOINTS[5],
            "joint7": IIWA_RESET_JOINTS[6],
            "pdz_gripper_left_finger_joint": 0.0,  # start open (both fingers now [0,0.04], 0=closed .. 0.04=open)
            "pdz_gripper_right_finger_joint": 0.0,  # symmetric with left (build_iiwa_gripper_usd.py fixed its range)
        },
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        # Arm: torque-controlled via the OSC (zero PD; the USD's stiffness=2000 drive is overridden).
        # effort limits above the OSC's own +/-100 Nm clamp so the clamp governs uniformly; velocity
        # limits set high so they never clip the OSC (mirrors the Franka Forge actuator setup).
        "iiwa_arm1": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-4]"],
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
            effort_limit_sim=176.0,
            velocity_limit_sim=100.0,
        ),
        "iiwa_arm2": ImplicitActuatorCfg(
            joint_names_expr=["joint[5-7]"],
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
            effort_limit_sim=110.0,
            velocity_limit_sim=100.0,
        ),
        # Gripper: both finger joints PD-position-controlled (right is a PhysX mimic of left, but the
        # base env commands both indices to the same target, so driving both is consistent).
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["pdz_gripper_left_finger_joint", "pdz_gripper_right_finger_joint"],
            # Closing FORCE (found 2026-08-03): 100N drove the jaws THROUGH the 20mm SDF head (finger<->screw
            # penetration); 20N still penetrated ~2-3/50. 10N: drives fully closed (on/off), holds the 6g screw,
            # gentle enough not to ram through the head. Lower stiffness (2000) softens the approach. NOTE: if
            # low force causes grasp DROPS (screw not held) or won't hold during insertion contact, weld instead.
            effort_limit_sim=10.0,
            velocity_limit_sim=0.2,
            stiffness=2000.0,
            damping=200.0,
            friction=0.1,
            armature=0.0,
        ),
    },
)


@configclass
class IiwaCtrlCfg(ForgeCtrlCfg):
    # Only the robot-specific joint-space seeds change; the task-space OSC gains (default_task_prop_gains
    # etc.) are frame-space quantities and carry over from Forge. Retune here if smoke control is poor.
    reset_joints = IIWA_RESET_JOINTS
    default_dof_pos_tensor = IIWA_RESET_JOINTS  # OSC null-space posture bias


@configclass
class ForgeCoolingInsertIiwa(ForgeCoolingInsert):
    # Grasp geometry for the new gripper. Pressing axis is still fingertip-Y (fingers close along Y),
    # so the base env's grasp-misalignment tilt-about-Y is unchanged. franka_fingerpad_length is the
    # ONE grasp knob: held_screw z-offset = head_height - franka_fingerpad_length in the gripper_tcp frame.
    # SIGN (verified 2026-08-03 via grasp renders): SMALLER franka_fingerpad_length pushes the screw further
    # OUT of the jaws (more shaft protrudes below the finger tips); LARGER retracts it back UP into the jaws.
    # The Franka 0.0176 left the shaft protruding only ~2-3mm (fingers hit the socket first); 0.030 was WRONG
    # (retracted even further); 0.0025 OVERSHOT (head dropped BELOW the jaws -> screw dangled/dropped). The
    # depth sweep (scripts/render_grasp_grid.py --sweep) + user read put the head-clamped + shaft-out zone at
    # ~0.012-0.015 (h=0.017 too retracted, small values too far out). 0.012: pads on the head, shaft out, head
    # up in the jaws. VERIFY the physics grip across the randomized distribution with render_grasp_grid.py (grid v3).
    robot_cfg: RobotCfg = RobotCfg(robot_usd=ROBOT_USD, franka_fingerpad_length=0.003, friction=0.75)

    def __post_init__(self):
        # Move the insertion target OUT from the iiwa base (Franka: 0.60 m). The iiwa is a big arm; a
        # close-in target forces a tightly-folded, near-wrist-limit down-pose (torque/limit stalls).
        # 0.72 m centres the workspace on the iiwa's natural tool-down reach (where the reset posture
        # already points), so the reset IK stays well inside limits. Mutating this instance's own
        # fixed_asset only (the Franka task uses a different task class -> untouched).
        parent_post = getattr(super(), "__post_init__", None)
        if callable(parent_post):
            parent_post()
        self.fixed_asset.init_state.pos = (0.72, 0.0, 0.05)


@configclass
class ForgeTaskCoolingInsertIiwaCfg(ForgeTaskCoolingInsertCfg):
    """State-obs iiwa+gripper cooling insertion (Franka swapped for the real hardware)."""

    task = ForgeCoolingInsertIiwa()
    robot: ArticulationCfg = IIWA_GRIPPER_CFG
    ctrl: IiwaCtrlCfg = IiwaCtrlCfg()


@configclass
class ForgeTaskCoolingInsertIiwaCameraCfg(ForgeTaskCoolingInsertCameraCfg):
    """Vision (wrist RGB-D) iiwa+gripper cooling insertion — the sim2real training target.

    Inherits the full appearance-DR + hybrid-CNN camera pipeline; only the robot and the wrist-camera
    mount change. The camera EYE is the REAL Intel D405 pose read off the vendor USD (a ``Camera`` prim
    on gripper_base_link at (0.0681, 0.00005, 0.0447) -> gripper_tcp frame = (0.068, 0.0, -0.101)),
    so the sim mount matches the hardware. The aim point + exact orientation are render-verified.
    """

    task = ForgeCoolingInsertIiwa()
    robot: ArticulationCfg = IIWA_GRIPPER_CFG
    ctrl: IiwaCtrlCfg = IiwaCtrlCfg()

    # Wrist D405 eye in the gripper_tcp (fingertip) frame. The vendor USD Camera prim = gripper_base
    # (0.068,0,0.045) is BURIED inside the solid hand mesh (occluded -> grey). The real lens is at the
    # camera bump's APERTURE: marching forward along the vendor look axis (-0.514,0,0.857) it exits the
    # mesh at ~5cm -> gripper_base (0.042,0,0.088) = fingertip (0.042,0,-0.058). This keeps the cam ON the
    # D405 mount (not floated out) and un-occluded. (Note the vendor camera bump mesh is a placeholder
    # shape, not the actual D405.) TODO sim2real: set from the real D405 bracket + intrinsics.
    # NEW pdz D405 (2026-08-24): eye = the D405 COLOR/DEPTH optical origin read off the converted pdz USD,
    # expressed in the pdz_gripper_tcp frame (CAD: flange-frame [0.009,-0.05056,0.06293], tcp=flange+0.1355z).
    wrist_cam_offset_pos: tuple = (0.009, -0.05056, -0.07257)
    # Look-at is OVERRIDDEN by the verbatim quat below.
    wrist_cam_look_target_pos: tuple = (0.0, 0.0, 0.07)
    # quat=None -> the env builds the camera from the LOOK-AT (eye -> wrist_cam_look_target_pos) in the
    # correct OpenGL (-Z fwd) convention. The raw ROS optical quat pointed the camera BACKWARD (user
    # 2026-08-24), so use the look-at (eye = the real D405 mount pos, aim = the insertion point).
    wrist_cam_offset_quat: tuple | None = None

    def __post_init__(self):
        super().__post_init__()
        # FOV set from the REAL D405 color intrinsics (rs-enumerate-devices, serial 260522275434): the
        # 16:9 stream is 88.7deg(H) x 57.7deg(V); the policy image is a SQUARE CENTER-CROP -> its FOV =
        # the vertical FOV ~57.7deg -> focal = 20.955/(2*tan(57.7/2)) = 19.0mm. (Raw 88deg-wide = focal 11
        # only if the full frame were squished, not cropped.) Own-instance mutation; Franka cam untouched.
        self.tiled_camera.spawn.focal_length = 19.0


# --- END-TO-END VISUOMOTOR variant (supervisor-locked 2026-07-30) -------------------------------
# A NEW control + obs + reward scheme on the SAME iiwa hardware: pure end-to-end visuomotor insertion
# (RGB-D + force -> 5-DoF fingertip-relative delta), IK + high-gain joint PD instead of Forge's OSC.
# The residual ForgeTaskCoolingInsertIiwa*Cfg above are left untouched (validated fallback). Consumed
# by InsertionEnvE2EIiwa. See PLANNING.md "END-TO-END VISUOMOTOR INSERTION" / memory fabrica-graft-plan.

# High-gain joint-PD arm actuators for the IK+PD controller (supervisor: HIGH gains; contact-force risk
# on the brittle parts accepted for now). The residual env's arm actuators are torque-controlled
# (stiffness=0, damping=0, driven by the OSC); here the arm is POSITION-controlled by the implicit PD,
# with the IK solution as the joint target (see InsertionEnvE2EIiwa.generate_ctrl_signals). Gains scaled
# up ~3x from the Isaac Lab KUKA reference (kuka_allegro.py: joints 1-4 = 300/45). effort_limit stays at
# the iiwa 176/110 Nm clamp. >>> TUNE THESE ON THE FIRST SMOKE <<<: too stiff (with the effort clamp +
# tight-contact solver) can oscillate; too soft tracks the IK target sluggishly. gravity stays off.
IIWA_GRIPPER_E2E_CFG = IIWA_GRIPPER_CFG.replace(
    actuators={
        # 2026-08-19: split into 4 arm groups so J6/J7 get the real iiwa7 40 Nm effort cap (was lumped with
        # J5 at 110). Stiffness/damping here are the MIDPOINTS of the per-episode DR ranges (§10 table);
        # the actual per-env, per-episode gain DR is applied at reset (see InsertionEnvE2EIiwa dynamics DR).
        # EFFORT sticker: 176/110/110/40 Nm from the iiwa7 R800 datasheet -- >>> CONFIRM J6/J7 = 40 Nm <<<.
        "iiwa_arm_j12": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-2]"],
            stiffness=562.0,   # DR 375-750 midpoint
            damping=60.0,      # DR 50-70 midpoint
            friction=0.0,
            armature=0.0,
            effort_limit_sim=176.0,
            velocity_limit_sim=100.0,
        ),
        "iiwa_arm_j34": ImplicitActuatorCfg(
            joint_names_expr=["joint[3-4]"],
            stiffness=562.0,   # DR 375-750 midpoint
            damping=60.0,      # DR 50-70 midpoint
            friction=0.0,
            armature=0.0,
            effort_limit_sim=110.0,
            velocity_limit_sim=100.0,
        ),
        "iiwa_arm_j5": ImplicitActuatorCfg(
            joint_names_expr=["joint5"],
            stiffness=281.0,   # DR 187-375 midpoint
            damping=38.0,      # DR 32-45 midpoint
            friction=0.0,
            armature=0.0,
            effort_limit_sim=110.0,
            velocity_limit_sim=100.0,
        ),
        "iiwa_arm_j67": ImplicitActuatorCfg(
            joint_names_expr=["joint[6-7]"],
            stiffness=281.0,   # DR 187-375 midpoint
            damping=38.0,      # DR 32-45 midpoint
            friction=0.0,
            armature=0.0,
            effort_limit_sim=40.0,   # iiwa7 wrist J6/J7 cap (was 110 when lumped with J5)
            velocity_limit_sim=100.0,
        ),
        # Gripper unchanged (PhysX PD position control; both finger joints commanded to the same target).
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["pdz_gripper_left_finger_joint", "pdz_gripper_right_finger_joint"],
            # Closing FORCE (found 2026-08-03): 100N drove the jaws THROUGH the 20mm SDF head (finger<->screw
            # penetration); 20N still penetrated ~2-3/50. 10N: drives fully closed (on/off), holds the 6g screw,
            # gentle enough not to ram through the head. Lower stiffness (2000) softens the approach. NOTE: if
            # low force causes grasp DROPS (screw not held) or won't hold during insertion contact, weld instead.
            effort_limit_sim=10.0,
            velocity_limit_sim=0.2,
            stiffness=2000.0,
            damping=200.0,
            friction=0.1,
            armature=0.0,
        ),
    }
)


# Shared end-to-end knob docs (inlined into both cfgs below rather than a configclass mixin, whose field
# resolution across multiple bases is fragile):
#   action_space=5           -> [dx, dy, dz, rot_a, rot_b] (3 translation + 2 tilt axes, NO yaw)
#   e2e_pos/rot_action_scale -> ctrl_target = fingertip + pos_scale*a[0:3]; quat = fingertip (x) dRot(a[3:5]*rot_scale)
#   e2e_reward_mode          -> "l2_axis": -(w_pos*||tip->socket_bottom|| + w_rot*shaft_axis_error) [yaw-
#                               invariant orientation, matching the tilt-only action + success]; "squashing"
#                               reuses the Factory keypoint kernel. No seat bonus yet (near-seated stop OK).
# TUNE the action scales together with the PD gains on the first smoke.


def _rigidify_held_for_weld(task) -> None:
    """Convert a task's held-asset ArticulationCfg -> RigidObjectCfg (no articulation root, gravity ON) so it
    can be rigidly welded to the gripper (see InsertionEnv._weld_held_asset). Mirrors the same USD + rigid /
    mass / collision props; only the wrapper class changes and gravity is re-enabled (the weld carries the
    ~6g screw). Idempotent. Call BEFORE any solver-iter loop so the rigidified asset still gets it."""
    art = task.held_asset
    if isinstance(art, RigidObjectCfg):
        return
    # The screw USD carries a baked articulation root (it was authored to load as a 1-body Articulation);
    # RigidObject refuses that, so disable the root on spawn (articulation_enabled=False). Also gravity ON.
    spawn = art.spawn.replace(
        rigid_props=art.spawn.rigid_props.replace(disable_gravity=False),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
    )
    task.held_asset = RigidObjectCfg(
        prim_path=art.prim_path,
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=art.init_state.pos, rot=art.init_state.rot),
    )


@configclass
class ForgeTaskCoolingInsertIiwaE2ECfg(ForgeTaskCoolingInsertIiwaCfg):
    """STATE-FIRST debug scaffold: end-to-end control loop with the TRUE (error-free) delta as the policy
    obs (NO camera). Validates the RL + IK/PD controller + reward + 5-DoF action loop before the vision
    encoder is switched in. Uses the high-gain PD arm. Launch with the rl_games_e2e_state cfg."""

    action_space: int = 5
    robot: ArticulationCfg = IIWA_GRIPPER_E2E_CFG
    debug_true_delta_obs: bool = True

    e2e_pos_action_scale: float = 0.01  # metres (was 0.02 for the 85% run; halved -> gentler contact/deploy)
    e2e_rot_action_scale: float = 0.1   # radians (~5.7 deg) per unit tilt action
    # Reward: SQUASHING kernel (Factory multi-scale, sharply peaked near d=0) + seat bonus. The plain
    # "l2_axis" reward is flat near the goal, so the state-first policy learned to HOVER ~3cm above the
    # socket and never commit to the contact-rich descent (confirmed by render). The squashing kernel
    # gives a strong gradient in the last ~2cm and the seat bonus rewards actually bottoming out -- the
    # proven Factory/residual-env recipe. (Was l2_axis / bonus 0 for the first smoke.)
    e2e_reward_mode: str = "squashing"
    e2e_reward_pos_weight: float = 1.0
    e2e_reward_rot_weight: float = 0.25
    # Seat dead-zone (m): zero the position reward once the tip is within this of the target (= physically
    # seated), so the policy doesn't press past the ~2.5mm-unreachable residual ("crash down"). 0 => off.
    e2e_seat_deadzone: float = 0.0
    e2e_reward_center_weight: float = 0.0  # lateral-centering reward weight (hole-finding under noise); 0 => off
    e2e_keep_aux_label: bool = False  # keep the aux_label obs group -> enables the EXPLICIT ESTIMATOR
    e2e_success_bonus: float = 1.0

    # --- SPEED (debug scaffold ONLY) ---------------------------------------------------------------
    # The state-first debug validates the control/reward/action plumbing, NOT tight-contact fidelity, so
    # we cheapen the physics to iterate ~3-4x faster. The real VISION cfg keeps Factory's 192 iterations
    # for the 1mm insertion. Measured (profile_env_speed.py): 192->48 iters ~= 2.4x; the reset is the
    # iiwa hover-IK servo burning its retry budget on straggler envs, so a smaller budget cuts it further.
    debug_solver_iters: int = 48       # PhysX position-iteration count (0 = keep the inherited 192)
    debug_reset_ik_budget: int = 8     # iiwa reset hover-IK retry cap (InsertionEnvE2EIiwa._reset_idx; 25 default)
    weld_held_parent_body: str = "pdz_gripper_tcp"
    weld_gripper_closed_half_gap: float = 0.006  # q=0 collision faces are ~4mm apart, so contact q = r - 2mm

    # Reset hover-IK convergence tolerance (root-caused via diag_reset_ik.py): Factory demands 1mm/0.057deg,
    # which the iiwa DLS can't hit for ~30% of the random targets -> ~22 wasted retries. 5mm/2deg accepts
    # ~87% per attempt (=> ~2 passes). SAFE + also used by the real vision run (the hover is already
    # ±8mm/±45deg random). 0 = exact Factory behaviour.
    reset_ik_pos_tol: float = 0.005    # metres
    reset_ik_rot_tol: float = 0.035    # radians (~2 deg)

    def __post_init__(self):
        parent_post = getattr(super(), "__post_init__", None)
        if callable(parent_post):
            parent_post()
        # Rigid grasp: weld the screw to the gripper (RigidObject + per-env FixedJoint) -> kills the clamp
        # slip/penetration/drop confound. Before the solver-iter loop so the rigidified asset gets it too.
        self.weld_held_to_gripper = True
        _rigidify_held_for_weld(self.task)
        n = int(getattr(self, "debug_solver_iters", 0) or 0)
        if n > 0:
            self.sim.physx.max_position_iteration_count = n
            for art in (self.robot, self.task.fixed_asset, self.task.held_asset):
                for propname in ("rigid_props", "articulation_props"):
                    props = getattr(getattr(art, "spawn", None), propname, None)
                    if props is not None and hasattr(props, "solver_position_iteration_count"):
                        props.solver_position_iteration_count = n


@configclass
class ForgeTaskCoolingInsertIiwaE2EVisionCfg(ForgeTaskCoolingInsertIiwaCameraCfg):
    """The real end-to-end visuomotor task: wrist RGB + force -> 6-DoF delta, no proprio. Inherits the
    full appearance-DR + wrist-cam pipeline; adds the high-gain PD arm + E2E action/reward. The policy
    receives a full, aspect-preserving 320x180 downscale of the real D405 color stream."""

    # 6-DoF action (2026-08-19 rebuild): [dx, dy, dz, rot_a, rot_b, rot_yaw] -- yaw RE-ENABLED (the round
    # peg is yaw-invariant to SEAT, but the wrist must be able to counter-rotate to stay in reach / keep
    # the socket framed). Translation delta is rotated into the TCP frame; all 3 rot axes compose in the
    # body frame. See InsertionEnvE2EIiwa._pre_physics_step. The state debug cfg keeps action_space=5.
    action_space: int = 6
    e2e_disable_yaw: bool = False  # yaw ON (guarded by e2e_yaw_action_bound + e2e_max_joint_delta); NaN was the CNN, not yaw.
    # YAW BOUND (2026-08-20 fix): cumulatively clamp the commanded wrist yaw to +-this (rad) from the reset
    # orientation, so the free (unrewarded) yaw DOF can't wind into the iiwa wrist singularity / joint limit
    # -- the NaN chain (see PLANNING/memory). +-90deg lets the wrist counter-rotate for framing/reach but
    # stays in one branch (0==360 respected, no wrap). 0 => unbounded (the old NaN-prone behaviour).
    e2e_yaw_action_bound: float = 1.5708
    # Minimal per-substep IK joint-delta clamp (rad): caps how far any arm joint can move in one control
    # substep, so a near-singular DLS-IK demand can't produce a lurch -> huge angvel -> fp16 critic NaN.
    # ~0.1 rad only bites on extreme lurches (normal tracking is ~0.01-0.02 rad/substep). 0 => off.
    e2e_max_joint_delta: float = 0.1
    robot: ArticulationCfg = IIWA_GRIPPER_E2E_CFG
    debug_true_delta_obs: bool = False
    # FULL EE-RELATIVE action (2026-08-19): rotate the translation delta into the TCP frame before applying
    # (target = tip + R(tip_quat) . (pos_scale . a[0:3])) instead of adding it in the world frame. Default
    # False preserves the state-scaffold behaviour; True on the vision rebuild.
    e2e_translation_in_tcp_frame: bool = True
    # Force obs expressed in the TCP frame (2026-08-19): rotate the signed 3-axis wrist force into the
    # fingertip frame so the policy reads body-relative contact (matches the real gravity-compensated F/T,
    # which is reported in the tool frame). Default False keeps the world-frame force of the older path.
    e2e_force_in_tcp_frame: bool = True
    # ACTION SAFETY BOX (world frame, metres): clamp the IK target to +-this around the GOAL centre (the
    # noisy assembly goal G = fixed_pos + goal_err_pos) so the peg can't wander far from the hole (§4).
    e2e_action_safety_box: float = 0.05
    # GOAL / ANCHOR obs (§4): append the EE-frame 6-DoF delta to the NOISY assembly goal G to the policy obs
    # (the seated target pose as a delta from the current TCP). True on the vision rebuild; the critic gets
    # the CLEAN true goal (already via the true-delta term). Anchor-dropout corrupts G in the obs only.
    e2e_goal_obs: bool = True
    # Small yaw regularization (yaw is now actionable) so the wrist doesn't wind to its joint limit. Penalty
    # = w * (yaw_action)^2 in _get_rewards. Small so it doesn't suppress useful counter-rotation. 0 => off.
    e2e_reward_yaw_reg_weight: float = 0.01
    # Add proprio + NOISY socket-relative pose (+-8mm) back into the policy obs (res224 recipe). The pure-E2E
    # graft dropped these -> 2% vs res224's 65%; vision refines the coarse pose. Default False keeps the
    # locked pure-E2E task intact; set True per-run via `env.e2e_use_proprio_obs=True`. DEVIATES from the
    # supervisor-locked pure-E2E spec -> flag before committing the direction. Socket still varies (+-5cm/360deg).
    e2e_use_proprio_obs: bool = False
    # VELOCITY-ONLY obs (weekend Run C): append the EE-frame fingertip linear+angular velocity (6) to the
    # policy obs -- NO pose, NO absolute state (unlike e2e_use_proprio_obs's 13-dim pose+vel block). The raw
    # world-frame fingertip velocities are rotated into the TCP frame (quat_rotate_inverse) so the signal is
    # EE-relative and deploy-consistent (real iiwa: joint vel -> Jacobian -> EE vel). Small obs noise mimics a
    # real velocity estimate; the LSTM otherwise infers velocity from the per-step goal delta, so this tests
    # whether an explicit velocity channel helps near-contact damping. Default False (baseline unchanged).
    e2e_use_velocity_obs: bool = False
    e2e_velocity_obs_noise: float = 0.01   # std of gaussian obs noise on the EE-frame velocity (m/s, rad/s)
    # PER-STEP angular obs noise (deg) on the goal-axis estimate (supervisor 2026-08-22), resampled every
    # step, distinct from the per-episode tilt bias. 0 => off. ~0.5deg models a live perception-jitter.
    e2e_goal_ang_obs_noise_deg: float = 0.0
    # Factory-style socket-anchored action: confine the IK target to a +-bound box around the (noisy) socket
    # -> kills the hover / surface-search (res224's mechanism). Set per-run via `env.e2e_socket_anchored_action=True`.
    # Reuses the socket pose we already feed; `bound` = the deploy pose-error tolerance (bigger = safer but
    # weaker confinement/slower learning). Deploy-valid if the base part is fixtured/estimated within +-bound.
    e2e_socket_anchored_action: bool = False
    e2e_socket_action_bound: float = 0.08   # metres (+-8cm box; user asked 7-10cm)
    # Rigid one-pad grasp: __post_init__ converts the held screw to a RigidObject and welds it to one finger
    # pad. Both pads are reset/targeted at the calibrated head-contact aperture. See InsertionEnv._weld_held_asset.
    weld_held_to_gripper: bool = False
    weld_held_parent_body: str = "pdz_gripper_tcp"
    weld_gripper_closed_half_gap: float = 0.006

    e2e_pos_action_scale: float = 0.01  # metres (was 0.02 for the 85% run; halved -> gentler contact/deploy)
    e2e_rot_action_scale: float = 0.1   # radians (~5.7 deg) per unit tilt action
    # Reward = the VALIDATED squashing kernel + seat bonus (the state-first debug reached ~60% success
    # with this; the flat l2_axis reward gave 0% -- policy hovered above the hole). Same reward, vision obs.
    e2e_reward_mode: str = "squashing"
    e2e_reward_pos_weight: float = 1.0
    e2e_reward_rot_weight: float = 0.25
    # LATERAL-CENTERING reward weight -- pull the tip OVER the hole (attacks "reaches the area, misses the
    # hole", which dominates the seating failure). 2026-08-19 rebuild: 15 with a small saturating cap
    # (~1.5-2cm, see e2e_reward_center_cap) so max/step = 15*cap ~ 0.30, comparable to the main term.
    e2e_reward_center_weight: float = 15.0
    # CAP (m) on the centering distance so the penalty stays FLAT beyond the cap instead of growing without
    # bound (v1 blew up: uncapped wc*xy over long episodes hit ~ -1000/episode when the tip drifted to the
    # box edge -> entropy/sigma collapse). 0 => uncapped (legacy). v2 sets 0.03 (3cm = "not centered").
    e2e_reward_center_cap: float = 0.0
    e2e_keep_aux_label: bool = False  # keep the aux_label obs group -> enables the EXPLICIT ESTIMATOR
    e2e_success_bonus: float = 1.0

    # ANTI-JAM contact-force penalty (2026-08-13, deploy hardening): penalize SUSTAINED contact force above
    # e2e_contact_penalty_threshold N while NOT seated, so the policy force-searches instead of pressing
    # ~18N onto the fins. scale sized so the MAX per-step penalty is only a few % of the main reward term
    # (must NOT make contact timid). Tune the scale via runs; 0 disables it.
    e2e_contact_penalty_scale: float = 0.02  # modest scale (max/step a few % of the main term -- don't make contact timid)
    e2e_contact_penalty_threshold: float = 6.5   # N; sustained contact above this (unseated) is penalized (2026-08-19: was 12)
    e2e_contact_penalty_cap: float = 8.0         # N; cap on the overshoot so a spike can't dominate reward
    # SEAT SHAPING (fine-seat of the chamfer-free ~1mm hole):
    # (a) CENTER-THEN-DESCEND gate: penalize the tip going below the socket rim while laterally off-centre
    #     (> thresh) -> centre first, then descend. Weight 0 => off. Penalty ~ w * below-rim-depth(m); with
    #     socket depth ~0.0175m, w~15 gives a max ~0.26/step -- comparable to the main reward. Tune via runs.
    e2e_reward_descend_gate_weight: float = 0.0
    e2e_reward_descend_center_thresh: float = 0.005  # m; xy miss beyond this counts as "off-centre"
    # (b) CONTACT COMPLIANCE: scale the high-gain arm PD stiffness (<1 = softer -> the peg can slide into the
    #     hole instead of rigidly jamming). Damping scaled by sqrt to stay ~critically damped. 1.0 = current
    #     gains. SMOKE for control stability before using <1 on a full run (soft gains + tight-contact solver
    #     can oscillate). Applied in __post_init__ to a COPY of the robot cfg (shared object untouched).
    e2e_arm_stiffness_scale: float = 1.0
    # ISOLATION TEST-A (2026-08-20 NaN debug): restore the wrist (J6/J7) to the proven 85%-run actuator
    # (110 Nm effort, 375 stiffness, 45 damping) and turn OFF per-episode dynamics-gain DR, to test whether
    # the weakened/randomized wrist (not the yaw DOF) is what drove the violent contact -> fp16 NaN. Applied
    # in __post_init__. False => the rebuild's split wrist (40 Nm / 281 / gain-DR). NOTE: read in __post_init__
    # so it must be toggled via this DEFAULT (env.* hydra overrides land after __post_init__).
    e2e_wrist_proven: bool = False
    # ANCHOR DROPOUT (2026-08-16, staged for the post-deploy run): on this fraction of episodes the socket
    # anchor is corrupted IN THE POLICY OBS ONLY (big noise; the action box keeps the good anchor) so the
    # policy can't over-rely on it and must localise the hole from vision. Root fix for the tiny-anchor
    # fragility (orange peaked 70% then collapsed). 0 => off. Try ~0.3 with ~10cm dropout noise.
    anchor_dropout_prob: float = 0.0
    anchor_dropout_noise: float = 0.10   # m; std of the big obs-only goal-G noise on dropout episodes
    # Anchor-dropout schedule (§8): ramp 0 -> anchor_dropout_prob from anchor_dropout_start_steps over
    # anchor_dropout_curriculum_steps control steps (same ~ep800 schedule as image-dropout). Mutually
    # exclusive with image-dropout (enforced in _reset_idx).
    anchor_dropout_start_steps: int = 0
    anchor_dropout_curriculum_steps: int = 0
    # DYNAMICS DR (§10, 2026-08-19): per-env, per-EPISODE randomization of the arm joint PD gains
    # (stiffness/damping over the §10 ranges), joint friction, armature, and link mass/inertia (+-few %).
    # Effort stays FIXED at the physical iiwa7 caps (not randomized). Applied at reset via the articulation
    # write APIs (see InsertionEnvE2EIiwa._apply_dynamics_dr). 0/False => off (backward compatible).
    e2e_dynamics_dr: bool = False
    # Ramp the dynamics-DR STRENGTH 0 -> 1 over this many env-steps (0 => full DR from step 0). Early
    # lag/variation-free foothold, matching the tilt/latency/obs-noise curricula.
    e2e_dynamics_dr_curriculum_steps: int = 0
    e2e_dr_joint_friction_range: tuple = (0.0, 0.05)   # per-arm-joint Coulomb friction (Nm), U[lo,hi]
    e2e_dr_armature_range: tuple = (0.0, 0.02)          # per-arm-joint added armature (kg m^2), U[lo,hi]
    e2e_dr_mass_frac: float = 0.03                      # link mass/inertia +- this fraction (3%)
    # CONTROL-LATENCY DR: max action delay in control steps (0-2 = 0-133ms at 15Hz), random per episode.
    control_latency_max_steps: int = 2
    # Latency CURRICULUM: ramp the effective max delay 0 -> control_latency_max_steps over this many control
    # steps (0 => full from step 0). Part of the v2 early-foothold fix (learn lag-free first).
    control_latency_curriculum_steps: int = 0
    # Effective radial-clearance stopgap: bump the peg+socket collision rest_offset by this (m) to TIGHTEN
    # the ~1mm radial clearance toward the real ~0.5mm without regenerating the mesh. 0 => mesh clearance as
    # authored. NOTE: the CLEAN fix is regenerating cooling_screw/cooling_base (13mm shaft vs 14mm socket)
    # in scripts/convert_assets.py; this offset is a coarse approximation (also shifts seating depth).
    contact_rest_offset: float = 0.0

    # Real-run contact solver: supervisor OK'd cranking Factory's 192 down to 128 (~1.5x faster rollout,
    # still high-fidelity for the 1mm insertion). Applied in __post_init__ (sim + robot + both parts).
    real_solver_iters: int = 192  # was 128 (speed); back to 192 for the final run -> stiffer weld under contact impulse

    # Pre-insert tilt (deg), the weekend SWEEP knob -> override per run via `env.weekend_tilt_deg=X`.
    # Must be a declared field so hydra can override it. Read in __post_init__ -> self.task.pre_insert_tilt_max_deg.
    # 2026-08-19 rebuild default 10deg (was 15), half-normal biased to 0 + curriculumed (see __post_init__).
    weekend_tilt_deg: float = 10.0
    # Tilt CURRICULUM (declared for hydra: `env.tilt_curriculum_steps=76800 env.tilt_start_deg=0`). Ramp the
    # effective pre-insert tilt from tilt_start_deg -> weekend_tilt_deg over this many control steps
    # (~= iters*horizon_length; 128*600=76800 -> ~600 it). 0 => off (constant at weekend_tilt_deg).
    tilt_curriculum_steps: int = 0
    tilt_start_deg: float = 0.0

    # Reset hover-IK convergence tolerance (see the state cfg / diag_reset_ik.py). SAFE speedup for the
    # real run too: the hover pose is already ±8mm/±45deg random. Keeps Factory's 192 solver iters for
    # insertion fidelity; only the reset acceptance bar + retry cap are relaxed. With the loose tol ~89%
    # of envs converge on the first attempt, so a low retry budget (6, vs Factory's effective 25) is safe
    # -- it cuts the real-run reset ~10.2s -> ~2.8s. (The reset cost is #attempts x 0.25s, so the budget
    # is the actual time knob; the tolerance just keeps a low budget from hurting placement.)
    reset_ik_pos_tol: float = 0.005    # metres
    reset_ik_rot_tol: float = 0.035    # radians (~2 deg)
    debug_reset_ik_budget: int = 6     # hover-IK retry cap (name is historical; applies to the real run)

    # Full 16:9 single-frame RGB, downscaled to 320x180 without cropping or anisotropic distortion.
    # frame_stack stays 1 (the LSTM carries temporal state).
    # RGB-ONLY (2026-08-19 rebuild): 3 channels, no depth branch / no depth DR (see rgb_only + __post_init__).
    image_height: int = 180
    image_width: int = 320
    image_channels: int = 3
    rgb_only: bool = True

    def __post_init__(self):
        super().__post_init__()
        # Rigid grasp: weld the screw to the gripper (RigidObject + per-env FixedJoint). Removes the clamp
        # slip/penetration/drop confound; keeps grasp-misalign DR baked as a per-env tilt. Before the
        # real_solver_iters loop below so the rigidified held asset gets the iteration count too.
        self.weld_held_to_gripper = True
        _rigidify_held_for_weld(self.task)
        self.tiled_camera.width = 320
        self.tiled_camera.height = 180
        # RGB-ONLY (2026-08-19): render RGB only (drop "depth" from the camera data_types -> no depth
        # buffer allocated/rendered).
        self.tiled_camera.data_types = ["rgb"]
        # Full D405 FOV at its native 16:9 aspect. With the 20.955mm aperture, 11mm gives approximately
        # 87deg horizontal and 57deg vertical FOV, matching the measured color intrinsics without squashing.
        self.tiled_camera.spawn.focal_length = 11.0
        # Pre-insert tilt is now a SWEEP knob (2026-07-31): the first vision run showed the wrist camera
        # resolves POSITION (tip closing) but not fine ORIENTATION -- shaft-axis stuck ~35-40deg, screw
        # arrives angled, won't seat. So the weekend maps success-vs-tilt instead of running one config
        # blind. __post_init__ reads `weekend_tilt_deg` (override it per run) -> the override propagates
        # through here regardless of default. Lateral kept +-10mm (vision handles position OK).
        # §4 GOAL/ANCHOR angular budget: the injected shaft-axis error ramps 0 -> 25deg (half-normal, biased
        # to 0). With the small baked grasp misalign (3deg, §10) the TOTAL start misalignment stays within
        # the 30deg cap (vector sum; smoke-verified worst ~30), ramping from ~0 over ~1000it. This SUBSUMES
        # the old standalone pre-insert tilt (0->10deg). Curriculum ~128000 control steps (128*1000). Tuning
        # knob (§4): drop this below 25 if the policy stalls on orientation.
        self.task.pre_insert_tilt_max_deg = 25.0
        self.task.pre_insert_tilt_start_deg = 0.0
        self.task.pre_insert_tilt_curriculum_steps = 128000
        self.task.pre_insert_tilt_halfnormal = True
        # §4 GOAL/ANCHOR lateral+z estimate error, injected PHYSICALLY at reset (peg starts off the true
        # hole; obs delta-to-noisy-G ~ 0). Radial xy <= 1.75cm, z <= 0.8cm, both ramp 0 -> max over ~1000it.
        self.task.goal_anchor_injection = True
        self.task.goal_anchor_lat_max = 0.0175
        self.task.goal_anchor_z_max = 0.008
        self.task.goal_anchor_curriculum_steps = 128000
        # §4/§6 WRIST YAW start = half-normal (biased to 0) about vertical, +-45deg, injected in the reset
        # re-pose (Factory's uniform hover yaw is turned OFF below). Yaw-symmetric -> rotates the image only.
        self.task.wrist_yaw_halfnormal = True
        self.task.wrist_yaw_max_deg = 45.0
        # Screw (held_asset) friction DR (2026-08-05): was FIXED 0.75 (num_buckets=1). PLA-on-PLA is ~0.3-0.5;
        # randomize static+dynamic over a buffered 0.3-0.9 so the CONTACT PAIR is robust to the real
        # (unmeasurable/drifting) friction instead of overfitting one value. Socket (fixed_asset) is already
        # randomized 0.25-1.25 by Forge. Self-disables if the Forge material event isn't present.
        # Recentre the CONTACT PAIR friction on ~0.5 (PLA-on-PLA), still a DR band so it's robust to the
        # real (unmeasurable/drifting) value. Applied to both the held screw and, if present, the socket.
        if hasattr(self, "events") and hasattr(getattr(self, "events"), "held_physics_material"):
            _hp = self.events.held_physics_material.params
            _hp["static_friction_range"] = (0.3, 0.8)
            _hp["dynamic_friction_range"] = (0.3, 0.8)
            _hp["num_buckets"] = 64
        for _fixed_evt in ("fixed_physics_material", "fixed_asset_physics_material"):
            if hasattr(self, "events") and hasattr(getattr(self, "events"), _fixed_evt):
                _fp = getattr(self.events, _fixed_evt).params
                _fp["static_friction_range"] = (0.4, 0.7)
                _fp["dynamic_friction_range"] = (0.4, 0.7)
                _fp["num_buckets"] = 64
        # Grasp realism DR (2026-08-05, user): the real clamp grasp isn't perfect/repeatable ->
        # (a) misalign tilt 10->25deg (screw crooked in the pads about the pressing axis -- the DOMINANT,
        #     realistic grasp error). FIXED per-env, cold from step 0: the rigid weld bakes it ONCE at init
        #     and it cannot be curriculum-ramped without the vetoed per-reset D6 drive. Pre-insert (gripper)
        #     tilt is the SMALL secondary error (weekend_tilt_deg=12) and IS the curriculum-ramped one.
        # (b) grasp POSITION jitter: axial +-3mm (shaft depth, never retracts the shaft into the pads)
        # + lateral +-2mm (in-pad plane). VERIFY in-bounds with render_grasp_grid.py.
        # FORCE-OBS GRAVITY COMPENSATION: the real F/T is gravity-compensated, so subtract the per-env
        # pre-contact gravity wrench from the force obs -> pure CONTACT (plan requirement; was silently off
        # in the weekend runs). Safe: the stateful baseline is nan_to_num-guarded in _get_observations
        # (the NaN-amplifier gotcha [[e2e-nan-gravity-comp]] is fixed).
        self.use_gravity_comp = True
        # §10 DYNAMICS DR: randomize arm PD gains / friction / armature / mass per episode (see the env).
        self.e2e_dynamics_dr = True
        # Ramp DR strength 0->1 over the first ~150k env-steps (early foothold before full spread). Tune vs
        # the tilt curriculum (~205k) if the early gains feel too soft/hard.
        self.e2e_dynamics_dr_curriculum_steps = 150000
        # PER-STEP angular obs noise on the goal-axis estimate (supervisor: ~0.5deg/step live jitter).
        self.e2e_goal_ang_obs_noise_deg = 0.5
        # §10/§4: keep the BAKED grasp misalign SMALL (3deg) so baked + the curriculumed injected angular
        # (0->27deg) stays <= the 30deg total start-misalignment budget. Secondary out-of-plane -> 0.
        self.task.grasp_misalign_max_deg = 3.0
        # Grasp position jitter. NOTE: the weld BAKES grasp DR per-env at init (cold from step 0), so it
        # CANNOT be curriculum-ramped. v1 used 5/5mm + 7deg secondary and never got an early foothold (13%
        # @it500 vs w2's 50%); v2 eases it back toward the w2-proven 3/2mm and a small 3deg secondary so the
        # cold grasp is learnable, while keeping the dominant 25deg pressing-axis misalign. The axial (+z)
        # direction retracts the shaft INTO the pads -- keep it small.
        self.task.grasp_pos_jitter_axial_mm = 3.0
        self.task.grasp_pos_jitter_lateral_mm = 2.0
        self.task.grasp_misalign_secondary_deg = 0.0  # §4: fold all angular error into the 30deg budget
        # §6 START HEIGHT: screw tip z = rim + 0.04 +- 0.01 -> TCP = tip + 17.5mm shaft = rim + 0.0575. The
        # hand_init_pos is the FINGERTIP (TCP) target above the socket opening, so z = 0.0575.
        self.task.hand_init_pos = [0.0, 0.0, 0.0575]
        # §6 LATERAL SPAWN = small servo SETTLING only (radial <= ~0.3cm). The lateral/z GOAL-ESTIMATE error
        # is now injected physically by the goal anchor (above), so the peg starts OFF the true hole and the
        # obs delta-to-noisy-G ~ 0. This 0.3cm is just the residual servo settling about that estimate. z
        # noise +-0.01 = the rim+0.04+-0.01 height band (tip 4cm above rim).
        self.task.hand_init_pos_noise = [0.003, 0.003, 0.010]
        # §6 WRIST YAW START is now injected as a HALF-NORMAL in the reset re-pose (wrist_yaw_halfnormal
        # above), so turn OFF Factory's uniform hover yaw here to avoid double-applying it.
        self.task.hand_init_orn_noise = [0.0, 0.0, 0.0]
        self.hand_init_yaw_curriculum_steps = 0  # the half-normal injection is the yaw source now
        # Widen the SOCKET placement so the restored socket-relative pose obs sees real spatial variation
        # (not a memorisable constant): X band 0.65-0.75m (nominal 0.70 +- 0.05, recentred 2cm nearer than
        # the 0.72 default per user), Y +-15cm, Z +-5cm. Y is the roomy axis -- at x~0.70 it's a base-yaw
        # swing (worst-corner radial ~sqrt(0.75^2+0.15^2)=0.765m, well under the ~0.8m reach). X stays a
        # narrow band: the iiwa base is at origin and ~0.72m is the tool-down SWEET SPOT (author's note L149)
        # -- closer folds the arm into torque/wrist-limit stalls, farther exceeds reach. Keeping the whole
        # workspace comfortably inside reach should also cut the near-singular-IK NaN crashes (the resume
        # cause). VALIDATE the corners with scripts/diag_reset_ik.py (reset-only) if reset failures spike.
        # Socket PLACEMENT region = how the base actually gets put on the table IRL: a hand-placed spot in
        # a workspace REGION (x 0.50-0.75m, y +-40cm) at ~table height (z 0-0.05m), FLAT (only yaw varies,
        # 360deg; roll/pitch=0 since it sits flat -- matches reality). Encoded as center +- uniform noise.
        #   x 0.50-0.75 -> center 0.625 +-0.125 ; y +-0.40 -> 0.0 +-0.40 ; z 0-0.05 -> 0.025 +-0.025
        # This is the PHYSICAL pose; the actor's noisy ESTIMATE of it is fixed_asset_pos_obs_noise_bound
        # (the +-3cm anchor / the ~5mm deploy CV) -- a SEPARATE knob.
        # !!! REACH WARNING: the iiwa reach is ~0.8m radial from its base at the origin. The far corners of
        # this box EXCEED it: (x=0.75, y=+-0.40) -> radial sqrt(0.75^2+0.40^2)=0.85m > 0.8m. Those resets
        # will fail hover-IK (fall back to the vertical grasp) and risk the near-singular-IK NaN crashes that
        # caused past resume loops. VALIDATE with scripts/diag_reset_ik.py BEFORE the full run; if the
        # corners fail, cap y to ~+-0.30 or x to ~0.70 (radial back under 0.8m).
        # §5 CUSTOM PLATE PLACEMENT (2026-08-19): x~U[0.40,0.65], y~U[-0.55,-0.20], z~U[-0.02,+0.01], flat
        # (roll=pitch=0). Yaw = 80% nominal +-3deg, 20% wide +-[3,45]deg. Reject radial > 0.72m so the far
        # corners of the box (which exceed the ~0.8m iiwa reach) never spawn. Replaces the bimodal-Y clusters.
        # See InsertionEnv._sample_custom_fixed_xy / _apply_fixed_yaw_bimodal. VALIDATE corners with
        # scripts/diag_reset_ik.py (near (0.40,-0.20) + the trimmed far corner) before a full run.
        self.task.fixed_asset_custom_placement = True
        self.task.base_y_clusters = None
        self.task.fixed_asset_xy_ranges = [0.40, 0.65, -0.55, -0.20]
        self.task.fixed_asset_z_center = -0.005   # U[-0.02, +0.01] centre (nominal ~ -0.01)
        self.task.fixed_asset_z_noise = 0.015
        self.task.fixed_asset_reach_radius = 0.72
        self.task.fixed_asset_yaw_nominal_deg = 90.0  # holes along world-Y [VERIFY vs USD default + mesh frame]
        self.task.fixed_asset_yaw_nominal_halfwidth_deg = 3.0
        self.task.fixed_asset_yaw_wide_frac = 0.20
        self.task.fixed_asset_yaw_wide_deg = [3.0, 45.0]
        # Crank the contact solver 192 -> real_solver_iters (128) for the real run (supervisor-approved).
        n = int(getattr(self, "real_solver_iters", 0) or 0)
        if n > 0:
            self.sim.physx.max_position_iteration_count = n
            for art in (self.robot, self.task.fixed_asset, self.task.held_asset):
                for propname in ("rigid_props", "articulation_props"):
                    props = getattr(getattr(art, "spawn", None), propname, None)
                    if props is not None and hasattr(props, "solver_position_iteration_count"):
                        props.solver_position_iteration_count = n

        # --- 2026-08-13 real-robot DEPLOY HARDENING (starting defaults; tune each via runs) --------------
        # RGB-ONLY (2026-08-19 rebuild): the depth channel + ALL depth DR are REMOVED (real D405 depth was
        # ~45% invalid and structured -- an unmodellable gap; the rebuild drops depth entirely and leans on
        # RGB appearance DR instead). depth_corruption_mode stays "uniform" (unused: no depth is rendered)
        # and every depth-DR knob is left at its no-op default. Do NOT re-enable depth DR on the RGB-only path.
        # §13 CURRICULA (2026-08-19 rebuild): all ramp from ~0 to full over the schedules below. No depth
        # curriculum (RGB-only).
        #   GOAL ANCHOR lateral 0->1.75cm / z 0->0.8cm / angular 0->27deg (+3deg baked = 30 total) ~128k (~1000 it)
        #   appearance         ~180k (~1400 it)  -- vision + image DR foothold
        #   control latency    0->2 steps        ~230k (~1800 it)  -- actuation lag last
        #   image + anchor dropout 0->10% / 0->30%  from ~ep800 (102400 steps), mutually exclusive
        self.appearance_curriculum_steps = 180000        # ~1400 it
        self.control_latency_curriculum_steps = 230000   # ~1800 it (0->2 steps)
        # §8/§13 IMAGE DROPOUT: ramp 0 -> 10% starting at ~epoch 800 (128*800 = 102400 control steps), over
        # ~700 epochs (~90k steps) to full. Forces a force-guided fallback when vision is occluded/dropped.
        self.image_dropout_prob = 0.10
        self.image_dropout_start_steps = 102400
        self.image_dropout_curriculum_steps = 90000
        # §8/§4 ANCHOR DROPOUT: corrupt the goal G in the obs only (30% of episodes at full), same ~ep800
        # ramp, MUTUALLY EXCLUSIVE with image-dropout -> the policy always keeps >=1 localisation channel.
        self.anchor_dropout_prob = 0.30
        self.anchor_dropout_noise = 0.05   # m; std of the big obs-only goal-G corruption on dropout episodes
        self.anchor_dropout_start_steps = 102400
        self.anchor_dropout_curriculum_steps = 90000
        # REWARD FIX (2026-08-19): small saturating centering cap (~1.5-2cm) so beyond it the tip is just
        # "not centered" and the penalty stays FLAT (no unbounded drift blow-up). 12s episodes give search
        # room over w2's 10s without v1's 18s over-amplification.
        self.e2e_reward_center_cap = 0.02
        # HEAVY APPEARANCE DR (base is vivid BLUE; explicit-estimator over-trusted a clean look):
        self.rgb_noise_std = 0.03
        self.photo_gain_rgb = 0.30            # per-channel gain in [0.7,1.3]
        self.photo_brightness = 0.10          # additive exposure in [-0.1,0.1]
        # LIGHT COLOUR-TEMPERATURE DR (plan/deploy-hardening: warm 3000K -> cool 8000K). Was None (off) =
        # legacy grey-only jitter, a silent plan miss like the colours/gravity-comp. Tints dome+key per reset.
        self.light_color_temp_range_k = (3000.0, 8000.0)
        # Camera ROLL jitter: plan is +-2-3 deg; base default 2.0 sat at the low edge -> mid-range.
        self.cam_rot_jitter_deg = 2.5
        # SUCCESS = FULL-SEAT (decision 2026-08-24, agent-authority): keep the plan's head-flush bar so the
        # metric is honest/deployable + comparable to w2's 83%, AND the success bonus doesn't let the policy
        # satisfice at a partial insertion. success_threshold 0.25 => is_seated when tip within height*0.25
        # =4.375mm of the socket-bottom ref (~13mm inserted, ~2mm shy of the 15mm physical stop). Half-seat
        # progress is still visible via the engaged/insertion_depth diagnostics (no need to loosen the metric).
        self.task.success_threshold = 0.25
        # Seat dead-zone: stop rewarding the last ~3mm the tip cannot close (kills the "crash down" press).
        self.e2e_seat_deadzone = 0.003
        
