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

ROBOT_USD = os.path.join(ASSET_DIR, "robots", "kuka_iiwa7_y_gripper.usd")

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
            "left_finger_joint": 0.04,  # start open (both fingers now [0,0.04], 0=closed .. 0.04=open)
            "right_finger_joint": 0.04,  # symmetric with left (build_iiwa_gripper_usd.py fixed its range)
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
            joint_names_expr=["left_finger_joint", "right_finger_joint"],
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
    robot_cfg: RobotCfg = RobotCfg(robot_usd=ROBOT_USD, franka_fingerpad_length=0.012, friction=0.75)

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
    wrist_cam_offset_pos: tuple = (0.042, 0.0, -0.058)
    # Rigid-mount orientation via a FIXED look-at: eye -> this aim point, both fixed in the fingertip
    # frame (rigid GoPro-style mount, not socket-tracking). Aim toward the socket/insertion region so the
    # workspace sits centred in the wrist view (sweep-tuned with the moved-out mount).
    wrist_cam_look_target_pos: tuple = (0.0, 0.0, 0.07)
    # Real D405 orientation from the vendor USD Camera prim, ROLLED 180deg about the optical axis to match
    # how the D405 is physically mounted (the real camera image shows the fingers at the BOTTOM converging
    # up; the un-rolled vendor quat put them at the TOP). vendor (0.18869,-0.68132,0.68158,-0.18881) *
    # 180deg-about-local-z (0,0,0,1) = below. look dir stays (-0.514,0,0.857) down-and-forward along the
    # tool. InsertionEnvIiwa uses this verbatim (overrides the look-at above). quat (w,x,y,z).
    wrist_cam_offset_quat: tuple | None = (0.18881, 0.68158, 0.68132, 0.18869)

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
        "iiwa_arm1": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-4]"],
            stiffness=750.0,   # softened from 1000 (2026-08-05) -> gentler contact / less weld-snap; still ~2.5x KUKA ref
            damping=70.0,
            friction=0.0,
            armature=0.0,
            effort_limit_sim=176.0,
            velocity_limit_sim=100.0,
        ),
        "iiwa_arm2": ImplicitActuatorCfg(
            joint_names_expr=["joint[5-7]"],
            stiffness=375.0,   # softened from 500 (2026-08-05), same ~25% cut as arm1
            damping=45.0,
            friction=0.0,
            armature=0.0,
            effort_limit_sim=110.0,
            velocity_limit_sim=100.0,
        ),
        # Gripper unchanged (PhysX PD position control; both finger joints commanded to the same target).
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["left_finger_joint", "right_finger_joint"],
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
    weld_held_parent_body: str = "left_finger_link"
    weld_gripper_closed_half_gap: float = 0.002  # q=0 collision faces are ~4mm apart, so contact q = r - 2mm

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
    """The real end-to-end visuomotor task: wrist RGB-D + force -> 5-DoF delta, no proprio. Inherits the
    full appearance-DR + wrist-cam pipeline; adds the high-gain PD arm + E2E action/reward. 224px RGB-D
    (the confirmed resolution win + ResNet-18-native). Launch with the rl_games_e2e_vision cfg."""

    action_space: int = 5
    robot: ArticulationCfg = IIWA_GRIPPER_E2E_CFG
    debug_true_delta_obs: bool = False
    # Add proprio + NOISY socket-relative pose (+-8mm) back into the policy obs (res224 recipe). The pure-E2E
    # graft dropped these -> 2% vs res224's 65%; vision refines the coarse pose. Default False keeps the
    # locked pure-E2E task intact; set True per-run via `env.e2e_use_proprio_obs=True`. DEVIATES from the
    # supervisor-locked pure-E2E spec -> flag before committing the direction. Socket still varies (+-5cm/360deg).
    e2e_use_proprio_obs: bool = False
    # Factory-style socket-anchored action: confine the IK target to a +-bound box around the (noisy) socket
    # -> kills the hover / surface-search (res224's mechanism). Set per-run via `env.e2e_socket_anchored_action=True`.
    # Reuses the socket pose we already feed; `bound` = the deploy pose-error tolerance (bigger = safer but
    # weaker confinement/slower learning). Deploy-valid if the base part is fixtured/estimated within +-bound.
    e2e_socket_anchored_action: bool = False
    e2e_socket_action_bound: float = 0.08   # metres (+-8cm box; user asked 7-10cm)
    # Rigid one-pad grasp: __post_init__ converts the held screw to a RigidObject and welds it to one finger
    # pad. Both pads are reset/targeted at the calibrated head-contact aperture. See InsertionEnv._weld_held_asset.
    weld_held_to_gripper: bool = False
    weld_held_parent_body: str = "left_finger_link"
    weld_gripper_closed_half_gap: float = 0.002

    e2e_pos_action_scale: float = 0.01  # metres (was 0.02 for the 85% run; halved -> gentler contact/deploy)
    e2e_rot_action_scale: float = 0.1   # radians (~5.7 deg) per unit tilt action
    # Reward = the VALIDATED squashing kernel + seat bonus (the state-first debug reached ~60% success
    # with this; the flat l2_axis reward gave 0% -- policy hovered above the hole). Same reward, vision obs.
    e2e_reward_mode: str = "squashing"
    e2e_reward_pos_weight: float = 1.0
    e2e_reward_rot_weight: float = 0.25
    # LATERAL-CENTERING reward weight -- pull the tip OVER the hole (attacks "reaches the area, misses the
    # hole", which dominates the seating failure). Hardened default 0.5 (start; tune 0.25-1.0 via runs).
    e2e_reward_center_weight: float = 0.5
    e2e_keep_aux_label: bool = False  # keep the aux_label obs group -> enables the EXPLICIT ESTIMATOR
    e2e_success_bonus: float = 1.0

    # ANTI-JAM contact-force penalty (2026-08-13, deploy hardening): penalize SUSTAINED contact force above
    # e2e_contact_penalty_threshold N while NOT seated, so the policy force-searches instead of pressing
    # ~18N onto the fins. scale sized so the MAX per-step penalty is only a few % of the main reward term
    # (must NOT make contact timid). Tune the scale via runs; 0 disables it.
    e2e_contact_penalty_scale: float = 0.01
    e2e_contact_penalty_threshold: float = 12.0  # N; sustained contact above this (unseated) is penalized
    e2e_contact_penalty_cap: float = 8.0         # N; cap on the overshoot so a spike can't dominate reward
    # CONTROL-LATENCY DR: max action delay in control steps (0-2 = 0-133ms at 15Hz), random per episode.
    control_latency_max_steps: int = 2
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
    weekend_tilt_deg: float = 15.0
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

    # 224px single-frame RGB-D (ResNet-18 ImageNet-native; the 224>160 win from prior screens). Heavier
    # than 160px -> may need fewer envs (memory). frame_stack stays 1 (the LSTM carries temporal state).
    image_height: int = 224
    image_width: int = 224

    def __post_init__(self):
        super().__post_init__()
        # Rigid grasp: weld the screw to the gripper (RigidObject + per-env FixedJoint). Removes the clamp
        # slip/penetration/drop confound; keeps grasp-misalign DR baked as a per-env tilt. Before the
        # real_solver_iters loop below so the rigidified held asset gets the iteration count too.
        self.weld_held_to_gripper = True
        _rigidify_held_for_weld(self.task)
        self.tiled_camera.width = 224
        self.tiled_camera.height = 224
        # Pre-insert tilt is now a SWEEP knob (2026-07-31): the first vision run showed the wrist camera
        # resolves POSITION (tip closing) but not fine ORIENTATION -- shaft-axis stuck ~35-40deg, screw
        # arrives angled, won't seat. So the weekend maps success-vs-tilt instead of running one config
        # blind. __post_init__ reads `weekend_tilt_deg` (override it per run) -> the override propagates
        # through here regardless of default. Lateral kept +-10mm (vision handles position OK).
        self.task.pre_insert_tilt_max_deg = float(getattr(self, "weekend_tilt_deg", 15.0))
        # Tilt curriculum -> task (ramp start_deg -> weekend_tilt_deg over tilt_curriculum_steps control steps).
        self.task.pre_insert_tilt_start_deg = float(getattr(self, "tilt_start_deg", 0.0))
        self.task.pre_insert_tilt_curriculum_steps = int(getattr(self, "tilt_curriculum_steps", 0))
        # Screw (held_asset) friction DR (2026-08-05): was FIXED 0.75 (num_buckets=1). PLA-on-PLA is ~0.3-0.5;
        # randomize static+dynamic over a buffered 0.3-0.9 so the CONTACT PAIR is robust to the real
        # (unmeasurable/drifting) friction instead of overfitting one value. Socket (fixed_asset) is already
        # randomized 0.25-1.25 by Forge. Self-disables if the Forge material event isn't present.
        # Recentre the CONTACT PAIR friction on ~0.5 (PLA-on-PLA), still a DR band so it's robust to the
        # real (unmeasurable/drifting) value. Applied to both the held screw and, if present, the socket.
        if hasattr(self, "events") and hasattr(getattr(self, "events"), "held_physics_material"):
            _hp = self.events.held_physics_material.params
            _hp["static_friction_range"] = (0.4, 0.7)
            _hp["dynamic_friction_range"] = (0.4, 0.7)
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
        self.task.grasp_misalign_max_deg = 25.0
        # Grasp position jitter widened to +-5mm (2026-08-13 deploy hardening; was 3/2mm). NOTE the axial
        # (+z) direction retracts the shaft INTO the pads -- verify with render_grasp_grid.py that +5mm
        # never pulls the shaft shoulder inside the jaws; drop axial to ~4mm if it does.
        self.task.grasp_pos_jitter_axial_mm = 5.0
        self.task.grasp_pos_jitter_lateral_mm = 5.0
        # Secondary out-of-plane grasp tilt (about fingertip X): ~7deg from pad compliance / off-centre head.
        self.task.grasp_misalign_secondary_deg = 7.0
        # Lateral pre-insert offset widened +-8->+-15mm (keep z +-20mm).
        self.task.hand_init_pos_noise = [0.015, 0.015, 0.020]
        # Full +-180deg wrist YAW: deploy tool yaw (~50deg) exceeds the old +-45deg band and A7 limits block
        # preinserting to nominal, so train the full range. The hole is yaw-symmetric -> this only rotates
        # the wrist image (round peg), no geometric change to the insertion.
        self.task.hand_init_orn_noise = [0.0, 0.0, 3.1416]
        # Widen the SOCKET placement so the restored socket-relative pose obs sees real spatial variation
        # (not a memorisable constant): X band 0.65-0.75m (nominal 0.70 +- 0.05, recentred 2cm nearer than
        # the 0.72 default per user), Y +-15cm, Z +-5cm. Y is the roomy axis -- at x~0.70 it's a base-yaw
        # swing (worst-corner radial ~sqrt(0.75^2+0.15^2)=0.765m, well under the ~0.8m reach). X stays a
        # narrow band: the iiwa base is at origin and ~0.72m is the tool-down SWEET SPOT (author's note L149)
        # -- closer folds the arm into torque/wrist-limit stalls, farther exceeds reach. Keeping the whole
        # workspace comfortably inside reach should also cut the near-singular-IK NaN crashes (the resume
        # cause). VALIDATE the corners with scripts/diag_reset_ik.py (reset-only) if reset failures spike.
        self.task.fixed_asset.init_state.pos = (0.70, 0.0, 0.05)
        # Socket PLACEMENT spread; X/Z raised 0.05->0.06 (2026-08-13). Y stays 0.15 (the roomy base-yaw
        # axis). NOTE this is the PHYSICAL socket spread; the actor's noisy socket ESTIMATE that the policy
        # must correct is fixed_asset_pos_obs_noise_bound (the +-2.5cm anchor) -- tune that separately.
        self.task.fixed_asset_init_pos_noise = [0.08, 0.08, 0.08]
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
        # REALISTIC DEPTH: the confirmed dominant sim2real gap (real D405 ~45% invalid, structured on fin-
        # slots/edges + a missing socket interior; sim trained only 2% uniform). Swap to structured
        # corruption + per-episode modality dropout + 5mm range noise + occasional hole-fill.
        self.depth_corruption_mode = "structured"
        self.depth_modality_dropout = True
        self.depth_noise_std = 0.005          # 5mm range noise on valid returns (was 2mm)
        self.depth_fill_prob = 0.30           # ~30% of episodes get interpolation-filled holes
        # DEPTH-REALISM CURRICULUM: ramp structural corruption + modality-dropout prevalence 0->full over
        # ~160k control steps (~1250 it @ horizon 128, ~50% of a 2500-it run) so the estimator bootstraps
        # hole-prediction on good depth first (the biggest cold-start de-risk of the new regime).
        self.depth_corruption_curriculum_steps = 160000
        # WRIST-YAW CURRICULUM: ramp +-45deg -> the full +-180deg over the same window.
        self.hand_init_yaw_curriculum_steps = 160000
        self.hand_init_yaw_start_deg = 45.0
        # HEAVY APPEARANCE DR (base is vivid BLUE; explicit-estimator over-trusted a clean look):
        self.rgb_noise_std = 0.03
        self.photo_gain_rgb = 0.30            # per-channel gain in [0.7,1.3]
        self.photo_brightness = 0.10          # additive exposure in [-0.1,0.1]
        self.photo_gamma = 0.20               # tone curve in [0.8,1.2]
        self.photo_contrast = 0.20            # contrast in [0.8,1.2]
        self.material_hue_randomize = True    # full-hue materials (spans vivid blue)
        self.light_color_temp_range_k = (3000.0, 8000.0)  # warm->cool white balance
        self.light_intensity_range = (1000.0, 3000.0)     # dome ~ +-50%
        self.key_light_intensity_range = (1000.0, 3000.0)  # key ~ +-50%
        self.cam_rot_jitter_deg = 3.0         # wrist mount roll jitter +-3deg (was 2)
        # LONGER EPISODES: allow force-guided search (was 10s).
        self.episode_length_s = 18.0
        self.task.duration_s = 18.0
        # CONTACT REALISM: effective radial-clearance stopgap (0 => authored mesh clearance; see the field
        # doc -- the clean fix is regenerating the 13mm-shaft/14mm-socket mesh in convert_assets.py).
        _ro = float(getattr(self, "contact_rest_offset", 0.0) or 0.0)
        if _ro > 0.0:
            for art in (self.task.held_asset, self.task.fixed_asset):
                cp = getattr(getattr(art, "spawn", None), "collision_props", None)
                if cp is not None and hasattr(cp, "rest_offset"):
                    cp.rest_offset = _ro
