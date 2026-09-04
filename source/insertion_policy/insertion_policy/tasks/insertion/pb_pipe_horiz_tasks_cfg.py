"""pb-assembly HORIZONTAL PIPE insertion task (2nd generalization target — "across parts + direction").

Scenario (2026-08-28): the pb_pipe is grasped near one end across its flat faces (from above, gripper Z
down) so its long axis sticks out ~90deg to gripper Z (HORIZONTAL), and seated in pb_base's central
halfpipe. The channel is mesh-local Y; yawing the base by -90deg maps it to world +X while the base's
mesh-local X long axis reads as world Y. This is modelled EXACTLY like the cooling E2E-iiwa visuomotor task
(ForgeTaskCoolingInsertIiwaE2E*Cfg) — same iiwa7+pdz robot, IK+PD controller, RGB wrist cam, appearance /
dynamics DR, goal-anchor noise, squashing reward + all curricula — with only TWO things changed per the
supervisor directive: (1) the PARTS (pb_pipe + pb_base bore, see assets_cfg), and (2) the INCOMING
    DIRECTION (horizontal). The shared env stays on its screw defaults unless these cfg fields opt into
    midpoint/bore-axis geometry for this task.

>>> SMOKE / RENDER PASS REQUIRED before training <<< (mirrors how pb_screw was left):
  * Geometry (socket depth/diameter, socket_offsets_local, hand_init_pos height) are mesh-scan first cuts
    marked TODO-tune. Confirm the pipe seats in the bore and the grasp is stable with render_grasp_grid /
    a viz_camera pass, then retune.
  * The HORIZONTAL grasp: the pipe long axis (mesh-local Y) must end up perpendicular to gripper Z. Factory's
    reset/weld transform decides the actual pipe-in-gripper pose; VERIFY the first grasp render and adjust the
    relative grasp offset / base yaw if the wrong end points at the bore.
  * Base placement: the halfpipe channel must run along world X while the base long axis runs along world Y.
    fixed_asset_yaw_nominal_deg performs that mesh-to-world mapping; VERIFY in the reset and goal renders.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.factory.factory_tasks_cfg import FactoryTask
from isaaclab_tasks.direct.forge.forge_tasks_cfg import ForgeTask

from .assets_cfg import PbBasePipeSocket, PbPipe
from .cooling_iiwa_tasks_cfg import (
    ForgeCoolingInsertIiwa,
    ForgeTaskCoolingInsertIiwaE2ECfg,
    ForgeTaskCoolingInsertIiwaE2EVisionCfg,
    _rigidify_held_for_weld,
)
from .cooling_tasks_cfg import _RIGID_PROPS


# ---------------------------------------------------------------------------------------------------
# Task geometry (parts + horizontal direction). Everything else (robot, ctrl, reward, DR, curricula) is
# inherited from the cooling E2E-iiwa cfgs below, so this class only carries what actually differs.
# ---------------------------------------------------------------------------------------------------
@configclass
class PbPipeHorizInsert(FactoryTask):
    # "peg_insert" = the Factory behaviour selector (peg-in-hole grasp/keypoint/success machinery), same as
    # cooling. The distinct identity is the gym id (Isaac-Insertion-PbPipe-Iiwa-E2E-*).
    name = "peg_insert"
    fixed_asset_cfg = PbBasePipeSocket()
    held_asset_cfg = PbPipe()
    asset_size = 50.0  # ~Ø50mm bore (keypoint scaling); cooling used the socket Ø in mm. TODO-tune.
    duration_s = 13.0

    # Seated channel frame in the base mesh. The central halfpipe is a circular cradle extruded along local Y.
    # Mesh scan: centre=(X=0, Z=+19.408 mm), radius=24.997 mm. Keep the commanded pipe midpoint 20 mm
    # upstream of the geometric channel centre so the angled gripper/pads clear the fixture's open face.
    # With fixture yaw=-90deg, local -Y is world -X.
    socket_offsets_local: list = [(0.0, -0.020, 0.019408)]
    held_ref_offset_local: tuple = (0.0, 0.0, 0.0)  # pipe root/origin = tube midpoint, and that is the seated goal
    held_axis_local: tuple = (0.0, 1.0, 0.0)        # pipe long axis in its mesh/root frame
    socket_axis_local: tuple = (0.0, 1.0, 0.0)      # central halfpipe/channel axis
    # Keep reset 65 mm upstream of the GEOMETRIC channel centre. Because the commanded seated goal above
    # is already 20 mm upstream, its relative opening offset is another 45 mm (20 + 45 = 65 mm absolute).
    # This preserves the approved reachable reset pose while shortening only the final insertion by 20 mm.
    # With fixture yaw=-90deg, local -Y is world -X; local-Z=0 keeps the pipe at opening height.
    socket_opening_offset_local: tuple = (0.0, -0.045, 0.0)
    reset_held_ref_to_socket_opening: bool = True
    num_keypoints: int = 4
    reward_keypoint_offsets_local: list = [
        (0.0, -0.0400, 0.0),
        (0.0, -0.0133, 0.0),
        (0.0, 0.0133, 0.0),
        (0.0, 0.0400, 0.0),
    ]

    success_bonus: float = 1.0
    success_orientation_threshold: float = 0.1745  # ~10deg pipe-axis error
    axis_alignment_abs: bool = True  # +Y and -Y pipe directions are both collinear with the through-bore
    success_axial_abs: bool = True  # midpoint should coincide with the bore centre, not merely pass it
    pre_insert_tilt_max_deg: float = 25.0

    # HELD-ASSET GRASP GEOMETRY. The grasp-depth formula (InsertionEnv.get_handheld_asset_relative_pose)
    # HARD-SETS relative_pos.z = (height - screw_shaft_length) - franka_fingerpad_length -- the z-offset of the
    # held-asset ORIGIN below the fingertip (TCP). The pipe mesh is CENTERED at its origin, so to sit the tube
    # CENTERED at the pad plane we want that z-offset ~= 0, i.e. (height - screw_shaft_length) ~= fingerpad.
    # With fingerpad = 0.005 (the cooling default) set screw_shaft_length = height - 0.005 = 0.075. (The name
    # is historical -- here it is just the depth-formula knob, NOT a real shaft.) The pipe is gripped at its
    # CENTER for this first cut (symmetric tube -> still a valid horizontal grasp); gripping near one END is a
    # later refinement (needs a fingertip-X offset the env formula does not expose without touching InsertionEnv).
    # TODO-tune at render together with franka_fingerpad_length.
    screw_shaft_length: float = 0.075  # = height(0.080) - fingerpad(0.005) -> centered vertical grip

    # ALONG-AXIS GRASP OFFSET (grip NEAR ONE END, not the centre). The pipe must insert until the pipe's
    # MIDPOINT coincides with the bore's centre (user spec 2026-08-28); if gripped at the centre, the trailing
    # half + the gripper would collide with the base before the midpoint seats. So shift the grip toward one
    # end. This is in the relative pose frame consumed by Factory's placement; because Factory later inverts
    # that transform, +30mm here has probed as held-root-at -30mm in the fingertip frame (desired near-end grip).
    grasp_offset_local: tuple = (0.0, 0.030, 0.0)
    # Cant the gripper 45deg toward world +X while the pipe remains level along world X. The sign is chosen
    # for the desired side-view lean; the cfg-gated transform is PB-only.
    grasp_fixed_tool_tilt_deg: float = -45.0

    # HORIZONTAL orientation (fixed 2026-08-28 after the first render): the pipe's LONG axis is mesh-local Y.
    # A rotation about X maps mesh-Y onto the fingertip Z (the APPROACH axis) -> the pipe stays VERTICAL (the
    # bug in the first render). The correct rotation is R_z(-90deg): mesh-Y -> fingertip-X (HORIZONTAL, toward
    # the bore), mesh-X (the flat faces) -> fingertip-Y (the pad pressing axis, so the pads grip the flats),
    # mesh-Z stays vertical. Baked into held_asset.init_state.rot below = quat (0.7071, 0, 0, -0.7071).
    # (Factory's held_asset_rot_init is IGNORED for peg_insert, so the spawn rot is the lever.)

    # Reset approach: pipe already runs along world X (the halfpipe channel) but starts fully outside its
    # open end. The policy drives it along its long axis into the central cradle.
    hand_init_pos: list = [0.0, 0.0, 0.067]        # fingertip above the bore mouth -- TODO-tune (pipe is big)
    reset_ref_ik_iters: int = 30                   # enough IK passes to reach the level along-axis reset pose
    reset_ref_ik_pos_tol: float = 0.002             # refine the pipe centreline to within 2 mm
    reset_ref_ik_rot_tol_deg: float = 1.0           # refine toward the level-pipe / configured grasp target
    reset_ref_ik_fail_pos_tol: float = 0.008         # angled grasp converges to ~5.1mm; keep it instead of hover fallback
    reset_level_toolstraight: bool = True          # force the reset pipe LEVEL; weld geometry sets tool angle
    reset_level_pitch_comp_deg: float = 0.0         # stable reset branch tracks the canonical level/tool-down frame
    hand_init_pos_noise: list = [0.003, 0.003, 0.010]
    # tool-down + yaw so the pipe long axis -> world X. The pure tool-down hover settled with the pipe tilted
    # ~16deg UP from level (realized pipe world z-component ~+0.27); add a small pitch to counter it so the
    # RESET pipe is LEVEL (horizontal), matching the seated goal. Sign/magnitude tuned at render.
    hand_init_orn: list = [math.pi, 0.0, -math.pi / 2.0]  # tool-down; pipe long axis -> world X
    hand_init_orn_noise: list = [0.0, 0.0, 0.0]

    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.0]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 0.0
    fixed_asset_custom_placement: bool = True
    base_y_clusters: list | None = None
    # Solve the low tool-down IK with the fixture staged at X=56.5 cm, then translate only the kinematic
    # fixture +15 mm after the grasp reset. This leaves the level arm solution untouched while putting the
    # final fixture at X=58 cm and increasing the realized pipe-to-base setback by 15 mm. Y=-25 cm selects
    # the reachable iiwa branch for the gripper's +X lean; lower-Y probes (-0.30 and -0.38 m) fell back.
    fixed_asset_xy_ranges: list = [0.565, 0.565, -0.25, -0.25]
    fixed_asset_post_reset_world_shift: tuple = (0.015, 0.0, 0.0)
    fixed_asset_z_center: float = 0.0325
    fixed_asset_z_noise: float = 0.0
    fixed_asset_reach_radius: float = 0.72
    fixed_asset_yaw_nominal_deg: float = -90.0  # mesh X (base long) -> world -Y; mesh Y (channel) -> world +X
    fixed_asset_yaw_nominal_halfwidth_deg: float = 0.0
    fixed_asset_yaw_wide_frac: float = 0.0
    fixed_asset_yaw_wide_deg: list = [3.0, 45.0]

    held_asset_pos_noise: list = [0.003, 0.0, 0.003]
    grasp_misalign_max_deg: float = 3.0

    keypoint_coef_baseline: list = [5, 4]
    keypoint_coef_coarse: list = [50, 2]
    keypoint_coef_fine: list = [100, 0]
    success_threshold: float = 0.25
    engage_threshold: float = 0.9

    # CRITICAL: the parent (CoolingInsert) freezes fixed_asset/held_asset -> the COOLING USDs at class
    # definition time; overriding *_cfg alone does NOT rebuild them. Redeclare the PB fixed base as a kinematic
    # rigid object (hard-glued) and the held pipe as the normal one-body asset that the weld helper rigidifies.
    fixed_asset: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=fixed_asset_cfg.usd_path,
            activate_contact_sensors=True,
            # Truly glued: a kinematic RigidObject cannot be moved or toppled by pipe contacts. The converted
            # USD carries a one-body articulation root, so disable that root when loading it as a rigid object.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, disable_gravity=True, **_RIGID_PROPS
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # STANDS on its 16x4cm footprint (UP = mesh-Z, height 6.5cm) so the halfpipe channel (mesh-Y) is HORIZONTAL.
            # z = half the 65mm height so the bottom rests on the table (z=0). rot=identity keeps mesh-Z up.
            # The E2E vision cfg's custom placement repositions xy + yaw per env (yaw keeps the bore horizontal).
            pos=(0.565, -0.25, 0.0325), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )
    held_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=held_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, **_RIGID_PROPS),
            mass_props=sim_utils.MassPropertiesCfg(mass=held_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # HORIZONTAL PIPE: rot = R_z(-90deg) = quat (0.7071, 0, 0, -0.7071). Maps the pipe's mesh-local-Y
            # long axis onto fingertip-X (HORIZONTAL, toward the bore) and the flat mesh-X faces onto the pad
            # pressing axis (fingertip-Y). (The first cut used R_x(-90) which mapped Y onto the fingertip Z =
            # the APPROACH axis, leaving the pipe pointing DOWN -- the "too low / vertical" render bug.)
            # VERIFY in the grasp render; flip to +90 = (0.7071,0,0,0.7071) if the pipe points away from the bore.
            pos=(0.0, 0.4, 0.1), rot=(0.70710678, 0.0, 0.0, -0.70710678), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


@configclass
class ForgePbPipeHorizInsertIiwa(PbPipeHorizInsert, ForgeCoolingInsertIiwa, ForgeTask):
    """Robot + grasp geometry inherited from the cooling iiwa task (pdz gripper, franka_fingerpad_length,
    the 0.72m workspace __post_init__); only the parts + horizontal geometry above differ."""


# ---------------------------------------------------------------------------------------------------
# STATE-FIRST debug scaffold: true-delta obs, no camera. Smoke the horizontal control/reward loop cheaply
# BEFORE spending GPU on the vision run. Inherits every E2E knob (action scale, PD arm, squashing reward,
# weld, reset-IK tolerances, DR) from the cooling state cfg; only the task is swapped.
# ---------------------------------------------------------------------------------------------------
@configclass
class ForgeTaskPbPipeHorizInsertIiwaE2ECfg(ForgeTaskCoolingInsertIiwaE2ECfg):
    task = ForgePbPipeHorizInsertIiwa()

    def __post_init__(self):
        super().__post_init__()
        # Re-rigidify + re-weld for the NEW held part (super() already did it for the cooling task instance;
        # our task is a different instance, so redo it). weld flag + solver-iter loop are inherited.
        self.weld_held_to_gripper = True
        # The near-end pipe grasp puts the TCP about 11 cm behind the seated midpoint at reset. Cooling's
        # inherited +/-5 cm goal-centred safety box would therefore pull a zero-action reset forward.
        self.e2e_action_safety_box = 0.13
        _rigidify_held_for_weld(self.task)


# ---------------------------------------------------------------------------------------------------
# The REAL weekend training target: RGB wrist-cam E2E visuomotor, horizontal pb_pipe. Inherits the ENTIRE
# newest-run recipe (appearance + dynamics DR, goal-anchor noise + curricula, force/anti-jam penalties,
# descend-gate + recenter shaping, latency, gravity comp, squashing reward + seat bonus) from the cooling
# E2E vision cfg. ONLY the task (parts + horizontal direction) and the base placement region change.
# ---------------------------------------------------------------------------------------------------
@configclass
class ForgeTaskPbPipeHorizInsertIiwaE2EVisionCfg(ForgeTaskCoolingInsertIiwaE2EVisionCfg):
    task = ForgePbPipeHorizInsertIiwa()

    def __post_init__(self):
        super().__post_init__()
        # Re-rigidify + re-weld for the pb_pipe held part (see the state cfg note).
        self.weld_held_to_gripper = True
        # Cover the full pipe approach pose: midpoint is 8 cm upstream and the near-end TCP another 3 cm
        # upstream. Otherwise the first zero action is clamped toward the goal and destroys the reset gap.
        self.e2e_action_safety_box = 0.13
        _rigidify_held_for_weld(self.task)
        # Re-apply the E2E task geometry that super() set on the COOLING task instance (our task is a
        # different instance): the goal-anchor / tilt / wrist-yaw curricula + grasp DR live on self.task.
        # We mirror the cooling values so the DR/curricula are IDENTICAL; only the numbers that are part-
        # specific (grasp jitter magnitudes, tip trim, start height) are the pipe's.
        self.task.pre_insert_tilt_max_deg = 25.0
        self.task.pre_insert_tilt_start_deg = 0.0
        self.task.pre_insert_tilt_curriculum_steps = 128000
        self.task.pre_insert_tilt_halfnormal = True
        self.task.goal_anchor_injection = True
        self.task.goal_anchor_lat_max = 0.0175
        self.task.goal_anchor_z_max = 0.008
        self.task.goal_anchor_curriculum_steps = 128000
        self.task.wrist_yaw_halfnormal = True
        self.task.wrist_yaw_max_deg = 45.0
        self.task.grasp_misalign_max_deg = 3.0
        self.task.grasp_misalign_secondary_deg = 0.0
        # Grasp position jitter — pipe is much bigger than the screw; start from the cooling values and widen
        # at smoke if the render shows the grasp is too tight/loose.
        self.task.grasp_pos_jitter_axial_mm = 3.0
        self.task.grasp_pos_jitter_lateral_mm = 2.0
        self.task.success_threshold = 0.25

        # --- HORIZONTAL-PIPE placement (user spec: base long axis -> world Y; halfpipe channel -> world X) ---
        # Same custom-placement machinery + reach guard as cooling, but the pipe base is a TALL bracket (6.5cm)
        # that STANDS on its footprint, so it must NOT inherit cooling's z_center=-0.005 (which sinks the tall
        # base BELOW the floor -> the contact solver ejects + TUMBLES it, the "base on its side" the user saw).
        # Stand it with its bottom ON the table: center z = half height = 0.0325, near-zero z noise.
        self.task.fixed_asset_z_center = 0.0325   # base centre at half its 6.5cm height -> bottom on the table
        self.task.fixed_asset_z_noise = 0.0       # no z jitter -> never penetrates the floor / never tumbles
        # Cooling's __post_init__ widens this box on its task instance. Reapply the PB deployment band after
        # super(): keep the reachable angled-tool staging pose (56.5 cm front, 25 cm negative-Y). The
        # task's post-reset shift then moves the final fixture pose to X=58 cm without perturbing the arm.
        self.task.fixed_asset_xy_ranges = [0.565, 0.565, -0.25, -0.25]
        self.task.fixed_asset_reach_radius = 0.70
        self.task.fixed_asset_yaw_nominal_deg = -90.0  # mesh X (base long) -> world -Y; mesh Y (channel) -> world +X
        self.task.fixed_asset_yaw_nominal_halfwidth_deg = 0.0
        # No WIDE-yaw tail for the pipe (keep the bore reliably facing the pipe; wide mis-fixture is a later DR).
        self.task.fixed_asset_yaw_wide_frac = 0.0
