"""pb-assembly SCREW insertion task config (generalization target — "across parts").

FIRST CUT: pb_screw (Ø9.5mm shank + head) into pb_base's Ø10mm hole at x=+/-40mm. The pre-assembled
pb_top on the base + the head-flush success criterion are the documented NEXT step (this version reuses
cooling's tip-at-bottom success, which is only approximate for the long pb shank -> retune at smoke).

Reuses InsertionEnv + Forge machinery wholesale; only the parts + task geometry differ. The vision
variant inherits the cooling wrist-cam + appearance-DR config (and the aux-head agent yaml), so the
grasp/hole aux heads generalize to pb for free. Geometry values marked TODO-tune are first-bring-up
approximations, refined once the env smoke-steps cleanly (cf. cooling_tasks_cfg note).
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.factory.factory_tasks_cfg import FactoryTask
from isaaclab_tasks.direct.forge.forge_tasks_cfg import ForgeTask

from .assets_cfg import PbBase, PbScrew
from .cooling_tasks_cfg import _RIGID_PROPS, ForgeTaskCoolingInsertCameraCfg, ForgeTaskCoolingInsertCfg


@configclass
class PbScrewInsert(FactoryTask):
    # "peg_insert" = Factory behavior-selector (same peg-in-hole grasp/keypoint/success machinery as
    # cooling); the distinct identity is the gym ID (Isaac-Insertion-PbScrew-*).
    name = "peg_insert"
    fixed_asset_cfg = PbBase()
    held_asset_cfg = PbScrew()
    asset_size = 10.0  # Ø10 socket in mm (keypoint scaling)
    duration_s = 10.0

    # pb_base Ø10 screw holes at x=+/-40mm. Socket BOTTOM in the centered base mesh frame (z ~+5mm,
    # blind hole). TODO-tune z + PbBase.height (depth) against the rendered geometry at smoke.
    socket_offsets_local: list = [(0.040, 0.0, 0.005), (-0.040, 0.0, 0.005)]
    # pb_screw: Ø9.5 shank ~47mm below the head (origin = head/shaft shoulder) -> head_height = 0.023.
    screw_shaft_length: float = 0.047

    success_bonus: float = 1.0
    success_orientation_threshold: float = 0.1745  # ~10deg shaft-axis error
    pre_insert_tilt_max_deg: float = 25.0

    # Raised vs cooling's 0.063: the pb shank is 47mm (vs 17.5mm), so the tip hangs ~30mm lower -> the
    # fingertip must start higher to keep the shaft tip hovering +10..+50mm above the socket opening
    # (smoke showed tip 1.6mm BELOW the opening at 0.063). 0.095 -> ~30mm hover (+/-20mm z noise).
    hand_init_pos: list = [0.0, 0.0, 0.095]
    hand_init_pos_noise: list = [0.008, 0.008, 0.020]  # lateral +/-8mm
    hand_init_orn: list = [3.1416, 0.0, 0.0]
    hand_init_orn_noise: list = [0.0, 0.0, 0.785]

    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 360.0

    held_asset_pos_noise: list = [0.003, 0.0, 0.003]
    held_asset_rot_init: float = 0.0
    grasp_misalign_max_deg: float = 10.0  # unobservable grasp tilt (the aux-head target)

    keypoint_coef_baseline: list = [5, 4]
    keypoint_coef_coarse: list = [50, 2]
    keypoint_coef_fine: list = [100, 0]
    # TODO-tune: pb seats HEAD-FLUSH on the plate (shank >> hole depth), NOT cooling's tip-at-bottom.
    # This placeholder lets the env run; the success/reward geometry is reworked once it smoke-steps.
    success_threshold: float = 0.25
    engage_threshold: float = 0.9

    fixed_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=fixed_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, **_RIGID_PROPS),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # z ~ half the 65mm height so the base bottom rests on the table; randomize_initial_state
            # repositions per env. TODO-tune at smoke.
            pos=(0.6, 0.0, 0.033), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
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
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


@configclass
class ForgePbScrewInsert(PbScrewInsert, ForgeTask):
    contact_penalty_scale: float = 0.0  # pure neg-L2 residual reward (matches cooling); raise for sim2real


@configclass
class ForgeTaskPbScrewInsertCfg(ForgeTaskCoolingInsertCfg):
    """State-obs pb screw insertion. Inherits the cooling env (obs noise + socket obs-noise bound);
    only the task (parts/geometry) is swapped."""

    task_name = "peg_insert"
    task = ForgePbScrewInsert()


@configclass
class ForgeTaskPbScrewInsertCameraCfg(ForgeTaskCoolingInsertCameraCfg):
    """Vision pb screw insertion: reuses the cooling wrist-cam + appearance-DR config (and the aux-head
    agent yaml) wholesale; only the task differs. The camera framing (offset/aim) is tuned for the
    cooling screw -- RE-TUNE at smoke/viz for the larger pb screw (47mm shaft vs 17.5mm)."""

    task = ForgePbScrewInsert()
