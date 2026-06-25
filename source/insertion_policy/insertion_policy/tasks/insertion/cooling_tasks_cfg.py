"""Cooling-screw insertion task config, forked from Forge's PegInsert.

Reuses Forge's env (`ForgeEnv`), controller, force sensing, and pose-noise machinery; only the
held/fixed assets and task geometry are swapped to the cooling parts. Geometry values marked
TODO-tune are approximate for first bring-up and refined once the env steps cleanly.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.factory.factory_tasks_cfg import FactoryTask
from isaaclab_tasks.direct.forge.forge_env_cfg import ForgeEnvCfg
from isaaclab_tasks.direct.forge.forge_tasks_cfg import ForgeTask

from .assets_cfg import CoolingBase, CoolingScrew

# Shared rigid-body props (same tuning Factory/Forge use for stable tight-contact insertion).
_RIGID_PROPS = dict(
    max_depenetration_velocity=5.0,
    linear_damping=0.0,
    angular_damping=0.0,
    max_linear_velocity=1000.0,
    max_angular_velocity=3666.0,
    enable_gyroscopic_forces=True,
    solver_position_iteration_count=192,
    solver_velocity_iteration_count=1,
    max_contact_impulse=1e32,
)


@configclass
class CoolingInsert(FactoryTask):
    # NOTE: internal behavior-selector must be "peg_insert" — Factory's env branches on this
    # name for the held-asset grasp pose, keypoint targets, and success logic, and our task is
    # geometrically a peg-in-hole insertion. The distinct task identity lives in the gym ID
    # (Isaac-Insertion-CoolingPeg-Direct-v0). When we add residual/path-centric logic we'll
    # override those methods directly.
    name = "peg_insert"
    fixed_asset_cfg = CoolingBase()
    held_asset_cfg = CoolingScrew()
    asset_size = 14.0  # socket diameter in mm (used for keypoint scaling)
    duration_s = 10.0

    # --- Cooling-specific (consumed by InsertionEnv) ---
    # Socket BOTTOM positions in the cooling_base local (centered) frame, meters. The platform
    # sockets sit at x=+/-30mm; socket spans z in [0, +17.5mm] so bottom is at z=0. Multi-socket:
    # each env samples one of these at reset.
    socket_offsets_local: list = [(0.030, 0.0, 0.0), (-0.030, 0.0, 0.0)]
    # Held screw: shaft tip is 17.5mm below the part origin (origin = head/shaft shoulder).
    screw_shaft_length: float = 0.0175
    # Reward shaping (pure-residual: negative L2 to socket target + seat bonus).
    success_bonus: float = 1.0
    success_orientation_threshold: float = 0.1745  # rad, about 10deg shaft-axis error

    # Angular pre-insert error injected at reset: the grasped screw is tilted about its shaft tip
    # (tip stays over the socket mouth) by a uniform angle in [0, this], matching the ~20-25 deg
    # upstream orientation uncertainty. 0.0 disables it (recovers the vertical-only reset).
    pre_insert_tilt_max_deg: float = 25.0

    # Robot start, relative to the fixed-asset tip (socket opening). The screw shaft tip sits
    # roughly 30mm below the fingertip in the current Franka grasp, so this gives about 1-5cm of
    # shaft-tip clearance above the socket opening.
    hand_init_pos: list = [0.0, 0.0, 0.063]
    hand_init_pos_noise: list = [0.007, 0.007, 0.020]
    hand_init_orn: list = [3.1416, 0.0, 0.0]
    hand_init_orn_noise: list = [0.0, 0.0, 0.785]

    # Fixed asset placement noise (the upstream-uncertainty source the policy corrects).
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 360.0

    # Held asset noise in the gripper.
    held_asset_pos_noise: list = [0.003, 0.0, 0.003]
    held_asset_rot_init: float = 0.0
    # Grasp misalignment: the screw is not perfectly aligned in the gripper. A random shaft-axis
    # tilt in [0, this] (deg) about a random horizontal axis is baked into the grasp at reset, so the
    # TRUE shaft pose differs from what the fingertip-based proprio obs implies. The actor cannot
    # observe this (held pose is critic-only/privileged) -> it must be inferred from vision. Distinct
    # from pre_insert_tilt (a known commanded arm pose); this is unknown grasp error. 0.0 disables it.
    grasp_misalign_max_deg: float = 5.0

    # Reward keypoint coefficients (kept from PegInsert; revisit for head-flush success).
    keypoint_coef_baseline: list = [5, 4]
    keypoint_coef_coarse: list = [50, 2]
    keypoint_coef_fine: list = [100, 0]
    success_threshold: float = 0.25  # fraction of socket height (~4.4mm)  # TODO-tune for head-flush seat
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
            pos=(0.6, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
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
class ForgeCoolingInsert(CoolingInsert, ForgeTask):
    # Contact-force penalty in the reward: OFF by default so the pure neg-L2 residual reward is
    # used (matches the validated learnability run). Set >0 (Forge uses 0.05-0.2) to penalize
    # over-force for sim-to-real; re-tune against the neg-L2 magnitude when enabling.
    contact_penalty_scale: float = 0.0


@configclass
class ForgeTaskCoolingInsertCfg(ForgeEnvCfg):
    task_name = "peg_insert"  # matches task.name (Factory behavior-selector); see CoolingInsert
    task = ForgeCoolingInsert()
    episode_length_s = 10.0


@configclass
class ForgeTaskCoolingInsertCameraCfg(ForgeTaskCoolingInsertCfg):
    """Vision variant: adds a wrist-mounted RGB-D camera to the validated state-obs task.

    Everything geometric/reward/tilt is inherited unchanged; we only add a TiledCamera parented
    to the Franka wrist (panda_hand) and the image-obs metadata. The state-obs task above is left
    untouched so the validated pipeline stays available for ablations. Train with --enable_cameras.
    """

    # RGB-D render resolution. 160px @ a ~6cm zoomed view => ~0.37mm/px (~2.7 px/mm), which
    # oversamples the ~1mm alignment threshold (Nyquist: need >=2 px/mm to resolve 1mm). 4 channels =
    # RGB (3) + depth (1), stacked in InsertionEnv._get_camera_image. Render is ~2x slower than 64px.
    image_height: int = 160
    image_width: int = 160
    image_channels: int = 4
    # Stage-1 debug: dump a rendered RGB/depth frame to disk to sanity-check the mount pose.
    write_image_to_file: bool = False
    # Ablation: feed the policy a BLANK (zeroed) image instead of the rendered one. Same network +
    # camera/render overhead, but the CNN gets no signal -> isolates whether the image CONTENT
    # contributes (blank ~= full means the image isn't helping). Off by default.
    blank_image: bool = False

    # Wrist camera. A standalone per-env prim whose world pose is driven every step to follow the
    # fingertip (InsertionEnv._update_camera_pose), rather than parented under the articulation link
    # -- parenting a camera under the moving robot trips Isaac's Fabric-hierarchy pose resolution.
    # The camera rides a fixed offset from the fingertip frame, defined below.
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/wrist_cam",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            # focal_length=42mm zooms to a ~6cm-wide view at the ~12cm camera->socket distance
            # (focal = distance*aperture/view_width = 120*20.955/60 ~= 42), so the socket + a
            # tilt-swung shaft fill the frame at ~2.7 px/mm. Verify framing with a render check before
            # the long run (working distance shifts with pose). Far clip 0.3m clips distant background.
            focal_length=42.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 0.3)
        ),
        width=160,
        height=160,
    )
    # Camera EYE position in the fingertip-midpoint frame (orientation comes from a look-at on the
    # active socket opening, see InsertionEnv._update_camera_pose). An oblique side mount: offset to
    # the side and above the fingertip so the camera sees the shaft entering the socket without the
    # screw head occluding it. Tuned in the Stage-1 render sanity check (keep the camera->socket
    # working distance > the D405 ~7cm min depth). At nominal grasp local x->world x, local z->-world z.
    wrist_cam_offset_pos: tuple = (-0.06, 0.0, -0.03)  # 6cm to the side, 3cm above the fingertip

    # Optional third-person debug camera, set only by scripts/viz_camera.py to render the scene from
    # outside (to see how the wrist cam is mounted). None in training -> zero impact.
    scene_camera: TiledCameraCfg | None = None

    # Domain randomization (vision-specific): per-episode camera eye-position jitter (meters, stddev)
    # modelling mount / hand-eye-calibration uncertainty for sim-to-real. 0.0 = OFF -- keep off for
    # the first clean vision baseline (so we can tell whether vision helps), then enable for the
    # sim-to-real / robustness runs. Dynamics DR (friction, mass, dead-zone) is already active via
    # Forge's EventCfg; appearance DR (per-env lights / textures) needs per-env lights + GPU iteration
    # and is deferred to the sim-to-real phase.
    cam_pos_jitter: float = 0.0

    def __post_init__(self):
        # Keep Fabric ENABLED (GPU PhysX is stable on Fabric; disabling it caused CUDA-700 crashes /
        # hangs in vision runs). This build's usdrt lacks the `hierarchy` submodule that Isaac Lab's
        # Fabric world-pose path needs, but we route ONLY the camera's xform view to the USD pose path
        # via a targeted monkeypatch in insertion_env.py (physics keeps Fabric). See that patch.
        super().__post_init__() if hasattr(super(), "__post_init__") else None
        # Render once per env step (decimation=8) instead of once per physics substep — 8x less render
        # work; all the policy needs.
        self.sim.render_interval = self.decimation
