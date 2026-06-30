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
from isaaclab_tasks.direct.forge.forge_env_cfg import ForgeEnvCfg, ForgeObsRandCfg
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

    # Robot start, relative to the fixed-asset tip (socket opening). This is the FINGERTIP target:
    # nominally 6.3cm above the socket, with +/-2cm z noise -> fingertip starts about 4.3-8.3cm
    # above the opening. The screw shaft tip is lower because it hangs below the fingertip.
    hand_init_pos: list = [0.0, 0.0, 0.063]
    hand_init_pos_noise: list = [0.008, 0.008, 0.020]  # lateral +/-8mm (realistic upstream target)
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
    grasp_misalign_max_deg: float = 10.0  # realistic upstream grasp error (was 5.0 for the old A/B)

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

    # Proprio/observation noise (actor-side only; the critic keeps the clean privileged state). Forge
    # applies this in _compute_intermediate_values -> the policy sees noisy_fingertip_pos/quat/force.
    # Forge's defaults are LIGHT (0.25mm / 0.1deg); bump to MODERATE so the state-vs-vision comparison
    # is fair: real proprioception is noisy, and a clean signal would flatter the state policy on a
    # cue it won't have on hardware. Velocity noise is free (ee_lin/ang vel are finite-differenced from
    # the noisy fingertip pose). F/T is the genuinely noisy channel, so its noise is the largest.
    # Shared by both tasks (the vision cfg subclasses this). Re-tune from measured hardware noise later.
    obs_rand: ForgeObsRandCfg = ForgeObsRandCfg(
        fingertip_pos=0.0005,    # 0.5 mm stddev (was 0.25 mm)
        fingertip_rot_deg=0.5,   # 0.5 deg stddev (was 0.1 deg)
        ft_force=2.0,            # larger force-sensor noise (was 1.0)
    )

    # Actor-side socket/goal observation error. Factory's inherited fixed_asset_pos noise is Gaussian
    # (~1mm stddev by default), which makes the state policy's target estimate too clean for the
    # upstream pose-estimation error. Use bounded per-episode uniform noise instead: x/y are lateral
    # target error "up to 8mm"; z is smaller because the base is table-supported, but nonzero for
    # pose-estimation/table-height/calibration error.
    fixed_asset_pos_obs_noise_bound: list = [0.008, 0.008, 0.002]


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
            # focal_length=28mm -> ~41deg FOV (2*atan(aperture/2/focal)). Wider than the old 42mm so
            # BOTH the shaft tip and the socket opening fit in frame (they subtend ~17deg apart from
            # the oblique mount) with margin for lateral/tilt/approach drift. The old narrow 42mm
            # centred only the hole and clipped the hovering shaft -- losing the grasp-misalignment cue.
            # ~8-9cm working distance, ~2.5 px/mm. Far clip 0.3m clips distant background. Verify framing
            # with a render check (viz_camera) before the long run; working distance shifts with pose.
            focal_length=28.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 0.3)
        ),
        width=160,
        height=160,
    )
    # RIGID wrist mount ("GoPro on the wrist"): the camera EYE, AIM POINT, and full ORIENTATION are
    # fixed in the fingertip-midpoint frame, so the camera is bolted to the gripper and never re-aims
    # (see InsertionEnv._update_camera_pose). Oblique mount at ~45deg azimuth (equal -x,-y offset) and
    # ~37deg off vertical: the -y component looks roughly PERPENDICULAR to the grasp-tilt plane so the
    # shaft's grasp-misalignment tilt is visible (not foreshortened), while the -x component keeps the
    # gripper profile thin enough to avoid occluding the hole. The Franka pressing axis is fingertip
    # +y; RE-DERIVE this azimuth for the custom gripper (viz_camera prints the pressing axis + framing).
    # Keep the camera->socket working distance > the D405 ~7cm min depth. Verify framing with a render.
    wrist_cam_offset_pos: tuple = (-0.041, -0.041, -0.045)  # eye: ~45deg azimuth, ~6cm out, 4.5cm back
    # Aim point in the fingertip frame: the INSERTION REGION between the shaft tip (z_local~0.018) and
    # the nominal socket opening (z_local~0.058, measured mean over envs via viz_camera) -- aiming at
    # the gap keeps BOTH the shaft and the hole in frame, so the shaft's tilt/offset (the grasp-
    # misalignment cue) is visible alongside the socket-drift (lateral) cue. The camera points here
    # RIGIDLY (wrist-fixed); both drift off-centre under error -- the transfer-valid cues. NOT the true
    # socket (that was the old non-physical look-at, which also clipped the shaft out the top).
    wrist_cam_look_target_pos: tuple = (0.0, 0.0, 0.033)

    # Optional third-person debug camera, set only by scripts/viz_camera.py to render the scene from
    # outside (to see how the wrist cam is mounted). None in training -> zero impact.
    scene_camera: TiledCameraCfg | None = None

    # Domain randomization (vision-specific): per-episode camera eye-position jitter (meters, stddev),
    # sampled at reset, modelling mount / hand-eye-calibration uncertainty for sim-to-real. Via the
    # rigid-mount look-at toward the fixed aim point it also tilts the optical axis slightly, so it is
    # a combined position+angle mount perturbation. ON (5mm) for the hardened realistic batch.
    # Dynamics DR (friction, mass, dead-zone) is already active via Forge's EventCfg; appearance DR
    # (scene materials/lighting + per-env photometric aug + camera sensor noise) is the block below.
    cam_pos_jitter: float = 0.005
    # Per-episode camera ROLL jitter about the optical axis (deg, stddev), sampled at reset. Models
    # mount/bracket roll-calibration error on top of the rigid mount. Small -- the mount is rigid, this
    # is just realistic uncertainty (NOT the old world-up roll artifact, which has been removed).
    cam_rot_jitter_deg: float = 2.0

    # --- Appearance domain randomization (sim-to-real for the wrist RGB-D) -----------------------
    # FOUR complementary layers, all default-ON, each a no-op when its knob is 0/empty (so the
    # state-obs task and the deterministic render scripts are unaffected). Expect the absolute sim
    # success % to DROP vs the no-appearance-DR run -- that is correct; we are buying transfer, not a
    # higher sim number. See InsertionEnv._randomize_appearance / _get_camera_image.

    # (1) SCENE-MATERIAL DR (PER-ENV, matte). Parts are matte 3D-printed plastic -> albedo (colour) is
    # the dominant material cue and the real filament colour is only a GUESS. EACH env gets its own
    # material (colours vary WITHIN a batch). Per env we draw ONE scene colour (base +/- color_jitter),
    # then screw and base = that colour +/- a SMALL part_color_jitter: usually similar (matching reality
    # -- same filament => no screw-vs-base colour contrast to exploit), occasionally diverging (robust).
    # Geometry-correct specular/shading under the moving wrist light -- which the image aug can't fake.
    # Fail-safe: self-disables on any USD error (can't kill a run). See _setup_part_materials.
    randomize_part_materials: bool = True
    material_base_color: tuple = (0.33, 0.33, 0.40)  # darker centre -> richer/less-washed colours (was 0.45/0.5 = pale)
    material_color_jitter: float = 0.25             # per-env scene-colour half-range (the between-env DR)
    material_part_color_jitter: float = 0.05        # screw-vs-base divergence within an env (small => usually same)
    material_roughness_range: tuple = (0.55, 0.95)  # matte 3D-printed plastic (high roughness -> diffuse colour dominates, less white specular)
    material_metallic: float = 0.0

    # (2) SCENE-LIGHTING DR: a DOME (soft ambient fill: intensity + colour) PLUS a directional KEY light
    # (DistantLight) whose DIRECTION is re-pointed per reset (random azimuth, elevation off straight-down)
    # -> the moving shadows/specular highlights a dome alone can't give (the real cell has a directional
    # source). Distant = parallel/infinite, so it lights every env identically (no per-env brightness
    # confound a positioned point light would add). Distance/falloff intentionally omitted (negligible
    # over a ~10cm workspace). See _setup_key_light / _randomize_scene_light.
    light_intensity_range: tuple = (700.0, 2500.0)   # dome ambient lux (dome+key stack; keep combined ~ original 2000)
    light_color_jitter: float = 0.12                 # dome +/- per-RGB-channel about the 0.75 grey baseline
    key_light_intensity_range: tuple = (700.0, 2500.0)  # directional key lux
    key_light_elev_range_deg: tuple = (15.0, 60.0)   # key direction: tilt off straight-down (deg)
    key_light_angle: float = 1.0                     # angular size (deg) -> shadow softness

    # (3) PER-ENV PHOTOMETRIC IMAGE AUG (resampled per episode PER ENV; applied every step). Gives the
    # within-batch appearance diversity a single global light/material can't. Pure GPU tensor ops (no
    # USD/instancing risk). NOTE the env re-centres each image (rgb - per-channel mean), which already
    # grants brightness-invariance, so we lean on terms that SURVIVE that: contrast, gamma, per-channel
    # gain (white balance); brightness is applied BEFORE gamma so it survives nonlinearly. Half-ranges.
    photo_brightness: float = 0.06  # additive exposure shift (pre-gamma), +/- this in [0,1] units
    photo_contrast: float = 0.25    # contrast gain in [1-this, 1+this] about the per-channel mean
    photo_gain_rgb: float = 0.12    # independent per-channel gain in [1-this, 1+this] (white balance)
    photo_gamma: float = 0.3        # tone-curve gamma in [1-this, 1+this]

    # (4) CAMERA SENSOR NOISE (every step), modelling the real D405: gaussian RGB noise; gaussian depth
    # range noise (metres, applied to real returns only) + per-pixel depth DROPOUT (no-return holes ->
    # 0, like a real depth sensor). Keeps the blank-image control truly blank (added after that branch).
    rgb_noise_std: float = 0.02     # stddev on the [0,1] RGB image
    depth_noise_std: float = 0.002  # metres (2mm) on real depth returns, before far-clip normalisation
    depth_dropout_prob: float = 0.02  # fraction of depth pixels zeroed (sensor holes)

    # (5) TRAINING IMAGE GALLERY (observability, not DR): every cam_log_interval camera reads, dump a
    # tiled montage (all envs) of exactly what the policy SEES to cam_log_dir -> watch framing / DR
    # aggressiveness / occlusion DURING an 8h run without a second GPU job. 0 disables. renders/ is
    # gitignored. (For rollout BEHAVIOUR video, use train.py --video; see auto_resume_train.sh VIDEO=1.)
    cam_log_interval: int = 2000    # camera reads between montage dumps (~15 it at 64 envs); 0 = off
    cam_log_dir: str = "renders/train_cam"

    def __post_init__(self):
        # Keep Fabric ENABLED (GPU PhysX is stable on Fabric; disabling it caused CUDA-700 crashes /
        # hangs in vision runs). This build's usdrt lacks the `hierarchy` submodule that Isaac Lab's
        # Fabric world-pose path needs, but we route ONLY the camera's xform view to the USD pose path
        # via a targeted monkeypatch in insertion_env.py (physics keeps Fabric). See that patch.
        super().__post_init__() if hasattr(super(), "__post_init__") else None
        # Render once per env step (decimation=8) instead of once per physics substep — 8x less render
        # work; all the policy needs.
        self.sim.render_interval = self.decimation
