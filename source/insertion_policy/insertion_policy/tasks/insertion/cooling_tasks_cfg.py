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
    # Tilt CURRICULUM: linearly ramp the effective max tilt from ``pre_insert_tilt_start_deg`` to
    # ``pre_insert_tilt_max_deg`` over the first ``pre_insert_tilt_curriculum_steps`` env CONTROL steps
    # (self.common_step_counter; ~= iterations * horizon_length, so e.g. 128*600=76800 ramps over ~600 it
    # at horizon 128). 0 steps => no curriculum (constant at max, the current behaviour). Lets the policy
    # learn to SEAT aligned first, then generalize to tilt -- the direct attack on the axis-error binding
    # constraint. NOTE: common_step_counter resets to 0 on a resumed run, so set the ramp to finish well
    # inside one run segment (re-ramp on a mid-anneal crash is a minor transient).
    pre_insert_tilt_start_deg: float = 0.0
    pre_insert_tilt_curriculum_steps: int = 0

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
    # Secondary out-of-plane grasp tilt (deg) about the fingertip X axis (perpendicular to the pressing
    # axis). Flat parallel pads mostly BLOCK this, but pad compliance / an off-centre screw head allow a
    # few deg in practice. Baked per-env into the weld like the primary tilt. 0.0 => off (pure single-axis
    # pressing-axis tilt, the physically-strict model). Set ~6-8 for the hardened sim2real grasp DR.
    grasp_misalign_secondary_deg: float = 0.0
    # Grasp POSITION DR (2026-08-05): the real clamp doesn't grab the screw at the exact same spot.
    # Per-env jitter of where the head sits between the pads -- axial (shaft depth) + lateral (in-pad
    # plane); y (pressing axis) is clamp-fixed. 0 => off (residual tasks unaffected). Welded task only.
    grasp_pos_jitter_axial_mm: float = 0.0
    grasp_pos_jitter_lateral_mm: float = 0.0

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
    # target error "up to ~1cm"; z is smaller because the base is table-supported, but nonzero for
    # pose-estimation/table-height/calibration error. Bounded PER-EPISODE (fixed within an episode) ->
    # models the real INIT socket-pose estimate (base is static during a seat, so no per-step tracking
    # needed); set to the real init-estimator accuracy (~1cm per user, 2026-08-04).
    fixed_asset_pos_obs_noise_bound: list = [0.025, 0.025, 0.005]
    # Anchor-noise CURRICULUM (hydra: env.fixed_asset_pos_obs_noise_curriculum_steps=N + _start=[...]).
    # Ramp the socket-anchor obs noise from _start -> _bound over N control steps; 0 => constant at _bound.
    # 2.5cm from scratch is uncrackable (s192=29%), so learn at 1cm first and extend.
    fixed_asset_pos_obs_noise_start: list = [0.010, 0.010, 0.002]
    fixed_asset_pos_obs_noise_curriculum_steps: int = 0
    # GRAVITY COMP: subtract the pre-contact gravity wrench from ft_force so the policy's force obs is pure
    # CONTACT, matching the real robot's gravity-compensated F/T (hydra: env.use_gravity_comp=True).
    use_gravity_comp: bool = False

    # --- 6-axis F/T: expose the CONTACT TORQUE channels in the policy obs ---------------------
    # Forge already feeds the 3-axis contact FORCE (`ft_force` = noisy `force_sensor_smooth[:,0:3]`)
    # into the policy, but the matching 3 TORQUE channels (`force_sensor_smooth[:,3:6]`, the contact
    # MOMENT a tilted shaft makes in the hole) are computed by the base env and left UNUSED. Turning
    # this on appends the (noisy) torque triplet to the policy proprio vector -> 24-d becomes 27-d,
    # giving the policy the 6-axis wrench. The hybrid net auto-infers the wider proprio from the obs
    # space (no net change); the aux-label group is untouched. Toggle so it screens cleanly on/off.
    use_torque_obs: bool = False
    # Gaussian obs noise (stddev) on the torque channels, mirroring Forge's `obs_rand.ft_force` on the
    # force channels. Kept a SEPARATE knob because torque (N*m) and force (N) have different units and
    # magnitudes -- coupling them 1:1 could swamp the (small) torque signal. Default matches the bumped
    # force noise (2.0); re-tune once the smoke prints the realized force-vs-torque magnitudes.
    torque_obs_noise: float = 2.0


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
    image_channels: int = 4  # per-FRAME channels: RGB (3) + depth (1)
    # Temporal frame-stack: stack the last N wrist frames along the channel axis (env-side ring buffer)
    # so the CNN sees intra-observation MOTION (approach speed, contact-onset dynamics) the single frame
    # lacks -- complements the LSTM's cross-step memory. 1 = OFF (current single-frame behaviour); N>1
    # feeds the CNN 4*N channels (its first conv auto-adapts from the widened image obs space, no net
    # change). The buffer is cleared per-episode at reset so no frames bleed across episodes. Toggle so
    # it screens on/off; N=2-3 is the intended range. Note: N*image memory -- may need fewer envs @224px.
    frame_stack: int = 1
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
    depth_dropout_prob: float = 0.02  # fraction of depth pixels zeroed (LEGACY uniform mode; sensor holes)

    # (6) REALISTIC (D405-like) STRUCTURED DEPTH CORRUPTION. The real D405 depth measured ~45% INVALID,
    # structured on fin-slots/edges + a missing socket interior -- NOT the 2% uniform dropout above; that
    # gap was the confirmed killer of the deployed vision behaviour. Setting depth_corruption_mode
    # "structured" swaps the uniform dropout for clustered blobs + edge/gradient holes + dark-slot dropout
    # (+ optional per-episode modality dropout + hole-fill). Default "uniform" keeps every existing task
    # unchanged; the sim2real E2E-vision cfg turns it on. Tune the knobs (via env.<field>=...) to hit the
    # ~0.35-0.50 invalid target. See InsertionEnv._corrupt_depth_structured.
    depth_corruption_mode: str = "uniform"   # "uniform" (legacy per-pixel) | "structured" (D405-like)
    depth_invalid_target_frac: float = 0.40  # documentation target for total invalid fraction (0.35-0.50)
    depth_blob_frac: float = 0.20            # coverage of clustered dropout blobs (low-freq-noise threshold)
    depth_blob_scale_px: int = 20            # characteristic blob size in px (coarse-noise cell size ~5-30)
    depth_edge_drop_prob: float = 0.5        # drop prob along strong depth/luma gradients (edges)
    depth_edge_grad_thresh: float = 0.015    # normalized depth+luma gradient magnitude that counts as an edge
    depth_dark_drop_prob: float = 0.6        # drop prob where RGB is dark (the deep fin slots)
    depth_dark_value_thresh: float = 0.30    # RGB luma below this = "dark" (fin-slot) region
    depth_heavy_strength: float = 1.6        # corruption multiplier for the "heavy" modality episodes
    depth_fill_prob: float = 0.0             # per-episode prob the synthetic holes are interpolation-filled
    depth_fill_iters: int = 4                # 3x3 propagation passes used by the hole-fill variant
    # MODALITY DROPOUT (per-EPISODE): p(RGB-only / heavily-corrupted depth / normal-corrupted depth). Stops
    # the policy + estimator over-trusting the (now noisy) depth channel -> forces reading the hole from RGB
    # when depth is bad. Off by default (all "normal"); the sim2real cfg turns it on with (0.15,0.35,0.50).
    depth_modality_dropout: bool = False
    depth_modality_probs: tuple = (0.15, 0.35, 0.50)  # (rgb_only, heavy, normal); need not sum to 1
    # DEPTH-REALISM CURRICULUM: ramp the STRUCTURAL corruption (blob/edge/dark drop fractions) AND the
    # modality-dropout prevalence from ~0 -> full over this many control steps (common_step_counter ~=
    # iters*horizon_length; 160000 @ horizon 128 ~= 1250 it). 0 => full corruption from step 0 (no ramp).
    # Lets the estimator first learn hole-prediction on good depth (the w2 regime), THEN adapt to the real
    # ~45%-invalid + 15%-RGB-only sensor -- the cold-start de-risk for the hardest new gap. See
    # InsertionEnv._depth_curric_frac / _corrupt_depth_structured / randomize_initial_state.
    depth_corruption_curriculum_steps: int = 0

    # WRIST-YAW CURRICULUM: ramp the reset yaw noise from +-hand_init_yaw_start_deg -> the full
    # hand_init_orn_noise[2] over this many control steps (0 => full range from step 0). Yaw is irrelevant
    # to seating a round peg but rotates the wrist IMAGE, so a full-range cold start makes the CNN/estimator
    # job harder; this eases them in. See InsertionEnv.randomize_initial_state.
    hand_init_yaw_curriculum_steps: int = 0
    hand_init_yaw_start_deg: float = 45.0

    # (7) FULL-HUE MATERIAL DR: the real base is vivid BLUE, which the narrow RGB-jitter-about-grey band
    # (material_base_color/material_color_jitter) does NOT span. Enabling this draws a full-range HUE per
    # env (HSV), so blue -- and every colour -- appears, with configurable saturation/value; the screw
    # stays hue-correlated to the base (no screw-vs-base colour cue). Off by default (legacy RGB jitter).
    material_hue_randomize: bool = False
    material_saturation_range: tuple = (0.4, 1.0)
    material_value_range: tuple = (0.3, 0.9)
    material_hue_part_jitter: float = 0.02   # screw hue +/- this about the base hue (small => same filament)
    # (8) LIGHT COLOUR TEMPERATURE (K): when set, tints the dome + key lights warm(3000K)->cool(8000K)
    # instead of the grey +/- light_color_jitter tint, spanning the white balances the D405 sees. Empty/None
    # => legacy grey jitter. See InsertionEnv._color_temp_to_rgb / _randomize_scene_light.
    light_color_temp_range_k: tuple | None = None

    # (5) TRAINING IMAGE GALLERY (observability, not DR): every cam_log_interval camera reads, dump a
    # tiled montage (all envs) of exactly what the policy SEES to cam_log_dir -> watch framing / DR
    # aggressiveness / occlusion DURING an 8h run without a second GPU job. 0 disables. renders/ is
    # gitignored. (For rollout BEHAVIOUR video, use train.py --video; see auto_resume_train.sh VIDEO=1.)
    cam_log_interval: int = 2000    # camera reads between montage dumps (~15 it at 64 envs); 0 = off
    cam_log_dir: str = "renders/train_cam"
    # When dumping the gallery, stamp the TRUE socket opening on the policy-seen wrist RGB+depth (projected
    # via the wrist camera) so you can see if the hole is in-frame and whether depth is void/corrupted
    # there. Off by default. (The estimator's PREDICTED hole vs GT is overlaid offline by
    # scripts/render_estimator_overlay.py, which has the trained aux head.)
    cam_log_overlay: bool = False

    def __post_init__(self):
        # Keep Fabric ENABLED (GPU PhysX is stable on Fabric; disabling it caused CUDA-700 crashes /
        # hangs in vision runs). This build's usdrt lacks the `hierarchy` submodule that Isaac Lab's
        # Fabric world-pose path needs, but we route ONLY the camera's xform view to the USD pose path
        # via a targeted monkeypatch in insertion_env.py (physics keeps Fabric). See that patch.
        super().__post_init__() if hasattr(super(), "__post_init__") else None
        # Render once per env step (decimation=8) instead of once per physics substep — 8x less render
        # work; all the policy needs.
        self.sim.render_interval = self.decimation
