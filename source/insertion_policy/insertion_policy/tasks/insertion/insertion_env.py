"""InsertionEnv: socket-aware residual insertion env, subclassing Forge.

Two jobs on top of Forge:

1. Off-center sockets. Factory/Forge assume the socket sits at the fixed-asset XY origin, but the
   cooling_base sockets are at x=+/-30mm. We redefine ``self.fixed_pos`` as the ACTIVE socket frame
   (base origin (+) per-env socket offset) inside ``_compute_intermediate_values``. Because every
   downstream computation (tip frame, hand placement, observations, success/reward target) is built
   from ``self.fixed_pos``, they all become socket-centric automatically. Multi-socket: each env
   samples one of ``cfg_task.socket_offsets_local`` at reset.

2. Pure-residual reward (per supervisor). Reward = multi-keypoint squashing kernel (Factory/
   IndustReal: dense, bounded, smooth near the goal) on the distance from the screw to the target
   seated pose, replacing Factory's centered-socket keypoint geometry. The first keypoint is the
   shaft tip and the rest run up the screw axis, so tilt is penalized even when the tip is close to
   the socket bottom. A binary seat bonus is added on top.

The fixed_pos socket frame is the socket BOTTOM; with CoolingBase.height = socket depth and
base_height = 0, Factory's "tip" lands at the socket opening (entry) and the target stays at the
bottom — matching peg-insert semantics with the screw's shaft tip as the held reference.
"""

import math

import carb
import gymnasium as gym
import torch

import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import TiledCamera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.direct.factory import factory_utils
from isaaclab_tasks.direct.forge.forge_env import ForgeEnv

from .cooling_tasks_cfg import ForgeTaskCoolingInsertCfg

# --- Fabric/usdrt workaround (keep Fabric ON for physics; route only the camera to USD poses) -----
# This Isaac Sim build's usdrt lacks the `hierarchy` submodule, so Isaac Lab's Fabric world-pose path
# (XformPrimView._get_world_poses_fabric -> _initialize_fabric -> usdrt.hierarchy) crashes whenever a
# camera reads/writes its pose. Disabling Fabric globally "fixes" that but makes GPU PhysX unstable
# (CUDA-700 crashes / hangs in vision runs). Instead we keep Fabric ENABLED (stable physics) and force
# ONLY XformPrimView (used by cameras / xform-sensors, NOT the physics articulation views) onto the
# USD XformCache pose path, which needs no usdrt.hierarchy. Physics is untouched.
from isaaclab.sim.views import XformPrimView as _XformPrimView  # noqa: E402

if not getattr(_XformPrimView, "_insertion_usd_pose_patch", False):
    _orig_xpv_init = _XformPrimView.__init__

    def _xpv_init_usd_poses(self, *args, **kwargs):
        _orig_xpv_init(self, *args, **kwargs)
        self._use_fabric = False  # USD pose path (no usdrt.hierarchy); physics keeps Fabric

    _XformPrimView.__init__ = _xpv_init_usd_poses
    _XformPrimView._insertion_usd_pose_patch = True

# --- rl_games dict-obs input-normalization fix --------------------------------------------------
# rl_games normalizes dict observations with torch.jit.script(RunningMeanStdObs(obs_shape)), but this
# rl_games/torch version can't infer the Dict[str,Tensor] input type when scripting -> compile error
# ("'Tensor' object has no attribute 'items'"), so normalize_input=True crashes our hybrid
# (proprio+image) dict obs at model build. The EAGER module works fine; only the JIT compile fails.
# We intercept torch.jit.script for that one class and return it un-scripted, so we can normalize the
# proprio + privileged critic state (which Forge needs; force obs are large) while the image stays
# pre-normalized in-env. Everything else still scripts normally.
import torch as _torch  # noqa: E402
from rl_games.algos_torch.running_mean_std import RunningMeanStdObs as _RMSObs  # noqa: E402

if not getattr(_torch.jit, "_insertion_rmsobs_patch", False):
    _orig_jit_script = _torch.jit.script

    def _jit_script_skip_rmsobs(obj, *args, **kwargs):
        if isinstance(obj, _RMSObs):
            return obj  # eager; TorchScript can't infer the Dict input type in this rl_games version
        return _orig_jit_script(obj, *args, **kwargs)

    _torch.jit.script = _jit_script_skip_rmsobs
    _torch.jit._insertion_rmsobs_patch = True


class InsertionEnv(ForgeEnv):
    cfg: ForgeTaskCoolingInsertCfg

    def __init__(self, cfg: ForgeTaskCoolingInsertCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Vision variant: a wrist RGB-D camera adds an "image" observation group alongside Forge's
        # "policy" (proprio) and "critic" (state) groups. The rl_games wrapper consumes it via
        # obs_groups={"obs": ["policy", "image"], "states": ["critic"]} + concate_obs_groups=False.
        self._has_camera = getattr(self.cfg, "tiled_camera", None) is not None
        if self._has_camera:
            h, w, c = self.cfg.image_height, self.cfg.image_width, self.cfg.image_channels
            self.single_observation_space["image"] = gym.spaces.Box(low=0.0, high=1.0, shape=(h, w, c))
            # Camera eye offset in the fingertip frame; the world pose (look-at) is recomputed each step.
            self._cam_offset_pos = torch.tensor(
                self.cfg.wrist_cam_offset_pos, device=self.device, dtype=torch.float32
            ).repeat(self.num_envs, 1)
            # Per-episode camera extrinsics jitter (DR; resampled at reset, zero when cam_pos_jitter=0).
            self._cam_jitter = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)

        # Socket offsets (socket BOTTOM, base-local frame); each env holds the active one.
        self._socket_table = torch.tensor(
            self.cfg_task.socket_offsets_local, device=self.device, dtype=torch.float32
        )  # (n_sockets, 3)
        self.socket_offset_local = self._socket_table[0].repeat(self.num_envs, 1).clone()

        # Held reference = screw shaft tip (shaft_length below the part origin).
        self.held_base_offset = torch.tensor(
            [0.0, 0.0, -self.cfg_task.screw_shaft_length], device=self.device
        ).repeat(self.num_envs, 1)

        self._identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self._local_z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)

        self._reward_keypoint_offsets = torch.zeros(
            (self.cfg_task.num_keypoints, 3), device=self.device, dtype=torch.float32
        )
        self._reward_keypoint_offsets[:, 2] = torch.linspace(
            0.0, self.cfg_task.held_asset_cfg.height, self.cfg_task.num_keypoints, device=self.device
        )

    # --- scene (adds the wrist camera for the vision variant) -------------------------------
    def _setup_scene(self):
        """Factory's scene plus a wrist-mounted TiledCamera (vision variant only).

        Mirrors ``FactoryEnv._setup_scene`` but instantiates the camera BEFORE
        ``clone_environments`` so the per-env camera prims are created under each cloned robot.
        When the cfg has no ``tiled_camera`` (the state-obs task), this is just Factory's setup.
        (We override rather than call super because the camera must exist before the clone, which
        the parent performs internally; the cooling task never uses the gear-mesh assets.)
        """
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))
        table_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        )
        table_cfg.func(
            "/World/envs/env_.*/Table", table_cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        self._robot = Articulation(self.cfg.robot)
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
        self._held_asset = Articulation(self.cfg_task.held_asset)

        self._tiled_camera = None
        if getattr(self.cfg, "tiled_camera", None) is not None:
            # NOTE: this Isaac Sim build's usdrt lacks the ``hierarchy`` submodule Isaac Lab's Fabric
            # world-pose path needs. We keep Fabric ON (GPU-stable physics) and monkeypatch only the
            # camera's XformPrimView to the USD pose path (see top of file), so the camera works.
            self._tiled_camera = TiledCamera(self.cfg.tiled_camera)

        # Optional third-person debug camera (only set by scripts/viz_camera.py; None in training).
        self._scene_camera = None
        if getattr(self.cfg, "scene_camera", None) is not None:
            self._scene_camera = TiledCamera(self.cfg.scene_camera)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset
        if self._tiled_camera is not None:
            self.scene.sensors["tiled_camera"] = self._tiled_camera
        if self._scene_camera is not None:
            self.scene.sensors["scene_camera"] = self._scene_camera

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # --- socket selection -------------------------------------------------------------------
    def _sample_sockets(self, env_ids):
        n = self._socket_table.shape[0]
        idx = torch.randint(0, n, (len(env_ids),), device=self.device)
        self.socket_offset_local[env_ids] = self._socket_table[idx]

    def randomize_initial_state(self, env_ids):
        # Pick which socket each resetting env targets, then let Forge place everything relative to
        # the (now socket-centric) fixed frame. (Guard: socket tensors are allocated after the
        # parent __init__, which may itself trigger a reset.)
        if hasattr(self, "_socket_table"):
            self._sample_sockets(env_ids)
        super().randomize_initial_state(env_ids)
        # Forge's reset leaves the screw vertical (yaw noise only), so the upstream angular error is
        # absent. Add it now by tilting the grasped screw about its shaft tip.
        if hasattr(self, "_socket_table"):
            self._apply_pre_insert_tilt(env_ids)
        # Resample per-episode camera extrinsics jitter (DR; no-op when cam_pos_jitter == 0).
        if getattr(self, "_has_camera", False) and self.cfg.cam_pos_jitter > 0.0:
            self._cam_jitter[env_ids] = self.cfg.cam_pos_jitter * torch.randn(
                (len(env_ids), 3), device=self.device
            )

    def get_handheld_asset_relative_pose(self):
        """Grip the screw HEAD between the fingerpads, then add the rotational grasp misalignment.

        Position fix. Factory's peg formula sets ``relative_pos.z = held_height - fingerpad_length``,
        which assumes the asset ORIGIN is at its BASE. Our cooling_screw origin is at the head/shaft
        SHOULDER, with the shaft (``screw_shaft_length``) hanging below it -- so the formula places
        the screw ~one shaft-length too low and the gripper closes on empty air below the head (the
        screw then only dangles, wobbles, and can slip out). We instead grip the HEAD treated as the
        graspable cylinder (its base = the shoulder = our origin), so ``relative_pos.z =
        head_height - fingerpad_length`` with ``head_height = full_height - screw_shaft_length``.

        Misalignment. Factory grasps the peg perfectly aligned. Real grasps are not: the screw sits
        slightly tilted in the jaws. We tilt the shaft OFF the finger-pointing direction by a uniform
        angle in [0, grasp_misalign_max_deg] about a random horizontal axis (the shaft axis no longer
        points straight along the gripper), so the TRUE screw pose deviates from the fingertip-based
        estimate by an amount the actor cannot observe (held pose is privileged/critic-only) -- it
        must be recovered from vision. Resampled every reset. Independent of the pre-insert tilt.
        """
        held_asset_relative_pos, held_asset_relative_quat = super().get_handheld_asset_relative_pose()
        # Grip the head, not below it (correct for the shoulder-origin, two-diameter screw).
        head_height = self.cfg_task.held_asset_cfg.height - self.cfg_task.screw_shaft_length
        held_asset_relative_pos[:, 2] = head_height - self.cfg_task.robot_cfg.franka_fingerpad_length
        max_deg = getattr(self.cfg_task, "grasp_misalign_max_deg", 0.0)
        if max_deg > 0.0:
            n = self.num_envs
            mag = math.radians(max_deg) * torch.rand(n, device=self.device)
            az = 2.0 * math.pi * torch.rand(n, device=self.device)
            axis = torch.stack([torch.cos(az), torch.sin(az), torch.zeros_like(az)], dim=1)
            misalign_quat = torch_utils.quat_from_angle_axis(mag, axis)
            held_asset_relative_quat = torch_utils.quat_mul(misalign_quat, held_asset_relative_quat)
        return held_asset_relative_pos, held_asset_relative_quat

    def _apply_pre_insert_tilt(self, env_ids):
        """Inject the angular pre-insert error (the upstream orientation uncertainty).

        Factory's reset leaves the screw vertical (only yaw noise), so the ~20-25 deg angular error
        never appears and the orientation-aware reward/success are never exercised. Here we rigidly
        rotate the already-grasped screw AND the wrist together about the screw's shaft tip, by a
        uniform angle in [0, pre_insert_tilt_max_deg] about a random horizontal axis. Pivoting about
        the tip keeps it over the socket mouth (preserving the lateral pre-insert offset) while the
        body angles -- a realistic angled pre-insert, not a tip swung out of the socket. Uses the
        live grasp poses, so it stays correct after the iiwa/gripper swap (no hard-coded geometry).
        """
        if self.cfg_task.pre_insert_tilt_max_deg <= 0.0:
            return

        # No gravity while we re-pose (mirror Factory's reset; the screw also has gravity disabled).
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        # Sample an isotropic tilt of the vertical screw axis: magnitude in [0, max], random azimuth.
        n = self.num_envs
        max_rad = math.radians(self.cfg_task.pre_insert_tilt_max_deg)
        mag = max_rad * torch.rand(n, device=self.device)
        az = 2.0 * math.pi * torch.rand(n, device=self.device)
        axis = torch.stack([torch.cos(az), torch.sin(az), torch.zeros_like(az)], dim=1)
        tilt_quat = torch_utils.quat_from_angle_axis(mag, axis)

        # Capture the vertical grasp BEFORE moving (resets are all-env at once): pivot (shaft tip),
        # the screw pose, and the arm joints, so envs whose IK can't reach the tilted pose fall back
        # to this known-good vertical grasp instead of an invalid one.
        tip, _ = self._held_base_pose()
        ft_pos, ft_quat = self.fingertip_midpoint_pos.clone(), self.fingertip_midpoint_quat.clone()
        held_pos0, held_quat0 = self.held_pos.clone(), self.held_quat.clone()
        saved_joint_pos = self.joint_pos.clone()

        # Tilt the wrist target about the tip and IK the arm there (teleports joints; screw floats).
        new_ft_pos = tip + torch_utils.quat_apply(tilt_quat, ft_pos - tip)
        new_ft_quat = torch_utils.quat_mul(tilt_quat, ft_quat)
        pos_err, aa_err = self.set_pos_inverse_kinematics(new_ft_pos, new_ft_quat, env_ids)
        ik_failed = (torch.linalg.norm(pos_err, dim=1) > 1e-3) | (torch.linalg.norm(aa_err, dim=1) > 1e-3)

        # Carry the screw rigidly with the same tilt about the tip (tip stays put, body angles by the
        # sampled angle). Envs where IK failed keep the original vertical grasp -> arm + screw stay
        # consistent (no contact shove, no out-of-range tilt).
        new_held_pos = tip + torch_utils.quat_apply(tilt_quat, held_pos0 - tip)
        new_held_quat = torch_utils.quat_mul(tilt_quat, held_quat0)
        new_held_pos[ik_failed] = held_pos0[ik_failed]
        new_held_quat[ik_failed] = held_quat0[ik_failed]
        if ik_failed.any():
            self.joint_pos[ik_failed] = saved_joint_pos[ik_failed]
            self.joint_vel[:] = 0.0
            self._robot.write_joint_state_to_sim(self.joint_pos, self.joint_vel)
            self.ctrl_target_joint_pos[:, 0:7] = self.joint_pos[:, 0:7]
            self._robot.set_joint_position_target(self.ctrl_target_joint_pos)

        held_state = self._held_asset.data.default_root_state.clone()
        held_state[:, 0:3] = new_held_pos + self.scene.env_origins
        held_state[:, 3:7] = new_held_quat
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7])
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:])
        self._held_asset.reset()
        self.step_sim_no_action()

        # Restore gravity; clear the reset-induced finite-difference velocities so step 0 is clean.
        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()
        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()
        self.ee_linvel_fd[:, :] = 0.0
        self.ee_angvel_fd[:, :] = 0.0

    # --- socket-centric frame ----------------------------------------------------------------
    def _compute_intermediate_values(self, dt):
        super()._compute_intermediate_values(dt)
        if not hasattr(self, "socket_offset_local"):
            return  # parent __init__ may call this before our tensors exist
        # Shift the fixed frame from base origin to the active socket bottom (same orientation).
        _, socket_pos = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self._identity_quat, self.socket_offset_local
        )
        self.fixed_pos = socket_pos
        self.fixed_pos_obs_frame[:] = self._socket_opening_pos()
        self._update_camera_pose()

    def _update_camera_pose(self):
        """Drive the wrist camera each step (it is not parented to the robot).

        Eye = fingertip position + a rigid offset expressed in the fingertip frame (so the camera
        rides the gripper at an oblique side mount); it looks AT the active socket opening. Looking
        at the socket gives a stable, occlusion-free oblique view of the shaft entering the hole --
        the pre-insert lateral/angular error stays visible as the shaft's offset/tilt in the image.
        Guarded so the state-obs task and the parent __init__'s early calls are unaffected.
        """
        if getattr(self, "_tiled_camera", None) is None:
            return
        # Eye: rigid position offset on the wrist (+ per-episode DR jitter), look-at the socket.
        eye = self.fingertip_midpoint_pos + torch_utils.quat_apply(
            self.fingertip_midpoint_quat, self._cam_offset_pos + self._cam_jitter
        )
        target = self.fixed_pos_obs_frame  # active socket opening (entry)
        self._tiled_camera.set_world_poses_from_view(
            eye + self.scene.env_origins, target + self.scene.env_origins
        )

    def _socket_opening_pos(self):
        """World position of the active socket opening/entry frame."""
        fixed_tip_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
        fixed_tip_pos_local[:, 2] = self.cfg_task.fixed_asset_cfg.height + self.cfg_task.fixed_asset_cfg.base_height
        _, socket_opening_pos = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self._identity_quat, fixed_tip_pos_local
        )
        return socket_opening_pos

    def _held_base_pose(self):
        """World pose of the screw shaft tip (the point that seats at the socket bottom)."""
        held_base_quat, held_base_pos = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self._identity_quat, self.held_base_offset
        )
        return held_base_pos, held_base_quat

    def _keypoint_distances(self):
        """Mean keypoint distance between current screw pose and target seated pose."""
        held_base_pos, held_base_quat = self._held_base_pose()
        target_base_pos = self.fixed_pos
        target_base_quat = self.fixed_quat

        keypoints_held = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
        keypoints_target = torch.zeros_like(keypoints_held)
        for idx, keypoint_offset in enumerate(self._reward_keypoint_offsets):
            offset = keypoint_offset.repeat(self.num_envs, 1)
            keypoints_held[:, idx] = torch_utils.tf_combine(
                held_base_quat, held_base_pos, self._identity_quat, offset
            )[1]
            keypoints_target[:, idx] = torch_utils.tf_combine(
                target_base_quat, target_base_pos, self._identity_quat, offset
            )[1]

        keypoint_dist = torch.linalg.vector_norm(keypoints_held - keypoints_target, dim=-1).mean(dim=-1)
        tip_dist = torch.linalg.vector_norm(held_base_pos - target_base_pos, dim=1)
        return keypoint_dist, tip_dist

    def _shaft_axis_error(self):
        """Angle between the screw shaft axis and the active socket axis."""
        _, held_base_quat = self._held_base_pose()
        held_axis = torch_utils.quat_apply(held_base_quat, self._local_z_axis)
        socket_axis = torch_utils.quat_apply(self.fixed_quat, self._local_z_axis)
        axis_cos = torch.clamp(torch.sum(held_axis * socket_axis, dim=1), -1.0, 1.0)
        return torch.acos(axis_cos)

    # --- success + reward (pure residual, squashing kernel) ---------------------------------
    def _get_curr_successes(self, success_threshold, check_rot=False):
        held_base_pos, _ = self._held_base_pose()
        target = self.fixed_pos  # socket bottom
        xy_dist = torch.linalg.vector_norm(target[:, 0:2] - held_base_pos[:, 0:2], dim=1)
        z_disp = held_base_pos[:, 2] - target[:, 2]
        is_centered = xy_dist < 0.0025
        height_threshold = self.cfg_task.fixed_asset_cfg.height * success_threshold
        is_seated = z_disp < height_threshold
        is_aligned = self._shaft_axis_error() < self.cfg_task.success_orientation_threshold
        return torch.logical_and(torch.logical_and(is_centered, is_seated), is_aligned)

    def _get_rewards(self):
        keypoint_dist, tip_dist = self._keypoint_distances()
        shaft_axis_error = self._shaft_axis_error()
        curr_successes = self._get_curr_successes(self.cfg_task.success_threshold)

        # Multi-scale squashing kernel 1/(exp(a*d)+b+exp(-a*d)) on the mean keypoint distance:
        # each scale peaks (bounded) at d=0 and decays smoothly. baseline (small a) shapes the whole
        # approach; coarse/fine (large a) sharpen the gradient near the seated pose. This replaces the
        # unbounded neg-L2 + binary-dominated signal with the dense bounded reward Factory/IndustReal
        # use, while keeping our orientation-aware shaft-tip->top keypoints (so tilt still costs).
        a0, b0 = self.cfg_task.keypoint_coef_baseline
        a1, b1 = self.cfg_task.keypoint_coef_coarse
        a2, b2 = self.cfg_task.keypoint_coef_fine
        kp_baseline = factory_utils.squashing_fn(keypoint_dist, a0, b0)
        kp_coarse = factory_utils.squashing_fn(keypoint_dist, a1, b1)
        kp_fine = factory_utils.squashing_fn(keypoint_dist, a2, b2)

        rew_buf = kp_baseline + kp_coarse + kp_fine + self.cfg_task.success_bonus * curr_successes.float()

        # Restore Forge's success-prediction term that our override otherwise drops: train action[:,6]
        # to predict whether the insertion is currently successful. The penalty scale ramps from 0 to
        # 1 only once a delay_until_ratio fraction of envs have succeeded (so it doesn't fight early
        # learning), matching ForgeEnv._get_rewards. Needed before sim-to-real (the policy's done-signal).
        policy_success_pred = (self.actions[:, 6] + 1) / 2  # [-1,1] -> [0,1]
        success_pred_error = (curr_successes.float() - policy_success_pred).abs()
        if curr_successes.float().mean() >= self.cfg_task.delay_until_ratio:
            self.success_pred_scale = 1.0
        rew_buf = rew_buf - self.success_pred_scale * success_pred_error

        log_dict = {
            "kp_baseline": kp_baseline,
            "kp_coarse": kp_coarse,
            "kp_fine": kp_fine,
            "neg_keypoint_l2": -keypoint_dist,
            "neg_tip_l2": -tip_dist,
            "shaft_axis_error": shaft_axis_error,
            "success_pred_error": success_pred_error,
            "success": curr_successes.float(),
        }

        # Optional Forge contact-force penalty (off by default). Penalizes estimated EE contact
        # force above the per-env threshold -> discourages jamming/over-force, important for
        # sim-to-real and not damaging the part. Enable with cfg_task.contact_penalty_scale > 0;
        # the scale likely needs tuning against the neg-L2 reward magnitude (~0.05-0.3).
        if self.cfg_task.contact_penalty_scale > 0.0:
            contact_force = torch.linalg.vector_norm(self.force_sensor_smooth[:, 0:3], dim=-1)
            contact_penalty = torch.nn.functional.relu(contact_force - self.contact_penalty_thresholds)
            rew_buf = rew_buf - self.cfg_task.contact_penalty_scale * contact_penalty
            log_dict["contact_force"] = contact_force
            log_dict["contact_penalty"] = contact_penalty

        self.prev_actions = self.actions.clone()
        self._log_factory_metrics(log_dict, curr_successes)
        return rew_buf

    # --- observations (adds the RGB-D image group for the vision variant) -------------------
    def _get_observations(self):
        obs = super()._get_observations()  # {"policy": proprio, "critic": state}
        if getattr(self, "_has_camera", False):
            obs["image"] = self._get_camera_image()
        return obs

    def _get_camera_image(self):
        """Wrist RGB-D as a (num_envs, H, W, 4) tensor: normalized RGB (3) + normalized depth (1).

        RGB is scaled to [0,1] and mean-subtracted per image (as in Isaac Lab's camera examples);
        depth has inf (no-hit) zeroed and is scaled by the camera far-clip into ~[0,1]. The CNN in
        the hybrid network consumes this directly (it permutes HWC->CHW).
        """
        if getattr(self.cfg, "blank_image", False):
            # Ablation: zeroed image (camera still renders; the policy just gets no visual signal).
            h, w, c = self.cfg.image_height, self.cfg.image_width, self.cfg.image_channels
            return torch.zeros((self.num_envs, h, w, c), device=self.device)

        out = self._tiled_camera.data.output
        rgb_raw = out["rgb"][..., :3].float() / 255.0
        rgb = rgb_raw - torch.mean(rgb_raw, dim=(1, 2), keepdim=True)

        depth = out["depth"].clone()
        depth[~torch.isfinite(depth)] = 0.0
        far = float(self.cfg.tiled_camera.spawn.clipping_range[1])
        depth = torch.clamp(depth, 0.0, far) / far
        if depth.dim() == 3:
            depth = depth.unsqueeze(-1)

        if getattr(self.cfg, "write_image_to_file", False):
            from isaaclab.sensors import save_images_to_file

            save_images_to_file(rgb_raw, "/tmp/wrist_rgb.png")
            save_images_to_file(depth, "/tmp/wrist_depth.png")

        return torch.cat([rgb, depth], dim=-1)
