"""InsertionEnv: socket-aware residual insertion env, subclassing Forge.

Two jobs on top of Forge:

1. Off-center sockets. Factory/Forge assume the socket sits at the fixed-asset XY origin, but the
   cooling_base sockets are at x=+/-30mm. We redefine ``self.fixed_pos`` as the ACTIVE socket frame
   (base origin (+) per-env socket offset) inside ``_compute_intermediate_values``. Because every
   downstream computation (tip frame, hand placement, observations, success/reward target) is built
   from ``self.fixed_pos``, they all become socket-centric automatically. Multi-socket: each env
   samples one of ``cfg_task.socket_offsets_local`` at reset.

2. Pure-residual reward (per supervisor). Reward = multi-keypoint negative L2 from the screw to
   the target seated pose, replacing Factory's centered-socket keypoint reward. The first keypoint
   is the shaft tip and the rest run up the screw axis, so tilt is penalized even when the tip is
   close to the socket bottom.

The fixed_pos socket frame is the socket BOTTOM; with CoolingBase.height = socket depth and
base_height = 0, Factory's "tip" lands at the socket opening (entry) and the target stays at the
bottom — matching peg-insert semantics with the screw's shaft tip as the held reference.
"""

import math

import carb
import torch

import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils

from isaaclab_tasks.direct.forge.forge_env import ForgeEnv

from .cooling_tasks_cfg import ForgeTaskCoolingInsertCfg


class InsertionEnv(ForgeEnv):
    cfg: ForgeTaskCoolingInsertCfg

    def __init__(self, cfg: ForgeTaskCoolingInsertCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

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

    # --- success + reward (pure residual, neg-L2) -------------------------------------------
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

        rew_buf = -keypoint_dist + self.cfg_task.success_bonus * curr_successes.float()

        log_dict = {
            "neg_keypoint_l2": -keypoint_dist,
            "neg_tip_l2": -tip_dist,
            "shaft_axis_error": shaft_axis_error,
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
