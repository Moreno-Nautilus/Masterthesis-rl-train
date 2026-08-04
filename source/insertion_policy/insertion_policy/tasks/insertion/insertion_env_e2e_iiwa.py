"""InsertionEnvE2EIiwa: the PURE END-TO-END visuomotor variant of the iiwa cooling-insertion env.

Supervisor-locked design (2026-07-30, see PLANNING.md "END-TO-END VISUOMOTOR INSERTION" and the
memory ``fabrica-graft-plan``). We learn the FULL pre-insert -> seated motion from scratch as an
end-to-end visuomotor policy: NO residual, NO nominal plan, NO path-centric frame, NO explicit delta
in the POLICY obs. This subclasses ``InsertionEnvIiwa`` (all task geometry -- sockets, pre-insert
tilt, grasp misalignment, wrist RGB-D + appearance DR, iiwa body-name / jacobian fixes -- is inherited
unchanged) and overrides ONLY the four things the end-to-end design changes:

  1. ACTION (``_pre_physics_step`` + ``_apply_action``): a 5-DoF delta relative to the CURRENT
     fingertip = [dx, dy, dz, rot_a, rot_b] (3 translation + 2 tilt axes, NO yaw -- the round peg's
     yaw is irrelevant). ``ctrl_target_pos = fingertip + pos_scale * a[0:3]``;
     ``ctrl_target_quat = fingertip_quat (x) dRot(a[3:5] * rot_scale)`` (body-frame tilt). The target
     is computed ONCE per env step (decision time) and held across the decimation substeps, so the
     IK+PD servos toward a fixed setpoint instead of re-targeting (which with stiff PD would run away).

  2. CONTROLLER (``generate_ctrl_signals``): Forge's task-space impedance (OSC torque) is replaced by
     IK + joint PD, HIGH gains (supervisor). Each substep we take one damped-least-squares differential
     IK step toward the target fingertip pose (reusing ``factory_control.get_delta_dof_pos`` -- the same
     DLS the reset servo uses) and command the resulting joint positions to the implicit-actuator PD
     (see the high-gain arm actuators in ``IIWA_GRIPPER_E2E_CFG``). The arm's feedforward torque is
     zeroed so control is pure joint PD. Contact-force risk on the brittle parts is accepted for now.

  3. OBSERVATIONS (``_get_observations``): the policy is END-TO-END with NO proprio --
       * vision run:  image (wrist RGB-D) + wrist F/T force (3) + prev_action (5).
       * state-first debug run (``debug_true_delta_obs``): the TRUE (error-free) delta (6) + force (3)
         + prev_action (5), NO image -- validates the RL + controller + reward + action loop before
         the vision encoder is switched in. Set the flag off (and add the camera) for the real runs.
     The critic is PRIVILEGED (asymmetric): true delta + true socket + fingertip pose + velocities +
     contact wrench + prev_action.

  4. REWARD (``_get_rewards``): ``-(w_pos * ||tip->socket_bottom|| + w_rot * shaft_axis_error)`` to the
     seated pose (near-seated stop OK for now; no seat bonus yet). The orientation term is the
     SHAFT-AXIS error, NOT a full quaternion geodesic: the action has no yaw DOF and success is
     yaw-invariant (round peg + random socket yaw), so penalizing yaw would handicap a cue the policy
     cannot fix. The Forge/Factory squashing-kernel keypoint reward is kept behind ``e2e_reward_mode``.

The residual ``InsertionEnvIiwa`` and its registered tasks are left completely untouched (validated
fallback). Everything here keys off the E2E cfg flags, so nothing leaks back into the residual path.
"""

import gymnasium as gym
import torch

import isaacsim.core.utils.torch as torch_utils

from isaaclab_tasks.direct.factory import factory_control, factory_utils
from isaaclab_tasks.direct.factory.factory_env import FactoryEnv

from .insertion_env_iiwa import InsertionEnvIiwa


class InsertionEnvE2EIiwa(InsertionEnvIiwa):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Per-env-step control setpoint (fixed across the decimation substeps); filled in
        # _pre_physics_step. Guarded in _apply_action so an out-of-order call just holds position.
        self._ctrl_target_pos = None
        self._ctrl_target_quat = None
        self._zero_effort = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        # Rebuild the observation/critic spaces for the pure end-to-end policy (the base built Forge's
        # proprio + a privileged Forge state + the aux-label group; none of those apply here). Mirrors
        # the space-rebuild pattern InsertionEnv uses for the torque/image groups.
        self._debug_true_delta = bool(getattr(self.cfg, "debug_true_delta_obs", False))
        # Restore the res224-style state block into the POLICY obs. The pure-E2E graft dropped it (force+
        # image only) -> 2% vs res224's 65%; this puts back the coarse-localisation signal so the wrist cam
        # only has to REFINE it, not find the socket from scratch. Block = NOISY socket-relative fingertip
        # pos (+-8mm, the factory_env recipe -> an imperfect upstream estimate) + proprio pose/vel. The
        # socket still varies (+-5cm + 360deg yaw), so this rel-pose is not a memorisable constant.
        self._use_proprio = bool(getattr(self.cfg, "e2e_use_proprio_obs", False))
        proprio_dim = 13 if self._use_proprio else 0  # rel_fixed(3) + quat(4) + linvel(3) + angvel(3)

        # Optional Factory-style socket-anchored action: confine the IK target to a +-bound box around the
        # (noisy) socket so the peg can't wander/hover far from the hole -> shrinks exploration to the goal.
        self._socket_anchored = bool(getattr(self.cfg, "e2e_socket_anchored_action", False))
        self._socket_action_bound = float(getattr(self.cfg, "e2e_socket_action_bound", 0.08))

        force_dim, act_dim, delta_dim = 3, self.cfg.action_space, 6
        policy_dim = (delta_dim if self._debug_true_delta else 0) + proprio_dim + force_dim + act_dim
        self.single_observation_space["policy"] = gym.spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(policy_dim,)
        )
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space["policy"], self.num_envs)

        # Privileged critic: true delta (6) + socket pose (7) + fingertip pose (7) + fingertip vel (6)
        # + contact wrench (6) + prev_action (act_dim).
        critic_dim = delta_dim + 7 + 7 + 6 + 6 + act_dim
        self.single_observation_space["critic"] = gym.spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(critic_dim,)
        )
        self.state_space = gym.vector.utils.batch_space(self.single_observation_space["critic"], self.num_envs)

        # The end-to-end policy has no auxiliary heads: drop the aux-label group the vision base added.
        self.single_observation_space.spaces.pop("aux_label", None)

    # --- action: 5-DoF fingertip-relative delta ---------------------------------------------------
    def _pre_physics_step(self, action):
        """EMA-smooth the action, then compute the FIXED per-env-step IK+PD setpoint from the delta."""
        super()._pre_physics_step(action)  # EMA smoothing into self.actions (+ reset-buffer bookkeeping)
        a = self.actions
        pos_scale = float(getattr(self.cfg, "e2e_pos_action_scale", 0.02))
        rot_scale = float(getattr(self.cfg, "e2e_rot_action_scale", 0.1))

        # Translation: setpoint = current fingertip + pos_scale * a[0:3] (frozen for the substep loop).
        self._ctrl_target_pos = self.fingertip_midpoint_pos + pos_scale * a[:, 0:3]
        if self._socket_anchored:
            # Factory-style CONFINEMENT: re-express the target relative to the (noisy) socket and clip it to
            # a +-bound box, so a zero/exploratory action stays NEXT TO the hole instead of hovering far
            # away. Reuses the SAME noisy socket fed to the policy -> no new sim2real dependency. The true
            # socket must lie inside the box (bound >> estimate error); vision servos the residual.
            socket = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
            self._ctrl_target_pos = socket + torch.clamp(
                self._ctrl_target_pos - socket, -self._socket_action_bound, self._socket_action_bound
            )

        # Rotation: 2 tilt axes only (fingertip x, y); NO yaw. Compose in the BODY frame so the tilt is
        # about the gripper's own axes (current_quat (x) dRot), matching the pre-insert-tilt convention.
        rotvec = torch.zeros((self.num_envs, 3), device=self.device)
        rotvec[:, 0] = a[:, 3] * rot_scale
        rotvec[:, 1] = a[:, 4] * rot_scale
        angle = torch.linalg.norm(rotvec, dim=1)
        axis = rotvec / angle.unsqueeze(-1).clamp(min=1e-6)
        dquat = torch_utils.quat_from_angle_axis(angle, axis)
        dquat = torch.where(angle.unsqueeze(-1) > 1e-6, dquat, self._identity_quat)
        self._ctrl_target_quat = torch_utils.quat_mul(self.fingertip_midpoint_quat, dquat)

    def _apply_action(self):
        """Drive the IK+PD controller toward the frozen per-env-step setpoint (each decimation substep)."""
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)
        if self._ctrl_target_pos is None:  # belt-and-suspenders: hold position if called out of order
            self._ctrl_target_pos = self.fingertip_midpoint_pos.clone()
            self._ctrl_target_quat = self.fingertip_midpoint_quat.clone()
        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=self._ctrl_target_pos,
            ctrl_target_fingertip_midpoint_quat=self._ctrl_target_quat,
            ctrl_target_gripper_dof_pos=0.0,
        )

    # --- controller: differential IK + joint PD (replaces Forge's OSC torque) ---------------------
    def generate_ctrl_signals(
        self, ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, ctrl_target_gripper_dof_pos
    ):
        """One DLS differential-IK step to the target fingertip pose, commanded to the joint-PD actuators.

        Replaces ``FactoryEnv.generate_ctrl_signals`` (task-space impedance / OSC torque). The arm
        actuators are high-gain implicit-PD (position control, see ``IIWA_GRIPPER_E2E_CFG``): we set
        their position target to the IK solution and ZERO the feedforward arm torque so control is pure
        joint PD. The gripper joints keep their PhysX PD position target (close-in-place during grasp,
        held during the episode). Called both while stepping (via ``_apply_action``) and during the
        reset grasp (``close_gripper_in_place``), so both paths stay PD-consistent.
        """
        pos_error, axis_angle_error = factory_control.get_pose_error(
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            jacobian_type="geometric",
            rot_error_type="axis_angle",
        )
        delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
        delta_dof_pos = factory_control.get_delta_dof_pos(
            delta_pose=delta_hand_pose,
            ik_method="dls",
            jacobian=self.fingertip_midpoint_jacobian,
            device=self.device,
        )
        self.ctrl_target_joint_pos[:, 0:7] = self.joint_pos[:, 0:7] + delta_dof_pos[:, 0:7]
        if getattr(self, "_weld_held", False):
            ctrl_target_gripper_dof_pos = self._weld_gripper_dof_pos()
        self.ctrl_target_joint_pos[:, 7:9] = ctrl_target_gripper_dof_pos

        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        # Pure joint PD: no feedforward arm torque (the implicit high-gain PD does the work). Zeroing
        # every substep also clears any stale OSC/reset effort target that would otherwise superpose.
        self._robot.set_joint_effort_target(self._zero_effort)

    # --- geometry helpers -------------------------------------------------------------------------
    def _axis_error_vec(self):
        """World-frame rotation vector that aligns the screw shaft axis to the socket axis.

        Its NORM is the shaft-axis error (yaw-invariant, matches the yaw-invariant success + the 2
        tilt-only action DOFs); its direction is the tilt axis. Used both as the orientation term of
        the reward (its norm) and as the rotation half of the true-delta obs (the full vector).
        """
        _, held_base_quat = self._held_base_pose()
        held_axis = torch_utils.quat_apply(held_base_quat, self._local_z_axis)
        socket_axis = torch_utils.quat_apply(self.fixed_quat, self._local_z_axis)
        v = torch.cross(held_axis, socket_axis, dim=1)
        s = torch.clamp(torch.sum(held_axis * socket_axis, dim=1), -1.0, 1.0)
        angle = torch.atan2(torch.linalg.norm(v, dim=1), s)  # [0, pi], = shaft_axis_error
        axis = v / torch.linalg.norm(v, dim=1, keepdim=True).clamp(min=1e-6)
        return axis * angle.unsqueeze(-1)

    def _true_delta(self):
        """Error-free 6-D delta from the screw tip to the seated pose: [dpos(3), tilt_err_vec(3)], world.

        Position = socket_bottom - shaft_tip; rotation = the shaft-axis-alignment vector (yaw-free).
        This is the privileged quantity fed to the critic always and to the POLICY only under the
        state-first debug flag (error-free there; the real vision policy never sees it).
        """
        tip_pos, _ = self._held_base_pose()
        dpos = self.fixed_pos - tip_pos
        return torch.cat([dpos, self._axis_error_vec()], dim=-1)

    # --- observations: end-to-end policy + privileged critic --------------------------------------
    def _get_observations(self):
        prev_actions = self.actions.clone()
        force_policy = self.noisy_force  # noisy 3-axis wrist contact force (NOT proprio)

        policy_parts = []
        if self._debug_true_delta:
            policy_parts.append(self._true_delta())  # state-first scaffold: error-free delta (6)
        if self._use_proprio:
            # res224 recipe: NOISY socket-relative fingertip pos (+-8mm obs noise) + proprio pose/vel.
            noisy_fixed = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
            policy_parts += [
                self.fingertip_midpoint_pos - noisy_fixed,  # fingertip_pos_rel_fixed (3, noisy)
                self.fingertip_midpoint_quat,               # 4
                self.ee_linvel_fd,                          # 3
                self.ee_angvel_fd,                          # 3
            ]
        policy_parts += [force_policy, prev_actions]
        obs = {"policy": torch.cat(policy_parts, dim=-1)}

        # Privileged critic (asymmetric): give it everything, clean.
        obs["critic"] = torch.cat(
            [
                self._true_delta(),                      # 6
                self.fixed_pos,                          # 3  true socket bottom
                self.fixed_quat,                         # 4
                self.fingertip_midpoint_pos,             # 3
                self.fingertip_midpoint_quat,            # 4
                self.fingertip_midpoint_linvel,          # 3  clean velocities
                self.fingertip_midpoint_angvel,          # 3
                self.force_sensor_smooth[:, 0:6],        # 6  full contact wrench
                prev_actions,                            # act_dim
            ],
            dim=-1,
        )

        if getattr(self, "_has_camera", False):
            obs["image"] = self._get_camera_image()  # wrist RGB-D (reuses InsertionEnv machinery)
        return obs

    # --- reward: -L2 position + shaft-axis orientation to the seated pose --------------------------
    def _get_rewards(self):
        keypoint_dist, tip_dist = self._keypoint_distances()
        shaft_axis_error = self._shaft_axis_error()
        curr_successes = self._get_curr_successes(self.cfg_task.success_threshold)

        mode = str(getattr(self.cfg, "e2e_reward_mode", "l2_axis"))
        if mode == "squashing":
            # Optional: the Factory/IndustReal multi-scale squashing kernel on the keypoint distance
            # (kept behind the flag; no success-prediction term -- the E2E action has no success DOF).
            a0, b0 = self.cfg_task.keypoint_coef_baseline
            a1, b1 = self.cfg_task.keypoint_coef_coarse
            a2, b2 = self.cfg_task.keypoint_coef_fine
            rew_buf = (
                factory_utils.squashing_fn(keypoint_dist, a0, b0)
                + factory_utils.squashing_fn(keypoint_dist, a1, b1)
                + factory_utils.squashing_fn(keypoint_dist, a2, b2)
            )
        else:
            # Dense -L2 position + orientation error to the seated pose.
            w_pos = float(getattr(self.cfg, "e2e_reward_pos_weight", 1.0))
            w_rot = float(getattr(self.cfg, "e2e_reward_rot_weight", 0.25))
            if mode == "l2_geodesic":
                # The spec's literal FULL quaternion geodesic (includes yaw). NOTE: the action has no yaw
                # DOF and success is yaw-invariant (round peg + random socket yaw), so this penalises a
                # yaw the policy cannot correct -- kept selectable, but "l2_axis" is the default for a reason.
                _, held_base_quat = self._held_base_pose()
                dq = torch_utils.quat_mul(self.fixed_quat, torch_utils.quat_conjugate(held_base_quat))
                orient_err = 2.0 * torch.atan2(torch.linalg.norm(dq[:, 1:4], dim=1), dq[:, 0].abs())
            else:  # "l2_axis" (default): yaw-invariant shaft-axis error, matching the tilt-only action
                orient_err = shaft_axis_error
            rew_buf = -(w_pos * tip_dist + w_rot * orient_err)

        bonus = float(getattr(self.cfg, "e2e_success_bonus", 0.0))
        if bonus != 0.0:
            rew_buf = rew_buf + bonus * curr_successes.float()

        log_dict = {
            "neg_tip_l2": -tip_dist,
            "neg_keypoint_l2": -keypoint_dist,
            "shaft_axis_error": shaft_axis_error,
            "success": curr_successes.float(),
        }
        self.prev_actions = self.actions.clone()
        self._log_factory_metrics(log_dict, curr_successes)
        return rew_buf

    # --- reset speed: loosen the hover-IK convergence tolerance --------------------------------------
    def set_pos_inverse_kinematics(self, *args, **kwargs):
        """Report the reset hover-IK 'converged' at a LOOSER tolerance than Factory's 1 mm / 0.057 deg.

        Root-caused via scripts/diag_reset_ik.py: the iiwa DLS servo can't hit 1 mm AND 0.057 deg for
        ~30% of the random hover targets in 0.25 s (the 0.057 deg angle bar is ~20x tighter than it
        achieves), so the base reset loop rejection-samples fresh targets until all envs pass -> ~22
        wasted passes. We zero the reported error for any env already within ``reset_ik_pos_tol`` /
        ``reset_ik_rot_tol`` so the loop stops re-rolling those. Safe: the hover pose is ALREADY ±8 mm /
        ±45 deg random, so a few mm / deg of slack is negligible (and 2 deg of gripper tilt is nothing
        next to the 25 deg pre-insert tilt). Off (0) => exact Factory behaviour.
        """
        pos_error, axis_angle_error = super().set_pos_inverse_kinematics(*args, **kwargs)
        pos_tol = float(getattr(self.cfg, "reset_ik_pos_tol", 0.0) or 0.0)
        rot_tol = float(getattr(self.cfg, "reset_ik_rot_tol", 0.0) or 0.0)
        if pos_tol > 0.0 or rot_tol > 0.0:
            within = torch.ones(pos_error.shape[0], dtype=torch.bool, device=self.device)
            if pos_tol > 0.0:
                within &= torch.linalg.norm(pos_error, dim=1) < pos_tol
            if rot_tol > 0.0:
                within &= torch.linalg.norm(axis_angle_error, dim=1) < rot_tol
            pos_error = pos_error.clone()
            axis_angle_error = axis_angle_error.clone()
            pos_error[within] = 0.0
            axis_angle_error[within] = 0.0
        return pos_error, axis_angle_error

    # --- reset: zero the delta action, keep only the Forge episode-DR we still use ----------------
    def _reset_idx(self, env_ids):
        """End-to-end reset.

        We call ``FactoryEnv._reset_idx`` directly (which still dispatches our iiwa/InsertionEnv
        overrides for the asset/robot pose, grasp, pre-insert tilt, camera + appearance DR) and SKIP
        ``ForgeEnv._reset_idx``, whose extra block re-initialises a 7-D ABSOLUTE Forge action
        (indexing actions[:, 5:7]) that does not exist for our 5-D delta action space. The delta action
        resets to zero (no motion), which ``randomize_initial_state`` already does. We then re-add only
        the Forge per-episode DR the end-to-end path still uses (EMA factor, contact threshold, force
        smoothing reset).
        """
        # iiwa hover-IK retry cap (see InsertionEnvIiwa._reset_idx). Cfg-overridable so the debug
        # scaffold can cap it low (straggler envs fall back to the vertical grasp) to speed the reset.
        self._reset_ik_budget = int(getattr(self.cfg, "debug_reset_ik_budget", 25) or 25)
        FactoryEnv._reset_idx(self, env_ids)

        contact_rand = torch.rand((self.num_envs,), device=self.device)
        contact_lower, contact_upper = self.cfg.task.contact_penalty_threshold_range
        self.contact_penalty_thresholds = contact_lower + contact_rand * (contact_upper - contact_lower)

        self.force_sensor_world_smooth[:, :] = 0.0

        ema_rand = torch.rand((self.num_envs, 1), device=self.device)
        ema_lower, ema_upper = self.cfg.ctrl.ema_factor_range
        self.ema_factor = ema_lower + ema_rand * (ema_upper - ema_lower)

        self.actions[:] = 0.0
        self.prev_actions[:] = 0.0
        self._ctrl_target_pos = None
        self._ctrl_target_quat = None
