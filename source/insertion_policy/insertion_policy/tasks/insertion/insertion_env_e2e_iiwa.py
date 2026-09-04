"""InsertionEnvE2EIiwa: the PURE END-TO-END visuomotor variant of the iiwa cooling-insertion env.

Supervisor-locked design (2026-07-30, see PLANNING.md "END-TO-END VISUOMOTOR INSERTION" and the
memory ``fabrica-graft-plan``). We learn the FULL pre-insert -> seated motion from scratch as an
end-to-end visuomotor policy: NO residual, NO nominal plan, NO path-centric frame, NO explicit delta
in the POLICY obs. This subclasses ``InsertionEnvIiwa`` (all task geometry -- sockets, pre-insert
tilt, grasp misalignment, wrist RGB-D + appearance DR, iiwa body-name / jacobian fixes -- is inherited
unchanged) and overrides ONLY the four things the end-to-end design changes:

  1. ACTION (``_pre_physics_step`` + ``_apply_action``): a full EE-relative delta relative to the CURRENT
     fingertip. The 2026-08-19 RGB-only rebuild uses 6-DoF = [dx, dy, dz, rot_a, rot_b, rot_yaw] (yaw
     RE-ENABLED so the wrist can counter-rotate to stay in reach / keep the socket framed; the state
     scaffold cfg keeps 5-DoF, and the yaw term is width-guarded). The translation delta is rotated into
     the TCP frame (``ctrl_target_pos = fingertip + R(fingertip_quat) . (pos_scale * a[0:3])``) and the
     rotation composes in the BODY frame (``ctrl_target_quat = fingertip_quat (x) dRot(a[3:] * rot_scale)``).
     An optional world-frame safety box clamps the target around the goal (PENDING-ANCHOR centre). The
     target is computed ONCE per env step and held across the decimation substeps, so the IK+PD servos
     toward a fixed setpoint instead of re-targeting (which with stiff PD would run away).

  2. CONTROLLER (``generate_ctrl_signals``): Forge's task-space impedance (OSC torque) is replaced by
     IK + joint PD, HIGH gains (supervisor). Each substep we take one damped-least-squares differential
     IK step toward the target fingertip pose (reusing ``factory_control.get_delta_dof_pos`` -- the same
     DLS the reset servo uses) and command the resulting joint positions to the implicit-actuator PD
     (see the high-gain arm actuators in ``IIWA_GRIPPER_E2E_CFG``). The arm's feedforward torque is
     zeroed so control is pure joint PD. Contact-force risk on the brittle parts is accepted for now.

  3. OBSERVATIONS (``_get_observations``): the policy is END-TO-END --
       * vision run:  image (wrist RGB, 3ch) + EE-frame wrist force (3) + prev_action (6) + the EE-frame
         6-DoF delta to the NOISY assembly goal G (§4). Depth is gone (RGB-only rebuild).
       * state-first debug run (``debug_true_delta_obs``): the TRUE (error-free) delta (6) + force (3)
         + prev_action, NO image -- validates the RL + controller + reward + action loop before
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
        # VELOCITY-ONLY obs (Run C): EE-frame fingertip linvel(3) + angvel(3); no pose. See cfg flag.
        self._use_velocity = bool(getattr(self.cfg, "e2e_use_velocity_obs", False))
        velocity_dim = 6 if self._use_velocity else 0

        # Optional Factory-style socket-anchored action: confine the IK target to a +-bound box around the
        # (noisy) socket so the peg can't wander/hover far from the hole -> shrinks exploration to the goal.
        self._socket_anchored = bool(getattr(self.cfg, "e2e_socket_anchored_action", False))
        self._socket_action_bound = float(getattr(self.cfg, "e2e_socket_action_bound", 0.08))

        # GOAL/ANCHOR obs (§4): EE-frame 6-DoF delta to the NOISY assembly goal G (seated target as a delta
        # from the current TCP). Built each step from the injected goal error (see _get_observations).
        self._goal_obs = bool(getattr(self.cfg, "e2e_goal_obs", False))
        goal_dim = 6 if self._goal_obs else 0
        # Per-episode injected goal-estimate error (world frame), set by _apply_pre_insert_tilt under
        # goal_anchor_injection: e_pos (3) + the tilt-only rotation quat (4). Default = no error.
        self._goal_err_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._goal_err_rot_quat = self._identity_quat.clone()
        # Cumulative commanded wrist yaw (rad) since episode start, bounded in _pre_physics_step (yaw box).
        self._yaw_accum = torch.zeros(self.num_envs, device=self.device)

        force_dim, act_dim, delta_dim = 3, self.cfg.action_space, 6
        policy_dim = (delta_dim if self._debug_true_delta else 0) + proprio_dim + velocity_dim + force_dim + act_dim + goal_dim
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

        # The end-to-end policy has no auxiliary heads by default: drop the aux-label group the vision base
        # added. KEEP it for the EXPLICIT ESTIMATOR (env.e2e_keep_aux_label=True): the aux head then predicts
        # the true hole gap (aux_label[1:4]) and feeds it into the policy (see insertion_hybrid_network).
        if not bool(getattr(self.cfg, "e2e_keep_aux_label", False)):
            self.single_observation_space.spaces.pop("aux_label", None)

        # CONTROL-LATENCY DR: delay the APPLIED action by k in [0, control_latency_max_steps] control steps
        # (random per episode), modelling the ~0-133ms round-trip control latency at 15Hz. The policy still
        # OBSERVES its true commanded action (prev_action); only the arm's setpoint responds late. A ring
        # buffer of the last (max+1) commanded actions; k=0 => no delay. 0 disables it (backward compatible).
        self._latency_max = int(getattr(self.cfg, "control_latency_max_steps", 0) or 0)
        if self._latency_max > 0:
            self._action_delay_buf = torch.zeros(
                (self.num_envs, self._latency_max + 1, self.cfg.action_space), device=self.device
            )
            self._action_delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # ANCHOR DROPOUT: per-episode mask (sampled at reset). When set, the socket anchor is corrupted in
        # the POLICY OBS ONLY (the action box still uses the good anchor), so the policy can't over-rely on
        # the anchor and must localise the hole from vision -- the root fix for the tiny-anchor fragility
        # (orange peaked 70% then collapsed). All-False by default => no-op.
        self._anchor_dropout = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # --- action: 5-DoF fingertip-relative delta ---------------------------------------------------
    def _pre_physics_step(self, action):
        """EMA-smooth the action, then compute the FIXED per-env-step IK+PD setpoint from the delta."""
        super()._pre_physics_step(action)  # EMA smoothing into self.actions (+ reset-buffer bookkeeping)
        a = self.actions
        # Control-latency DR: the arm servos toward a DELAYED copy of the commanded action (per-env k).
        # self.actions (what the policy observes as prev_action / EMA history) stays the true command.
        if getattr(self, "_latency_max", 0) > 0:
            self._action_delay_buf = torch.roll(self._action_delay_buf, shifts=1, dims=1)
            self._action_delay_buf[:, 0] = self.actions
            idx = self._action_delay_steps.view(-1, 1, 1).expand(-1, 1, a.shape[1])
            a = self._action_delay_buf.gather(1, idx).squeeze(1)
        pos_scale = float(getattr(self.cfg, "e2e_pos_action_scale", 0.02))
        rot_scale = float(getattr(self.cfg, "e2e_rot_action_scale", 0.1))

        # Translation: setpoint = current fingertip + pos_scale * a[0:3] (frozen for the substep loop).
        # FULL EE-RELATIVE (e2e_translation_in_tcp_frame): rotate the delta into the TCP frame so [dx,dy,dz]
        # are along the gripper's own axes (matches the deploy controller); else add it in the world frame.
        pos_delta = pos_scale * a[:, 0:3]
        if bool(getattr(self.cfg, "e2e_translation_in_tcp_frame", False)):
            pos_delta = torch_utils.quat_apply(self.fingertip_midpoint_quat, pos_delta)
        self._ctrl_target_pos = self.fingertip_midpoint_pos + pos_delta
        # ACTION SAFETY BOX (world frame): clamp the IK target to +-e2e_action_safety_box around the GOAL
        # centre so the peg can't run away from the hole. PENDING-ANCHOR: the centre is the socket-relative
        # goal defined in the deferred Goal/Anchor section -- structure wired here, centre left as a TODO.
        _box = float(getattr(self.cfg, "e2e_action_safety_box", 0.0) or 0.0)
        if _box > 0.0:
            # Centre the box on the NOISY assembly goal G = true socket bottom + injected goal-estimate
            # error (§4). The action box uses the GOOD G even under anchor-dropout (only the obs is dropped).
            _center = self.fixed_pos + self._goal_err_pos
            self._ctrl_target_pos = _center + torch.clamp(self._ctrl_target_pos - _center, -_box, _box)
        if self._socket_anchored:
            # Factory-style CONFINEMENT: re-express the target relative to the (noisy) socket and clip it to
            # a +-bound box, so a zero/exploratory action stays NEXT TO the hole instead of hovering far
            # away. Reuses the SAME noisy socket fed to the policy -> no new sim2real dependency. The true
            # socket must lie inside the box (bound >> estimate error); vision servos the residual.
            socket = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
            self._ctrl_target_pos = socket + torch.clamp(
                self._ctrl_target_pos - socket, -self._socket_action_bound, self._socket_action_bound
            )

        # Rotation: tilt axes (fingertip x, y) + yaw (fingertip z, RE-ENABLED for the 6-DoF rebuild).
        # Compose in the BODY frame so the rotation is about the gripper's own axes (current_quat (x) dRot),
        # matching the pre-insert-tilt convention. Width-aware: a 5-DoF action (state scaffold) has no yaw.
        rotvec = torch.zeros((self.num_envs, 3), device=self.device)
        rotvec[:, 0] = a[:, 3] * rot_scale
        rotvec[:, 1] = a[:, 4] * rot_scale
        if a.shape[1] >= 6 and not bool(getattr(self.cfg, "e2e_disable_yaw", False)):
            yaw_inc = a[:, 5] * rot_scale  # per-step yaw increment about the fingertip z axis
            # CUMULATIVE YAW BOUND: clamp the TOTAL commanded yaw to +-e2e_yaw_action_bound from the reset
            # orientation so the (unrewarded) yaw DOF can't wind into the iiwa wrist singularity / joint limit
            # (the NaN chain). yaw_inc is reduced to the amount that keeps the accumulator in range. Bounding
            # to <180deg keeps it in one branch -> 0==360 respected, no wrap. 0 => unbounded (legacy).
            _yb = float(getattr(self.cfg, "e2e_yaw_action_bound", 0.0) or 0.0)
            if _yb > 0.0:
                new_accum = (self._yaw_accum + yaw_inc).clamp(-_yb, _yb)
                yaw_inc = new_accum - self._yaw_accum
                self._yaw_accum = new_accum
            rotvec[:, 2] = yaw_inc
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
        # Minimal per-substep joint-delta clamp: a near-singular DLS-IK demand (e.g. the wrist near the
        # J6~0 singularity) can otherwise produce a large joint step -> lurch -> huge angvel -> fp16 critic
        # NaN. Capping the per-substep move bounds the lurch. ~0.1 rad only bites on extremes. 0 => off.
        _mjd = float(getattr(self.cfg, "e2e_max_joint_delta", 0.0) or 0.0)
        if _mjd > 0.0:
            delta_dof_pos = delta_dof_pos.clamp(-_mjd, _mjd)
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
        held_axis = torch_utils.quat_apply(held_base_quat, self._held_axis_local)
        socket_axis = torch_utils.quat_apply(self.fixed_quat, self._socket_axis_local)
        if bool(getattr(self.cfg_task, "axis_alignment_abs", False)):
            same_dir = torch.sum(held_axis * socket_axis, dim=1, keepdim=True) >= 0.0
            socket_axis = torch.where(same_dir, socket_axis, -socket_axis)
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

    def _goal_delta_ee(self):
        """EE-frame 6-DoF delta from the current TCP to the NOISY assembly goal G (§4).

        G = the seated target pose (screw tip at the socket bottom, shaft on the hole axis) SHIFTED by the
        per-episode injected goal-estimate error (self._goal_err_pos + self._goal_err_rot_quat). The peg was
        physically spawned at G's approach, so this delta is ~0 at reset ("the policy thinks it's at the
        goal"); the CLEAN true seated pose is up to goal_anchor_lat_max / angular away. Returned as
        [pos(3), rot(3)] expressed in the fingertip (TCP) frame.
        """
        tip_pos, held_base_quat = self._held_base_pose()
        # translation: current tip -> noisy seated tip (= true socket bottom + injected lateral/z error).
        pos_world = (self.fixed_pos + self._goal_err_pos) - tip_pos
        # orientation: align the current shaft axis to the NOISY goal axis (= true socket axis tilted by the
        # injected angular error). Same cross/dot axis-angle as _axis_error_vec, but to the noisy target.
        socket_axis = torch_utils.quat_apply(self.fixed_quat, self._socket_axis_local)
        goal_axis = torch_utils.quat_apply(self._goal_err_rot_quat, socket_axis)
        held_axis = torch_utils.quat_apply(held_base_quat, self._held_axis_local)
        if bool(getattr(self.cfg_task, "axis_alignment_abs", False)):
            same_dir = torch.sum(held_axis * goal_axis, dim=1, keepdim=True) >= 0.0
            goal_axis = torch.where(same_dir, goal_axis, -goal_axis)
        v = torch.cross(held_axis, goal_axis, dim=1)
        s = torch.clamp(torch.sum(held_axis * goal_axis, dim=1), -1.0, 1.0)
        angle = torch.atan2(torch.linalg.norm(v, dim=1), s)
        rot_world = v / torch.linalg.norm(v, dim=1, keepdim=True).clamp(min=1e-6) * angle.unsqueeze(-1)
        pos_ee = torch_utils.quat_rotate_inverse(self.fingertip_midpoint_quat, pos_world)
        rot_ee = torch_utils.quat_rotate_inverse(self.fingertip_midpoint_quat, rot_world)
        return torch.cat([pos_ee, rot_ee], dim=-1)

    # --- observations: end-to-end policy + privileged critic --------------------------------------
    def _get_observations(self):
        prev_actions = self.actions.clone()
        force_policy = self.noisy_force  # noisy 3-axis wrist contact force (NOT proprio)
        # GRAVITY COMP: subtract the pre-contact gravity baseline so force_policy is pure CONTACT, like the
        # real gravity-compensated F/T. get_link_incoming_joint_force() reports the wrench IN THE SENSOR'S
        # CHILD (BODY) FRAME (Isaac docstring), NOT world -- so the gravity load rotates with the wrist. A
        # single BODY-frame baseline captured at the reset pose therefore only cancels gravity at that
        # orientation and leaks a phantom force ~|g|*(1-cos theta) once the wrist tilts (curriculum -> 25deg).
        # FIX: store the baseline in the WORLD frame (constant there), and at every step rotate that constant
        # world-frame gravity back into the CURRENT sensor-body frame before subtracting -> zero force at any
        # orientation, with no contact. self._grav_base now holds the WORLD-frame gravity wrench.
        if getattr(self.cfg, "use_gravity_comp", False) and hasattr(self, "_grav_pending"):
            _p = self._grav_pending
            _had_pending = bool(_p.any())
            _sensor_quat = self._robot.data.body_quat_w[:, self.force_sensor_body_idx]  # sensor body -> world
            if _had_pending:
                # Snapshot the reset-env mask BEFORE clearing _grav_pending (both point to the same
                # tensor, so the in-place False-set below would otherwise lose the mask).
                _reset = _p.clone()
                # Use the RAW (unsmoothed) force: force_sensor_smooth starts from 0 after reset and
                # is only 25% of the true gravity load on the first obs (EMA alpha=0.25), which would
                # bake a 0.75*gravity_weight error into the baseline. The raw reading is the correct
                # gravity load immediately after the first physics step.
                raw_body = self._robot.root_physx_view.get_link_incoming_joint_force()[
                    :, self.force_sensor_body_idx
                ]
                # Warm-start the EMA buffer so the smoothed force has no transient for the rest of
                # the episode (without this, the smoothed force takes ~10 steps to reach gravity).
                self.force_sensor_world_smooth[_reset] = raw_body[_reset]
                # Rotate the RAW body-frame gravity reading into WORLD (constant there) and store it.
                # SANITIZE: guard against a rare PhysX transient spike at reset.
                _grav_world_reset = torch_utils.quat_apply(_sensor_quat[_reset], raw_body[_reset, 0:3])
                self._grav_base[_reset] = torch.nan_to_num(_grav_world_reset, nan=0.0, posinf=0.0, neginf=0.0)
                self._grav_pending[_reset] = False

            # Rotate the stored WORLD-frame gravity baseline into the CURRENT sensor-body frame so it lines
            # up with force_policy (which is body-frame). At the reset orientation this reproduces the old
            # behaviour exactly; at any tilt it correctly tracks the rotated gravity -> pure contact force.
            grav_base_body = torch_utils.quat_rotate_inverse(_sensor_quat, self._grav_base)
            compensated = torch.nan_to_num(force_policy - grav_base_body, nan=0.0, posinf=0.0, neginf=0.0)
            if _had_pending:
                # Reset envs: noisy_force still carries the attenuated EMA value (0.25*gravity) from the
                # pre-obs _compute_intermediate_values, so use the raw body reading for a clean 0 on step 1.
                compensated = compensated.clone()
                compensated[_reset] = torch.nan_to_num(
                    raw_body[_reset, 0:3] - grav_base_body[_reset], nan=0.0, posinf=0.0, neginf=0.0
                )
            force_policy = compensated

        # FORCE FRAME (e2e_force_in_tcp_frame): express the signed 3-axis wrist force in the TCP (fingertip)
        # frame so the policy reads body-relative contact -- matching the real robot's gravity-compensated
        # F/T, which is reported in the tool frame. VERIFY sign at deploy: the sim wrist sensor should read
        # +z (out of the tool, along the shaft) on a press into the socket; flip here if the hardware differs.
        if bool(getattr(self.cfg, "e2e_force_in_tcp_frame", False)):
            force_policy = torch_utils.quat_rotate_inverse(self.fingertip_midpoint_quat, force_policy)

        policy_parts = []
        if self._debug_true_delta:
            policy_parts.append(self._true_delta())  # state-first scaffold: error-free delta (6)
        if self._use_proprio:
            # res224 recipe: NOISY socket-relative fingertip pos (+-8mm obs noise) + proprio pose/vel.
            noisy_fixed = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
            # ANCHOR DROPOUT: on flagged episodes, replace the OBS anchor with a big-noise (useless) one so
            # the policy must localise from vision. The ACTION box (in _pre_physics_step) keeps the good
            # anchor, so the task stays seatable -- only the policy's SOCKET ESTIMATE is dropped.
            if bool(self._anchor_dropout.any()):
                _big = float(getattr(self.cfg, "anchor_dropout_noise", 0.10)) * torch.randn_like(noisy_fixed)
                noisy_fixed = torch.where(self._anchor_dropout.unsqueeze(-1), self.fixed_pos_obs_frame + _big, noisy_fixed)
            policy_parts += [
                self.fingertip_midpoint_pos - noisy_fixed,  # fingertip_pos_rel_fixed (3, noisy)
                self.fingertip_midpoint_quat,               # 4
                self.ee_linvel_fd,                          # 3
                self.ee_angvel_fd,                          # 3
            ]
        # GOAL / ANCHOR obs (§4): EE-frame 6-DoF delta to the NOISY assembly goal G. ANCHOR DROPOUT: on
        # flagged episodes, corrupt G in the OBS ONLY (big noise) so the policy can't over-rely on the goal
        # estimate and must localise the hole from vision; the action box still uses the good G. Mutually
        # exclusive with image-dropout (sampled in _reset_idx).
        if self._goal_obs:
            goal_ee = self._goal_delta_ee()
            if bool(self._anchor_dropout.any()):
                _big = float(getattr(self.cfg, "anchor_dropout_noise", 0.10)) * torch.randn_like(goal_ee)
                goal_ee = torch.where(self._anchor_dropout.unsqueeze(-1), goal_ee + _big, goal_ee)
            # PER-STEP angular perception noise on the goal-axis estimate (supervisor 2026-08-22: ~0.5deg/step
            # obs noise, DISTINCT from the per-EPISODE tilt bias baked into _goal_err_rot_quat). Resampled
            # every step; jitters ONLY the rotation channels (3:6, axis-angle rad), not the position delta.
            _gan = float(getattr(self.cfg, "e2e_goal_ang_obs_noise_deg", 0.0)) * 0.017453292519943295
            if _gan > 0.0:
                goal_ee = goal_ee.clone()
                goal_ee[:, 3:6] = goal_ee[:, 3:6] + _gan * torch.randn_like(goal_ee[:, 3:6])
            policy_parts.append(goal_ee)
        if self._use_velocity:
            # EE-frame fingertip velocity (rotate the clean world-frame vels into the TCP frame), + obs noise.
            lin_ee = torch_utils.quat_rotate_inverse(self.fingertip_midpoint_quat, self.fingertip_midpoint_linvel)
            ang_ee = torch_utils.quat_rotate_inverse(self.fingertip_midpoint_quat, self.fingertip_midpoint_angvel)
            vel_ee = torch.cat([lin_ee, ang_ee], dim=-1)
            _vn = float(getattr(self.cfg, "e2e_velocity_obs_noise", 0.0))
            if _vn > 0.0:
                vel_ee = vel_ee + _vn * torch.randn_like(vel_ee)
            policy_parts.append(torch.nan_to_num(vel_ee, nan=0.0, posinf=0.0, neginf=0.0))
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
        # EXPLICIT ESTIMATOR: expose the privileged hole-gap label so the aux head learns to predict it
        # (and its prediction is fed into the policy in-network). Excluded from the policy input by the
        # network (privileged label_key), so it never leaks into the deploy-time observation.
        if bool(getattr(self.cfg, "e2e_keep_aux_label", False)):
            obs["aux_label"] = self._get_aux_label()
        # FINAL SAFETY NET (unattended 72h run): a single non-finite obs poisons the whole PPO batch
        # permanently (NaN grad -> NaN weights). The feature fixes above address the known source
        # (gravity-comp/centering); this is defense-in-depth for anything unforeseen (e.g. a rare NaN
        # depth frame). Sanitize + warn (rate-limited) so a transient is SURVIVABLE and VISIBLE, not
        # silently masked -- if [obs-guard] ever prints, investigate that source.
        for _k in ("policy", "image", "critic"):
            _v = obs.get(_k)
            if _v is not None and not torch.isfinite(_v).all():
                _c = getattr(self, "_nan_sanitized_count", 0) + 1
                self._nan_sanitized_count = _c
                if _c <= 10 or _c % 1000 == 0:
                    print(f"[obs-guard] sanitized {int((~torch.isfinite(_v)).sum())} non-finite in "
                          f"obs[{_k}] at step {int(self.common_step_counter)} (count={_c})", flush=True)
                obs[_k] = torch.nan_to_num(_v, nan=0.0, posinf=0.0, neginf=0.0)
        # gated NaN probe (DEBUG_NAN=1): print the first obs group + force-pipeline stage that goes NaN.
        import os as _os
        if _os.environ.get("DEBUG_NAN") and not getattr(self, "_nan_reported", False):
            for _k, _v in obs.items():
                if torch.isnan(_v).any() or torch.isinf(_v).any():
                    _fss = self.force_sensor_smooth
                    print(f"[DEBUG_NAN] step={int(self.common_step_counter)} obs[{_k}] NaN={int(torch.isnan(_v).sum())} "
                          f"Inf={int(torch.isinf(_v).sum())} | force_policy_nan={int(torch.isnan(force_policy).sum())} "
                          f"grav_base_nan={int(torch.isnan(self._grav_base).sum()) if hasattr(self,'_grav_base') else 'na'} "
                          f"grav_pending={int(self._grav_pending.sum()) if hasattr(self,'_grav_pending') else 'na'} "
                          f"noisy_force_nan={int(torch.isnan(self.noisy_force).sum())} "
                          f"force_sensor_smooth_nan={int(torch.isnan(_fss).sum())}/{_fss.numel()}", flush=True)
                    self._nan_reported = True
                    break
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
            # SEAT DEAD-ZONE (2026-08-24): the reward target (fixed_pos) sits ~2.5mm BELOW the deepest the tip
            # can physically reach (head jams on rim at ~15mm), so without this the policy keeps PRESSING to
            # close an uncloseable residual -> the "crash down hard" motion. Zero the position error once the
            # tip is within e2e_seat_deadzone of the target (= physically seated) so there's no press incentive
            # past the stop. Contact force is already low (~2.3N) so this is behaviour polish, not a force fix.
            _dz = float(getattr(self.cfg, "e2e_seat_deadzone", 0.0))
            tip_dist_rew = torch.nn.functional.relu(tip_dist - _dz) if _dz > 0.0 else tip_dist
            rew_buf = -(w_pos * tip_dist_rew + w_rot * orient_err)

        # PER-TERM REWARD LOGGING (2026-08-26): record each term's per-step contribution so the tensorboard
        # shows the reward BREAKDOWN, not just the aggregate -> makes the term balance (esp. the anti-jam vs
        # base tug-of-war) directly visible instead of inferred. Values are batch means; logged via log_dict.
        _rt = {"rew_term_base": rew_buf.clone()}

        bonus = float(getattr(self.cfg, "e2e_success_bonus", 0.0))
        if bonus != 0.0:
            _b = bonus * curr_successes.float()
            rew_buf = rew_buf + _b
            _rt["rew_term_success_bonus"] = _b

        # Lateral-centering reward: pull the tip OVER the hole -- attacks the "reaches the area, misses
        # the hole" failure that dominates under the 2.5cm anchor noise. Behind a weight (default 0 => no-op).
        _wc = float(getattr(self.cfg, "e2e_reward_center_weight", 0.0))
        if _wc > 0.0:
            _, _, _xyd = self._goal_error_components()
            # CAP the centering distance (m): beyond the cap the tip is already "not centered", so keep the
            # penalty FLAT instead of growing without bound. Without this, drift to the +-8cm socket-anchor
            # box edge over a long episode makes wc*xy accumulate to ~ -1000/episode -> high-variance
            # punishing returns that collapse the action sigma (entropy) and destabilize learning (v1 bug).
            _cap = float(getattr(self.cfg, "e2e_reward_center_cap", 0.0))
            if _cap > 0.0:
                _xyd = _xyd.clamp(max=_cap)
            # sanitize: a NaN held-base pose would put a NaN in the reward buffer -> NaN advantages/losses.
            _center = -_wc * torch.nan_to_num(_xyd, nan=0.0, posinf=0.0, neginf=0.0)
            rew_buf = rew_buf + _center
            _rt["rew_term_center"] = _center

        # ANTI-JAM contact-force penalty: penalize SUSTAINED contact force above a threshold while NOT
        # seated -> discourages "press down on the fins" and nudges toward a force-guided lateral search.
        # Sized so the MAX per-step penalty is only a few % of the main reward term (must not make contact
        # timid). Weight 0 => off (default; backward compatible). See e2e_contact_penalty_scale/threshold.
        contact_force = torch.linalg.vector_norm(self.force_sensor_smooth[:, 0:3], dim=-1)
        _wj = float(getattr(self.cfg, "e2e_contact_penalty_scale", 0.0))
        if _wj > 0.0:
            thr = float(getattr(self.cfg, "e2e_contact_penalty_threshold", 12.0))
            cap = float(getattr(self.cfg, "e2e_contact_penalty_cap", 8.0))  # cap the overshoot (N) so a
            over = torch.nn.functional.relu(contact_force - thr).clamp(max=cap) * (~curr_successes).float()  # force spike can't dominate the reward
            _antijam = -_wj * torch.nan_to_num(over, nan=0.0, posinf=0.0, neginf=0.0)
            rew_buf = rew_buf + _antijam
            _rt["rew_term_antijam"] = _antijam

        # SEAT SHAPING -- CENTER-THEN-DESCEND gate: penalize the shaft tip going BELOW the socket rim while
        # laterally OFF-CENTRE, so the policy must CENTRE first and only then commit the descent. Directly
        # attacks the observed "commit down on the rim -> jam" failure. Penalty = w * below_depth while
        # xy-miss > thresh (proportional to how far below the rim it is; zero once centred). Weight 0 => off.
        _wg = float(getattr(self.cfg, "e2e_reward_descend_gate_weight", 0.0))
        if _wg > 0.0:
            # CAP below-rim depth at the socket depth (deeper is unphysical / a stray pose) so this term
            # stays bounded -- an uncapped "below" spiked the episodic reward to ~ -1825 (40x) in the hard run.
            _cap_d = float(getattr(self.cfg, "e2e_reward_descend_cap", 0.0)) or float(self.cfg_task.fixed_asset_cfg.height)
            below = torch.nn.functional.relu(self._insertion_depth_from_opening()).clamp(max=_cap_d)
            _, _, xy = self._goal_error_components()
            thr = float(getattr(self.cfg, "e2e_reward_descend_center_thresh", 0.005))
            gate = below * (xy > thr).float()                                   # below-rim depth while off-centre
            _descend = -_wg * torch.nan_to_num(gate, nan=0.0, posinf=0.0, neginf=0.0)
            rew_buf = rew_buf + _descend
            _rt["rew_term_descend_gate"] = _descend

        # SEARCH / LIFT incentive (2026-08-26): a SMALL positive reward for REDUCING the lateral xy-miss to the
        # hole axis WHILE IN CONTACT -> teaches "slide toward centre on a rim-strike" instead of pressing down
        # (the deploy failure: "leans into contact, never lifts/searches"). Reward = w * max(0, prev_xy - xy)
        # gated on contact_force > thresh and unseated. Kept LOW so it can't out-compete seating. Weight 0 => off.
        _wr = float(getattr(self.cfg, "e2e_reward_recenter_weight", 0.0))
        if _wr > 0.0:
            _, _, _xy_now = self._goal_error_components()
            if not hasattr(self, "_prev_tip_xy"):
                self._prev_tip_xy = _xy_now.clone()
            _in_contact = (contact_force > float(getattr(self.cfg, "e2e_reward_recenter_force_thresh", 3.0)))
            _improve = torch.nn.functional.relu(self._prev_tip_xy - _xy_now)      # metres of xy-miss reduced
            _recenter = _improve * (_in_contact & (~curr_successes)).float()
            _search = _wr * torch.nan_to_num(_recenter, nan=0.0, posinf=0.0, neginf=0.0)
            rew_buf = rew_buf + _search
            _rt["rew_term_search_lift"] = _search
            self._prev_tip_xy = _xy_now.detach().clone()

        # YAW REGULARIZATION (yaw is now actionable): a small penalty on the commanded yaw action so the
        # wrist doesn't slowly wind toward its joint limit. Squared -> only meaningfully penalizes large
        # sustained yaw, leaving small counter-rotations free. Width-guarded (5-DoF scaffold has no yaw).
        _wy = float(getattr(self.cfg, "e2e_reward_yaw_reg_weight", 0.0))
        if _wy > 0.0 and self.actions.shape[1] >= 6:
            _yaw = -_wy * torch.nan_to_num(self.actions[:, 5] ** 2, nan=0.0, posinf=0.0, neginf=0.0)
            rew_buf = rew_buf + _yaw
            _rt["rew_term_yaw_reg"] = _yaw

        # DIAGNOSTIC SUB-METRICS (2026-08-24): split the opaque 3D tip distance into the two sub-skills so
        # the plots show WHICH part is missing -- lateral centring vs insertion depth -- plus two coverage
        # fractions. `success` stays EXACTLY as before (unchanged headline).
        _, _, tip_xy_dist = self._goal_error_components()
        insertion_depth = self._insertion_depth_from_opening()
        _socket_r = 0.5 * float(self.cfg_task.fixed_asset_cfg.diameter)                 # ~7mm
        _engage_thr = float(getattr(self.cfg, "e2e_engage_depth_thresh", 0.005))        # 5mm shaft in
        centered = (tip_xy_dist < _socket_r).float()                                    # tip over the hole
        engaged = (insertion_depth > _engage_thr).float()                              # shaft meaningfully in

        log_dict = {
            "neg_tip_l2": -tip_dist,
            "neg_keypoint_l2": -keypoint_dist,
            "shaft_axis_error": shaft_axis_error,
            "contact_force": contact_force,  # realized wrist contact-force magnitude (N) -- verify ~10-20N
            "success": curr_successes.float(),
            "tip_xy_dist": tip_xy_dist,          # lateral distance to socket axis (m)
            "insertion_depth": insertion_depth,  # tip depth below the rim (m); ~15mm at full seat
            "centered": centered,                # fraction with tip over the hole (xy < socket radius)
            "engaged": engaged,                  # fraction with the shaft inserted > engage threshold
        }
        # PER-TERM REWARD BREAKDOWN: merge the recorded per-step reward contributions (batch means logged by
        # _log_factory_metrics as logs_rew_rew_term_*). Lets the tensorboard show WHICH term drives the return
        # -- watch rew_term_antijam vs rew_term_base to catch the "policy went timid" failure early.
        log_dict.update(_rt)
        self.prev_actions = self.actions.clone()
        self._log_factory_metrics(log_dict, curr_successes)
        return rew_buf

    # --- dynamics DR: per-episode arm PD gains / friction / armature / mass (§10) ------------------
    def _apply_dynamics_dr(self, env_ids):
        """Per-env, per-episode randomization of the arm joint PD gains + friction/armature + link mass.

        §10: resample per episode from uniform ranges. Effort stays FIXED at the physical iiwa7 caps (set in
        IIWA_GRIPPER_E2E_CFG, NOT randomized). Gains DR the 7 ARM joints only (indices 0:6); the 2 gripper
        joints keep their fixed 2000/200 PhysX PD. Mass DR scales every robot link by a per-env factor
        (PhysX scales inertia with mass, so this covers mass/inertia together). Best-effort + gated by
        e2e_dynamics_dr; a stray API/version mismatch self-disables so it can never kill a run.
        """
        if not bool(getattr(self.cfg, "e2e_dynamics_dr", False)) or getattr(self, "_dyn_dr_failed", False):
            return
        try:
            m = len(env_ids)
            arm = list(range(7))  # arm joints J1..J7

            # CURRICULUM: ramp the DR STRENGTH 0 -> 1 over e2e_dynamics_dr_curriculum_steps env-steps so the
            # policy gets an early lag/variation-free foothold before the full domain spread kicks in (v2
            # early-foothold fix, matching the tilt/latency/obs-noise curricula). 0 => full DR from step 0.
            _dr_steps = int(getattr(self.cfg, "e2e_dynamics_dr_curriculum_steps", 0) or 0)
            frac = 1.0 if _dr_steps <= 0 else min(1.0, max(0.0, float(self.common_step_counter) / float(_dr_steps)))

            def _u(lo_list, hi_list):  # per-(env,joint) uniform, deviation from the MIDPOINT scaled by frac
                lo = torch.tensor(lo_list, device=self.device)
                hi = torch.tensor(hi_list, device=self.device)
                mid = 0.5 * (lo + hi)
                val = lo + (hi - lo) * torch.rand((m, len(lo_list)), device=self.device)
                return mid + (val - mid) * frac  # frac=0 -> nominal midpoint gains (no variation)

            # Stiffness / damping per-joint ranges (§10 table): J1-4 higher, J5-7 lower.
            stiff = _u([375, 375, 375, 375, 187, 187, 187], [750, 750, 750, 750, 375, 375, 375])
            damp = _u([50, 50, 50, 50, 32, 32, 32], [70, 70, 70, 70, 45, 45, 45])
            self._robot.write_joint_stiffness_to_sim(stiff, joint_ids=arm, env_ids=env_ids)
            self._robot.write_joint_damping_to_sim(damp, joint_ids=arm, env_ids=env_ids)

            fr_lo, fr_hi = (float(x) for x in getattr(self.cfg, "e2e_dr_joint_friction_range", (0.0, 0.0)))
            if fr_hi > 0.0:
                fric = fr_lo + (fr_hi - fr_lo) * frac * torch.rand((m, 7), device=self.device)
                self._robot.write_joint_friction_coefficient_to_sim(fric, joint_ids=arm, env_ids=env_ids)
            ar_lo, ar_hi = (float(x) for x in getattr(self.cfg, "e2e_dr_armature_range", (0.0, 0.0)))
            if ar_hi > 0.0:
                arm_val = ar_lo + (ar_hi - ar_lo) * frac * torch.rand((m, 7), device=self.device)
                self._robot.write_joint_armature_to_sim(arm_val, joint_ids=arm, env_ids=env_ids)

            # Link mass/inertia DR: scale every body's default mass by a per-env factor (1 +- mass_frac*frac).
            mfrac = float(getattr(self.cfg, "e2e_dr_mass_frac", 0.0)) * frac
            if mfrac > 0.0 and getattr(self._robot.data, "default_mass", None) is not None:
                view = self._robot.root_physx_view
                masses = view.get_masses().clone()  # (num_envs, num_bodies) CPU tensor
                base = self._robot.data.default_mass.to(masses.device)
                fac = 1.0 + (2.0 * torch.rand((m, 1)) - 1.0) * mfrac
                masses[env_ids.cpu()] = base[env_ids.cpu()] * fac
                view.set_masses(masses, indices=env_ids.cpu())
        except Exception as exc:  # noqa: BLE001 -- never let a DR API mismatch kill a run
            print(f"[dynamics-dr] disabled after error: {exc}", flush=True)
            self._dyn_dr_failed = True

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

        # SEARCH/LIFT reward: clear the per-env previous-xy baseline for reset envs so the first post-reset
        # step doesn't score a spurious "improvement" off the previous episode's tip position (which could be
        # anywhere). Seeding with the fresh post-reset xy makes the first-step delta 0 (no reward).
        if hasattr(self, "_prev_tip_xy"):
            _, _, radial = self._goal_error_components()
            self._prev_tip_xy[env_ids] = radial[env_ids]

        ema_rand = torch.rand((self.num_envs, 1), device=self.device)
        ema_lower, ema_upper = self.cfg.ctrl.ema_factor_range
        self.ema_factor = ema_lower + ema_rand * (ema_upper - ema_lower)

        self.actions[:] = 0.0
        self.prev_actions[:] = 0.0
        self._ctrl_target_pos = None
        self._ctrl_target_quat = None
        self._yaw_accum[env_ids] = 0.0  # reset the cumulative-yaw bound for the new episode
        # §10 per-episode dynamics DR (arm PD gains / friction / armature / mass). Gated + self-disabling.
        self._apply_dynamics_dr(env_ids)
        # ANCHOR DROPOUT (§8): flag this episode's envs whose OBS goal G will be corrupted (vision-only seat).
        # Ramp the rate 0 -> anchor_dropout_prob from anchor_dropout_start_steps over anchor_dropout_
        # curriculum_steps (same ~ep800 schedule as image-dropout). MUTUALLY EXCLUSIVE with image-dropout
        # (sampled earlier in this reset via randomize_initial_state): never drop both the image and the
        # anchor in the same episode, so the policy always has at least one localisation channel.
        _adp = float(getattr(self.cfg, "anchor_dropout_prob", 0.0))
        if _adp > 0.0:
            _astart = int(getattr(self.cfg, "anchor_dropout_start_steps", 0) or 0)
            _aramp = int(getattr(self.cfg, "anchor_dropout_curriculum_steps", 0) or 0)
            if self.common_step_counter < _astart:
                _rate = 0.0
            elif _aramp <= 0:
                _rate = _adp
            else:
                _rate = _adp * min(1.0, max(0.0, (float(self.common_step_counter) - float(_astart)) / float(_aramp)))
            _draw = torch.rand(len(env_ids), device=self.device) < _rate
            _img = getattr(self, "_image_dropout", None)
            if _img is not None:
                _draw = _draw & (~_img[env_ids])  # mutual exclusivity with image-dropout
            self._anchor_dropout[env_ids] = _draw
        else:
            self._anchor_dropout[env_ids] = False
        # Control-latency DR: resample each env's delay and clear its action history for the new episode.
        # CURRICULUM: ramp the effective max delay 0 -> control_latency_max_steps over
        # control_latency_curriculum_steps control steps (0 => full from the start), so the policy first
        # learns without actuation lag, then adapts -- part of the v2 early-foothold fix.
        if getattr(self, "_latency_max", 0) > 0:
            _eff_max = self._latency_max
            _lsteps = int(getattr(self.cfg, "control_latency_curriculum_steps", 0) or 0)
            if _lsteps > 0:
                _eff_max = int(round(self._latency_max * min(1.0, float(self.common_step_counter) / float(_lsteps))))
            self._action_delay_steps[env_ids] = torch.randint(
                0, _eff_max + 1, (len(env_ids),), device=self.device
            )
            self._action_delay_buf[env_ids] = 0.0
