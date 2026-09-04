from __future__ import annotations

import time
import warnings

# #5: the VENDORED HIL serl_launcher stack (ChunkingWrapper.space_stack, train_rlpd's
# RecordEpisodeStatistics) is GYMNASIUM-based, so the env MUST be a gymnasium.Env producing
# gymnasium spaces. (The classic serl_launcher was legacy-gym; we are now on the HIL launcher.)
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from iiwa_serl.config import IiwaInsertionConfig
from iiwa_serl.robot import LocalBackendClient, MockIiwaBackend, RobotServerClient
from iiwa_serl.robot.cameras import make_camera_provider
from iiwa_serl.teleop.pose_target import TeleopPoseTarget
from iiwa_serl.utils.transformations import (
    clip_pose7,
    clip_pose7_relative,
    compose_delta_pose,
    goal_delta_in_tcp_frame,
    pose6_to_pose7,
    pose7_to_pose6,
)


class BaseIiwaSERLEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        config: IiwaInsertionConfig | None = None,
        include_image: bool = True,
        fake_env: bool = False,
    ):
        super().__init__()
        self.config = config or IiwaInsertionConfig()
        self.include_image = include_image
        self.fake_env = fake_env
        self.goal_pose7 = pose6_to_pose7(self.config.target_pose)
        self.reset_pose7 = pose6_to_pose7(self.config.reset_pose)
        # Per-insert JOINT configs (from the handoff): goal pose + insertion axis derived from the
        # robot's OWN FK (frame-consistent, fix #2/#3). Goal pose cached (fixed); reset origin +
        # axis + nominal endpoints rebuilt EVERY reset from the measured pre-insert pose (#4).
        self._goal_pose7_cache = None
        self.prev_action = np.zeros(self.config.action_dim, dtype=np.float32)
        self.curr_path_length = 0
        self.latest_images = {}
        self.last_cycle_start = time.time()
        self._manual_success = False  # one-shot external success flag (manual mode / demo button)
        self._teleop_target: TeleopPoseTarget | None = None
        self._camera_stale = False
        self._last_camera_warning = 0.0
        self._stop_forward = False  # operator L1 (set by PS4Intervention); freezes nominal clock
        # deadman (R1): when _require_deadman, the nominal only advances while _deadman is True
        # (so the episode WAITS at reset until R1 is pressed). Opt-in via config.require_deadman.
        self._deadman = False
        self._require_deadman = bool(getattr(self.config, "require_deadman", False))

        # Residual-on-nominal controller (None in E2E / free-delta mode). Built here with the
        # config endpoints; rebuilt frame-correctly in _resolve_goal_from_fk() once the robot's
        # own FK of the goal joints is available.
        self._nominal = None
        if getattr(self.config, "residual_enable", False):
            self._build_nominal()

        if fake_env:
            backend = MockIiwaBackend(
                initial_pose6=self.config.reset_pose.tolist(),
                reset_pose6=self.config.reset_pose.tolist(),
                reset_joints=self.config.reset_joints.tolist(),
            )
            self.client = LocalBackendClient(backend)
            self.camera_provider = make_camera_provider(self.config.cameras, real=False)
        else:
            self.client = RobotServerClient(self.config.server_url)
            self.camera_provider = make_camera_provider(self.config.cameras, real=include_image)

        state_dim = 6 + 6 + 3 + 3 + self.config.action_dim
        # #9: in residual mode the nominal progress + contact force-increment are otherwise hidden
        # controller state (the policy would be non-Markov under compliant lag). Append them to the
        # state vector so the observation is Markov. Gated; +5 dims (progress, force_incr, lat(3)).
        self._augment_obs = bool(
            getattr(self.config, "residual_enable", False)
            and getattr(self.config, "residual_obs_augment", True)
        )
        if self._augment_obs:
            # [progress, force_increment, lateral_offset(3)] = 5 dims (#8/#9): the full hidden
            # controller state so the obs is Markov under compliant lag (measured pose can differ
            # from the commanded nominal+lateral target).
            state_dim += 5
        obs_spaces: dict[str, spaces.Space] = {
            "state": spaces.Box(-np.inf, np.inf, shape=(state_dim,), dtype=np.float32)
        }
        if include_image:
            # Register only POLICY-VISIBLE cameras in the obs space. record_only cams (the fixed
            # ZED scene backup) still stream into the obs dict via _get_images so the recorder can
            # save them, but are excluded here so they never become a training/policy input.
            for cam in self.config.cameras:
                if getattr(cam, "record_only", False):
                    continue
                obs_spaces[cam.image_key] = spaces.Box(
                    0, 255, shape=(cam.height, cam.width, 3), dtype=np.uint8
                )
        self.observation_space = spaces.Dict(obs_spaces)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.config.action_dim,),
            dtype=np.float32,
        )

    def _randomized_reset_pose7(self, base_pose7: np.ndarray | None = None) -> np.ndarray:
        pose6 = (
            np.asarray(self.config.reset_pose, dtype=np.float64).copy()
            if base_pose7 is None
            else pose7_to_pose6(np.asarray(base_pose7, dtype=np.float64))
        )
        if self.config.random_reset:
            pose6[0] += self.np_random.uniform(
                -self.config.random_xy_range, self.config.random_xy_range
            )
            pose6[1] += self.np_random.uniform(
                -self.config.random_xy_range, self.config.random_xy_range
            )
            pose6[5] += self.np_random.uniform(
                -self.config.random_yaw_range, self.config.random_yaw_range
            )
        pose7 = pose6_to_pose7(pose6)
        xyz_half_range = np.asarray(
            getattr(self.config, "reset_noise_xyz_m", np.zeros(3)), dtype=np.float64
        )
        if xyz_half_range.shape != (3,) or np.any(xyz_half_range < 0.0):
            raise ValueError("reset_noise_xyz_m must contain three non-negative metre values")
        pose7[:3] += self.np_random.uniform(-xyz_half_range, xyz_half_range)
        # Constant +Z lift of the reset pose (base frame) so the arm starts higher above the seat,
        # giving more descent room. Config reset_z_offset_m (metres).
        pose7[2] += float(getattr(self.config, "reset_z_offset_m", 0.0))
        return pose7

    def _get_images(self) -> dict[str, np.ndarray]:
        if not self.include_image:
            return {}
        try:
            self.latest_images = self.camera_provider.read()
            # ROSCameraProvider deliberately returns its last good image through a
            # topic gap.  Surface that cached-frame condition in diagnostics even
            # though read() itself succeeded.
            self._camera_stale = bool(
                getattr(self.camera_provider, "last_read_stale", False)
            )
        except RuntimeError as exc:
            if not self.config.reuse_last_camera_frame or not self.latest_images:
                raise
            self._camera_stale = True
            now = time.monotonic()
            if now - self._last_camera_warning >= 5.0:
                warnings.warn(f"Camera read failed; reusing the last good frame: {exc}", RuntimeWarning)
                self._last_camera_warning = now
        return self.latest_images

    def _state_vector(self, state) -> np.ndarray:
        goal_delta = goal_delta_in_tcp_frame(state.pose, self.goal_pose7)
        base = np.concatenate(
            [goal_delta, state.vel, state.force, state.torque, self.prev_action.astype(np.float64)]
        ).astype(np.float32)
        if getattr(self, "_augment_obs", False):
            # #8/#9: append [progress, force_increment, lateral_offset(3)] so the FULL hidden
            # controller state (commanded target = nominal(progress)+lateral) is observable and the
            # obs is Markov under compliant lag. Zero-safe before the nominal exists.
            if self._nominal is not None:
                progress = float(self._nominal.progress)
                lat = np.asarray(self._nominal.lateral_offset, dtype=np.float32).reshape(3)
            else:
                progress, lat = 0.0, np.zeros(3, dtype=np.float32)
            force_incr = float(np.linalg.norm(np.asarray(state.force, dtype=np.float64))
                               - getattr(self, "_force_baseline_n", 0.0))
            base = np.concatenate(
                [base, np.array([progress, force_incr], dtype=np.float32), lat]
            )
        return base

    def _make_obs(self, state) -> dict[str, np.ndarray]:
        obs = {"state": self._state_vector(state)}
        obs.update(self._get_images())
        return obs

    def set_manual_success(self, value: bool = True) -> None:
        """Externally mark the current step as a success (used by reward_mode='manual'
        and by the demo recorder's operator success-button). Consumed on the next
        _success() call."""
        self._manual_success = bool(value)

    def set_stop_forward(self, value: bool) -> None:
        """Operator stop-forward (L1): freeze the nominal clock while held. Set each
        step by PS4Intervention (recording + HIL training). Not used at inference."""
        self._stop_forward = bool(value)

    def set_deadman(self, value: bool) -> None:
        """Operator deadman (R1): when False, freeze the nominal clock (no forward push).
        Recording/HIL only — lets the episode WAIT at reset until R1 is pressed. When
        _require_deadman is False (default / inference), this is ignored and the nominal
        advances freely as before."""
        self._deadman = bool(value)

    def _effective_nominal_n(self, insertion_len_m: float | None) -> int:
        """Number of nominal steps so the forward push runs at ~nominal_speed_mm_s (#7).

        We do NOT use the handoff's raw waypoint count (0.18 mm/step -> 280-349 steps, which
        over-discounts the terminal reward and is needlessly fine). Steps = distance / (speed *
        dt); dt = 1/hz. Same physical motion, coarser sampling, still FRI-safe."""
        # explicit override (config.nominal_steps_override > 0): use exactly this many steps
        # regardless of length/speed — for very short inserts where distance/speed gives too
        # many sub-mm steps that the compliant controller can't resolve (k=0, 5mm).
        override = int(getattr(self.config, "nominal_steps_override", 0) or 0)
        if override > 0:
            return max(override, 2)
        length_m = insertion_len_m
        if length_m is None:
            length_m = float(np.linalg.norm(
                np.asarray(self.config.nominal_goal_pose6[:3])
                - np.asarray(self.config.nominal_reset_pose6[:3])
            ))
        speed_mm_s = max(float(getattr(self.config, "nominal_speed_mm_s", 4.0)), 1e-3)
        hz = float(self.config.hz)
        # #10: we need `advances` forward steps; NominalResidual advances over (N-1) intervals, so
        # N = advances + 1 to hit the requested speed exactly (else the shortest insert runs ~5% fast).
        advances = int(np.ceil((length_m * 1000.0) / speed_mm_s * hz))
        return max(advances + 1, 2)

    def _build_nominal(self, insertion_len_m: float | None = None) -> None:
        from iiwa_serl.envs.nominal_residual import NominalResidual
        n_eff = self._effective_nominal_n(insertion_len_m)
        # forward_gain in NORMALIZED-action units (#16): a full-negative along-axis action (=-1)
        # must reach the halt/back-off clamp. dxyz metres for action=-1 along-axis is
        # action_scale_xyz_m*mult; convert the config's per-metre gain intent into a per-normalized
        # gain so the clamp is actually reachable. We express residual_forward_gain as the
        # forward-scale change for a FULL along-axis action, applied directly (unit-free).
        self._nominal = NominalResidual(
            reset_pose6=self.config.nominal_reset_pose6,
            goal_pose6=self.config.nominal_goal_pose6,
            insertion_axis=self.config.insertion_axis,
            nominal_n=n_eff,
            forward_gain=self.config.residual_forward_gain,
            min_forward_scale=self.config.residual_min_forward_scale,
            max_forward_scale=self.config.residual_max_forward_scale,
            lateral_gain=self.config.residual_lateral_gain,
            lateral_limit=self.config.residual_lateral_limit,
            rotation_enable=self.config.residual_rotation_enable,
        )
        self._nominal_n_eff = n_eff
        # Per-insert horizon = effective nominal steps + lateral-search margin (#7). Distinguish
        # this time-limit truncation from task success in step().
        margin = int(getattr(self.config, "episode_search_margin", 150))
        self.config.max_episode_length = n_eff + margin

    def _resolve_goal_from_fk(self) -> None:
        """Derive goal pose + insertion axis + nominal endpoints from the robot's OWN FK, in the
        robot's base/TCP frame (fixes the handoff-frame/flange-vs-TCP bug #2/#3).

        #4: the goal FK (from fixed goal_joints) is cached, but the RESET ORIGIN is re-measured
        EVERY reset — so the nominal line/axis is rebuilt from THIS episode's actual pre-insert
        pose (the arm lands slightly differently each time). Called from reset() after the
        measured origin is captured. No-op if goal_joints not provided."""
        goal_q = getattr(self.config, "goal_joints", None)
        if goal_q is None:
            return
        # goal pose is fixed -> FK once and cache.
        if getattr(self, "_goal_pose7_cache", None) is None:
            self._goal_pose7_cache = np.asarray(self.client.fk(goal_q), dtype=np.float64)
        goal_pose7 = self._goal_pose7_cache

        # reset origin: use THIS episode's measured pre-insert pose (re-derived every reset).
        reset_q = getattr(self.config, "reset_joints_target", None)
        if getattr(self, "_episode_origin", None) is not None:
            reset_pose7 = np.asarray(self._episode_origin, dtype=np.float64)
        else:
            reset_pose7 = np.asarray(
                self.client.fk(reset_q if reset_q is not None else self.config.reset_joints),
                dtype=np.float64,
            )
        self.goal_pose7 = goal_pose7
        axis_vec = goal_pose7[:3] - reset_pose7[:3]
        length = float(np.linalg.norm(axis_vec))
        axis_unit = axis_vec / length if length > 1e-6 else np.asarray(self.config.insertion_axis)
        self.config.insertion_axis = axis_unit
        self.config.nominal_reset_pose6 = pose7_to_pose6(reset_pose7)
        self.config.nominal_goal_pose6 = pose7_to_pose6(goal_pose7)
        if self._nominal is not None:
            self._build_nominal(insertion_len_m=length)  # rebuild per-episode with fresh origin

    def _success_geometric(self, state) -> bool:
        delta = np.abs(goal_delta_in_tcp_frame(state.pose, self.goal_pose7))
        return bool(np.all(delta < self.config.reward_threshold))

    def _success_force_depth(self, state) -> bool:
        """Success = EE reached seated insertion depth AND wrist force shows the
        seating signature (but below the jam/collision safety cap). This is measured
        from robot signals only — independent of the camera and of the noisy socket
        pose estimate. Thresholds are calibrated at the rig (see config)."""
        delta = goal_delta_in_tcp_frame(state.pose, self.goal_pose7)
        # TCP-frame z-delta to the seated target: |dz| small => at seated depth.
        depth_ok = abs(float(delta[2])) < self.config.seat_depth_z_m
        force_mag = float(np.linalg.norm(np.asarray(state.force, dtype=np.float64)))
        force_ok = self.config.seat_force_thresh_n <= force_mag <= self.config.seat_force_max_n
        return bool(depth_ok and force_ok)

    def _success_reach(self, state) -> bool:
        """Trivial reach task: EE within `reach_radius_m` of the target position
        (position only). Used for the rig hello-world to prove RLPD learns on hardware."""
        delta = goal_delta_in_tcp_frame(state.pose, self.goal_pose7)
        return bool(np.linalg.norm(delta[:3]) < self.config.reach_radius_m)

    def _success(self, state) -> bool:
        mode = getattr(self.config, "reward_mode", "geometric")
        # A one-shot external/manual success always wins (demo button, manual mode).
        if getattr(self, "_manual_success", False):
            return True
        if mode == "manual":
            return False
        if mode == "reach":
            return self._success_reach(state)
        if mode == "force_depth":
            return self._success_force_depth(state)
        return self._success_geometric(state)

    def _reward(self, state, success: bool) -> float:
        if self.config.sparse_reward:
            return float(self.config.success_bonus if success else 0.0)
        # Dense shaping is only meaningful for the geometric mode (needs a trusted
        # pose target). For force_depth/manual, dense-to-a-noisy-target would mislead,
        # so fall back to sparse there.
        if getattr(self.config, "reward_mode", "geometric") != "geometric":
            return float(self.config.success_bonus if success else 0.0)
        delta = goal_delta_in_tcp_frame(state.pose, self.goal_pose7)
        pos_cost = np.linalg.norm(delta[:3])
        rot_cost = np.linalg.norm(delta[3:])
        reward = -(self.config.dense_pos_weight * pos_cost + self.config.dense_rot_weight * rot_cost)
        if success:
            reward += self.config.success_bonus
        return float(reward)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.curr_path_length = 0
        self.prev_action[:] = 0.0
        self._manual_success = False
        self._stop_forward = False
        # Require a fresh R1 sample after every reset; never inherit an enabled
        # deadman state from the terminal step of the preceding episode.
        self._deadman = False
        if self._nominal is not None:
            self._nominal.reset()
        self.client.clear_errors()
        if self.config.manual_reset:
            # Manual pre-insert workflow: DON'T auto-move. The operator has already jogged the
            # peg to the pre-insert pose. Capture it as the episode origin for the relative box.
            self._episode_origin = self.client.get_state().pose.copy()
        else:
            # AUTOMATED reset: drive to the per-insert pre-insert JOINT config (frame-independent),
            # then use the MEASURED pose as the origin. No bogus Cartesian move through a stale
            # absolute box (that was fix #1 — the handoff pre-insert lies outside the old box).
            if self.config.joint_reset_on_reset:
                target_q = getattr(self.config, "reset_joints_target", None)
                self.client.joint_reset(
                    target_q=(target_q if target_q is not None else self.config.reset_joints)
                )
            reset_pose7 = self.client.get_state().pose.copy()
            xyz_half_range = np.asarray(
                getattr(self.config, "reset_noise_xyz_m", np.zeros(3)), dtype=np.float64
            )
            if np.any(xyz_half_range > 0.0):
                # The target is sampled once, then repeated briefly so a compliant position
                # loop reaches the same absolute pose instead of resampling or accumulating.
                reset_pose7 = self._randomized_reset_pose7(reset_pose7)
                for _ in range(3):
                    self.client.move_pose(reset_pose7)
                    time.sleep(max(0.0, 1.0 / float(self.config.hz)))
            self._episode_origin = self.client.get_state().pose.copy()
        if self.config.sticky_gripper_closed:
            self.client.close_gripper()
        time.sleep(max(0.0, 1.0 / float(self.config.hz)))
        state = self.client.get_state()
        # Baseline (no-contact) force magnitude for the force-retract safety net. The pre-insert
        # pose is hovering above the socket, so this captures the peg+gripper gravity offset.
        self._force_baseline_n = float(np.linalg.norm(np.asarray(state.force, dtype=np.float64)))
        # Frame-consistent goal/axis from the robot's own FK (once), now that we have a measured
        # pre-insert pose to anchor against (#2/#3).
        if not self.config.manual_reset:
            self._resolve_goal_from_fk()
        if self._nominal is not None:
            self._nominal.reset()
        if self.config.integrate_action_target:
            self._teleop_target = TeleopPoseTarget(
                self.config.action_scale_xyz_m * self.config.action_scale_xyz_mult,
                self.config.action_scale_rot_rad,
                base_frame_actions=self.config.base_frame_actions,
            )
            self._teleop_target.reset(state.pose)
        else:
            self._teleop_target = None
        return self._make_obs(state), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        deadman_blocked = bool(self._require_deadman and not self._deadman)
        if deadman_blocked:
            # Do not accumulate a policy/residual command that would jump into
            # effect when R1 is pressed again.
            action = np.zeros_like(action)
        current_state = self.client.get_state()
        force_mag = float(np.linalg.norm(np.asarray(current_state.force, dtype=np.float64)))
        baseline = getattr(self, "_force_baseline_n", 0.0)
        contact = (force_mag - baseline) > getattr(self.config, "nominal_pause_force_n", 1e9)

        # Contact-retract trigger (per-insert axis via the nominal; NOT hardcoded +Z).
        self._force_retracting = bool(
            not deadman_blocked
            and getattr(self.config, "force_retract_enable", False)
            and (force_mag - baseline) > self.config.force_retract_thresh_n
        )

        axis_unit = np.asarray(self.config.insertion_axis, dtype=np.float64)
        axis_unit = axis_unit / max(float(np.linalg.norm(axis_unit)), 1e-9)

        dxyz = action[:3] * self.config.action_scale_xyz_m * self.config.action_scale_xyz_mult
        drot = action[3:6] * self.config.action_scale_rot_rad
        # normalized along-axis action component (frame-consistent: raw action xyz dot axis) — drives
        # the forward-speed scale so the halt clamp is reachable regardless of metric scale (#16).
        action_along_norm = float(np.dot(np.asarray(action[:3], dtype=np.float64), axis_unit))

        # --- Residual-on-nominal composition ----------------------------------------------
        # residual_enable: the nominal owns the along-axis forward push (SCALE the residual
        # modulates: 0->nominal, negative->halt/back off); orthogonal residual ACCUMULATES a
        # PERSISTENT lateral offset; rotation OFF unless residual_rotation_enable. Forward pauses on
        # contact OR L1. On a force-retract we drive the nominal ITSELF backward (retract_progress)
        # so the NEXT step doesn't re-command the old forward target (no oscillation, #13).
        # E2E path (self._nominal is None) is unchanged.
        nominal_target6 = None
        if self._nominal is not None:
            if self._force_retracting:
                # ease the nominal back along the axis and hold lateral; no forward this step.
                self._nominal.retract_progress(self.config.force_retract_step_m)
                nominal_target6 = self._nominal.target_pose6()
                drot = np.zeros(3)
            else:
                # deadman: when required and R1 not held, freeze the nominal (treat as stop_forward)
                _freeze = bool(self._stop_forward) or (
                    self._require_deadman and not self._deadman
                )
                rstep = self._nominal.step(
                    dxyz_m=dxyz, drot_rad=drot,
                    contact=contact, stop_forward=_freeze,
                    action_along_norm=action_along_norm,
                )
                nominal_target6 = self._nominal.target_pose6()
                drot = rstep.drot
        elif self._force_retracting:
            # E2E mode retract: back off along -axis via a metric override (no nominal state).
            dxyz = (-axis_unit) * self.config.force_retract_step_m
            drot = np.zeros(3)

        command_kind = "move"
        # SAFETY DEADMAN: when required and R1 not held, HOLD the current measured pose instead of
        # commanding the nominal/goal target. Freezing only the clock still re-sent the goal pose
        # every step -> the arm kept driving toward it with R1 released (runaway). Holding the
        # measured pose makes releasing R1 a true freeze.
        if self._require_deadman and not self._deadman:
            target_pose7 = current_state.pose.copy()
        elif nominal_target6 is not None:
            base_pose7 = pose6_to_pose7(nominal_target6)
            target_pose7 = compose_delta_pose(
                base_pose7, delta_xyz=np.zeros(3), delta_rot_xyz=drot
            )
            target_pose7[:3] = base_pose7[:3]  # translation fully described by nominal_target6
        elif self._teleop_target is not None:
            command = self._teleop_target.update(action, current_state.pose)
            command_kind = command.kind
            target_pose7 = command.target_pose
        else:
            if getattr(self.config, "base_frame_actions", False):
                # Base-frame translation (matches the teleop jog feel): add xyz directly in base
                # coords instead of rotating it into the tool frame. Rotation still tool-frame.
                target_pose7 = compose_delta_pose(
                    current_state.pose, delta_xyz=np.zeros(3), delta_rot_xyz=drot
                )
                target_pose7[:3] = current_state.pose[:3] + dxyz
            else:
                target_pose7 = compose_delta_pose(current_state.pose, delta_xyz=dxyz, delta_rot_xyz=drot)
        unclipped_target_pose7 = target_pose7.copy()
        if nominal_target6 is not None:
            # RESIDUAL mode (#8): the nominal+lateral controller is the authority and is ALREADY
            # bounded (forward reaches the goal along the per-insert axis; lateral by
            # residual_lateral_limit). Re-clipping through the small relative box would truncate the
            # legitimate seat (e.g. k=2 travels 64mm > the old +/-30mm box). So skip the box here.
            pass
        elif self.config.relative_pose_limit and getattr(self, "_episode_origin", None) is not None:
            # Keep the translation box in the base frame and the rotation box in
            # the captured pre-insert tool frame. Absolute Euler clipping here made
            # right-stick rotations hit coupled/wrapped limits unpredictably.
            target_pose7 = clip_pose7_relative(
                target_pose7,
                self._episode_origin,
                self.config.rel_limit_low,
                self.config.rel_limit_high,
            )
        else:
            target_pose7 = clip_pose7(
                target_pose7,
                self.config.abs_pose_limit_low,
                self.config.abs_pose_limit_high,
            )
        position_clipped = not np.allclose(
            target_pose7[:3], unclipped_target_pose7[:3], atol=1e-9, rtol=0.0
        )
        quat_dot = abs(float(np.dot(target_pose7[3:7], unclipped_target_pose7[3:7])))
        workspace_clipped = command_kind == "move" and (
            position_clipped or quat_dot < 1.0 - 1e-10
        )
        if self._teleop_target is not None and command_kind == "move":
            # Workspace saturation must also saturate the integrator.  Otherwise a
            # target can wind up outside the box and feel dead after stick reversal.
            self._teleop_target.sync_target(target_pose7)

        if deadman_blocked:
            # Pin measured joints instead of repeatedly solving toward a Cartesian
            # target. This cancels interpolation and guarantees no commanded motion
            # while R1 is released.
            target_pose7 = current_state.pose.copy()
            command_kind = "hold"
            workspace_clipped = False

        cycle_start = time.time()
        if getattr(self.config, "debug_step", False):
            # Prove the record-phase integration story: is the measured pose (what we add
            # deltas onto) lagging behind the target we command? If |target-measured| grows
            # while pushing the stick, integrating on the measured pose is stalling the arm.
            m = current_state.pose
            gap = float(np.linalg.norm(target_pose7[:3] - m[:3]))
            step_dxyz = float(np.linalg.norm(dxyz))
            print(
                f"[step] a_xyz={action[:3].round(2)} a_rot={action[3:6].round(2)} "
                f"| dxyz_cmd={step_dxyz*1000:.1f}mm drot={np.linalg.norm(drot):.3f}rad "
                f"| meas_xyz={m[:3].round(3)} tgt_xyz={target_pose7[:3].round(3)} "
                f"| |tgt-meas|={gap*1000:.1f}mm cmd={command_kind}"
            )
        if command_kind == "move":
            self.client.move_pose(target_pose7)
        elif command_kind == "hold":
            self.client.hold_position()
        if not deadman_blocked:
            self.curr_path_length += 1
        self.prev_action = action.copy()
        dt = time.time() - cycle_start
        time.sleep(max(0.0, (1.0 / float(self.config.hz)) - dt))

        state = self.client.get_state()
        obs = self._make_obs(state)
        success = self._success(state)          # consumes the one-shot manual flag
        reward = self._reward(state, success)
        self._manual_success = False            # clear after this step's success is resolved
        # #7: distinguish task success (terminated) from the time-limit (truncated).
        terminated = bool(success)
        truncated = self.curr_path_length >= self.config.max_episode_length
        info = {
            "success": success,
            "target_pose": self.goal_pose7.copy(),
            "command_kind": command_kind,
            "commanded_pose": target_pose7.copy(),
            "measured_pose_before": current_state.pose.copy(),
            "measured_pose_after": state.pose.copy(),
            "command_gap_mm": float(np.linalg.norm(target_pose7[:3] - state.pose[:3]) * 1000.0),
            "camera_stale": bool(self._camera_stale),
            "workspace_clipped": workspace_clipped,
            "force_retracting": bool(self._force_retracting),
            "deadman_held": bool(self._deadman),
            "deadman_blocked": deadman_blocked,
        }
        if self._nominal is not None:
            info["nominal_progress"] = float(self._nominal.progress)
            info["nominal_lateral_offset"] = self._nominal.lateral_offset
            info["nominal_paused"] = bool(contact or self._stop_forward or deadman_blocked)
        # #8 (safe, non-behavioural): FLAG steps where the executed motion differs from the raw
        # action — force-retract override or L1 forward-freeze. Stored in the transition so these
        # "shielded" steps CAN be filtered/down-weighted later, without changing current learning.
        info["shielded"] = bool(
            self._force_retracting or self._stop_forward or deadman_blocked
        )
        return obs, reward, terminated, truncated, info

    def render(self, mode="rgb_array"):
        if not self.include_image or not self.latest_images:
            return None
        return next(iter(self.latest_images.values()))

    def close(self):
        self.camera_provider.close()
