from __future__ import annotations

import time

try:
    import gym
except ImportError:
    import gymnasium as gym
import numpy as np
from gym import spaces

from iiwa_serl.config import IiwaInsertionConfig
from iiwa_serl.robot import LocalBackendClient, MockIiwaBackend, RobotServerClient
from iiwa_serl.robot.cameras import make_camera_provider
from iiwa_serl.utils.transformations import clip_pose7, compose_delta_pose, goal_delta_in_tcp_frame, pose6_to_pose7


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
        self.prev_action = np.zeros(self.config.action_dim, dtype=np.float32)
        self.curr_path_length = 0
        self.latest_images = {}
        self.last_cycle_start = time.time()
        self._manual_success = False  # one-shot external success flag (manual mode / demo button)

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
        obs_spaces: dict[str, spaces.Space] = {
            "state": spaces.Box(-np.inf, np.inf, shape=(state_dim,), dtype=np.float32)
        }
        if include_image:
            cam = self.config.cameras[0]
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

    def _randomized_reset_pose7(self) -> np.ndarray:
        pose6 = np.asarray(self.config.reset_pose, dtype=np.float64).copy()
        if self.config.random_reset:
            pose6[0] += np.random.uniform(-self.config.random_xy_range, self.config.random_xy_range)
            pose6[1] += np.random.uniform(-self.config.random_xy_range, self.config.random_xy_range)
            pose6[5] += np.random.uniform(-self.config.random_yaw_range, self.config.random_yaw_range)
        return pose6_to_pose7(pose6)

    def _get_images(self) -> dict[str, np.ndarray]:
        if not self.include_image:
            return {}
        self.latest_images = self.camera_provider.read()
        return self.latest_images

    def _state_vector(self, state) -> np.ndarray:
        goal_delta = goal_delta_in_tcp_frame(state.pose, self.goal_pose7)
        return np.concatenate(
            [goal_delta, state.vel, state.force, state.torque, self.prev_action.astype(np.float64)]
        ).astype(np.float32)

    def _make_obs(self, state) -> dict[str, np.ndarray]:
        obs = {"state": self._state_vector(state)}
        obs.update(self._get_images())
        return obs

    def set_manual_success(self, value: bool = True) -> None:
        """Externally mark the current step as a success (used by reward_mode='manual'
        and by the demo recorder's operator success-button). Consumed on the next
        _success() call."""
        self._manual_success = bool(value)

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
        self.client.clear_errors()
        if self.config.joint_reset_on_reset:
            self.client.joint_reset()
        reset_pose7 = clip_pose7(
            self._randomized_reset_pose7(),
            self.config.abs_pose_limit_low,
            self.config.abs_pose_limit_high,
        )
        self.client.move_pose(reset_pose7)
        if self.config.sticky_gripper_closed:
            self.client.close_gripper()
        time.sleep(max(0.0, 1.0 / float(self.config.hz)))
        state = self.client.get_state()
        return self._make_obs(state), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        current_state = self.client.get_state()
        target_pose7 = compose_delta_pose(
            current_state.pose,
            delta_xyz=action[:3] * self.config.action_scale_xyz_m,
            delta_rot_xyz=action[3:6] * self.config.action_scale_rot_rad,
        )
        target_pose7 = clip_pose7(
            target_pose7,
            self.config.abs_pose_limit_low,
            self.config.abs_pose_limit_high,
        )

        cycle_start = time.time()
        self.client.move_pose(target_pose7)
        self.curr_path_length += 1
        self.prev_action = action.copy()
        dt = time.time() - cycle_start
        time.sleep(max(0.0, (1.0 / float(self.config.hz)) - dt))

        state = self.client.get_state()
        obs = self._make_obs(state)
        success = self._success(state)          # consumes the one-shot manual flag
        reward = self._reward(state, success)
        self._manual_success = False            # clear after this step's success is resolved
        terminated = success or self.curr_path_length >= self.config.max_episode_length
        info = {"success": success, "target_pose": self.goal_pose7.copy()}
        return obs, reward, terminated, False, info

    def render(self, mode="rgb_array"):
        if not self.include_image or not self.latest_images:
            return None
        return next(iter(self.latest_images.values()))

    def close(self):
        self.camera_provider.close()
