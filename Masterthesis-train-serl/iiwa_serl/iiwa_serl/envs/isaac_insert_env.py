from __future__ import annotations

from typing import Any

try:
    import gym
except ImportError:
    import gymnasium as gym
import numpy as np
from gym import spaces

from iiwa_serl.config import IsaacAdapterConfig


def _extract_obs(payload: Any) -> dict[str, Any]:
    obs = payload[0] if isinstance(payload, tuple) else payload
    if isinstance(obs, dict) and "obs" in obs and isinstance(obs["obs"], dict):
        obs = obs["obs"]
    return obs


def _maybe_unbatch(value: Any):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim > 0 and value.shape[0] == 1:
        return value[0]
    return value


class IsaacLabInsertionSERLEnv(gym.Env):
    def __init__(
        self,
        config: IsaacAdapterConfig | None = None,
        include_image: bool = True,
    ):
        super().__init__()
        self.config = config or IsaacAdapterConfig()
        self.include_image = include_image

        import gymnasium as gymnasium
        import isaaclab_tasks  # noqa: F401
        import insertion_policy.tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg

        env_cfg = parse_env_cfg(self.config.task, device=None, num_envs=self.config.num_envs)
        self._env = gymnasium.make(self.config.task, cfg=env_cfg, render_mode=None)

        reset_obs = _extract_obs(self._env.reset())
        state = _maybe_unbatch(reset_obs["policy"]).astype(np.float32)
        obs_spaces: dict[str, spaces.Space] = {
            self.config.state_key: spaces.Box(-np.inf, np.inf, shape=state.shape, dtype=np.float32)
        }
        self._image_present = include_image and "image" in reset_obs
        if self._image_present:
            image = _maybe_unbatch(reset_obs["image"])
            if self.config.keep_rgb_only and image.shape[-1] > 3:
                image = image[..., :3]
            image = np.asarray(image, dtype=np.uint8)
            obs_spaces[self.config.image_key] = spaces.Box(
                0, 255, shape=image.shape, dtype=np.uint8
            )

        act_dim = int(_maybe_unbatch(self._env.unwrapped.cfg.action_space))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(act_dim,), dtype=np.float32)
        self.observation_space = spaces.Dict(obs_spaces)

    def _convert_obs(self, obs_payload: Any) -> dict[str, np.ndarray]:
        obs = _extract_obs(obs_payload)
        converted = {
            self.config.state_key: _maybe_unbatch(obs["policy"]).astype(np.float32)
        }
        if self._image_present:
            image = _maybe_unbatch(obs["image"])
            if self.config.keep_rgb_only and image.shape[-1] > 3:
                image = image[..., :3]
            converted[self.config.image_key] = np.asarray(image, dtype=np.uint8)
        return converted

    def reset(self, *, seed=None, options=None):
        obs = self._env.reset()
        return self._convert_obs(obs), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(1, -1)
        obs, reward, terminated, truncated, info = self._env.step(action)
        reward = float(_maybe_unbatch(reward))
        terminated = bool(_maybe_unbatch(terminated))
        truncated = bool(_maybe_unbatch(truncated))
        if self.config.sparse_reward:
            reward = 1.0 if reward >= self.config.success_reward_threshold else 0.0
        return self._convert_obs(obs), reward, terminated, truncated, info

    def close(self):
        self._env.close()
