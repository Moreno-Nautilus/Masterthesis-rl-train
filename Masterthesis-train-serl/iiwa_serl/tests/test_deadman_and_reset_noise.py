"""Offline checks for the R1 motion gate and one-millimetre reset jitter."""

from __future__ import annotations

import numpy as np

from iiwa_serl.config import IiwaInsertionConfig
from iiwa_serl.envs.hil_wrappers import PS4Intervention
from iiwa_serl.envs.iiwa_insert_env import KukaIiwaInsertionEnv


class _Teleop:
    def __init__(self, *, r1: bool, action=None):
        self.r1 = r1
        self._action = np.zeros(6, dtype=np.float32) if action is None else np.asarray(action)

    def get_action(self):
        return self._action.copy()

    def debug_snapshot(self):
        return {"r1": int(self.r1)}

    def is_success(self):
        return False

    def is_failure(self):
        return False

    def is_stop_forward(self):
        return False

    def close(self):
        pass


def test_ps4_r1_gates_policy_and_human_actions():
    env = KukaIiwaInsertionEnv(
        config=IiwaInsertionConfig(reward_mode="manual", require_deadman=True),
        include_image=False,
        fake_env=True,
    )
    teleop = _Teleop(r1=False, action=[0.8, 0, 0, 0, 0, 0])
    wrapper = PS4Intervention(env, teleop=teleop)
    policy = np.array([0.4, 0, 0, 0, 0, 0], dtype=np.float32)
    np.testing.assert_array_equal(wrapper.action(policy), np.zeros(6))
    assert env._deadman is False

    teleop.r1 = True
    np.testing.assert_allclose(wrapper.action(policy), teleop._action)
    assert env._deadman is True


def test_deadman_release_holds_without_advancing_episode():
    cfg = IiwaInsertionConfig(
        reward_mode="manual",
        residual_enable=True,
        require_deadman=True,
        force_retract_enable=False,
        nominal_pause_force_n=1e9,
    )
    env = KukaIiwaInsertionEnv(config=cfg, include_image=False, fake_env=True)
    env.reset(seed=7)
    before = env.client.get_state().pose.copy()

    _, _, _, truncated, info = env.step(np.ones(6, dtype=np.float32))

    np.testing.assert_allclose(env.client.get_state().pose, before)
    assert info["command_kind"] == "hold"
    assert info["deadman_blocked"] is True
    assert info["nominal_progress"] == 0.0
    assert env.curr_path_length == 0
    assert truncated is False


def test_reset_noise_is_seeded_and_bounded_to_one_millimetre_xyz():
    cfg = IiwaInsertionConfig(
        reward_mode="manual",
        random_reset=False,
        reset_noise_xyz_m=np.full(3, 0.001),
        force_retract_enable=False,
    )
    env = KukaIiwaInsertionEnv(config=cfg, include_image=False, fake_env=True)
    base = env.client.backend.reset_pose7.copy()

    env.reset(seed=123)
    first = env._episode_origin.copy()
    env.reset(seed=123)
    second = env._episode_origin.copy()

    delta = first[:3] - base[:3]
    np.testing.assert_allclose(first, second, atol=1e-12)
    assert np.all(np.abs(delta) <= 0.001 + 1e-12)
    assert np.any(np.abs(delta) > 1e-6)
    np.testing.assert_allclose(first[3:], base[3:], atol=1e-12)
