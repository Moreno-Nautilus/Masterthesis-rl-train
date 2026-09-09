"""Offline regression checks for reset clearance and PS4 episode boundaries.

Run with the interpreter exported by franka_env.sh. No ROS, camera, pad, or motion.
"""

import contextlib
import io
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np

from franka_env.envs.wrappers import PS4RewardWrapper
from franka_env.spacemouse.ps4_expert import PS4Expert
from experiments.franka_plumbers.wrapper import FrankaPlumbersEnv
from experiments.franka_plumbers.preflight import check_backend_wired


def expert_without_hardware(**events):
    expert = object.__new__(PS4Expert)
    expert.latest_data = dict(heartbeat=time.monotonic(), success=False,
                              abort=False, regrasp=False)
    expert.latest_data.update(events)
    expert.process = SimpleNamespace(is_alive=lambda: True)
    expert._lock = threading.Lock()
    expert._warned_dead = False
    return expert


class DummyEnv(gym.Env):
    def __init__(self, expert):
        self.expert = expert
        self.should_regrasp = False
        self.reset_calls = 0
        self.regrasps = 0

    def reset(self, **kwargs):
        self.reset_calls += 1
        self.regrasps += int(self.should_regrasp)
        self.should_regrasp = False
        return {}, {}


class RigSafetyTests(unittest.TestCase):
    def test_timeout_is_not_a_confirmed_empty_read(self):
        expert = expert_without_hardware(success=True)
        with expert._lock, patch('franka_env.spacemouse.ps4_expert.LOCK_TIMEOUT_S', 0):
            self.assertFalse(expert.get_events()['read_ok'])
            self.assertTrue(expert.latest_data['success'])
        event = expert.get_events()
        self.assertTrue(event['read_ok'])
        self.assertTrue(event['success'])
        self.assertFalse(expert.latest_data['success'])

    def test_two_failed_reads_do_not_confirm_drain(self):
        expert = expert_without_hardware(success=True)
        env = PS4RewardWrapper(DummyEnv(expert), require_expert=True)
        acquire = expert._acquire
        attempts = iter([False, False])

        def transient_contention(timeout):
            result = next(attempts, None)
            return acquire(timeout) if result is None else result

        with patch.object(expert, '_acquire', side_effect=transient_contention):
            drained, _ = env._drain()
        self.assertTrue(drained)
        self.assertFalse(expert.latest_data['success'])

    def test_failed_drain_refuses_reset_motion(self):
        expert = expert_without_hardware(success=True)
        inner = DummyEnv(expert)
        env = PS4RewardWrapper(inner, require_expert=True)
        with expert._lock, patch('franka_env.spacemouse.ps4_expert.LOCK_TIMEOUT_S', 0):
            with self.assertRaisesRegex(RuntimeError, 'before reset'):
                env.reset()
        self.assertEqual(inner.reset_calls, 0)
        self.assertTrue(expert.latest_data['success'])

    def test_circle_before_reset_applied_once(self):
        expert = expert_without_hardware(regrasp=True)
        inner = DummyEnv(expert)
        env = PS4RewardWrapper(inner, require_expert=True)
        env.reset()
        env.reset()
        self.assertEqual(inner.regrasps, 1)

    def test_failed_post_reset_drain_refuses_new_episode(self):
        inner = DummyEnv(expert_without_hardware())
        env = PS4RewardWrapper(inner, require_expert=True)
        with patch.object(env, '_drain', side_effect=[(True, False), (False, True)]):
            with self.assertRaisesRegex(RuntimeError, 'after reset'):
                env.reset()
        self.assertTrue(env._regrasp_pending)

    def test_circle_during_reset_survives_to_next_reset(self):
        expert = expert_without_hardware()
        inner = DummyEnv(expert)
        env = PS4RewardWrapper(inner, require_expert=True)
        original_reset = inner.reset

        def reset_with_circle(**kwargs):
            result = original_reset(**kwargs)
            expert.latest_data['regrasp'] = True
            return result

        with patch.object(inner, 'reset', side_effect=reset_with_circle):
            env.reset()
        self.assertEqual(inner.regrasps, 0)
        env.reset()
        env.reset()
        self.assertEqual(inner.regrasps, 1)

    def test_full_retract_or_refusal_before_followup_motion(self):
        env = object.__new__(FrankaPlumbersEnv)
        env.currpos = np.array([0.5, 0, 0.3, 0, 0, 0, 1.0])
        env.retract_dir = np.array([0, 0, 1.0])
        env.retract_dist = 0.04
        env.xyz_bounding_box = gym.spaces.Box(
            np.array([0., -1., 0.]), np.array([1., 1., 1.]), dtype=np.float64)
        env.rpy_bounding_box = gym.spaces.Box(
            np.array([0., -np.pi, -np.pi]), np.full(3, np.pi), dtype=np.float64)
        env._update_currpos = Mock()
        np.testing.assert_allclose(env._retract_target(0.04)[:3], [0.5, 0, 0.34])
        env.xyz_bounding_box.high[2] = 0.31  # only 1cm available, 4cm required
        env._send_pos_command = Mock()
        env.interpolate_move = Mock()
        env.robot = Mock()
        env.config = SimpleNamespace(PRECISION_PARAM={})
        with patch('experiments.franka_plumbers.wrapper.time.sleep'):
            with self.assertRaisesRegex(RuntimeError, 'shorten or rotate'):
                env.go_to_reset()
        env.interpolate_move.assert_not_called()
        self.assertEqual(env._send_pos_command.call_count, 1)  # initial hold only

    def test_preflight_rejects_uninspectable_and_empty_methods(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with patch('inspect.getsource', side_effect=OSError('unavailable')):
                self.assertFalse(check_backend_wired())
            for body in ('pass', '"docstring only"', 'raise NotImplementedError'):
                with patch('inspect.getsource', return_value=f'def method(self):\n    {body}\n'):
                    self.assertFalse(check_backend_wired())


if __name__ == '__main__':
    unittest.main()
