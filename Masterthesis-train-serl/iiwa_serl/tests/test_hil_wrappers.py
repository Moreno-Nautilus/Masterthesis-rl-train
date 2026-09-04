"""Offline unit tests for the HIL-SERL wrappers (intervention + manual reward).

Runs fully offline against the mock backend (no robot, no PS4 hardware, CPU-only).
A stub teleop provider stands in for `PS4TeleopProvider` so we can script exact
deltas/button edges and assert the training-loop contract:

  * operator delta present  -> info["intervene_action"] emitted, action overridden
  * operator delta absent   -> policy action passes through, no intervene_action
  * X (success) edge        -> reward 1.0, done, env success flag set
  * Square/Triangle (abort) -> reward 0.0, done
  * one event pump per step -> reward wrapper reads the SAME edges (no double poll)

    conda activate serl
    JAX_PLATFORMS=cpu python -m pytest iiwa_serl/tests/test_hil_wrappers.py -q
    # or: JAX_PLATFORMS=cpu python iiwa_serl/tests/test_hil_wrappers.py
"""

from __future__ import annotations

import numpy as np

from iiwa_serl.config import IiwaInsertionConfig
from iiwa_serl.envs.iiwa_insert_env import KukaIiwaInsertionEnv
from iiwa_serl.envs.hil_wrappers import PS4Intervention, HumanReward


class StubTeleop:
    """Scriptable stand-in for PS4TeleopProvider.

    Each call to `get_action()` pops the next scripted delta and latches the
    scripted success/abort edges (mirroring the real provider's one-shot,
    consume-on-read semantics). `get_action_calls` counts pumps so a test can
    prove exactly one pump happens per env step."""

    def __init__(self, script):
        # script: list of (delta6, success_bool, abort_bool)
        self._script = list(script)
        self._i = 0
        self._success = False
        self._abort = False
        self.get_action_calls = 0

    def get_action(self):
        self.get_action_calls += 1
        delta, succ, ab = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        self._success, self._abort = succ, ab
        return np.asarray(delta, dtype=np.float32)

    def is_success(self):
        f, self._success = self._success, False
        return f

    def is_failure(self):
        f, self._abort = self._abort, False
        return f

    def is_stop_forward(self):
        return False

    def close(self):
        pass


def _wrap(env, script):
    teleop = StubTeleop(script)
    inner = PS4Intervention(env, teleop=teleop)
    return HumanReward(inner), teleop


def _base_env():
    cfg = IiwaInsertionConfig(reward_mode="manual")
    return KukaIiwaInsertionEnv(config=cfg, include_image=False, fake_env=True)


def test_intervention_overrides_action_and_emits_info():
    env = _base_env()
    delta = [0.5, -0.2, 0.1, 0.0, 0.0, 0.0]
    wrapped, _ = _wrap(env, [(delta, False, False)])
    wrapped.reset()
    policy_action = np.zeros(6, dtype=np.float32)
    _, rew, done, _, info = wrapped.step(policy_action)
    assert "intervene_action" in info, "human takeover must emit intervene_action"
    np.testing.assert_allclose(info["intervene_action"], np.asarray(delta), atol=1e-6)
    assert not done and rew == 0.0, "plain intervention step should not end the episode"


def test_no_intervention_passes_policy_action_through():
    env = _base_env()
    wrapped, _ = _wrap(env, [([0, 0, 0, 0, 0, 0], False, False)])
    wrapped.reset()
    _, _, _, _, info = wrapped.step(np.array([0.3, 0, 0, 0, 0, 0], dtype=np.float32))
    assert "intervene_action" not in info, "no takeover -> no intervene_action"


def test_manual_success_gives_reward_and_done():
    env = _base_env()
    wrapped, _ = _wrap(env, [([0, 0, 0, 0, 0, 0], True, False)])  # X pressed
    wrapped.reset()
    _, rew, done, _, info = wrapped.step(np.zeros(6, dtype=np.float32))
    assert rew == 1.0 and done, "X should give reward 1.0 and terminate"
    assert info.get("succeed") is True


def test_manual_abort_ends_without_reward():
    env = _base_env()
    wrapped, _ = _wrap(env, [([0, 0, 0, 0, 0, 0], False, True)])  # Square/Triangle
    wrapped.reset()
    _, rew, done, _, info = wrapped.step(np.zeros(6, dtype=np.float32))
    assert rew == 0.0 and done, "abort should end the episode with no reward"
    assert info.get("succeed") is False


def test_abort_wins_over_success_tie():
    env = _base_env()
    wrapped, _ = _wrap(env, [([0, 0, 0, 0, 0, 0], True, True)])  # both pressed
    wrapped.reset()
    _, rew, done, _, _ = wrapped.step(np.zeros(6, dtype=np.float32))
    assert rew == 0.0 and done, "abort must win the tie (matches actor semantics)"


def test_single_event_pump_per_step():
    env = _base_env()
    wrapped, teleop = _wrap(env, [([0.5, 0, 0, 0, 0, 0], True, False)])
    wrapped.reset()
    teleop.get_action_calls = 0
    wrapped.step(np.zeros(6, dtype=np.float32))
    assert teleop.get_action_calls == 1, (
        f"exactly one teleop pump per step, got {teleop.get_action_calls} "
        "(double-poll would race the button edges)"
    )


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all HIL wrapper tests passed")
