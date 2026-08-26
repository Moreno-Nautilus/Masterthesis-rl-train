"""Unit tests for the swappable SERL success/reward detector.

Runs fully offline against the mock backend (no robot, CPU-only). Exercises all
three reward_mode paths with synthetic pose/force states by driving the mock
backend's cached RobotState directly.

    conda activate serl
    JAX_PLATFORMS=cpu python -m pytest iiwa_serl/tests/test_reward_modes.py -q
    # or just: JAX_PLATFORMS=cpu python iiwa_serl/tests/test_reward_modes.py
"""

from __future__ import annotations

import numpy as np

from iiwa_serl.config import IiwaInsertionConfig
from iiwa_serl.envs.iiwa_insert_env import KukaIiwaInsertionEnv
from iiwa_serl.utils.transformations import pose6_to_pose7


def _make_env(reward_mode: str, **overrides) -> KukaIiwaInsertionEnv:
    cfg = IiwaInsertionConfig(reward_mode=reward_mode, **overrides)
    return KukaIiwaInsertionEnv(config=cfg, include_image=False, fake_env=True)


def _state_at(env, pose7, force_mag_n):
    """Return a RobotState positioned at pose7 with an isotropic force of the given
    magnitude, using the env's mock backend as a factory."""
    st = env.client.get_state()
    st.pose = np.asarray(pose7, dtype=np.float64)
    # spread the magnitude across 3 axes
    f = np.ones(3) * (force_mag_n / np.sqrt(3.0))
    st.force = f
    return st


# ---------------------------------------------------------------------------
# geometric mode
# ---------------------------------------------------------------------------
def test_geometric_success_at_target():
    env = _make_env("geometric")
    goal = env.goal_pose7
    st = _state_at(env, goal, force_mag_n=0.0)  # exactly at target
    assert env._success(st) is True
    assert env._reward(st, True) == env.config.success_bonus


def test_geometric_failure_far_from_target():
    env = _make_env("geometric")
    far = env.goal_pose7.copy()
    far[0] += 0.10  # 10 cm off in x
    st = _state_at(env, far, force_mag_n=0.0)
    assert env._success(st) is False


# ---------------------------------------------------------------------------
# force_depth mode  (the real-task default)
# ---------------------------------------------------------------------------
def test_force_depth_success_when_seated_and_force_present():
    env = _make_env("force_depth", seat_depth_z_m=0.008, seat_force_thresh_n=5.0, seat_force_max_n=40.0)
    # at seated depth (pose == target => dz==0) with a seating-level force
    st = _state_at(env, env.goal_pose7, force_mag_n=10.0)
    assert env._success(st) is True


def test_force_depth_fail_no_force():
    env = _make_env("force_depth", seat_force_thresh_n=5.0)
    st = _state_at(env, env.goal_pose7, force_mag_n=1.0)  # at depth but no seating force
    assert env._success(st) is False


def test_force_depth_fail_not_deep_enough():
    env = _make_env("force_depth", seat_depth_z_m=0.008)
    # 3 cm above the seated z (in TCP frame this shows as a large dz) but with force
    high = env.goal_pose7.copy()
    high[2] += 0.03
    st = _state_at(env, high, force_mag_n=10.0)
    assert env._success(st) is False


def test_force_depth_fail_force_too_high_is_a_jam():
    env = _make_env("force_depth", seat_force_max_n=40.0)
    st = _state_at(env, env.goal_pose7, force_mag_n=80.0)  # jam/collision, not a clean seat
    assert env._success(st) is False


# ---------------------------------------------------------------------------
# reach mode  (rig hello-world)
# ---------------------------------------------------------------------------
def test_reach_success_within_radius():
    env = _make_env("reach", reach_radius_m=0.03)
    near = env.goal_pose7.copy()
    near[0] += 0.02  # 2 cm off, inside 3 cm radius
    st = _state_at(env, near, force_mag_n=0.0)
    assert env._success(st) is True


def test_reach_failure_outside_radius():
    env = _make_env("reach", reach_radius_m=0.03)
    far = env.goal_pose7.copy()
    far[0] += 0.10  # 10 cm off
    st = _state_at(env, far, force_mag_n=0.0)
    assert env._success(st) is False


def test_reach_ignores_force():
    # reach must NOT require force (unlike force_depth)
    env = _make_env("reach", reach_radius_m=0.03)
    st = _state_at(env, env.goal_pose7, force_mag_n=0.0)
    assert env._success(st) is True


# ---------------------------------------------------------------------------
# manual mode + one-shot external success
# ---------------------------------------------------------------------------
def test_manual_mode_never_auto_fires():
    env = _make_env("manual")
    st = _state_at(env, env.goal_pose7, force_mag_n=10.0)  # would pass force_depth
    assert env._success(st) is False


def test_manual_success_flag_overrides_any_mode():
    for mode in ("geometric", "force_depth", "manual", "reach"):
        env = _make_env(mode)
        far = env.goal_pose7.copy()
        far[0] += 0.20
        st = _state_at(env, far, force_mag_n=0.0)  # clearly a non-success state
        env.set_manual_success(True)
        assert env._success(st) is True, f"manual flag ignored in mode={mode}"


def test_manual_flag_is_one_shot_across_step():
    # Drive a real step: set manual success, step once -> success True, flag cleared.
    env = _make_env("manual")
    env.reset()
    env.set_manual_success(True)
    obs, reward, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
    assert info["success"] is True and reward == env.config.success_bonus and terminated
    # next step without re-setting -> no success
    env.reset()
    obs, reward, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
    assert info["success"] is False


if __name__ == "__main__":
    import sys
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} reward-mode tests passed")
    sys.exit(0 if passed == len(fns) else 1)
