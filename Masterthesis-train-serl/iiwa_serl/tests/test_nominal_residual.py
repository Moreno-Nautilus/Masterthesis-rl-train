"""Offline tests for the reworked residual controller (HIL_RESIDUAL_SPEC.md §5).

New semantics: persistent accumulated lateral offset + along-axis forward-step scaling
(clamped) + rotation off by default. Validates the per-insert axis against the REAL
handoff (esp. plumbers k=2 = +Y horizontal).

    JAX_PLATFORMS=cpu python iiwa_serl/tests/test_nominal_residual.py
"""

from __future__ import annotations

import numpy as np

from iiwa_serl.envs.nominal_residual import NominalResidual, project_out_axis
from iiwa_serl.envs.insertion_handoff import load_insert_spec


def _nr(n=101, axis=(0, 0, -1), **kw):
    reset = np.array([0.0, 0.0, 0.10, 0, 0, 0], float)
    goal = np.array([0.0, 0.0, 0.05, 0, 0, 0], float)  # 5cm travel along -Z
    return NominalResidual(reset, goal, np.asarray(axis, float), n, **kw)


def test_zero_residual_advances_at_nominal_rate():
    nr = _nr(n=101)  # base_rate = 1/100
    nr.step(np.zeros(3), np.zeros(3), contact=False, stop_forward=False)
    assert abs(nr.progress - 0.01) < 1e-9
    for _ in range(99):
        nr.step(np.zeros(3), np.zeros(3), contact=False, stop_forward=False)
    assert nr.at_goal and abs(nr.progress - 1.0) < 1e-9


def test_progress_clamps_at_one():
    nr = _nr(n=3)
    for _ in range(10):
        nr.step(np.zeros(3), np.zeros(3), contact=False, stop_forward=False)
    assert nr.progress == 1.0


def test_forward_scale_driven_by_normalized_action_and_halt_reachable():
    # forward_scale = clip(1 + gain*action_along_norm, min, max). gain=1.5 -> full pull (-1)
    # reaches the -0.5 halt clamp; neutral -> 1.0; full push (+1) clamped to max 1.0 (#16).
    nr = _nr(n=101, forward_gain=1.5, min_forward_scale=-0.5, max_forward_scale=1.0)
    r_neutral = nr.step(np.zeros(3), np.zeros(3), contact=False, stop_forward=False,
                        action_along_norm=0.0)
    assert abs(r_neutral.forward_scale - 1.0) < 1e-9
    r_halt = nr.step(np.zeros(3), np.zeros(3), contact=False, stop_forward=False,
                     action_along_norm=-1.0)
    assert abs(r_halt.forward_scale - (-0.5)) < 1e-9   # halt/back-off clamp IS reachable
    r_push = nr.step(np.zeros(3), np.zeros(3), contact=False, stop_forward=False,
                     action_along_norm=1.0)
    assert abs(r_push.forward_scale - 1.0) < 1e-9      # never faster than nominal


def test_retract_progress_eases_back_without_going_below_zero():
    nr = _nr(n=101)  # 5cm travel
    for _ in range(20):
        nr.step(np.zeros(3), np.zeros(3), contact=False, stop_forward=False)
    p_before = nr.progress
    nr.retract_progress(0.005)  # back off 5mm along a 50mm line -> ~0.1 progress
    assert nr.progress < p_before
    for _ in range(100):
        nr.retract_progress(0.005)
    assert nr.progress == 0.0   # clamps at 0, never negative


def test_contact_and_stop_forward_pause_progress():
    nr = _nr(n=101)
    p0 = nr.progress
    nr.step(np.zeros(3), np.zeros(3), contact=True, stop_forward=False)
    assert nr.progress == p0  # frozen (scale clamped to <=0, zero residual -> 0)
    nr.step(np.zeros(3), np.zeros(3), contact=False, stop_forward=True)
    assert nr.progress == p0
    nr.step(np.zeros(3), np.zeros(3), contact=False, stop_forward=False)
    assert nr.progress > p0


def test_lateral_offset_accumulates_and_persists():
    nr = _nr(n=101, lateral_gain=1.0, lateral_limit=0.03)
    # orthogonal (X) residual should accumulate; along (Z) should NOT touch lateral.
    for _ in range(3):
        nr.step(np.array([0.002, 0.0, 0.05]), np.zeros(3), contact=False, stop_forward=False)
    off = nr.lateral_offset
    assert abs(off[0] - 0.006) < 1e-9   # 3 * 0.002 accumulated in X
    assert abs(off[2]) < 1e-9           # Z (along axis) never accumulates laterally
    # and it PERSISTS with no further lateral input:
    nr.step(np.zeros(3), np.zeros(3), contact=False, stop_forward=False)
    assert abs(nr.lateral_offset[0] - 0.006) < 1e-9


def test_lateral_offset_bounded_by_box():
    nr = _nr(n=101, lateral_gain=1.0, lateral_limit=0.01)
    for _ in range(50):
        nr.step(np.array([0.005, 0.0, 0.0]), np.zeros(3), contact=False, stop_forward=False)
    assert abs(nr.lateral_offset[0] - 0.01) < 1e-9  # clamped at the box edge


def test_target_pose_includes_forward_and_lateral():
    nr = _nr(n=101, lateral_gain=1.0)
    # step once with lateral X and nominal forward
    nr.step(np.array([0.002, 0.0, 0.0]), np.zeros(3), contact=False, stop_forward=False)
    tgt = nr.target_pose6()
    # forward: progress 0.01 of 5cm along -Z => z moved 0.0005 down from 0.10
    assert abs(tgt[2] - (0.10 - 0.01 * 0.05)) < 1e-9
    # lateral: X shifted by 0.002
    assert abs(tgt[0] - 0.002) < 1e-9


def test_rotation_off_by_default_on_by_flag():
    nr_off = _nr(n=101)  # rotation_enable defaults False
    r = nr_off.step(np.zeros(3), np.array([0.1, -0.2, 0.3]), contact=False, stop_forward=False)
    np.testing.assert_allclose(r.drot, [0, 0, 0])
    nr_on = _nr(n=101, rotation_enable=True)
    r2 = nr_on.step(np.zeros(3), np.array([0.1, -0.2, 0.3]), contact=False, stop_forward=False)
    np.testing.assert_allclose(r2.drot, [0.1, -0.2, 0.3])


def test_reset_zeroes_state():
    nr = _nr(n=101, lateral_gain=1.0)
    nr.step(np.array([0.002, 0, -0.001]), np.zeros(3), contact=False, stop_forward=False)
    nr.reset()
    assert nr.progress == 0.0
    np.testing.assert_allclose(nr.lateral_offset, [0, 0, 0])


def test_retract_dir_is_negative_axis():
    nr = _nr(axis=(0, 1, 0))
    np.testing.assert_allclose(nr.retract_dir, [0, -1, 0], atol=1e-9)


def test_project_out_axis_helper():
    np.testing.assert_allclose(project_out_axis([1.0, 2.0, 3.0], [0, 0, 1.0]), [1, 2, 0], atol=1e-9)


def test_real_handoff_axes_plumbers():
    """Per-insert axis must match the real data: k=2 is HORIZONTAL (+Y)."""
    expected = {0: [0, 0, -1], 1: [0, 0, -1], 2: [0, 1, 0], 3: [0, 0, -1]}
    for k, ax in expected.items():
        spec = load_insert_spec(k, assembly="plumbers_block")
        np.testing.assert_allclose(np.round(spec.insertion_axis, 2), ax, atol=0.02,
                                   err_msg=f"insert k={k} axis {spec.insertion_axis} != {ax}")
        assert spec.reset_joints.shape == (7,) and spec.goal_pose6.shape == (6,)


def test_env_residual_mode_runs_and_advances():
    from iiwa_serl.config import IiwaInsertionConfig
    from iiwa_serl.envs.iiwa_insert_env import KukaIiwaInsertionEnv

    spec = load_insert_spec(2, assembly="plumbers_block")  # +Y horizontal
    cfg = IiwaInsertionConfig(
        reward_mode="manual", residual_enable=True,
        insertion_axis=spec.insertion_axis,
        nominal_reset_pose6=spec.reset_pose6, nominal_goal_pose6=spec.goal_pose6,
        nominal_n=spec.nominal_n, target_pose=spec.goal_pose6, reset_pose=spec.reset_pose6,
        force_retract_enable=False, nominal_pause_force_n=1e9,
    )
    env = KukaIiwaInsertionEnv(config=cfg, include_image=False, fake_env=True)
    # #8/#9: residual obs is 29-D = 24 base + [progress, force_incr, lateral_offset(3)].
    assert env.observation_space["state"].shape == (29,)
    obs, _ = env.reset()
    assert obs["state"].shape == (29,)
    # drive a lateral (+X) nudge and confirm the stored lateral_offset dims reflect it.
    obs, _, _, _, info = env.step(np.array([0.5, 0.0, 0.0, 0, 0, 0], dtype=np.float32))
    assert 0.0 < info["nominal_progress"] <= 1.0
    assert info["nominal_paused"] is False
    np.testing.assert_allclose(obs["state"][24], info["nominal_progress"], atol=1e-6)  # progress dim
    np.testing.assert_allclose(obs["state"][26:29], info["nominal_lateral_offset"], atol=1e-6)  # lat dims


def test_e2e_state_is_24d():
    from iiwa_serl.config import IiwaInsertionConfig
    from iiwa_serl.envs.iiwa_insert_env import KukaIiwaInsertionEnv
    env = KukaIiwaInsertionEnv(config=IiwaInsertionConfig(reward_mode="manual"),
                               include_image=False, fake_env=True)
    assert env.observation_space["state"].shape == (24,)


def test_env_fk_goal_gives_real_axis_and_horizon():
    """#12: with a DISTINCT-pose FK mock (reset != goal), the env derives the REAL insertion axis
    and a multi-step horizon from goal_joints — exercising the frame/horizon path, not the
    degenerate mock where FK(reset)==FK(goal)==current pose."""
    from iiwa_serl.config import IiwaInsertionConfig
    from iiwa_serl.envs.iiwa_insert_env import KukaIiwaInsertionEnv
    from iiwa_serl.utils.transformations import pose6_to_pose7

    reset_pose6 = np.array([0.6, 0.0, 0.30, np.pi, 0, 0], float)
    goal_pose6 = np.array([0.6, 0.0, 0.25, np.pi, 0, 0], float)   # 5cm along -Z
    cfg = IiwaInsertionConfig(
        reward_mode="manual", residual_enable=True,
        goal_joints=np.zeros(7), reset_joints_target=np.zeros(7),
        reset_joints=np.zeros(7), nominal_speed_mm_s=4.0, hz=10,
        force_retract_enable=False, nominal_pause_force_n=1e9,
    )
    env = KukaIiwaInsertionEnv(config=cfg, include_image=False, fake_env=True)
    # Patch the mock backend so FK(goal_joints) = goal pose AND the measured pose = reset pose,
    # so reset->goal spans a real 5cm -Z line (backend.get_state builds pose from this).
    goal7 = pose6_to_pose7(goal_pose6)
    reset7 = pose6_to_pose7(reset_pose6)
    env.client.fk = lambda q: goal7
    # the mock's joint_reset snaps state.pose back to backend.reset_pose7 — patch BOTH so the
    # measured pre-insert origin is our reset pose.
    env.client.backend.reset_pose7 = reset7
    env.client.backend.state.pose = reset7
    env.reset()
    # axis should be ~[0,0,-1]; horizon ~ 5cm/4mm/s*10hz + margin = ~13+150
    np.testing.assert_allclose(np.round(env.config.insertion_axis, 2), [0, 0, -1], atol=0.05)
    assert env._nominal_n_eff >= 10          # NOT the degenerate n=2
    assert env.config.max_episode_length == env._nominal_n_eff + 150
    _, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))
    assert info["nominal_progress"] < 1.0     # multi-step, not one-shot


def test_env_e2e_mode_has_no_nominal():
    from iiwa_serl.config import IiwaInsertionConfig
    from iiwa_serl.envs.iiwa_insert_env import KukaIiwaInsertionEnv

    cfg = IiwaInsertionConfig(reward_mode="manual")
    env = KukaIiwaInsertionEnv(config=cfg, include_image=False, fake_env=True)
    env.reset()
    _, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))
    assert "nominal_progress" not in info


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all nominal_residual tests passed")
