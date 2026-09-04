from __future__ import annotations

from argparse import Namespace
import csv
from types import SimpleNamespace

import numpy as np
import pytest

from iiwa_serl.robot.pure_down_probe import (
    MAX_TOTAL_TRAVEL_MM,
    ProbeLimits,
    _controller_state,
    make_down_target,
    pose_metrics,
    quaternion_angle_deg,
    run_probe,
    settled_step_passes,
    target_is_settled,
    validate_args,
)


def _args(**overrides):
    values = dict(
        confirm_right_arm=True,
        step_mm=1.0,
        steps=5,
        hz=20.0,
        active_timeout_s=5.0,
        stable_samples=3,
        down_threshold=0.5,
        other_axis_threshold=0.05,
        axis_tolerance_mm=0.2,
        cross_axis_tolerance_mm=0.2,
        orientation_tolerance_deg=0.1,
        server_url="http://unused",
        joystick_index=0,
        out=None,
    )
    values.update(overrides)
    return Namespace(**values)


def test_down_target_changes_only_z_by_exact_millimetres():
    origin = np.array([0.52, -0.31, 0.24, 0.1, -0.2, 0.3, 0.92736185])
    target = make_down_target(origin, step_index=2, step_mm=1.5)

    np.testing.assert_array_equal(
        target[[0, 1, 3, 4, 5, 6]], origin[[0, 1, 3, 4, 5, 6]]
    )
    assert target[2] == pytest.approx(origin[2] - 0.003, abs=1e-15)


def test_metrics_and_limits_accept_one_millimetre_pure_down():
    origin = np.array([0.5, -0.3, 0.2, 0.0, 0.0, 0.0, 1.0])
    target = make_down_target(origin, step_index=1, step_mm=1.0)
    measured = target.copy()
    measured[0] += 0.00005
    measured[1] -= 0.00008
    limits = ProbeLimits(1.0, 0.2, 0.2, 0.1)
    metrics = pose_metrics(measured, target, origin, origin)

    assert target_is_settled(metrics, limits)
    assert settled_step_passes(metrics, limits)
    assert metrics.down_step_mm == pytest.approx(1.0)
    assert metrics.cross_total_mm < 0.1


@pytest.mark.parametrize(
    "measured",
    [
        np.array([0.50021, -0.3, 0.199, 0.0, 0.0, 0.0, 1.0]),
        np.array([0.5, -0.3, 0.1997, 0.0, 0.0, 0.0, 1.0]),
    ],
)
def test_limits_reject_cross_axis_or_wrong_size(measured):
    origin = np.array([0.5, -0.3, 0.2, 0.0, 0.0, 0.0, 1.0])
    target = make_down_target(origin, step_index=1, step_mm=1.0)
    limits = ProbeLimits(1.0, 0.2, 0.2, 0.1)
    metrics = pose_metrics(measured, target, origin, origin)

    assert not settled_step_passes(metrics, limits)


def test_quaternion_angle_is_sign_invariant():
    quat = np.array([0.1, -0.2, 0.3, 0.9])
    assert quaternion_angle_deg(quat, -quat) == pytest.approx(0.0)


class _FakeController:
    def __init__(self, action, r1):
        self.action = np.asarray(action, dtype=np.float64)
        self.r1 = r1

    def get_action(self):
        return self.action.copy()

    def debug_snapshot(self):
        return {"r1": self.r1, "raw_l2": 1.0, "raw_r2": -1.0}


def test_controller_down_requires_r1_and_neutral_other_axes():
    _, _, r1, clean = _controller_state(
        _FakeController([0, 0, -1, 0, 0, 0], r1=1), 0.5, 0.05
    )
    assert r1 and clean

    _, _, r1, clean = _controller_state(
        _FakeController([0, 0, 0, 0, 0, 0], r1=0), 0.5, 0.05
    )
    assert not r1 and not clean

    _, _, _, clean = _controller_state(
        _FakeController([0.06, 0, -1, 0, 0, 0], r1=1), 0.5, 0.05
    )
    assert not clean


class _SequenceController:
    def __init__(self, samples):
        self.samples = list(samples)
        self.index = -1
        self.current = None
        self.closed = False

    def get_action(self):
        self.index += 1
        self.current = self.samples[min(self.index, len(self.samples) - 1)]
        return np.asarray(self.current[0], dtype=np.float64)

    def debug_snapshot(self):
        return {
            "r1": self.current[1],
            "raw_l2": 1.0 if self.current[0][2] < 0 else -1.0,
            "raw_r2": -1.0,
        }

    def close(self):
        self.closed = True


class _FakeClient:
    def __init__(self):
        self.pose = np.array([0.5, -0.3, 0.2, 0.0, 0.0, 0.0, 1.0])
        self.moves = []
        self.holds = 0

    def get_state(self):
        return SimpleNamespace(pose=self.pose.copy(), q=np.zeros(7))

    def move_pose(self, target):
        self.moves.append(np.asarray(target).copy())
        self.pose = np.asarray(target).copy()

    def hold_position(self):
        self.holds += 1


def test_probe_never_moves_while_r1_is_released_and_csv_contains_command(tmp_path):
    neutral = ([0, 0, 0, 0, 0, 0], 0)
    down = ([0, 0, -1, 0, 0, 0], 1)
    # Neutral arms the trigger edge, the first down starts the step, an R1
    # release in the active loop must HOLD, and the final down resumes it.
    controller = _SequenceController([neutral, down, neutral, down])
    client = _FakeClient()
    out = tmp_path / "down.csv"
    args = _args(
        steps=1,
        stable_samples=1,
        hz=10_000.0,
        active_timeout_s=0.5,
        out=str(out),
    )

    returned = run_probe(args, client=client, controller=controller)

    assert returned == out
    assert controller.closed
    assert len(client.moves) == 1
    assert client.holds >= 4  # initial, R1 release, settled step, final guard
    target = client.moves[0]
    np.testing.assert_array_equal(
        target[[0, 1, 3, 4, 5, 6]],
        np.array([0.5, -0.3, 0.0, 0.0, 0.0, 1.0]),
    )
    assert target[2] == pytest.approx(0.199)
    with out.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 1
    assert rows[0]["r1"] == "1"
    assert rows[0]["result"] == "PASS"


def test_argument_validation_enforces_step_and_total_travel(monkeypatch):
    monkeypatch.setenv("SERL_ARM_PREFIX", "lbr_two")
    validate_args(_args(step_mm=2.0, steps=int(MAX_TOTAL_TRAVEL_MM / 2.0)))

    with pytest.raises(ValueError, match="between 1.0 and 2.0"):
        validate_args(_args(step_mm=0.9))
    with pytest.raises(ValueError, match="total requested travel"):
        validate_args(_args(step_mm=2.0, steps=11))
    with pytest.raises(ValueError, match="confirm-right-arm"):
        validate_args(_args(confirm_right_arm=False))


def test_argument_validation_refuses_wrong_arm(monkeypatch):
    monkeypatch.setenv("SERL_ARM_PREFIX", "lbr_one")
    with pytest.raises(ValueError, match="expected 'lbr_two'"):
        validate_args(_args())
