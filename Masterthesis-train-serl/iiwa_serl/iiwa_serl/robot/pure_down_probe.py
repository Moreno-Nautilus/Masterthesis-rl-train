"""R1-gated, one-step-at-a-time acceptance test for pure base-frame -Z motion.

The controller is used only as a safety/step input:

* hold R1 and press L2 to request one down step;
* keep both held while that absolute Cartesian target settles;
* release L2 before requesting the next step;
* releasing R1 immediately sends the explicit joint-position hold command.

Stick and rotation values are never added to the target.  Every commanded pose
therefore has the initial X, Y, and quaternion, with only Z changed.  Each HTTP
pose command and the measured pose before/after it are flushed to CSV.

Run from the repository root after starting the right-arm (lbr_two) server::

    SERL_ARM_PREFIX=lbr_two python -m iiwa_serl.robot.pure_down_probe \
        --confirm-right-arm

The arm remains at the final down position.  This probe intentionally does not
command an automatic return move.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from iiwa_serl.robot.api import RobotServerClient

if TYPE_CHECKING:
    from iiwa_serl.teleop import PS4TeleopProvider


MAX_TOTAL_TRAVEL_MM = 20.0
POSE_NAMES = ("x", "y", "z", "qx", "qy", "qz", "qw")


class ProbeFailure(RuntimeError):
    """A motion or measurement failed the acceptance limits."""


@dataclass(frozen=True)
class ProbeLimits:
    step_mm: float
    axis_tolerance_mm: float
    cross_axis_tolerance_mm: float
    orientation_tolerance_deg: float


@dataclass(frozen=True)
class PoseMetrics:
    target_error_mm: np.ndarray
    from_origin_mm: np.ndarray
    from_previous_mm: np.ndarray
    down_from_origin_mm: float
    down_step_mm: float
    cross_total_mm: float
    cross_step_mm: float
    orientation_error_deg: float


def quaternion_angle_deg(q1: Sequence[float], q2: Sequence[float]) -> float:
    """Return the sign-invariant angular distance between xyzw quaternions."""
    qa = np.asarray(q1, dtype=np.float64)
    qb = np.asarray(q2, dtype=np.float64)
    na = float(np.linalg.norm(qa))
    nb = float(np.linalg.norm(qb))
    if na < 1e-12 or nb < 1e-12:
        return math.inf
    dot = min(1.0, abs(float(np.dot(qa / na, qb / nb))))
    return math.degrees(2.0 * math.acos(dot))


def make_down_target(origin: np.ndarray, step_index: int, step_mm: float) -> np.ndarray:
    """Build an absolute target with only base-frame Z changed."""
    target = np.asarray(origin, dtype=np.float64).copy()
    target[2] = float(origin[2]) - step_index * step_mm / 1000.0
    return target


def pose_metrics(
    measured: np.ndarray,
    target: np.ndarray,
    origin: np.ndarray,
    previous: np.ndarray,
) -> PoseMetrics:
    """Compute acceptance values in millimetres/degrees."""
    measured = np.asarray(measured, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    previous = np.asarray(previous, dtype=np.float64)
    target_error = (measured[:3] - target[:3]) * 1000.0
    from_origin = (measured[:3] - origin[:3]) * 1000.0
    from_previous = (measured[:3] - previous[:3]) * 1000.0
    return PoseMetrics(
        target_error_mm=target_error,
        from_origin_mm=from_origin,
        from_previous_mm=from_previous,
        down_from_origin_mm=-float(from_origin[2]),
        down_step_mm=-float(from_previous[2]),
        cross_total_mm=float(np.linalg.norm(from_origin[:2])),
        cross_step_mm=float(np.linalg.norm(from_previous[:2])),
        orientation_error_deg=quaternion_angle_deg(origin[3:7], measured[3:7]),
    )


def target_is_settled(metrics: PoseMetrics, limits: ProbeLimits) -> bool:
    """Check the instantaneous absolute-target tracking limits."""
    return bool(
        abs(float(metrics.target_error_mm[2])) <= limits.axis_tolerance_mm
        and abs(float(metrics.from_origin_mm[0])) <= limits.cross_axis_tolerance_mm
        and abs(float(metrics.from_origin_mm[1])) <= limits.cross_axis_tolerance_mm
        and metrics.orientation_error_deg <= limits.orientation_tolerance_deg
    )


def settled_step_passes(metrics: PoseMetrics, limits: ProbeLimits) -> bool:
    """Check that the newly achieved step was down-only and the right size."""
    return bool(
        abs(metrics.down_step_mm - limits.step_mm) <= limits.axis_tolerance_mm
        and abs(float(metrics.from_previous_mm[0])) <= limits.cross_axis_tolerance_mm
        and abs(float(metrics.from_previous_mm[1])) <= limits.cross_axis_tolerance_mm
        and abs(float(metrics.from_origin_mm[0])) <= limits.cross_axis_tolerance_mm
        and abs(float(metrics.from_origin_mm[1])) <= limits.cross_axis_tolerance_mm
        and metrics.orientation_error_deg <= limits.orientation_tolerance_deg
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Command one exact base-frame -Z step per R1+L2 press and record "
            "commanded/measured poses with strict pass/fail limits."
        )
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:5000")
    parser.add_argument("--joystick-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--step-mm",
        type=float,
        default=1.0,
        help="exact commanded distance per L2 press; constrained to 1..2 mm",
    )
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument(
        "--down-threshold",
        type=float,
        default=0.50,
        help="minimum decoded L2 down action needed to enable a step",
    )
    parser.add_argument("--axis-tolerance-mm", type=float, default=0.20)
    parser.add_argument("--cross-axis-tolerance-mm", type=float, default=0.20)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=0.10)
    parser.add_argument("--stable-samples", type=int, default=3)
    parser.add_argument(
        "--active-timeout-s",
        type=float,
        default=5.0,
        help="maximum R1+L2-held settling time for one step; pauses do not count",
    )
    parser.add_argument(
        "--other-axis-threshold",
        type=float,
        default=0.05,
        help="refuse to start/resume a step unless decoded non-Z axes are neutral",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="CSV path (default: diagnostics/controller_down_probe_TIMESTAMP.csv)",
    )
    parser.add_argument(
        "--confirm-right-arm",
        action="store_true",
        help="required acknowledgement that the HTTP server controls lbr_two",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.confirm_right_arm:
        raise ValueError(
            "Refusing hardware motion without --confirm-right-arm. Verify that the "
            "server was launched with SERL_ARM_PREFIX=lbr_two."
        )
    configured_arm = os.environ.get("SERL_ARM_PREFIX")
    if configured_arm not in (None, "lbr_two"):
        raise ValueError(
            f"SERL_ARM_PREFIX={configured_arm!r}, expected 'lbr_two' for this test."
        )
    if not 1.0 <= args.step_mm <= 2.0:
        raise ValueError("--step-mm must be between 1.0 and 2.0 mm")
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    if args.steps * args.step_mm > MAX_TOTAL_TRAVEL_MM:
        raise ValueError(
            f"total requested travel exceeds the {MAX_TOTAL_TRAVEL_MM:.0f} mm safety limit"
        )
    if args.hz <= 0.0 or args.active_timeout_s <= 0.0:
        raise ValueError("--hz and --active-timeout-s must be positive")
    if args.stable_samples < 1:
        raise ValueError("--stable-samples must be at least 1")
    if not 0.0 < args.down_threshold <= 1.0:
        raise ValueError("--down-threshold must be in (0, 1]")
    if not 0.0 <= args.other_axis_threshold <= 1.0:
        raise ValueError("--other-axis-threshold must be in [0, 1]")
    for name in (
        "axis_tolerance_mm",
        "cross_axis_tolerance_mm",
        "orientation_tolerance_deg",
    ):
        if getattr(args, name) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def _default_output_path() -> Path:
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path("diagnostics") / f"controller_down_probe_{stamp}.csv"


def _fieldnames() -> list[str]:
    fields = [
        "wall_time",
        "elapsed_s",
        "step_index",
        "command_index",
        "r1",
        "raw_l2",
        "raw_r2",
        "action_x",
        "action_y",
        "action_z",
        "action_rx",
        "action_ry",
        "action_rz",
    ]
    fields.extend(f"command_{name}" for name in POSE_NAMES)
    fields.extend(f"before_{name}" for name in POSE_NAMES)
    fields.extend(f"measured_{name}" for name in POSE_NAMES)
    fields.extend(f"joint_{i}" for i in range(1, 8))
    fields.extend(
        [
            "target_error_x_mm",
            "target_error_y_mm",
            "target_error_z_mm",
            "from_origin_x_mm",
            "from_origin_y_mm",
            "from_origin_z_mm",
            "step_x_mm",
            "step_y_mm",
            "step_z_mm",
            "down_from_origin_mm",
            "down_step_mm",
            "cross_total_mm",
            "cross_step_mm",
            "orientation_error_deg",
            "stable_samples",
            "result",
        ]
    )
    return fields


def _row(
    *,
    start_time: float,
    step_index: int,
    command_index: int,
    snapshot: dict[str, Any],
    action: np.ndarray,
    target: np.ndarray,
    before: Any,
    measured: Any,
    metrics: PoseMetrics,
    stable_samples: int,
    result: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "wall_time": dt.datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "elapsed_s": round(time.monotonic() - start_time, 6),
        "step_index": step_index,
        "command_index": command_index,
        "r1": int(bool(snapshot.get("r1", 0))),
        "raw_l2": snapshot.get("raw_l2", ""),
        "raw_r2": snapshot.get("raw_r2", ""),
        "stable_samples": stable_samples,
        "result": result,
    }
    for name, value in zip(("x", "y", "z", "rx", "ry", "rz"), action):
        row[f"action_{name}"] = float(value)
    for prefix, pose in (
        ("command", target),
        ("before", before.pose),
        ("measured", measured.pose),
    ):
        for name, value in zip(POSE_NAMES, pose):
            row[f"{prefix}_{name}"] = float(value)
    for index, value in enumerate(measured.q, start=1):
        row[f"joint_{index}"] = float(value)
    for prefix, values in (
        ("target_error", metrics.target_error_mm),
        ("from_origin", metrics.from_origin_mm),
        ("step", metrics.from_previous_mm),
    ):
        for name, value in zip(("x", "y", "z"), values):
            row[f"{prefix}_{name}_mm"] = float(value)
    row.update(
        down_from_origin_mm=metrics.down_from_origin_mm,
        down_step_mm=metrics.down_step_mm,
        cross_total_mm=metrics.cross_total_mm,
        cross_step_mm=metrics.cross_step_mm,
        orientation_error_deg=metrics.orientation_error_deg,
    )
    return row


def _controller_state(
    controller: "PS4TeleopProvider",
    down_threshold: float,
    other_axis_threshold: float,
) -> tuple[np.ndarray, dict[str, Any], bool, bool]:
    action = np.asarray(controller.get_action(), dtype=np.float64)
    snapshot = controller.debug_snapshot()
    r1 = bool(snapshot.get("r1", 0))
    other_axes = np.concatenate((action[:2], action[3:]))
    clean_down = bool(
        r1
        and action[2] <= -down_threshold
        and np.all(np.abs(other_axes) <= other_axis_threshold)
    )
    return action, snapshot, r1, clean_down


def run_probe(
    args: argparse.Namespace,
    *,
    client: RobotServerClient | None = None,
    controller: "PS4TeleopProvider | None" = None,
) -> Path:
    """Run the hardware probe and return its CSV path on success."""
    validate_args(args)
    limits = ProbeLimits(
        step_mm=float(args.step_mm),
        axis_tolerance_mm=float(args.axis_tolerance_mm),
        cross_axis_tolerance_mm=float(args.cross_axis_tolerance_mm),
        orientation_tolerance_deg=float(args.orientation_tolerance_deg),
    )
    client = client or RobotServerClient(args.server_url)
    if controller is None:
        from iiwa_serl.teleop import PS4TeleopProvider

        controller = PS4TeleopProvider(
            joystick_index=args.joystick_index,
            server_url=None,
        )
    output_path = Path(args.out) if args.out else _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    period = 1.0 / float(args.hz)
    origin: np.ndarray | None = None
    previous: np.ndarray | None = None
    command_index = 0
    start_time = time.monotonic()
    release_seen = False

    print(f"[DOWN TEST] CSV: {output_path.resolve()}")
    print(
        "[DOWN TEST] Verify lbr_two has clear space below the TCP. Release both "
        "triggers once to calibrate them."
    )
    print(
        f"[DOWN TEST] For each of {args.steps} steps: hold R1, press and KEEP "
        f"L2 held until PASS, then release L2. Each target is exactly "
        f"-{args.step_mm:.3f} mm in base Z. Sticks are ignored and must remain neutral."
    )
    print("[DOWN TEST] Releasing R1 sends HOLD immediately. Ctrl-C also sends HOLD.")

    try:
        client.hold_position()
        with output_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=_fieldnames())
            writer.writeheader()
            csv_file.flush()

            step_index = 1
            while step_index <= args.steps:
                # Require a released/neutral L2 sample before every new step.  A
                # trigger already held when the process starts cannot move the arm.
                print(f"[DOWN TEST] Step {step_index}/{args.steps}: release L2, then hold R1+L2.")
                while True:
                    loop_start = time.monotonic()
                    _, _, _, clean_down = _controller_state(
                        controller, args.down_threshold, args.other_axis_threshold
                    )
                    if not clean_down:
                        release_seen = True
                    elif release_seen:
                        break
                    time.sleep(max(0.0, period - (time.monotonic() - loop_start)))

                if origin is None:
                    origin = client.get_state().pose.copy()
                    previous = origin.copy()
                    print(
                        "[DOWN TEST] Frozen origin xyz="
                        f"{np.round(origin[:3], 6).tolist()}, quat="
                        f"{np.round(origin[3:7], 7).tolist()}"
                    )
                assert previous is not None
                target = make_down_target(origin, step_index, args.step_mm)
                # Structural guard: this is the defining invariant of the test.
                if not np.array_equal(target[[0, 1, 3, 4, 5, 6]], origin[[0, 1, 3, 4, 5, 6]]):
                    raise ProbeFailure("internal error: non-Z target component changed")

                stable = 0
                active_elapsed = 0.0
                paused = False
                while True:
                    loop_start = time.monotonic()
                    action, snapshot, r1, clean_down = _controller_state(
                        controller, args.down_threshold, args.other_axis_threshold
                    )
                    if not clean_down:
                        if not paused:
                            client.hold_position()
                            reason = "R1 released" if not r1 else "L2 released or another axis active"
                            print(f"[DOWN TEST] HOLD ({reason}); keep R1+L2 held to resume this step.")
                            paused = True
                        stable = 0
                        time.sleep(max(0.0, period - (time.monotonic() - loop_start)))
                        continue

                    paused = False
                    before = client.get_state()
                    client.move_pose(target)
                    command_index += 1
                    time.sleep(max(0.0, period - (time.monotonic() - loop_start)))
                    measured = client.get_state()
                    active_elapsed += time.monotonic() - loop_start
                    metrics = pose_metrics(measured.pose, target, origin, previous)
                    stable = stable + 1 if target_is_settled(metrics, limits) else 0

                    result = ""
                    if stable >= args.stable_samples:
                        result = "PASS" if settled_step_passes(metrics, limits) else "FAIL_STEP"
                    elif active_elapsed >= args.active_timeout_s:
                        result = "FAIL_TIMEOUT"

                    writer.writerow(
                        _row(
                            start_time=start_time,
                            step_index=step_index,
                            command_index=command_index,
                            snapshot=snapshot,
                            action=action,
                            target=target,
                            before=before,
                            measured=measured,
                            metrics=metrics,
                            stable_samples=stable,
                            result=result,
                        )
                    )
                    csv_file.flush()

                    if result == "FAIL_TIMEOUT":
                        raise ProbeFailure(
                            f"step {step_index} did not settle within {args.active_timeout_s:.1f} s; "
                            f"last target error={np.round(metrics.target_error_mm, 3).tolist()} mm, "
                            f"cross total={metrics.cross_total_mm:.3f} mm, "
                            f"orientation={metrics.orientation_error_deg:.3f} deg"
                        )
                    if result == "FAIL_STEP":
                        raise ProbeFailure(
                            f"step {step_index} settled but violated the down-only increment: "
                            f"delta={np.round(metrics.from_previous_mm, 3).tolist()} mm, "
                            f"orientation={metrics.orientation_error_deg:.3f} deg"
                        )
                    if result == "PASS":
                        client.hold_position()
                        print(
                            f"[DOWN TEST] PASS {step_index}: commanded -{args.step_mm:.3f} mm Z; "
                            f"measured step dx={metrics.from_previous_mm[0]:+.3f}, "
                            f"dy={metrics.from_previous_mm[1]:+.3f}, "
                            f"dz={metrics.from_previous_mm[2]:+.3f} mm; "
                            f"total XY={metrics.cross_total_mm:.3f} mm; "
                            f"orientation={metrics.orientation_error_deg:.3f} deg"
                        )
                        previous = measured.pose.copy()
                        step_index += 1
                        release_seen = False
                        break

        assert origin is not None and previous is not None
        final_metrics = pose_metrics(
            previous,
            make_down_target(origin, args.steps, args.step_mm),
            origin,
            origin,
        )
        print(
            f"[DOWN TEST] ALL {args.steps} STEPS PASSED. Total measured: "
            f"dx={final_metrics.from_origin_mm[0]:+.3f}, "
            f"dy={final_metrics.from_origin_mm[1]:+.3f}, "
            f"dz={final_metrics.from_origin_mm[2]:+.3f} mm."
        )
        print("[DOWN TEST] Arm is held at the final down pose; no return move was sent.")
        return output_path
    finally:
        try:
            client.hold_position()
        except Exception as exc:  # pragma: no cover - hardware/network failure path
            print(f"[DOWN TEST] WARNING: final HOLD request failed: {exc}", file=sys.stderr)
        controller.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_probe(args)
    except (ProbeFailure, ValueError) as exc:
        print(f"[DOWN TEST] FAIL: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[DOWN TEST] Interrupted; HOLD sent. No return move was sent.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
