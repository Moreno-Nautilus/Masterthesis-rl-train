"""Jog the FR3 with the PS4 while LOGGING everything to CSV, for tuning diagnosis.

Same control path as jog_and_capture.py (get_state -> stick delta -> send_pos), but every
control step appends a row so the commanded action, the commanded target and the MEASURED
response can be compared offline. Use this to answer:

  * does a pure-x stick command produce pure-x motion, or is it cross-coupled?
  * how far does the measured TCP lag the commanded target (tracking error)?
  * is the response oscillating (rumble) and at what frequency?
  * does the orientation drift while translating?

SAFETY: identical to jog_and_capture.py — motion only while R1 is held, lead-clamped,
E-stop is the real stop.

USAGE
    source /opt/ros/humble/setup.bash && source ~/fr3_ws/install/setup.bash
    source franka_env.sh
    export TELEOP_DEVICE=ps4 SDL_JOYSTICK_DEVICE=/dev/input/js0
    python3 .../jog_log.py [output.csv]

  Hold R1 + sticks and do DELIBERATE single-axis moves, e.g.
    1. push left stick +x only, ~2s, release
    2. push left stick +y only, ~2s, release
    3. hold L2 (down) ~2s, release; then R2 (up) ~2s, release
    4. right stick pitch only, ~2s
  Then Ctrl+C. Send me the CSV.
"""
import os
import sys
import time

import numpy as np

from franka_env.envs.robot_client import Ros2FrankaClient
from franka_env.spacemouse.ps4_expert import PS4Expert
from franka_env.utils.rotations import euler_2_quat, quat_2_euler

TRANS_STEP = 0.004
ROT_STEP = 0.025
RATE_HZ = 20.0
MAX_LEAD_M = 0.08
MAX_LEAD_RAD = 0.50

COLS = [
    "t",
    # what the operator commanded (post-deadzone stick vector) + deadman
    "act_x", "act_y", "act_z", "act_r", "act_p", "act_yaw", "r1",
    # the absolute target we published
    "tgt_x", "tgt_y", "tgt_z", "tgt_qx", "tgt_qy", "tgt_qz", "tgt_qw",
    # what the robot actually did
    "mes_x", "mes_y", "mes_z", "mes_qx", "mes_qy", "mes_qz", "mes_qw",
    # measured euler (convenience) + tcp velocity
    "mes_roll", "mes_pitch", "mes_yaw",
    "vel_x", "vel_y", "vel_z", "vel_r", "vel_p", "vel_yaw",
    # external wrench (contact / gravity signature)
    "fx", "fy", "fz", "tx", "ty", "tz",
    # joint angles (for jacobian / singularity analysis)
    "q0", "q1", "q2", "q3", "q4", "q5", "q6",
]


def _quat_mul(a, b):
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "diagnostics/jog_log.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    print("Connecting (read-only until R1 is held)...")
    robot = Ros2FrankaClient(config=None)
    expert = PS4Expert()

    st = robot.get_state()
    target = np.asarray(st["pose"], dtype=float).copy()
    print(f"  live TCP: {np.round(target[:3], 4)}")
    print("\nHOLD R1 + sticks. Do DELIBERATE single-axis moves (see the docstring).")
    print("Ctrl+C to stop and write the CSV.\n")

    rows = []
    dt = 1.0 / RATE_HZ
    t0 = time.monotonic()
    last_print = 0.0
    prev_target = target.copy()

    _last_warn = [0.0]

    def now_warn(period=2.0):
        """Rate-limit the lead-limit warnings so they don't spam the control loop."""
        t = time.monotonic()
        if t - _last_warn[0] > period:
            _last_warn[0] = t
            return True
        return False

    try:
        while True:
            loop_t = time.monotonic()
            st = robot.get_state()
            cur = np.asarray(st["pose"], dtype=float)

            a, _ = expert.get_action()
            a = np.asarray(a, dtype=float)
            held = bool(expert.deadman_held())

            if held:
                # snapshot so an over-limit request can be REFUSED wholesale (below)
                # without disturbing axes the operator did not command
                prev_target = target.copy()
                target[:3] = target[:3] + a[:3] * TRANS_STEP
                if np.linalg.norm(a[3:]) > 1e-6:
                    # Rotate in the TOOL frame, about the TCP — NOT the base frame.
                    # The FR3 tool points down (roll ~ +/-180 deg), so a base-frame delta
                    # appears INVERTED on roll/pitch (measured: cmd pitch+ gave pitch-),
                    # and rotating about the base origin swings the tip (measured: a 4.5
                    # deg pitch dragged 22.7 mm of z). Right-multiplying keeps the rotation
                    # body-fixed, so the TCP spins in place and the sign matches the stick.
                    dq = np.asarray(euler_2_quat(a[3:] * ROT_STEP), dtype=float)
                    q = _quat_mul(target[3:], dq)
                    target[3:] = q / np.linalg.norm(q)

                # Lead limit: REFUSE the increment rather than reprojecting the whole
                # target onto the measured pose. The old form
                #     target = cur + scaled(target - cur)
                # rewrote coordinates the operator never commanded: with a standing y
                # tracking error, a pure-x input moved the y TARGET too (~0.5mm/step).
                # Here an over-limit request is simply not applied, so uncommanded axes
                # and the orientation are preserved exactly.
                lead = target[:3] - cur[:3]
                if float(np.linalg.norm(lead)) > MAX_LEAD_M:
                    target[:3] = prev_target[:3].copy()
                    if now_warn():
                        print("\n  [jog] translation lead limit reached — arm is lagging; "
                              "release R1 and let it catch up.")
                dot = abs(float(np.dot(target[3:], cur[3:])))
                if 2.0 * np.arccos(np.clip(dot, -1.0, 1.0)) > MAX_LEAD_RAD:
                    target[3:] = prev_target[3:].copy()
                    if now_warn():
                        print("\n  [jog] rotation lead limit reached — request refused.")

            # publish every step (held or not) so the arm holds the last target on release
            robot.send_pos(target)

            eul = np.asarray(quat_2_euler(cur[3:]), dtype=float)
            rows.append([
                time.monotonic() - t0,
                a[0], a[1], a[2], a[3], a[4], a[5], 1.0 if held else 0.0,
                target[0], target[1], target[2], target[3], target[4], target[5], target[6],
                cur[0], cur[1], cur[2], cur[3], cur[4], cur[5], cur[6],
                eul[0], eul[1], eul[2],
                *[float(v) for v in st["vel"]],
                *[float(v) for v in st["force"]],
                *[float(v) for v in st["torque"]],
                *[float(v) for v in st["q"]],
            ])

            now = time.monotonic()
            if now - last_print > 0.25:
                err = np.linalg.norm(target[:3] - cur[:3]) * 1000.0
                sys.stdout.write(
                    "\r  n=%5d | xyz %+.4f %+.4f %+.4f | lag %5.1fmm | R1 %s  "
                    % (len(rows), cur[0], cur[1], cur[2], err, "HELD" if held else "--  "))
                sys.stdout.flush()
                last_print = now

            time.sleep(max(0.0, dt - (time.monotonic() - loop_t)))

    except KeyboardInterrupt:
        print("\n\nstopping...")
    finally:
        try:
            expert.close()
        except Exception:
            pass
        try:
            robot.close()
        except Exception:
            pass
        if rows:
            arr = np.array(rows, dtype=float)
            np.savetxt(out, arr, delimiter=",", header=",".join(COLS), comments="", fmt="%.6f")
            print(f"wrote {len(rows)} rows -> {out}")
        else:
            print("no rows recorded")


if __name__ == "__main__":
    main()
