"""Jog the FR3 with the PS4 and capture poses — WITHOUT constructing the env.

Why this exists: RESET_POSE / ABS_POSE_LIMIT_* ship as ZEROS, so calling env.reset()
before they are filled would drive the arm toward the base-frame ORIGIN. This tool talks
straight to Ros2FrankaClient (get_state + send_pos), so nothing commands a placeholder pose.

SAFETY
  * Motion ONLY while R1 (deadman) is held. Release -> the target is re-latched to the
    CURRENT pose, so the arm stops where it is.
  * Per-step translation/rotation are clamped (small), and the commanded target can never
    run more than MAX_LEAD_M ahead of the measured TCP -> no runaway if a stick sticks.
  * The E-STOP is still the real stop.

USAGE
    source /opt/ros/humble/setup.bash && source ~/fr3_ws/install/setup.bash
    source franka_env.sh
    export TELEOP_DEVICE=ps4 SDL_JOYSTICK_DEVICE=/dev/input/js0
    python3 .../jog_and_capture.py

  Hold R1 + sticks to jog.  Press:
    X        -> capture the current pose as RESET_POSE   (pre-insert)
    TRIANGLE -> capture the current pose as TARGET_POSE  (part seated)
    CIRCLE   -> print the config block for everything captured so far
  Ctrl+C to quit (prints the block too).
"""
import os
import sys
import time

import numpy as np

from franka_env.envs.robot_client import Ros2FrankaClient
from franka_env.spacemouse.ps4_expert import PS4Expert
from franka_env.utils.rotations import euler_2_quat, quat_2_euler

# --- jog tuning (deliberately small; this is hand-guiding, not the RL loop) ----------
# Reduced at the rig 2026-09-09: the arm was overshooting on every input and rotations
# were far too coarse to place a part. 1 mm/step at 20 Hz = 2 cm/s at full stick.
TRANS_STEP = 0.001      # m per control step at full stick   (was 0.004)
# 0.003 rad = 0.17 deg/step = ~3.4 deg/s at full stick. Rotations need to be MUCH finer
# than translation: a few degrees of attitude error is the difference between a part
# seating and jamming.
ROT_STEP = 0.003        # rad per control step at full stick  (was 0.025)
RATE_HZ = 20.0

# SINGLE-AXIS LOCK: while an axis is being commanded, the others are held EXACTLY.
# The operator asked for pure x / pure y / pure z moves that do not drift on the other
# axes and do not rotate at all. Rather than rely on the controller's decoupling (which
# leaks a few percent, and up to ~6 deg of pitch during fast translation), the jog target
# itself refuses to change the uncommanded coordinates.
SINGLE_AXIS_LOCK = True
# A stick axis counts as "commanded" above this (post-deadzone) magnitude.
AXIS_ACTIVE = 0.05

# INSERTION-AXIS FORCE CAP: refuse further motion INTO the socket once the measured
# contact force exceeds this. Protects the part/socket while hand-jogging into contact.
# Motion OUT is always allowed so the operator can always retreat.
#
# Which axis is "into the socket" depends on the insert, so set it here to match the
# EnvConfig you are capturing for (INS_SIGN mirrors RETRACT_DIR: OUT of the socket).
#   insert 0 (vertical descent): INS_AXIS = 2, INS_SIGN = +1.0   -> guards -Z
#   insert 1 (horizontal, -Y)  : INS_AXIS = 1, INS_SIGN = -1.0   -> guards +Y
#   insert 2 (cooling base, vertical descent): INS_AXIS = 2, INS_SIGN = +1.0
INS_AXIS = 2
INS_SIGN = +1.0
INS_NAME = "XYZ"[INS_AXIS]
INSERT_FORCE_LIMIT_N = 15.0
Z_FORCE_LIMIT_N = INSERT_FORCE_LIMIT_N  # legacy alias

# FR3 per-joint torque limits — joints 5-7 (wrist) have only a 12 Nm budget and are what
# actually trips the robot's joint-torque reflex. Shown live in the status line.
_JOINT_TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
# The commanded target legitimately leads the measured TCP under a compliant controller
# (that lead IS the spring force). Clamp only enough to stop a runaway, not enough to fight
# normal tracking lag — 3cm was too tight and made z jogging feel like it sprang back.
MAX_LEAD_M = 0.08
MAX_LEAD_RAD = 0.50


def _quat_mul(a, b):
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def _euler_to_quat_xyz(rpy):
    """Delegate to the repo's converter so the jog uses the SAME Euler convention as the
    env/config (a hand-rolled version disagreed on axis order — verified by test)."""
    return np.asarray(euler_2_quat(np.asarray(rpy, dtype=float)), dtype=float)


def _fmt_pose(p7):
    """[x,y,z,qx,qy,qz,qw] -> the 6-value [x,y,z,r,p,y] the config wants."""
    eul = quat_2_euler(np.asarray(p7[3:]))
    return np.concatenate([np.asarray(p7[:3]), np.asarray(eul)])


def _print_block(reset_pose, target_pose):
    print("\n" + "=" * 74)
    print("PASTE INTO experiments/franka_plumbers/config.py -> EnvConfigInsert0")
    print("=" * 74)
    if reset_pose is None or target_pose is None:
        print("  (need BOTH: X = RESET_POSE (pre-insert), TRIANGLE = TARGET_POSE (seated))")
        if reset_pose is not None:
            print("  RESET_POSE  = np.array([%s])" % ", ".join(f"{v:.5f}" for v in reset_pose))
        if target_pose is not None:
            print("  TARGET_POSE = np.array([%s])" % ", ".join(f"{v:.5f}" for v in target_pose))
        print("=" * 74)
        return

    d = reset_pose[:3] - target_pose[:3]
    n = float(np.linalg.norm(d))
    retract = d / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])

    lo = np.minimum(reset_pose[:3], target_pose[:3])
    hi = np.maximum(reset_pose[:3], target_pose[:3])
    # clearance for the 4cm reset / 8cm regrasp retraction, plus jitter margin
    pad_out = 0.10 * np.abs(retract)
    pad = np.full(3, 0.05)
    lo_b = lo - pad - pad_out
    hi_b = hi + pad + pad_out

    def arr(v, prec=5):
        return "np.array([%s])" % ", ".join(f"{x:.{prec}f}" for x in v)

    print(f"    TARGET_POSE = {arr(target_pose)}")
    print(f"    RESET_POSE  = {arr(reset_pose)}")
    print(f"    # retract dir = normalize(RESET[:3] - TARGET[:3]); |sep| = {n*1000:.1f} mm")
    print(f"    RETRACT_DIR = {arr(retract, 4)}")
    print(f"    ABS_POSE_LIMIT_LOW  = {arr(lo_b, 4)}")
    print(f"    ABS_POSE_LIMIT_HIGH = {arr(hi_b, 4)}")
    print("=" * 74)
    if n < 0.005:
        print("  !! WARNING: RESET and TARGET are <5mm apart — did you capture two DIFFERENT poses?")
    print("  CHECK the safety box by hand before the first env.reset(): it must cover the")
    print("  full 4cm reset / 8cm regrasp retraction with real physical clearance.")


def main():
    print("Connecting to the robot (read-only until you hold R1)...")
    robot = Ros2FrankaClient(config=None)
    expert = PS4Expert()

    st = robot.get_state()
    target = np.asarray(st["pose"], dtype=float).copy()
    print(f"  live TCP: {np.round(target[:3], 4)}  |q|={np.linalg.norm(target[3:]):.4f}")
    print("\nHOLD R1 + sticks to jog.  X=RESET_POSE  TRIANGLE=TARGET_POSE  CIRCLE=print")
    print("Release R1 to freeze. Ctrl+C to quit.\n")

    reset_pose = None
    target_pose = None
    dt = 1.0 / RATE_HZ
    last_print = 0.0
    prev_target = target.copy()
    warned_triggers = False
    prev_close = prev_open = 0

    _last_warn = [0.0]

    def now_warn(period=2.0):
        """Rate-limit warnings so they cannot spam the 20 Hz control loop."""
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

            ev = expert.get_events()
            if ev.get("success"):
                reset_pose = _fmt_pose(cur)
                print(f"\n  [X] RESET_POSE captured: {np.round(reset_pose, 5)}\n")
            if ev.get("abort"):
                target_pose = _fmt_pose(cur)
                print(f"\n  [TRIANGLE] TARGET_POSE captured: {np.round(target_pose, 5)}\n")
            if ev.get("regrasp"):
                _print_block(reset_pose, target_pose)

            a, btns = expert.get_action()
            held = expert.deadman_held()

            # GRIPPER: SQUARE = close (grasp the part), CIRCLE-adjacent OPEN button = open.
            # Edge-triggered so one press = one action; the gripper actions block briefly.
            close_now, open_now = int(btns[0]), int(btns[1])
            if close_now and not prev_close:
                print("\n  [gripper] CLOSING...", flush=True)
                robot.close_gripper()
                print("  [gripper] closed", flush=True)
            if open_now and not prev_open:
                print("\n  [gripper] OPENING...", flush=True)
                robot.open_gripper()
                print("  [gripper] open", flush=True)
            prev_close, prev_open = close_now, open_now

            # z is suppressed until both analog triggers have reported their rest value
            # once (otherwise an untouched trigger reads half-pressed and injects phantom
            # z). Tell the operator exactly what to do, once.
            if not expert.triggers_ready() and not warned_triggers:
                print("\n  [jog] SQUEEZE AND RELEASE BOTH L2 AND R2 once to initialise the\n"
                      "        triggers — up/down (z) is disabled until then.\n")
                warned_triggers = True

            if not held:
                # FREEZE: hold the LAST COMMANDED target; do NOT re-latch to the measured
                # pose. Re-latching made commanded motion spring back on release — under a
                # compliant controller the arm always lags the target (that lag IS the
                # spring force, and gravity makes it worst on z), so snapping the target
                # back to `cur` threw away exactly the motion you had just commanded.
                robot.send_pos(target)
            else:
                a = np.asarray(a, dtype=float)
                # snapshot so an over-limit request can be refused wholesale, without
                # disturbing axes the operator did not command
                prev_target = target.copy()

                # ---- SINGLE-AXIS LOCK -------------------------------------------------
                # Keep only the DOMINANT translation axis and zero the rest, so "only x"
                # really means only x. Rotation is dropped entirely whenever translation
                # is being commanded (and vice versa) — mixed inputs are what produced the
                # coupled motion the operator kept seeing.
                if SINGLE_AXIS_LOCK:
                    lin, rot = a[:3].copy(), a[3:].copy()
                    lin_max, rot_max = np.abs(lin).max(), np.abs(rot).max()
                    if lin_max >= AXIS_ACTIVE and lin_max >= rot_max:
                        keep = int(np.argmax(np.abs(lin)))
                        a = np.zeros(6)
                        a[keep] = lin[keep]              # pure translation on one axis
                    elif rot_max >= AXIS_ACTIVE:
                        keep = int(np.argmax(np.abs(rot)))
                        a = np.zeros(6)
                        a[3 + keep] = rot[keep]          # pure rotation about one axis
                    else:
                        a = np.zeros(6)

                # ---- INSERTION-AXIS FORCE CAP -----------------------------------------
                # Refuse further motion INTO the socket once contact force exceeds the
                # limit; motion OUT stays available so the operator can always back out.
                # 2026-09-10: was hardcoded to -z (correct only for a VERTICAL insert).
                # Insert 1 approaches along -Y, so the Y contact force was UNGUARDED here
                # while the Z guard watched an axis nothing was pushing on.
                f_ins = float(st["force"][INS_AXIS])
                if a[INS_AXIS] * INS_SIGN < 0.0 and abs(f_ins) > INSERT_FORCE_LIMIT_N:
                    a[INS_AXIS] = 0.0
                    if now_warn():
                        print(f"\n  [jog] {INS_NAME} BLOCKED — contact force {f_ins:+.1f} N "
                              f"exceeds {INSERT_FORCE_LIMIT_N:.0f} N. Back out to release.")

                target[:3] = target[:3] + a[:3] * TRANS_STEP
                # Only touch the orientation target when rotation is ACTUALLY commanded.
                # (Deadzone output is exactly 0.0 when idle, so this is a hard gate: pure
                # translation can never accumulate orientation drift through this path.)
                if np.linalg.norm(a[3:]) > 1e-6:
                    # Rotate in the TOOL frame, about the TCP — NOT the base frame.
                    # The FR3 tool points down (roll ~ +/-180 deg), so a base-frame delta
                    # appears INVERTED on roll/pitch, and rotating about the base origin
                    # swings the tip (measured: 4.5 deg pitch dragged 22.7 mm of z).
                    # Right-multiplying keeps the rotation body-fixed.
                    dq = _euler_to_quat_xyz(a[3:] * ROT_STEP)
                    q = _quat_mul(target[3:], dq)
                    target[3:] = q / np.linalg.norm(q)

                # Lead limit: REFUSE the increment rather than reprojecting the whole
                # target onto the measured pose. The old form
                #     target = cur + scaled(target - cur)
                # rewrote coordinates the operator never commanded: with a standing y
                # tracking error, a pure-x input also moved the y TARGET. Refusing keeps
                # uncommanded axes and the orientation untouched.
                # Lead limit. NOTE: reverting to prev_target DEADLOCKS — once the arm lags
                # past the limit, prev_target is over the limit too, so every later frame
                # reverts again, the target freezes forever and the arm stops responding
                # while the controller still reports "active" (rig-observed 2026-09-09).
                # Instead: only REFUSE THE NEW INCREMENT if it makes the lead worse. Motion
                # that reduces the lead is always allowed, so it self-heals.
                lead_new = float(np.linalg.norm(target[:3] - cur[:3]))
                lead_old = float(np.linalg.norm(prev_target[:3] - cur[:3]))
                if lead_new > MAX_LEAD_M and lead_new > lead_old:
                    target[:3] = prev_target[:3].copy()
                ang_new = 2.0 * np.arccos(np.clip(abs(float(np.dot(target[3:], cur[3:]))), -1.0, 1.0))
                ang_old = 2.0 * np.arccos(np.clip(abs(float(np.dot(prev_target[3:], cur[3:]))), -1.0, 1.0))
                if ang_new > MAX_LEAD_RAD and ang_new > ang_old:
                    target[3:] = prev_target[3:].copy()

                robot.send_pos(target)

            now = time.monotonic()
            if now - last_print > 0.25:
                e = _fmt_pose(cur)
                f = st["force"]
                grip = st.get("gripper_pos", [0.0])[0]
                # Wrist-torque utilisation: joints 5-7 have only a 12 Nm budget and are
                # what actually trips the FR3's joint-torque reflex (the EE wrench does
                # NOT capture it). FrankaEnv's torque guard fires at 40% — showing the
                # live number here tells the operator whether the torque guard or the
                # force cap is the thing refusing to let the part go in.
                tau_j = st.get("tau_j", None)
                if tau_j is not None and len(tau_j) == 7:
                    _u = np.abs(np.asarray(tau_j, dtype=float)) / _JOINT_TORQUE_LIMITS
                    _j = int(np.argmax(_u))
                    tau_s = "j%d %3.0f%%" % (_j + 1, 100.0 * _u[_j])
                else:
                    tau_s = "tau n/a"
                sys.stdout.write(
                    "\r  xyz %+.4f %+.4f %+.4f | rpy %+.3f %+.3f %+.3f | Fx%+5.1f Fy%+5.1f Fz%+5.1f | %s | grip %.2f | R1 %s | [%s%s]   "
                    % (e[0], e[1], e[2], e[3], e[4], e[5], f[0], f[1], f[2], tau_s, grip,
                       "HELD" if held else "--  ",
                       "R" if reset_pose is not None else "-",
                       "T" if target_pose is not None else "-"))
                sys.stdout.flush()
                last_print = now

            time.sleep(max(0.0, dt - (time.monotonic() - loop_t)))

    except KeyboardInterrupt:
        print("\n\ninterrupted.")
    finally:
        _print_block(reset_pose, target_pose)
        try:
            expert.close()
        except Exception:
            pass
        try:
            robot.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
