"""IK probe 2 — measure whether ORIENTATION error dominates translation in the DLS IK.

Commands a pure single-axis position target (orientation frozen = start) and, each step, reads
back the achieved pose and reports:
  - pos_err   : |target_pos - achieved_pos|      (how far position tracking lags)
  - ori_err   : angle between target quat and achieved quat (deg)  (orientation mismatch)
  - tilt      : angle between START quat and achieved quat (deg)   (how much the EE tilted)
If ori_err/tilt grow while you command pure translation, orientation is fighting the IK.

Run at a chosen joint-7 angle (jog the wrist first), then:
  python ik_probe2.py --axis y --dist 0.05
Repeat at a DIFFERENT joint-7 angle to see if the corruption depends on wrist angle.
"""
import argparse
import time
import numpy as np
from iiwa_serl.robot.api import RobotServerClient

AX = {"x": 0, "y": 1, "z": 2}


def quat_angle_deg(qa, qb):
    # angle between two xyzw quaternions, degrees
    qa = qa / np.linalg.norm(qa)
    qb = qb / np.linalg.norm(qb)
    d = abs(float(np.dot(qa, qb)))
    d = min(1.0, d)
    return np.degrees(2.0 * np.arccos(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server_url", default="http://127.0.0.1:5000")
    ap.add_argument("--axis", default="y", choices=["x", "y", "z"])
    ap.add_argument("--dist", type=float, default=0.05)
    ap.add_argument("--step", type=float, default=0.003)
    ap.add_argument("--hz", type=float, default=10.0)
    args = ap.parse_args()

    c = RobotServerClient(args.server_url)
    start = c.get_state().pose.copy()
    q_start = start[3:7].copy()
    print(f"[PROBE2] start pose = {np.round(start,4).tolist()}")
    print(f"[PROBE2] joint7 = {np.round(np.degrees(c.get_state().q[6]),1)} deg")
    print(f"[PROBE2] pure +{args.axis} {args.dist} m, orientation FROZEN.\n")

    ai = AX[args.axis]
    n = int(abs(args.dist) / args.step)
    target = start.copy()
    dt = 1.0 / args.hz
    print("step | tgt_axis achieved_axis | pos_err(mm) | ori_err_vs_tgt(deg) | tilt_vs_start(deg) | dz(mm)")
    for i in range(n):
        target[ai] += args.step
        c.move_pose(target)
        time.sleep(dt)
        p = c.get_state().pose
        pos_err = np.linalg.norm(target[:3] - p[:3]) * 1000
        ori_err = quat_angle_deg(target[3:7], p[3:7])
        tilt = quat_angle_deg(q_start, p[3:7])
        dz = (p[2] - start[2]) * 1000
        print(f"{i:3d} | tgt={target[ai]:.3f} got={p[ai]:.3f} | {pos_err:5.1f} | {ori_err:5.2f} | {tilt:5.2f} | {dz:+6.1f}")


if __name__ == "__main__":
    main()
