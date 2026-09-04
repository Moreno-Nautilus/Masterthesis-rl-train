"""Direct IK probe — NO teleop, NO controller. Sends a scripted pure +X Cartesian target to
the backend and measures what the arm actually does. Isolates the backend IK from all teleop logic.

Reads current pose, then commands target = current with ONLY x += step, orientation IDENTICAL,
repeatedly for a few cm. Logs commanded target vs achieved pose each step.

Usage (server running):
  python ik_probe.py --axis x --dist 0.05
  python ik_probe.py --axis z --dist 0.03
"""
import argparse
import time
import numpy as np
from iiwa_serl.robot.api import RobotServerClient

AX = {"x": 0, "y": 1, "z": 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server_url", default="http://127.0.0.1:5000")
    ap.add_argument("--axis", default="x", choices=["x", "y", "z"])
    ap.add_argument("--dist", type=float, default=0.05, help="total metres to move")
    ap.add_argument("--step", type=float, default=0.003, help="metres per command")
    ap.add_argument("--hz", type=float, default=10.0)
    args = ap.parse_args()

    c = RobotServerClient(args.server_url)
    start = c.get_state().pose.copy()
    print(f"[PROBE] start pose = {np.round(start,4).tolist()}")
    print(f"[PROBE] commanding pure +{args.axis} {args.dist} m in {args.step} m steps.")
    print("        target orientation is FROZEN = start orientation.\n")

    ai = AX[args.axis]
    n = int(abs(args.dist) / args.step)
    target = start.copy()
    dt = 1.0 / args.hz
    print("step | commanded target(xyz) | achieved pose(xyz) | achieved quat(xyzw) | dpos from start(xyz)")
    for i in range(n):
        target[ai] += args.step
        # orientation stays exactly start[3:7] — never recomputed
        c.move_pose(target)
        time.sleep(dt)
        p = c.get_state().pose
        dp = p[:3] - start[:3]
        print(
            f"{i:3d} | tgt=[{target[0]:.3f},{target[1]:.3f},{target[2]:.3f}]"
            f" | pos=[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}]"
            f" | quat=[{p[3]:.3f},{p[4]:.3f},{p[5]:.3f},{p[6]:.3f}]"
            f" | dpos=[{dp[0]:+.3f},{dp[1]:+.3f},{dp[2]:+.3f}]"
        )
    print(f"\n[PROBE] start quat = {np.round(start[3:7],4).tolist()}")
    print(f"[PROBE] end   quat = {np.round(c.get_state().pose[3:7],4).tolist()}  (want SAME as start)")


if __name__ == "__main__":
    main()
