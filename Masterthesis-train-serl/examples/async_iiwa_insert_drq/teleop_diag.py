"""Teleop diagnostic logger — records what the arm actually does per command so we can
diagnose the IK/nullspace behaviour offline.

Same drive loop as teleop_drive.py (base-frame linear, d-pad->A7 direct), but every step it
logs a CSV row: timestamp, the decoded action, the commanded target pose, and the RESULTING
robot state (pose7 + all 7 joint angles) read back AFTER the command. That lets us see, for a
pure X or pure tilt command, exactly which joints moved and whether the TCP pose held.

Usage (server + bringup + upsampler already running):
  export SDL_JOYSTICK_DEVICE=/dev/input/js1
  conda run -n serl --no-capture-output python teleop_diag.py --out /home/moreno/Masterthesis-rl-train/diagnostics/teleop_log.csv

Then do ONE clean motion at a time (hold R1):
  1. only left-stick UP (pure +X) for ~2s, release
  2. only left-stick SIDE (pure Y) for ~2s, release
  3. only right-stick UP (pure tilt) for ~2s, release
  4. only d-pad L/R (joint 7) for ~2s, release
Ctrl-C to stop. Send me the CSV.
"""
import argparse
import csv
import os
import time

import numpy as np

from iiwa_serl.config import IiwaInsertionConfig
from iiwa_serl.robot.api import RobotServerClient
from iiwa_serl.teleop import PS4TeleopProvider
from iiwa_serl.utils.transformations import compose_delta_pose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server_url", default="http://127.0.0.1:5000")
    ap.add_argument("--hz", type=int, default=20)
    ap.add_argument("--linear_scale", type=float, default=0.004)
    ap.add_argument("--angular_scale", type=float, default=0.03)
    ap.add_argument("--joint7_scale", type=float, default=0.03)
    ap.add_argument("--joystick_index", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))),
        "diagnostics", "teleop_log.csv"))
    args = ap.parse_args()

    cfg = IiwaInsertionConfig()
    client = RobotServerClient(args.server_url)
    teleop = PS4TeleopProvider(joystick_index=args.joystick_index, server_url=None)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    f = open(args.out, "w", newline="")
    w = csv.writer(f)
    w.writerow(
        ["t", "ax", "ay", "az", "arx", "ary", "arz",
         "tgt_x", "tgt_y", "tgt_z", "tgt_qx", "tgt_qy", "tgt_qz", "tgt_qw",
         "pose_x", "pose_y", "pose_z", "pose_qx", "pose_qy", "pose_qz", "pose_qw",
         "q1", "q2", "q3", "q4", "q5", "q6", "q7"]
    )

    print(f"[DIAG] logging to {args.out}")
    print("[DIAG] Hold R1. Do ONE clean motion at a time (X, then Y, then tilt, then d-pad). Ctrl-C to stop.")

    dt = 1.0 / args.hz
    t0 = time.time()
    try:
        while True:
            loop_t = time.time()
            action = teleop.get_action()
            state = client.get_state()

            tgt = state.pose.copy()
            drz = action[5]
            if abs(drz) > 1e-6:
                dq7 = np.zeros(7); dq7[6] = drz * args.joint7_scale
                client.move_joint_delta(dq7)

            rot_cart = action[3:6].copy(); rot_cart[2] = 0.0
            if np.any(action[:3] != 0.0) or np.any(rot_cart != 0.0):
                tgt = compose_delta_pose(state.pose, delta_xyz=np.zeros(3),
                                         delta_rot_xyz=rot_cart * args.angular_scale)
                tgt[:3] = state.pose[:3] + action[:3] * args.linear_scale
                client.move_pose(tgt)

            # log only when something is commanded
            if np.any(np.abs(action) > 1e-4):
                w.writerow(
                    [round(loop_t - t0, 3)] + [round(float(x), 5) for x in action]
                    + [round(float(x), 5) for x in tgt]
                    + [round(float(x), 5) for x in state.pose]
                    + [round(float(x), 5) for x in state.q]
                )
                f.flush()

            time.sleep(max(0.0, dt - (time.time() - loop_t)))
    except KeyboardInterrupt:
        print("\n[DIAG] stopped.")
    finally:
        f.close()
        teleop.close()


if __name__ == "__main__":
    main()
