"""Standalone PS4 teleoperation of the iiwa via the SERL HTTP robot server.

No gym env, no RL — just direct HTTP pose commands. Use this to:
  - Test that the backend + server are working before training
  - Get comfortable with the controller mapping and workspace limits
  - Manually position the robot for debugging

Prerequisites
-------------
  python -m iiwa_serl.robot_servers.iiwa_server --backend="your_module:RealIiwaBackend"

Usage
-----
  python teleop_drive.py                      # real robot at localhost:5000
  python teleop_drive.py --mock               # mock backend (no robot, state printed)
  python teleop_drive.py --server_url http://192.168.1.100:5000
"""

import argparse
import time

import numpy as np

from iiwa_serl.config import IiwaInsertionConfig
from iiwa_serl.robot.api import LocalBackendClient, RobotServerClient
from iiwa_serl.robot.backends import MockIiwaBackend
from iiwa_serl.teleop import PS4TeleopProvider
from iiwa_serl.teleop.pose_target import TeleopPoseTarget
from iiwa_serl.utils.transformations import clip_pose7


def main():
    parser = argparse.ArgumentParser(description="PS4 teleoperation drive for iiwa via SERL HTTP.")
    parser.add_argument("--server_url", default="http://127.0.0.1:5000")
    parser.add_argument("--mock", action="store_true", help="use a local mock backend (no server needed)")
    parser.add_argument("--hz", type=int, default=20, help="control loop rate")
    parser.add_argument("--linear_scale", type=float, default=0.015, help="m per action unit per step")
    parser.add_argument("--angular_scale", type=float, default=0.12, help="rad per action unit per step")
    parser.add_argument("--joystick_index", type=int, default=0)
    parser.add_argument("--force_feedback", action="store_true")
    parser.add_argument("--no_clip", action="store_true",
                        help="disable workspace clipping (limits in config are for a different arm/home)")
    parser.add_argument("--ee_frame", action="store_true",
                        help="EE/tool-frame linear motion (SERL default). Omit for intuitive base-frame teleop.")
    parser.add_argument("--joint7_scale", type=float, default=0.03,
                        help="rad per d-pad step for direct joint-7 rotation")
    parser.add_argument("--debug_actions", action="store_true", help="print each nonzero action")
    args = parser.parse_args()

    cfg = IiwaInsertionConfig()

    if args.mock:
        backend = MockIiwaBackend(
            initial_pose6=cfg.reset_pose.tolist(),
            reset_pose6=cfg.reset_pose.tolist(),
            reset_joints=cfg.reset_joints.tolist(),
        )
        client = LocalBackendClient(backend)
        print("[DRIVE] Using mock backend — no real robot")
    else:
        client = RobotServerClient(args.server_url)
        print(f"[DRIVE] Connected to robot server at {args.server_url}")

    teleop = PS4TeleopProvider(
        joystick_index=args.joystick_index,
        server_url=None if args.mock else args.server_url,
        force_feedback=args.force_feedback and not args.mock,
    )

    print("\n[DRIVE] Controls:")
    print("  R1 (hold)     → enable motion (deadman)")
    print("  Left stick    → linear X/Y")
    print("  L2 / R2       → linear Z (down / up)")
    print("  Right stick   → angular pitch / yaw")
    print("  D-pad L/R     → angular roll")
    print("  Circle (○)    → open gripper")
    print("  Square (□)    → close gripper")
    print("  Cross  (✕)    → print current pose")
    print("  Triangle (△)  → reset to home pose")
    print("  Ctrl-C        → quit\n")

    dt = 1.0 / args.hz
    client.hold_position()
    target_state = TeleopPoseTarget(
        args.linear_scale,
        args.angular_scale,
        base_frame_actions=not args.ee_frame,
    )
    target_state.reset(client.get_state().pose)
    try:
        while True:
            t0 = time.monotonic()

            action = teleop.get_action()
            if args.debug_actions and np.any(np.abs(action) > 1e-4):
                print(f"[DBG] action={np.round(action,3).tolist()}")
            state = client.get_state()

            if teleop.is_success():
                print(f"[DRIVE] Current pose: {np.round(state.pose, 4).tolist()}")

            if teleop.is_failure():
                print("[DRIVE] Resetting to home pose ...")
                client.joint_reset()
                state = client.get_state()
                client.hold_position()
                target_state.reset(state.pose)
                action = np.zeros(6, dtype=np.float32)

            command = target_state.update(action, state.pose)
            if command.kind == "move":
                target = command.target_pose
                if not args.no_clip:
                    target = clip_pose7(target, cfg.abs_pose_limit_low, cfg.abs_pose_limit_high)
                    target_state.sync_target(target)
                client.move_pose(target)
            elif command.kind == "hold":
                client.hold_position()

            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, dt - elapsed))

    except KeyboardInterrupt:
        print("\n[DRIVE] Stopped.")
    finally:
        teleop.close()


if __name__ == "__main__":
    main()
