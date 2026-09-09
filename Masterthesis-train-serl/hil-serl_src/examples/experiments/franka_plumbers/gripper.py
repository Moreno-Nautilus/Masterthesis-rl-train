"""Open / close the Franka Hand from the command line.

    source /opt/ros/humble/setup.bash && source ~/fr3_ws/install/setup.bash
    source franka_env.sh
    python3 .../gripper.py open      # release
    python3 .../gripper.py close     # grasp
    python3 .../gripper.py           # just report the current width

Uses the same Ros2FrankaClient the env uses, so it exercises the real
Move/Grasp action path (/franka_gripper/{move,grasp}).
"""
import sys

from franka_env.envs.robot_client import Ros2FrankaClient


def main():
    what = (sys.argv[1] if len(sys.argv) > 1 else "status").lower()
    if what not in ("open", "close", "status"):
        print(f"usage: {sys.argv[0]} [open|close|status]")
        return 2

    c = Ros2FrankaClient(config=None)
    try:
        before = c.get_state()["gripper_pos"][0]
        if what == "status":
            print(f"gripper width = {before:.3f}  (0.0 = closed, 1.0 = fully open)")
            return 0

        print(f"gripper {what}ing (was {before:.3f}) ...")
        if what == "open":
            c.open_gripper()
        else:
            c.close_gripper()
        after = c.get_state()["gripper_pos"][0]
        print(f"gripper {what}ed: {before:.3f} -> {after:.3f}")
    finally:
        c.close()   # NOTE: closes the ROS client, NOT the gripper
    return 0


if __name__ == "__main__":
    sys.exit(main())
