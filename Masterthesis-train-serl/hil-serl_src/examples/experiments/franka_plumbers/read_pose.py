"""Print the live TCP pose as a config-ready line. Read-only: commands NO motion.

Use after re-fixturing the base: hand-guide the arm (it is compliant) to the new
pre-insert pose, run this, paste the printed RESET_POSE into config.py.

    source /opt/ros/humble/setup.bash && source ~/fr3_ws/install/setup.bash
    source franka_env.sh
    python3 .../read_pose.py              # one shot
    python3 .../read_pose.py --watch      # live, Ctrl+C to stop
"""
import sys
import time

import numpy as np

from franka_env.envs.robot_client import Ros2FrankaClient
from franka_env.utils.rotations import quat_2_euler


def line(p7):
    e = np.concatenate([np.asarray(p7[:3]), np.asarray(quat_2_euler(np.asarray(p7[3:])))])
    return e, "np.array([%s])" % ", ".join(f"{v:.5f}" for v in e)


def main():
    watch = "--watch" in sys.argv
    c = Ros2FrankaClient(config=None)
    try:
        if watch:
            print("live pose (Ctrl+C to stop)\n")
            while True:
                st = c.get_state()
                e, s = line(st["pose"])
                f = st["force"]
                sys.stdout.write(
                    "\r  xyz %+.4f %+.4f %+.4f | rpy %+.3f %+.3f %+.3f | F %+5.1f %+5.1f %+5.1f  "
                    % (e[0], e[1], e[2], e[3], e[4], e[5], f[0], f[1], f[2]))
                sys.stdout.flush()
                time.sleep(0.1)
        else:
            st = c.get_state()
            e, s = line(st["pose"])
            print("\n  live TCP pose:")
            print(f"    xyz  {np.round(e[:3], 4)}")
            print(f"    rpy  {np.round(e[3:], 4)}  rad")
            print(f"    grip {st['gripper_pos'][0]:.3f}   force {np.round(st['force'], 1)}")
            print("\n  paste into EnvConfigInsert0:")
            print(f"    RESET_POSE  = {s}")
            print("  (and capture the SEATED pose the same way for TARGET_POSE)")
    except KeyboardInterrupt:
        st = c.get_state()
        e, s = line(st["pose"])
        print(f"\n\n    RESET_POSE  = {s}")
    finally:
        c.close()


if __name__ == "__main__":
    main()
