"""FrankaPlumbersEnv — reset/regrasp for the glued-base plumbers inserts on the FR3.

Copied from ram_insertion/wrapper.py (RAMEnv) with two Franka-pivot changes:

1. Robot I/O goes through the RobotClient seam (self.robot.*) instead of raw
   requests.post, so the same env drives EITHER the stock Flask server (ROBOT_CLIENT=
   "http") or the ROS2 cartesian_impedance_control controller (ROBOT_CLIENT="ros2").
   Behavior on the http path is byte-identical to RAMEnv.

2. regrasp() reflects the plumbers handoff: the part BASE is glued to the table and the
   HUMAN hands the part into the gripper (grasp varies). So there is no fixed GRASP_POSE
   pick sequence — release, wait for the human to place the part, close, return to reset.

Everything else (pull-up reset, RANDOM_RESET jitter, compliance/precision toggling) is
the stock RAMEnv procedure.
"""

import copy
import time
from franka_env.utils.rotations import euler_2_quat
import numpy as np

from franka_env.envs.franka_env import FrankaEnv


class FrankaPlumbersEnv(FrankaEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Regrasp is requested from the PS4 pad (Circle button) via PS4RewardWrapper, which
        # sets self.should_regrasp before reset() — NOT the old F1 keyboard (you shouldn't
        # need the keyboard at the rig). Post-reset settle wait gives you a beat before the
        # policy starts moving (and a window to hit Circle to regrasp first).
        self.should_regrasp = False
        self.post_reset_wait_s = float(getattr(self.config, "POST_RESET_WAIT_S", 1.5))

        # Per-insert EXTRACTION axis: unit vector in base frame pointing OUT of the socket.
        # Default +Z (vertical inserts). The HORIZONTAL pipe insert MUST override this, else
        # the retract pushes the part sideways into the socket wall.
        rd = np.asarray(getattr(self.config, "RETRACT_DIR", (0.0, 0.0, 1.0)), dtype=float)
        n = np.linalg.norm(rd)
        if n < 1e-9:
            raise ValueError("RETRACT_DIR must be a non-zero vector")
        self.retract_dir = rd / n
        self.retract_dist = float(getattr(self.config, "RETRACT_DIST", 0.04))
        self.regrasp_retract_dist = float(
            getattr(self.config, "REGRASP_RETRACT_DIST", max(self.retract_dist, 0.08)))

    def _retract_target(self, distance):
        """Require the FULL extraction inside the safety box; never shorten it silently.

        The box must accommodate reset/regrasp clearance as well as policy motion.
        Clipping a retract can leave the part engaged before the next reset motion.
        """
        self._update_currpos()
        direction = np.asarray(self.retract_dir, dtype=float)
        distance = float(distance)
        if (direction.shape != (3,) or not np.all(np.isfinite(direction))
                or not np.isclose(np.linalg.norm(direction), 1.0)
                or not np.isfinite(distance) or distance <= 0):
            raise ValueError("Retraction requires a finite unit 3-vector and positive distance")
        if (not np.all(np.isfinite(self.currpos))
                or np.any(self.currpos[:3] < self.xyz_bounding_box.low)
                or np.any(self.currpos[:3] > self.xyz_bounding_box.high)):
            raise RuntimeError("Current pose is invalid/outside the safety box; reset refused")
        target = copy.deepcopy(self.currpos)
        target[:3] = target[:3] + direction * distance
        clipped = self.clip_safety_box(target.copy())
        same_position = np.allclose(clipped[:3], target[:3], rtol=0, atol=1e-8)
        q = target[3:] / np.linalg.norm(target[3:])
        same_orientation = np.isclose(abs(np.dot(q, clipped[3:])), 1.0, rtol=0, atol=1e-8)
        if not same_position or not same_orientation:
            raise RuntimeError(
                "Safety box would shorten or rotate the retract; reset/regrasp refused. "
                "Verify full extraction clearance and bounds at the rig before retrying. "
                "Do not widen bounds without checking the physical workspace.")
        return target

    def set_reset_from_base_pose(self, base_pose_xyz_euler):
        """MOVING-BASE hook (not used for the fixed-spot task; here so it's a small add,
        not a redesign, if the base ever moves around the table).

        Given the measured base pose (6-vec xyz+euler, e.g. from an ArUco tag on the base
        or the vision pipeline), recompute RESET_POSE so it stays "aligned + offset" above/
        beside the socket wherever the base now sits. The socket's offset FROM the base is
        fixed per insert, so store it once (BASE_TO_RESET_OFFSET in the config) and add it
        here. Call this at the start of reset() before go_to_reset() once localization is
        wired. Left unimplemented on purpose — fixed-spot task needs none of it.
        """
        raise NotImplementedError(
            "Moving-base reset localization is a deferred feature — implement only if the "
            "base is not glued at a fixed spot (see FRANKA_PIVOT_PLAN / RIG_CHECKLIST)."
        )

    def go_to_reset(self, joint_reset=False):
        """Move to the rest position, pulling up first to clear the glued base."""
        # use compliance mode for coupled reset
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.3)
        self.robot.update_param(self.config.PRECISION_PARAM)

        # RETRACT along this insert's EXTRACTION axis before moving to reset.
        # NOT hardcoded +Z: a horizontally-seated part (e.g. the pb_pipe insert) would be
        # shoved sideways into the socket wall by a vertical pull-up. RETRACT_DIR is a unit
        # vector in base frame pointing OUT of the socket (default +Z = vertical inserts).
        self.interpolate_move(self._retract_target(self.retract_dist), timeout=1)

        # perform joint reset if needed
        if joint_reset:
            print("JOINT RESET")
            self.robot.joint_reset()
            time.sleep(0.5)

        # perform Cartesian reset
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)
            self._send_pos_command(reset_pose)
        else:
            reset_pose = self.resetpos.copy()
            self._send_pos_command(reset_pose)
        time.sleep(0.5)

        # Change to compliance mode
        self.robot.update_param(self.config.COMPLIANCE_PARAM)

    def regrasp(self):
        """Human hands the part into the gripper (grasp varies) — no fixed pick pose.

        The base is glued to the table, so there is nothing to pick up autonomously. We
        pull up, open, let the operator place the part in the Franka Hand, close, and
        return to RESET_POSE. Grasp variation is intentional (matches deployment).
        """
        # use compliance mode for coupled reset
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.3)
        self.robot.update_param(self.config.PRECISION_PARAM)

        # Retract along the EXTRACTION axis (not blindly +Z — see go_to_reset) to give the
        # operator room, then a little extra clearance for the hand-off.
        self.interpolate_move(self._retract_target(self.regrasp_retract_dist), timeout=1)

        # Command the gripper DIRECTLY via the robot client, bypassing
        # _send_gripper_command's binary/cooldown guards (which look at a stale cached
        # curr_gripper_pos and would silently DROP the close here — the RAMEnv-regrasp bug).
        input("Press enter to release gripper...")
        self.robot.open_gripper()
        time.sleep(self.gripper_sleep)
        input("Hand the part into the gripper and press enter to grasp...")
        self.robot.close_gripper()          # unconditional close, no guard
        self.last_gripper_act = time.time()
        time.sleep(2)
        self._update_currpos()              # refresh so the cached gripper state is correct

        # return to the fixed pre-insert reset pose
        self.interpolate_move(self.config.RESET_POSE, timeout=1)
        time.sleep(0.5)

    def reset(self, joint_reset=False, **kwargs):
        self.last_gripper_act = time.time()
        if self.save_video:
            self.save_video_recording()

        if self.should_regrasp:
            self.regrasp()
            self.should_regrasp = False

        self._recover()
        self.go_to_reset(joint_reset=joint_reset)  # honor caller's request (stock dropped it)
        self._recover()
        self.curr_path_length = 0
        self._axis_lock_pos = None
        self._axis_lock_rot = None

        self.robot.update_param(self.config.COMPLIANCE_PARAM)

        # Settle wait BEFORE capturing state/obs, so the first policy action sees a SETTLED
        # arm (not a mid-motion pose/image). Also the operator's window to press Circle —
        # PS4RewardWrapper polls (does not blindly drain) across this boundary.
        if not self.fake_env and self.post_reset_wait_s > 0:
            time.sleep(self.post_reset_wait_s)

        self._update_currpos()
        obs = self._get_obs()
        self.terminate = False
        return obs, {}
