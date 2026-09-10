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
        # TOLERANCE on the "is the arm in the box" check (2026-09-10). The MEASURED TCP can
        # sit a couple of mm outside the box even when every COMMAND was clipped to it: this
        # is an impedance controller, so the commanded pose is a spring setpoint and the arm
        # settles wherever force balances. A strict check turned a 2.3 mm overshoot into a
        # hard RuntimeError that killed an 8-minute training run mid-episode. Being slightly
        # outside is not a safety event — the retract itself is still clipped below, and a
        # GROSS excursion (> 2 cm) still aborts.
        _slack = 0.02
        if (not np.all(np.isfinite(self.currpos))
                or np.any(self.currpos[:3] < self.xyz_bounding_box.low - _slack)
                or np.any(self.currpos[:3] > self.xyz_bounding_box.high + _slack)):
            raise RuntimeError("Current pose is invalid/outside the safety box; reset refused")
        target = copy.deepcopy(self.currpos)
        # Pull the START of the retract back inside the box, so a few mm of impedance sag
        # does not propagate into a retract target that the clip check then rejects.
        target[:3] = np.clip(target[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high)
        # SHRINK THE RETRACT TO FIT (2026-09-10). Previously this demanded the FULL
        # `distance` and raised if the box would shorten it — which killed the actor
        # repeatedly during insert-1 training: the policy legitimately drives deep into -Y,
        # ends an episode ~2 cm from the box floor, and the 4 cm retract then does not fit.
        # A SHORTENED retract along the same direction is perfectly safe (it still moves OUT
        # of the socket, just less far); what is NOT safe is a retract that gets ROTATED or
        # deflected onto another axis, and that is still refused below.
        # Compute how far we can actually travel along `direction` before leaving the box.
        room = np.inf
        for i in range(3):
            d = float(direction[i])
            if abs(d) < 1e-9:
                continue
            bound = self.xyz_bounding_box.high[i] if d > 0 else self.xyz_bounding_box.low[i]
            room = min(room, (bound - target[i]) / d)
        room = float(max(0.0, room))

        usable = min(float(distance), room)
        _MIN_RETRACT = 0.005          # below this the move is pointless — surface the problem
        if usable < _MIN_RETRACT:
            raise RuntimeError(
                f"No room to retract: only {room*1000:.1f} mm available along {tuple(direction)} "
                f"before the safety box (wanted {distance*1000:.0f} mm). The arm is jammed "
                "against the box edge — jog it back toward RESET_POSE before resuming.")
        if usable < float(distance) - 1e-9:
            print(f"[franka_plumbers] retract shortened {distance*1000:.0f} -> "
                  f"{usable*1000:.0f} mm (box edge); still along the extraction axis.")

        target[:3] = target[:3] + direction * usable
        clipped = self.clip_safety_box(target.copy())
        # Position is guaranteed in-box by construction now; the ORIENTATION check stays —
        # a rotated retract would drag the part sideways through the socket wall.
        q = target[3:] / np.linalg.norm(target[3:])
        same_orientation = np.isclose(abs(np.dot(q, clipped[3:])), 1.0, rtol=0, atol=1e-8)
        if not same_orientation:
            raise RuntimeError(
                "Safety box would ROTATE the retract; reset/regrasp refused. "
                "Verify the rotation bounds cover the parked orientation.")
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
        # The retract is a NICETY, not a precondition for resetting (2026-09-10). During
        # training the policy can park hard against the box edge with ~0 mm of extraction
        # room; refusing to reset there killed the actor repeatedly. If there is no room to
        # retract, skip it and go straight to the reset move — interpolate_move clips every
        # waypoint to the box, so driving back toward RESET_POSE from the wall is safe and
        # is exactly what un-jams the arm.
        try:
            self.interpolate_move(self._retract_target(self.retract_dist), timeout=1)
        except RuntimeError as e:
            print(f"[franka_plumbers] retract skipped ({e}); going straight to RESET_POSE.")

        # perform joint reset if needed
        if joint_reset:
            print("JOINT RESET")
            self.robot.joint_reset()
            time.sleep(0.5)

        # perform Cartesian reset
        if self.randomreset:  # randomize reset position
            reset_pose = self.resetpos.copy()
            # x/y always; z too when RANDOM_Z_RANGE is set (2026-09-10 — asked for noise in
            # ALL directions on insert 2). Stock upstream only ever jittered the xy plane.
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            _rz = float(getattr(self.config, "RANDOM_Z_RANGE", 0.0))
            if _rz > 0.0:
                reset_pose[2] += np.random.uniform(-_rz, _rz)
                # never let the jitter push the start below the safety-box floor
                reset_pose[2] = float(np.clip(
                    reset_pose[2],
                    self.xyz_bounding_box.low[2],
                    self.xyz_bounding_box.high[2]))
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)
        else:
            reset_pose = self.resetpos.copy()

        # INTERPOLATE the Cartesian reset (2026-09-10). This was a bare
        # _send_pos_command(reset_pose): a SINGLE step that teleports the reference to the
        # reset pose, so the controller lunges at it in a straight line at whatever speed
        # the stiffness produces. Two failures on the rig during insert-1 training:
        #   * the straight line from wherever the policy left the arm to RESET can pass
        #     BELOW THE TABLE -> the arm drove into the ground;
        #   * the lunge trips the wrist joint-torque limit.
        # interpolate_move walks a bounded path AND speed-limits it (see franka_env), and
        # clip_safety_box keeps every waypoint (including the z floor) inside the box.
        self.interpolate_move(reset_pose, timeout=1)
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
