"""Robot-communication seam for FrankaEnv.

HIL-SERL centralizes all robot I/O in ~8 HTTP calls to the Flask `franka_server.py`
(`getstate`, `pose`, `clearerr`, `update_param`, `jointreset`, gripper open/close,
`set_load`). This module hides those behind a thin `RobotClient` interface so the
same env / wrappers / agent / configs can talk to EITHER:

- ``HttpFrankaClient`` — the STOCK path. Every method makes exactly the same
  ``requests.post(url + endpoint, ...)`` call the original code made, with the same
  payloads and the same return shapes. This must stay BYTE-IDENTICAL in behavior to
  upstream so the reference HTTP setup is unchanged.

- ``Ros2FrankaClient`` — the Franka-pivot path (2026-09-08). An rclpy node that talks
  to the student's ``cartesian_impedance_control`` ROS2 controller over franka_ros2 +
  FCI. Topic/action names below come from the docs and the student's PS4 teleop; any
  that differ get finalized at the rig once `ros2 topic list` / `ros2 action list` are
  available (see the ``TODO(rig)`` markers).

``FrankaEnv.__init__`` picks the backend via ``config.ROBOT_CLIENT`` ("http" | "ros2").
The env/wrappers/agent/PS4Expert/ZEDCapture/plumbers-configs are IDENTICAL either way —
only ``Ros2FrankaClient``'s topic/action names are stack-specific.
"""

import numpy as np
import requests


class RobotClient:
    """Abstract robot-communication interface used by FrankaEnv and its wrappers.

    Method contract mirrors the stock HTTP endpoints one-for-one:

    - ``get_state() -> dict`` with keys
      ``pose, vel, force, torque, jacobian, q, dq, gripper_pos`` (raw, as the Flask
      server returns them; the env reshapes jacobian to (6,7)).
    - ``send_pos(arr)``      — command an absolute Cartesian target pose (7-vec: xyz+quat).
    - ``recover()``          — clear/recover from robot error state (stock ``clearerr``).
    - ``update_param(params)``— set impedance/compliance params (COMPLIANCE/PRECISION/RESET).
    - ``joint_reset()``      — move to the nominal joint configuration.
    - ``open_gripper()`` / ``close_gripper()`` / ``close_gripper_slow()``.
    - ``set_load(params)``   — set the end-effector load (mass/CoM/inertia).
    """

    def get_state(self) -> dict:
        raise NotImplementedError

    def send_pos(self, arr) -> None:
        raise NotImplementedError

    def recover(self) -> None:
        raise NotImplementedError

    def update_param(self, params: dict) -> None:
        """Set impedance stiffness on cartesian_impedance_control via ROS parameters.

        Maps the stock COMPLIANCE/PRECISION/RESET dicts (translational_stiffness /
        rotational_stiffness) onto the controller node's parameters. FR3 range:
        transl 10-3000 N/m, rot 1-300 Nm/rad.
        """
        import subprocess
        sets = []
        for key, pname in (("translational_stiffness", "translational_stiffness"),
                           ("rotational_stiffness", "rotational_stiffness")):
            if key in params:
                sets.append((pname, float(params[key])))
        if not sets:
            return
        for pname, val in sets:
            try:
                subprocess.run(
                    ["ros2", "param", "set", self.CONTROLLER_NODE, pname, str(val)],
                    check=False, capture_output=True, timeout=5.0)
            except Exception as e:
                print(f"[Ros2FrankaClient] update_param({pname}={val}) failed: {e}")

    def joint_reset(self) -> None:
        """No-op for this stack.

        Stock HIL-SERL uses `jointreset` to recover a wedged arm via a joint-space move.
        The cartesian_impedance_control stack has no joint-space controller loaded, and
        the RL reset path (`go_to_reset`) drives Cartesian targets only. Doing nothing is
        correct AND safe here: silently switching controllers mid-episode would be worse.
        """
        return

    def _run_gripper_action(self, client, goal, timeout=5.0):
        """Send a gripper action goal and wait (bounded) for it to complete."""
        import time
        if not client.wait_for_server(timeout_sec=2.0):
            print(f"[Ros2FrankaClient] gripper action server unavailable: {client._action_name}")
            return False
        fut = client.send_goal_async(goal)
        deadline = time.monotonic() + timeout
        while not fut.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not fut.done():
            print("[Ros2FrankaClient] gripper goal send timed out")
            return False
        handle = fut.result()
        if handle is None or not handle.accepted:
            print("[Ros2FrankaClient] gripper goal rejected")
            return False
        res_fut = handle.get_result_async()
        while not res_fut.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        return res_fut.done()

    def open_gripper(self) -> None:
        goal = self._Move.Goal()
        goal.width = self.GRIPPER_MAX_WIDTH
        goal.speed = 0.1
        self._run_gripper_action(self._move_ac, goal)

    def close_gripper(self) -> None:
        goal = self._Grasp.Goal()
        goal.width = 0.0
        goal.speed = 0.1
        goal.force = 40.0
        goal.epsilon.inner = 0.08
        goal.epsilon.outer = 0.08
        self._run_gripper_action(self._grasp_ac, goal)

    def close_gripper_slow(self) -> None:
        goal = self._Grasp.Goal()
        goal.width = 0.0
        goal.speed = 0.02          # slow close for the regrasp (human hands the part in)
        goal.force = 40.0
        goal.epsilon.inner = 0.08
        goal.epsilon.outer = 0.08
        self._run_gripper_action(self._grasp_ac, goal, timeout=10.0)

    def set_load(self, params: dict) -> None:
        """No-op: the EE load is configured in the URDF / Desk for this stack.

        The live FrankaRobotState already reports inertia_ee (m=1.4kg incl. the hand), so
        the FCI-side load is set. Stock's HTTP set_load is a franka_ros1 service that has
        no equivalent here; silently doing nothing is correct rather than fail-closed,
        because the env calls it during reset.
        """
        return

    def close(self) -> None:
        pass


class HttpFrankaClient(RobotClient):
    """Stock HIL-SERL backend: HTTP POSTs to the Flask franka_server.

    Each method reproduces the exact ``requests.post`` call the original franka_env.py /
    ram_insertion wrapper.py made. Do NOT change endpoint names, payload keys, or return
    handling here — this path must stay behavior-identical to upstream.
    """

    def __init__(self, url: str):
        self.url = url

    def get_state(self) -> dict:
        # stock: requests.post(self.url + "getstate").json()
        return requests.post(self.url + "getstate").json()

    def send_pos(self, arr) -> None:
        # stock: requests.post(self.url + "pose", json={"arr": arr.tolist()})
        arr = np.array(arr).astype(np.float32)
        data = {"arr": arr.tolist()}
        requests.post(self.url + "pose", json=data)

    def recover(self) -> None:
        # stock: requests.post(self.url + "clearerr")
        requests.post(self.url + "clearerr")

    def update_param(self, params: dict) -> None:
        # stock: requests.post(self.url + "update_param", json=<PARAM>)
        requests.post(self.url + "update_param", json=params)

    def joint_reset(self) -> None:
        # stock: requests.post(self.url + "jointreset")
        requests.post(self.url + "jointreset")

    def open_gripper(self) -> None:
        # stock: requests.post(self.url + "open_gripper")
        requests.post(self.url + "open_gripper")

    def close_gripper(self) -> None:
        # stock: requests.post(self.url + "close_gripper")
        requests.post(self.url + "close_gripper")

    def close_gripper_slow(self) -> None:
        # stock: requests.post(self.url + "close_gripper_slow")
        requests.post(self.url + "close_gripper_slow")

    def set_load(self, params: dict) -> None:
        # stock: requests.post(self.url + "set_load", json=self.config.LOAD_PARAM)
        requests.post(self.url + "set_load", json=params)


class Ros2FrankaClient(RobotClient):
    """Franka-pivot backend: rclpy node driving `cartesian_impedance_control`.

    Talks to the student's ROS2 Cartesian-impedance controller over franka_ros2 + FCI.
    No separate F/T sensor: the external wrench comes from the franka_ros2 robot-state
    (``O_F_ext_hat_K`` / ``tau_ext_hat_filtered``).

    Topic/action names below are the DOCUMENTED ones (student's PS4 teleop drives
    ``/cartesian_position_controller/commands``). Anything marked ``TODO(rig)`` must be
    confirmed with ``ros2 topic list`` / ``ros2 action list`` on the robot PC before
    autonomous motion (see FRANKA_RIG_CHECKLIST.md).

    NOTE: import of rclpy is deferred to __init__ so this module imports fine at home
    (no ROS2 in the offline dry-run; learner uses fake_env and never constructs this).
    """

    # --- ROS2 interface names (VERIFIED LIVE at the rig 2026-09-09) -----------------
    # Command out: the student's controller subscribes to a geometry_msgs/Pose here.
    # NOTE: this is NOT the Float64MultiArray /cartesian_position_controller/commands the
    # old docs assumed — cartesian_impedance_control v0.1.8 (on franka_ros2 v2.7.1) uses
    # /cartesian_impedance_controller/reference_pose (confirmed via ros2 topic info).
    CMD_POSE_TOPIC = "/cartesian_impedance_controller/reference_pose"
    CONTROL_MODE_TOPIC = "/cartesian_impedance_controller/control_mode"
    # Robot state in: franka_msgs/msg/FrankaRobotState @1000Hz (confirmed).
    ROBOT_STATE_TOPIC = "/franka_robot_state_broadcaster/robot_state"
    # Franka Hand joint states -> gripper width (NOT in FrankaRobotState).
    GRIPPER_STATE_TOPIC = "/franka_gripper/joint_states"
    # Impedance stiffness: set as ROS parameters on the controller node.
    CONTROLLER_NODE = "/cartesian_impedance_controller"
    # Gripper action servers (confirmed via ros2 action list).
    GRIPPER_MOVE_ACTION = "/franka_gripper/move"     # franka_msgs/action/Move
    GRIPPER_GRASP_ACTION = "/franka_gripper/grasp"   # franka_msgs/action/Grasp
    GRIPPER_HOMING_ACTION = "/franka_gripper/homing"
    # Error recovery (confirmed via ros2 action list).
    ERROR_RECOVERY_ACTION = "/action_server/error_recovery"

    # Franka Hand: max opening 0.08 m total (two fingers x 0.04). Normalize to [0,1]
    # because _send_gripper_command compares curr_gripper_pos against 0.85.
    GRIPPER_MAX_WIDTH = 0.08
    # Nominal width (m) the Grasp action closes TO. 0.0 = squeeze fully shut; set this
    # to just UNDER the part diameter so the fingers load the part instead of
    # bottoming out. Measure the pipe and set accordingly.
    GRIPPER_GRASP_WIDTH = 0.0
    # get_state() refuses to return state older than this (never feed stale state to the policy).
    STATE_STALE_S = 0.5
    # bounded wait for the first state message on construction
    FIRST_STATE_TIMEOUT_S = 10.0

    def __init__(self, url: str = None, config=None, node_name: str = "hil_serl_franka_client"):
        # Deferred import: keep this module importable without ROS2 (home / learner).
        import threading
        import rclpy
        from rclpy.node import Node
        from rclpy.action import ActionClient
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.executors import MultiThreadedExecutor
        from geometry_msgs.msg import Pose
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Int16
        from franka_msgs.msg import FrankaRobotState
        from franka_msgs.action import Move, Grasp, Homing, ErrorRecovery

        self.config = config
        self._Pose = Pose
        self._Int16 = Int16
        self._Move, self._Grasp, self._Homing = Move, Grasp, Homing

        self._owns_context = False
        if not rclpy.ok():
            rclpy.init()
            self._owns_context = True
        self._rclpy = rclpy
        self.node = Node(node_name)
        cb = ReentrantCallbackGroup()

        # ---- command publishers -------------------------------------------------
        self._cmd_pub = self.node.create_publisher(Pose, self.CMD_POSE_TOPIC, 10)
        self._mode_pub = self.node.create_publisher(Int16, self.CONTROL_MODE_TOPIC, 10)

        # ---- state subscriptions ------------------------------------------------
        self._lock = threading.Lock()
        self._state_msg = None          # latest FrankaRobotState
        self._state_stamp = 0.0         # monotonic time it arrived
        self._gripper_width = None      # metres (sum of both fingers)

        self.node.create_subscription(
            FrankaRobotState, self.ROBOT_STATE_TOPIC, self._on_state, 10, callback_group=cb)
        self.node.create_subscription(
            JointState, self.GRIPPER_STATE_TOPIC, self._on_gripper, 10, callback_group=cb)

        # ---- action clients -----------------------------------------------------
        self._move_ac = ActionClient(self.node, Move, self.GRIPPER_MOVE_ACTION, callback_group=cb)
        self._grasp_ac = ActionClient(self.node, Grasp, self.GRIPPER_GRASP_ACTION, callback_group=cb)
        self._homing_ac = ActionClient(self.node, Homing, self.GRIPPER_HOMING_ACTION, callback_group=cb)
        self._recover_ac = ActionClient(self.node, ErrorRecovery, self.ERROR_RECOVERY_ACTION,
                                        callback_group=cb)

        # ---- spin in a background thread so callbacks flow continuously ----------
        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        # ---- block until the first real state arrives (never start on zeros) -----
        self._wait_for_first_state()

    def _wait_for_first_state(self):
        """Block (bounded) until FrankaRobotState AND gripper width have arrived."""
        import time
        deadline = time.monotonic() + self.FIRST_STATE_TIMEOUT_S
        while time.monotonic() < deadline:
            with self._lock:
                if self._state_msg is not None and self._gripper_width is not None:
                    return
            time.sleep(0.02)
        with self._lock:
            missing = []
            if self._state_msg is None:
                missing.append(self.ROBOT_STATE_TOPIC)
            if self._gripper_width is None:
                missing.append(self.GRIPPER_STATE_TOPIC)
        raise RuntimeError(
            "Ros2FrankaClient: no data on %s within %.1fs. Is the controller running "
            "(ros2 control list_controllers)?" % (", ".join(missing), self.FIRST_STATE_TIMEOUT_S))

    # -- helpers -------------------------------------------------------------------
    def _on_state(self, msg):
        """Cache the latest FrankaRobotState (called at 1000 Hz by the executor thread)."""
        import time
        with self._lock:
            self._state_msg = msg
            self._state_stamp = time.monotonic()

    def _on_gripper(self, msg):
        """Franka Hand width = sum of the two finger joints (each ~0..0.04 m)."""
        try:
            width = float(sum(msg.position[:2]))
        except Exception:
            return
        with self._lock:
            self._gripper_width = width

    @staticmethod
    def _jacobian_from_q(q):
        """Geometric Jacobian (6x7) of the FR3 flange in the base frame.

        FrankaRobotState does NOT carry the jacobian, so we compute it from the measured
        joint angles with the FR3 modified-DH parameters. Used only for tcp_vel = J.dq and
        as the env's `jacobian` field.
        """
        import numpy as np
        # FR3 modified DH: (a, d, alpha) per joint, then the flange transform.
        dh = [
            (0.0,      0.333,   0.0),
            (0.0,      0.0,    -np.pi / 2),
            (0.0,      0.316,   np.pi / 2),
            (0.0825,   0.0,     np.pi / 2),
            (-0.0825,  0.384,  -np.pi / 2),
            (0.0,      0.0,     np.pi / 2),
            (0.088,    0.0,     np.pi / 2),
        ]
        T = np.eye(4)
        origins = []   # joint origins in base frame
        z_axes = []    # joint axes in base frame
        for i, (a, d, alpha) in enumerate(dh):
            ct, st = np.cos(q[i]), np.sin(q[i])
            ca, sa = np.cos(alpha), np.sin(alpha)
            A = np.array([
                [ct,      -st,     0.0,    a],
                [st * ca,  ct * ca, -sa, -d * sa],
                [st * sa,  ct * sa,  ca,  d * ca],
                [0.0,      0.0,     0.0,  1.0],
            ])
            T = T @ A
            # Capture the joint axis/origin AFTER this link's transform. In modified
            # (Craig) DH the axis of joint i is the z of frame i, not frame i-1.
            # VERIFIED at the rig against finite differences: 2.2e-08 vs 5.4e-01 if
            # captured before the transform.
            origins.append(T[:3, 3].copy())
            z_axes.append(T[:3, 2].copy())
        # flange (link8, d=0.107) + Franka Hand TCP (0.1034) along z.
        # VALIDATED at the rig 2026-09-09 against the robot's own FK (TF base->fr3_hand_tcp):
        # 0.4 mm agreement. Stopping at the flange instead was off by ~104 mm.
        T = T @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.107 + 0.1034],
                          [0, 0, 0, 1.0]])
        # p_e must be the SAME point the FK ends at (the TCP), otherwise the lever arm
        # (z_i x (p_e - o_i)) is taken about the wrong point and J[:3] is wrong.
        p_e = T[:3, 3]
        J = np.zeros((6, 7))
        for i in range(7):
            z = z_axes[i]
            J[:3, i] = np.cross(z, p_e - origins[i])
            J[3:, i] = z
        return J

    # -- RobotClient interface -----------------------------------------------------
    def get_state(self) -> dict:
        """Return the same dict shape as the HTTP server's /getstate.

        Keys: pose(7 xyz+quat), vel(6), force(3), torque(3), jacobian(42 flat -> (6,7)),
        q(7), dq(7), gripper_pos(1 normalized to [0,1]).

        Sources (VERIFIED LIVE on the FR3 2026-09-09):
          pose        <- o_t_ee.pose (position + orientation quaternion)
          force/torque<- o_f_ext_hat_k.wrench.{force,torque}  (NESTED, base frame)
          q/dq        <- measured_joint_state.{position,velocity}
          vel         <- J(q) . dq   (the msg only exposes DESIRED twists)
          jacobian    <- computed from q (NOT in FrankaRobotState)
          gripper_pos <- /franka_gripper/joint_states, normalized by GRIPPER_MAX_WIDTH
        """
        import time
        import numpy as np

        with self._lock:
            msg = self._state_msg
            stamp = self._state_stamp
            width = self._gripper_width

        if msg is None or width is None:
            raise RuntimeError("Ros2FrankaClient.get_state(): no robot state received yet.")
        age = time.monotonic() - stamp
        if age > self.STATE_STALE_S:
            raise RuntimeError(
                "Ros2FrankaClient.get_state(): state is stale (%.2fs old, limit %.2fs). "
                "Is the controller still running?" % (age, self.STATE_STALE_S))

        p_ = msg.o_t_ee.pose.position
        o_ = msg.o_t_ee.pose.orientation
        pose = [p_.x, p_.y, p_.z, o_.x, o_.y, o_.z, o_.w]

        f_ = msg.o_f_ext_hat_k.wrench.force
        t_ = msg.o_f_ext_hat_k.wrench.torque
        force = [f_.x, f_.y, f_.z]
        torque = [t_.x, t_.y, t_.z]

        q = list(msg.measured_joint_state.position)[:7]
        dq = list(msg.measured_joint_state.velocity)[:7]

        # PER-JOINT external torques (Nm). NOT part of the stock /getstate contract, but the
        # "joint torque limit" reflex that faults the arm is per-joint tau_J, which the EE
        # wrench cannot see — the env's torque guard needs this. Extra key: harmless for
        # consumers that ignore it.
        try:
            tau_j = list(msg.tau_ext_hat_filtered.effort)[:7]
        except Exception:
            tau_j = [0.0] * 7

        J = self._jacobian_from_q(np.asarray(q, dtype=float))
        vel = (J @ np.asarray(dq, dtype=float)).tolist()

        gripper_pos = [float(np.clip(width / self.GRIPPER_MAX_WIDTH, 0.0, 1.0))]

        return {
            "pose": pose,
            "vel": vel,
            "force": force,
            "torque": torque,
            "jacobian": J.flatten().tolist(),
            "q": q,
            "dq": dq,
            "gripper_pos": gripper_pos,
            "tau_j": tau_j,
        }

    # A quaternion below this norm carries no usable orientation -> reject (degenerate).
    # Anything above it is NORMALIZED, not rejected: `interpolate_move` builds its path with
    # np.linspace over raw quaternion components (component-wise LERP, not slerp), so every
    # intermediate quaternion is legitimately sub-unit — a 20 deg reset dips to |q|~0.9962.
    # Rejecting those would abort every orientation-changing reset/regrasp on the rig.
    QUAT_MIN_NORM = 1e-6

    def send_pos(self, arr) -> None:
        """Publish an absolute Cartesian target [x,y,z,qx,qy,qz,qw] to the controller.

        Validated before publishing: wrong-length, non-finite, or degenerate-quaternion
        targets must NOT reach the arm. A merely non-unit quaternion is NORMALIZED (the
        intent is unambiguous — see QUAT_MIN_NORM) rather than rejected.
        """
        arr = np.asarray(arr, dtype=np.float64).ravel()
        if arr.shape != (7,):
            raise ValueError(
                f"send_pos expects a 7-vec [x,y,z,qx,qy,qz,qw], got shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"send_pos got non-finite values: {arr}")
        qnorm = float(np.linalg.norm(arr[3:]))
        if qnorm < self.QUAT_MIN_NORM:
            raise ValueError(
                f"send_pos quaternion is degenerate (|q|={qnorm:.3e}); refusing to command "
                "the arm with no usable orientation.")
        arr = arr.copy()
        arr[3:] /= qnorm  # normalize (LERP-interpolated resets are sub-unit by construction)
        # The controller expects a geometry_msgs/Pose on CMD_POSE_TOPIC (verified at rig).
        msg = self._Pose()
        msg.position.x, msg.position.y, msg.position.z = float(arr[0]), float(arr[1]), float(arr[2])
        msg.orientation.x = float(arr[3])
        msg.orientation.y = float(arr[4])
        msg.orientation.z = float(arr[5])
        msg.orientation.w = float(arr[6])
        self._cmd_pub.publish(msg)

    def recover(self) -> None:
        # Called EVERY _send_pos_command (~10Hz), so it must be CHEAP. The stock HTTP
        # `clearerr` is fire-and-forget. Do NOT implement this as a blocking action call
        # per step — that would enqueue/await an error-recovery action at control rate.
        # TODO(rig): make this a no-op in the nominal case; only trigger the franka_ros2
        # error-recovery action when the robot is actually in an error/reflex state (read
        # that flag from the robot-state and gate on it). For now = no-op (safe: the
        # controller itself is compliant; recovery is only needed after a reflex).
        pass

    # cartesian_impedance_control does NOT expose stiffness as ROS parameters — K and D are
    # compile-time constants in cartesian_impedance_controller.hpp. So every `ros2 param
    # set` here was guaranteed to fail, AND each one spawned a subprocess that had to boot
    # Python + rclpy and discover the graph (~seconds). The env calls update_param 3x per
    # reset (x2 params), which added ~20 s of dead time to EVERY reset during demo
    # recording (rig-measured 2026-09-09). Made a no-op.
    #
    # TO ACTUALLY CHANGE STIFFNESS: edit K/D in the controller header and rebuild:
    #   ~/fr3_ws/src/cartesian_impedance_control/include/.../cartesian_impedance_controller.hpp
    #   colcon build --packages-select cartesian_impedance_control   (then relaunch)
    # If the controller ever gains real ROS params, wire them with a persistent
    # AsyncParametersClient created in __init__ — never a per-call subprocess.
    _warned_update_param = False

    def update_param(self, params: dict) -> None:
        if not Ros2FrankaClient._warned_update_param:
            Ros2FrankaClient._warned_update_param = True
            print("[Ros2FrankaClient] update_param() is a no-op: cartesian_impedance_control "
                  "has no ROS stiffness params (compile-time constants). Edit the controller "
                  "header + rebuild to change gains.")
        return

    def joint_reset(self) -> None:
        """No-op for this stack.

        Stock HIL-SERL's `jointreset` recovers a wedged arm with a joint-space move. The
        cartesian_impedance_control stack has no joint-space controller loaded and the RL
        reset path (go_to_reset) drives Cartesian targets only. Doing nothing is correct
        AND safe: switching controllers mid-episode would be worse.
        """
        return

    def _run_gripper_action(self, client, goal, timeout=5.0):
        """Send a gripper action goal and wait (bounded) for completion."""
        import time
        if not client.wait_for_server(timeout_sec=2.0):
            print("[Ros2FrankaClient] gripper action server unavailable")
            return False
        fut = client.send_goal_async(goal)
        deadline = time.monotonic() + timeout
        while not fut.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not fut.done():
            print("[Ros2FrankaClient] gripper goal send timed out")
            return False
        handle = fut.result()
        if handle is None or not handle.accepted:
            print("[Ros2FrankaClient] gripper goal rejected")
            return False
        res_fut = handle.get_result_async()
        while not res_fut.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        return res_fut.done()

    def open_gripper(self) -> None:
        goal = self._Move.Goal()
        # Request slightly UNDER the mechanical maximum. Commanding exactly
        # GRIPPER_MAX_WIDTH (0.08) can fail to move at all — the hand will not drive to its
        # own hard stop, so the action returns without opening (rig 2026-09-10: open
        # reported 0.000 -> 0.000 while a direct Move to 0.06 worked immediately).
        goal.width = self.GRIPPER_MAX_WIDTH - 0.004
        # 0.05 m/s, NOT 0.1: at 0.1 the Move action reported success without the fingers
        # moving at all (rig 2026-09-10 — open logged 0.438 -> 0.438, while the identical
        # goal at 0.05 opened immediately).
        goal.speed = 0.05
        self._run_gripper_action(self._move_ac, goal)

    def close_gripper(self) -> None:
        goal = self._Grasp.Goal()
        # Target the PART width, not 0.0: franka's Grasp squeezes toward `width` and holds
        # `force` once it is within epsilon. Commanding 0.0 with a loose epsilon made it
        # stop early without loading the part. GRIPPER_GRASP_WIDTH is the nominal part
        # diameter (m) — override per-part if needed.
        goal.width = float(getattr(self, "GRIPPER_GRASP_WIDTH", 0.0))
        goal.speed = 0.1
        goal.force = 70.0        # 40 -> 60 -> 70 N: the pipe still slipped at 60 (rig
        # 2026-09-10). 70 N is the Franka Hand's continuous maximum — there is no more
        # force available, so if the pipe still slips the remaining levers are the finger
        # pads (friction) or GRIPPER_GRASP_WIDTH, not this number.
        # epsilon = how far from `width` the final opening may be and still count as a
        # successful grasp. 0.08 (the old value) is the FULL stroke, so the action reported
        # success the moment the fingers stopped and never kept squeezing — the operator saw
        # it "hold position instead of press". franka_msgs' own default is 0.005.
        # Wide OUTER tolerance lets a thicker-than-expected part still register as grasped;
        # tight INNER makes closing on nothing a FAILURE rather than a silent success.
        goal.epsilon.inner = 0.005
        goal.epsilon.outer = 0.04
        self._run_gripper_action(self._grasp_ac, goal)

    def close_gripper_slow(self) -> None:
        goal = self._Grasp.Goal()
        # Target the PART width, not 0.0: franka's Grasp squeezes toward `width` and holds
        # `force` once it is within epsilon. Commanding 0.0 with a loose epsilon made it
        # stop early without loading the part. GRIPPER_GRASP_WIDTH is the nominal part
        # diameter (m) — override per-part if needed.
        goal.width = float(getattr(self, "GRIPPER_GRASP_WIDTH", 0.0))
        goal.speed = 0.02       # slow close for the regrasp (human hands the part in)
        goal.force = 70.0        # 40 -> 60 -> 70 N: the pipe still slipped at 60 (rig
        # 2026-09-10). 70 N is the Franka Hand's continuous maximum — there is no more
        # force available, so if the pipe still slips the remaining levers are the finger
        # pads (friction) or GRIPPER_GRASP_WIDTH, not this number.
        # epsilon = how far from `width` the final opening may be and still count as a
        # successful grasp. 0.08 (the old value) is the FULL stroke, so the action reported
        # success the moment the fingers stopped and never kept squeezing — the operator saw
        # it "hold position instead of press". franka_msgs' own default is 0.005.
        # Wide OUTER tolerance lets a thicker-than-expected part still register as grasped;
        # tight INNER makes closing on nothing a FAILURE rather than a silent success.
        goal.epsilon.inner = 0.005
        goal.epsilon.outer = 0.04
        self._run_gripper_action(self._grasp_ac, goal, timeout=10.0)

    def set_load(self, params: dict) -> None:
        """No-op: the EE load is configured in the URDF / Desk for this stack.

        The live FrankaRobotState already reports inertia_ee (m=1.4kg incl. the hand), so
        the FCI-side load is set. Stock's HTTP set_load has no equivalent here; the env
        calls it during reset, so returning quietly is correct rather than fail-closed.
        """
        return

    def close(self) -> None:
        # stop the background executor first so callbacks aren't running on a dead node
        try:
            self._executor.shutdown()
        except Exception:
            pass
        try:
            self.node.destroy_node()
        except Exception:
            pass
        # Only shut down the context if this client created it (see _owns_context).
        if getattr(self, "_owns_context", False):
            try:
                self._rclpy.shutdown()
            except Exception:
                pass


def make_robot_client(config, url: str) -> RobotClient:
    """Factory: pick the backend from ``config.ROBOT_CLIENT`` (default "http")."""
    backend = getattr(config, "ROBOT_CLIENT", "http")
    if backend == "http":
        return HttpFrankaClient(url)
    elif backend == "ros2":
        return Ros2FrankaClient(url=url, config=config)
    else:
        raise ValueError(
            f"Unknown ROBOT_CLIENT={backend!r}; expected 'http' or 'ros2'."
        )
