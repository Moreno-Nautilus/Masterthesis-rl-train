from __future__ import annotations

import importlib
import threading
import time
from typing import Any

import numpy as np

from iiwa_serl.robot.api import RobotState
from iiwa_serl.utils.transformations import pose6_to_pose7


class MockIiwaBackend:
    def __init__(
        self,
        initial_pose6: list[float] | None = None,
        reset_pose6: list[float] | None = None,
        reset_joints: list[float] | None = None,
    ):
        self.initial_pose7 = pose6_to_pose7(
            np.asarray(initial_pose6 or [0.72, 0.0, 0.10, np.pi, 0.0, 0.0], dtype=np.float64)
        )
        self.reset_pose7 = pose6_to_pose7(
            np.asarray(reset_pose6 or [0.72, 0.0, 0.10, np.pi, 0.0, 0.0], dtype=np.float64)
        )
        self.reset_joint_target = np.asarray(reset_joints or np.zeros(7), dtype=np.float64)
        self.params: dict[str, Any] = {}
        self.gripper = 0.0
        self.state = RobotState(
            pose=self.initial_pose7.copy(),
            vel=np.zeros(6, dtype=np.float64),
            force=np.zeros(3, dtype=np.float64),
            torque=np.zeros(3, dtype=np.float64),
            q=self.reset_joint_target.copy(),
            dq=np.zeros(7, dtype=np.float64),
            jacobian=np.zeros((6, 7), dtype=np.float64),
            gripper=0.0,
            timestamp=time.time(),
        )

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def clear_errors(self) -> None:
        return None

    def get_state(self) -> RobotState:
        self.state.timestamp = time.time()
        self.state.gripper = self.gripper
        return self.state

    def move_pose(self, pose7: np.ndarray) -> None:
        pose7 = np.asarray(pose7, dtype=np.float64)
        self.state.vel = np.concatenate([pose7[:3] - self.state.pose[:3], np.zeros(3, dtype=np.float64)])
        self.state.pose = pose7
        self.state.timestamp = time.time()

    def reset_joints(self) -> None:
        self.state.pose = self.reset_pose7.copy()
        self.state.q = self.reset_joint_target.copy()
        self.state.dq = np.zeros_like(self.state.dq)
        self.state.vel = np.zeros_like(self.state.vel)
        self.state.timestamp = time.time()

    def activate_gripper(self) -> None:
        return None

    def reset_gripper(self) -> None:
        self.gripper = 1.0

    def open_gripper(self) -> None:
        self.gripper = 1.0

    def close_gripper(self) -> None:
        self.gripper = 0.0

    def move_gripper(self, value: float) -> None:
        self.gripper = float(np.clip(value, 0.0, 1.0))

    def update_params(self, params: dict[str, Any]) -> None:
        self.params.update(params)


class RealIiwaBackend:
    """Real iiwa7 backend — ROS2 bridge to the workspace's Franka control stack.

    This mirrors the interface `ps4_controller_teleop/cartesian_bridge.py` already
    uses, but exposed through the SERL backend contract (get_state / move_pose /
    gripper) so the SERL HTTP server drives the robot for RL.

    Topics / interfaces (all overridable via kwargs):
      STATE IN   : `FrankaRobotState` on `/franka_robot_state_broadcaster/robot_state`
                   -> EE pose from `msg.o_t_ee.pose`, ext joint torque from
                      `msg.tau_ext_hat_filtered.effort`.
      WRENCH IN  : optional `WrenchStamped` (gravity/bias-compensated CONTACT wrench,
                   e.g. from the `ft_compensation` package's `~/wrench_contact`)
                   -> Cartesian force/torque used by the SERL reward. Falls back to
                      zeros if no wrench topic is available.
      COMMAND OUT: `Float64MultiArray [x,y,z,qx,qy,qz,qw]` on
                   `/cartesian_position_controller/commands`. The cartesian_impedance
                   controller applies its own EMA/slerp/clamp smoothing, so we publish
                   the *absolute* target pose directly (the SERL env already composes
                   deltas into absolute targets — no integrator needed here).
      GRIPPER    : Franka `Move`/`Grasp` action servers under `/{gripper_prefix}`.

    Usage
    -----
        python -m iiwa_serl.robot_servers.iiwa_server \\
            --backend="iiwa_serl.robot.backends:RealIiwaBackend" \\
            --backend_kwargs_json='{"gripper_prefix": "fr3_gripper"}'

    Requires a sourced ROS2 environment with rclpy + franka_msgs on PYTHONPATH.
    A background thread spins the node so the Flask server stays responsive.
    """

    def __init__(
        self,
        robot_state_topic: str = "/franka_robot_state_broadcaster/robot_state",
        wrench_topic: str | None = "/franka_ft_compensation/wrench_contact",
        command_topic: str = "/cartesian_position_controller/commands",
        gripper_prefix: str = "fr3_gripper",
        reset_pose7: list[float] | None = None,
        lock_orientation: bool = True,
        fixed_orientation: list[float] | None = None,
        state_timeout_s: float = 2.0,
        **kwargs: Any,
    ):
        self._robot_state_topic = robot_state_topic
        self._wrench_topic = wrench_topic
        self._command_topic = command_topic
        self._gripper_prefix = gripper_prefix
        self._reset_pose7 = np.asarray(reset_pose7, dtype=np.float64) if reset_pose7 is not None else None
        self._lock_orientation = lock_orientation
        self._fixed_orientation = (
            np.asarray(fixed_orientation, dtype=np.float64) if fixed_orientation is not None else None
        )
        self._state_timeout_s = state_timeout_s
        self._kwargs = kwargs

        # Lazily created in start() so importing this module never requires ROS2.
        self._node = None
        self._executor = None
        self._spin_thread = None
        self._gripper = None

        # Cached latest state (thread-safe via a lock).
        self._lock = threading.Lock()
        self._pose7 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self._q = np.zeros(7, dtype=np.float64)
        self._dq = np.zeros(7, dtype=np.float64)
        self._tau_ext = np.zeros(7, dtype=np.float64)
        self._force = np.zeros(3, dtype=np.float64)
        self._torque = np.zeros(3, dtype=np.float64)
        self._gripper_state = 0.0
        self._state_stamp = 0.0
        self._got_state = False

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        import rclpy
        from rclpy.node import Node
        from rclpy.executors import SingleThreadedExecutor
        from franka_msgs.msg import FrankaRobotState
        from std_msgs.msg import Float64MultiArray
        from geometry_msgs.msg import WrenchStamped

        from iiwa_serl.robot.gripper import GripperActionClient  # local, ROS2-only

        if not rclpy.ok():
            rclpy.init()
        self._node = Node("iiwa_serl_backend")
        self._Float64MultiArray = Float64MultiArray

        self._node.create_subscription(
            FrankaRobotState, self._robot_state_topic, self._on_robot_state, 10
        )
        if self._wrench_topic:
            self._node.create_subscription(
                WrenchStamped, self._wrench_topic, self._on_wrench, 10
            )
        self._command_pub = self._node.create_publisher(
            Float64MultiArray, self._command_topic, 10
        )
        self._gripper = GripperActionClient(self._node, gripper_prefix=self._gripper_prefix)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()
        self._node.get_logger().info(
            f"iiwa_serl backend up. state<-{self._robot_state_topic}, "
            f"cmd->{self._command_topic}, wrench<-{self._wrench_topic}, gripper={self._gripper_prefix}"
        )

    def stop(self) -> None:
        try:
            if self._executor is not None:
                self._executor.shutdown()
            if self._node is not None:
                self._node.destroy_node()
            import rclpy

            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    def clear_errors(self) -> None:
        # The cartesian_impedance controller recovers on its own; nothing to send here.
        # If an error-recovery action is later exposed, call it in this method.
        pass

    # ------------------------------------------------------------------
    # ROS2 subscription callbacks
    # ------------------------------------------------------------------
    def _on_robot_state(self, msg) -> None:
        p = msg.o_t_ee.pose.position
        o = msg.o_t_ee.pose.orientation
        with self._lock:
            self._pose7 = np.array([p.x, p.y, p.z, o.x, o.y, o.z, o.w], dtype=np.float64)
            try:
                self._tau_ext = np.asarray(msg.tau_ext_hat_filtered.effort, dtype=np.float64)
            except Exception:
                pass
            try:
                self._q = np.asarray(msg.measured_joint_state.position, dtype=np.float64)
                self._dq = np.asarray(msg.measured_joint_state.velocity, dtype=np.float64)
            except Exception:
                pass
            self._state_stamp = time.time()
            self._got_state = True

    def _on_wrench(self, msg) -> None:
        f = msg.wrench.force
        t = msg.wrench.torque
        with self._lock:
            self._force = np.array([f.x, f.y, f.z], dtype=np.float64)
            self._torque = np.array([t.x, t.y, t.z], dtype=np.float64)

    # ------------------------------------------------------------------
    # SERL backend contract
    # ------------------------------------------------------------------
    def get_state(self) -> RobotState:
        with self._lock:
            return RobotState(
                pose=self._pose7.copy(),
                vel=np.zeros(6, dtype=np.float64),  # controller doesn't publish EE twist; zeros OK
                force=self._force.copy(),
                torque=self._torque.copy(),
                q=self._q.copy() if self._q.shape == (7,) else np.zeros(7),
                dq=self._dq.copy() if self._dq.shape == (7,) else np.zeros(7),
                jacobian=np.zeros((6, 7), dtype=np.float64),
                gripper=float(self._gripper_state),
                timestamp=self._state_stamp or time.time(),
            )

    def move_pose(self, pose7: np.ndarray) -> None:
        pose7 = np.asarray(pose7, dtype=np.float64)
        if self._lock_orientation and self._fixed_orientation is not None:
            pose7 = np.concatenate([pose7[:3], self._fixed_orientation])
        msg = self._Float64MultiArray()
        msg.data = pose7.tolist()
        self._command_pub.publish(msg)

    def reset_joints(self) -> None:
        # This stack has no joint-space reset topic; the SERL env commands the reset
        # *pose* right after joint_reset(), which the cartesian controller tracks. If a
        # reset_pose7 was provided we publish it here so reset works even if the env
        # doesn't. Otherwise this is a safe no-op.
        if self._reset_pose7 is not None:
            self.move_pose(self._reset_pose7)

    def activate_gripper(self) -> None:
        if self._gripper is not None:
            try:
                self._gripper.wait_for_servers(timeout_sec=2.0)
                self._gripper.home()
            except Exception as exc:  # noqa: BLE001
                if self._node is not None:
                    self._node.get_logger().warn(f"gripper activate failed: {exc}")

    def reset_gripper(self) -> None:
        self.open_gripper()

    def open_gripper(self) -> None:
        if self._gripper is not None:
            self._gripper.open()
        self._gripper_state = 1.0

    def close_gripper(self) -> None:
        if self._gripper is not None:
            self._gripper.close()
        self._gripper_state = 0.0

    def move_gripper(self, value: float) -> None:
        # PDZ / Franka Hand here is treated as open/close; threshold the fractional width.
        if float(value) >= 0.5:
            self.open_gripper()
        else:
            self.close_gripper()

    def update_params(self, params: dict) -> None:
        pass


def make_backend(name: str, **kwargs):
    if name == "mock":
        return MockIiwaBackend(**kwargs)
    if ":" not in name:
        raise ValueError(f"Unsupported backend '{name}'. Use 'mock' or 'module.path:ClassName'.")
    module_name, class_name = name.split(":", 1)
    module = importlib.import_module(module_name)
    backend_cls = getattr(module, class_name)
    return backend_cls(**kwargs)
