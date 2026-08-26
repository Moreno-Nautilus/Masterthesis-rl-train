"""ROS2-only gripper action client for the SERL iiwa backend.

Self-contained copy of ps4_controller_teleop's GripperActionClient so this package
has no cross-package Python dependency. Only imported from RealIiwaBackend.start(),
so importing iiwa_serl never requires ROS2.

Franka Hand open/close via the Move/Grasp action servers under /{gripper_prefix}.
If the custom PDZ gripper exposes different action servers, subclass and override
_send(), or point gripper_prefix at the right namespace.
"""

from __future__ import annotations

import threading
import time


class GripperActionClient:
    def __init__(self, node, gripper_prefix="fr3_gripper", speed=0.1, force=20.0,
                 max_width=0.08, epsilon_inner=0.08, epsilon_outer=0.08,
                 command_cooldown=0.5):
        from franka_msgs.action import Grasp, Homing, Move
        from rclpy.action import ActionClient

        self._node = node
        self._Grasp = Grasp
        self._Move = Move
        self.gripper_speed = speed
        self.gripper_force = force
        self.gripper_max_width = max_width
        self.gripper_epsilon_inner = epsilon_inner
        self.gripper_epsilon_outer = epsilon_outer
        self.gripper_command_cooldown = command_cooldown

        self._lock = threading.Lock()
        self._action_in_progress = False
        self._last_command_time = 0.0
        self.goal_state = "unknown"

        self.homing_client = ActionClient(node, Homing, f"/{gripper_prefix}/homing")
        self.move_client = ActionClient(node, Move, f"/{gripper_prefix}/move")
        self.grasp_client = ActionClient(node, Grasp, f"/{gripper_prefix}/grasp")

    def wait_for_servers(self, timeout_sec=2.0):
        for client, name in (
            (self.homing_client, "homing"),
            (self.move_client, "move"),
            (self.grasp_client, "grasp"),
        ):
            while not client.wait_for_server(timeout_sec=timeout_sec):
                self._node.get_logger().info(f"Waiting for {name} action server...")

    def home(self):
        from franka_msgs.action import Homing

        self.homing_client.send_goal_async(Homing.Goal())
        self.goal_state = "open"

    def open(self):
        self._send("open")

    def close(self):
        self._send("closed")

    def _send(self, target_state):
        with self._lock:
            now = time.time()
            if self._action_in_progress:
                return
            if now - self._last_command_time < self.gripper_command_cooldown:
                return
            if self.goal_state == target_state:
                return
            self._action_in_progress = True
            self._last_command_time = now

        try:
            if target_state == "open":
                goal = self._Move.Goal(width=self.gripper_max_width, speed=self.gripper_speed)
                future = self.move_client.send_goal_async(goal)
            else:
                goal = self._Grasp.Goal(
                    width=0.0, speed=self.gripper_speed, force=self.gripper_force
                )
                goal.epsilon.inner = self.gripper_epsilon_inner
                goal.epsilon.outer = self.gripper_epsilon_outer
                future = self.grasp_client.send_goal_async(goal)

            self.goal_state = target_state
            future.add_done_callback(self._on_goal_response)
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().error(f"Gripper command failed: {exc}")
            with self._lock:
                self._action_in_progress = False

    def _on_goal_response(self, future):
        try:
            goal_handle = future.result()
            if goal_handle.accepted:
                goal_handle.get_result_async().add_done_callback(self._on_goal_result)
                return
        except Exception:  # noqa: BLE001
            pass
        with self._lock:
            self._action_in_progress = False

    def _on_goal_result(self, future):
        with self._lock:
            self._action_in_progress = False
