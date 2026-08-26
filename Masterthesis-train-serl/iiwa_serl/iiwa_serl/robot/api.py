from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any

import numpy as np
import requests


@dataclass
class RobotState:
    pose: np.ndarray
    vel: np.ndarray
    force: np.ndarray
    torque: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    jacobian: np.ndarray
    gripper: float
    timestamp: float

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "RobotState":
        return cls(
            pose=np.asarray(payload.get("pose", np.zeros(7)), dtype=np.float64),
            vel=np.asarray(payload.get("vel", np.zeros(6)), dtype=np.float64),
            force=np.asarray(payload.get("force", np.zeros(3)), dtype=np.float64),
            torque=np.asarray(payload.get("torque", np.zeros(3)), dtype=np.float64),
            q=np.asarray(payload.get("q", np.zeros(7)), dtype=np.float64),
            dq=np.asarray(payload.get("dq", np.zeros(7)), dtype=np.float64),
            jacobian=np.asarray(payload.get("jacobian", np.zeros((6, 7))), dtype=np.float64),
            gripper=float(payload.get("gripper", payload.get("gripper_pos", 0.0))),
            timestamp=float(payload.get("timestamp", time.time())),
        )

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, np.ndarray):
                payload[key] = value.tolist()
        return payload


class RobotServerClient:
    def __init__(self, server_url: str, timeout_s: float = 2.0):
        self.server_url = server_url.rstrip("/")
        self.timeout_s = timeout_s

    def _post(self, route: str, payload: dict[str, Any] | None = None) -> Any:
        response = requests.post(
            f"{self.server_url}/{route.lstrip('/')}",
            json=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        if response.headers.get("content-type", "").startswith("application/json"):
            return response.json()
        return response.text

    def move_pose(self, pose7: np.ndarray) -> None:
        self._post("pose", {"pose": np.asarray(pose7, dtype=np.float64).tolist()})

    def get_state(self) -> RobotState:
        try:
            payload = self._post("getstate")
        except requests.HTTPError:
            payload = {
                "pose": self._post("getpos")["pose"],
                "vel": self._post("getvel")["vel"],
                "force": self._post("getforce")["force"],
                "torque": self._post("gettorque")["torque"],
                "q": self._post("getq")["q"],
                "dq": self._post("getdq")["dq"],
                "jacobian": self._post("getjacobian")["jacobian"],
                "gripper": self._post("get_gripper")["gripper"],
                "timestamp": time.time(),
            }
        return RobotState.from_json(payload)

    def joint_reset(self) -> None:
        self._post("jointreset")

    def clear_errors(self) -> None:
        self._post("clearerr")

    def activate_gripper(self) -> None:
        self._post("activate_gripper")

    def reset_gripper(self) -> None:
        self._post("reset_gripper")

    def open_gripper(self) -> None:
        self._post("open_gripper")

    def close_gripper(self) -> None:
        self._post("close_gripper")

    def move_gripper(self, value: float) -> None:
        self._post("move_gripper", {"gripper": float(value)})

    def update_param(self, params: dict[str, Any]) -> None:
        self._post("update_param", params)


class LocalBackendClient:
    def __init__(self, backend):
        self.backend = backend

    def move_pose(self, pose7: np.ndarray) -> None:
        self.backend.move_pose(np.asarray(pose7, dtype=np.float64))

    def get_state(self) -> RobotState:
        return self.backend.get_state()

    def joint_reset(self) -> None:
        self.backend.reset_joints()

    def clear_errors(self) -> None:
        self.backend.clear_errors()

    def activate_gripper(self) -> None:
        self.backend.activate_gripper()

    def reset_gripper(self) -> None:
        self.backend.reset_gripper()

    def open_gripper(self) -> None:
        self.backend.open_gripper()

    def close_gripper(self) -> None:
        self.backend.close_gripper()

    def move_gripper(self, value: float) -> None:
        self.backend.move_gripper(float(value))

    def update_param(self, params: dict[str, Any]) -> None:
        self.backend.update_params(params)
