from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from iiwa_serl.utils.transformations import compose_delta_pose


@dataclass(frozen=True)
class TeleopPoseCommand:
    """One stateful teleop target update.

    ``hold`` is deliberately distinct from a zero Cartesian move: the LBR backend can
    then pin the measured joints directly without running IK or a nullspace objective.
    """

    kind: Literal["move", "hold", "idle"]
    target_pose: np.ndarray


class TeleopPoseTarget:
    """Integrate teleop deltas onto a target and brake on every release edge.

    Targets accumulate while an axis is driven so plant lag does not turn commands
    into dropped motion.  When an axis changes between active and inactive, that axis
    is rebased once to the measured pose.  In particular, releasing Z cannot leave an
    old Z target queued while another axis remains active.
    """

    def __init__(
        self,
        linear_scale: np.ndarray | float,
        angular_scale: float,
        *,
        base_frame_actions: bool,
        action_epsilon: float = 1e-6,
    ) -> None:
        scale = np.asarray(linear_scale, dtype=np.float64)
        self.linear_scale = np.broadcast_to(scale, (3,)).copy()
        self.angular_scale = float(angular_scale)
        self.base_frame_actions = bool(base_frame_actions)
        self.action_epsilon = float(action_epsilon)
        self._target_pose: np.ndarray | None = None
        self._previous_linear_active = np.zeros(3, dtype=bool)
        self._previous_rotation_active = np.zeros(3, dtype=bool)
        self._active = False

    def reset(self, measured_pose: np.ndarray) -> None:
        self._target_pose = self._pose7(measured_pose)
        self._previous_linear_active[:] = False
        self._previous_rotation_active[:] = False
        self._active = False

    def sync_target(self, target_pose: np.ndarray) -> None:
        """Keep the integrator synchronized with a workspace-clipped target."""
        self._target_pose = self._pose7(target_pose)

    @property
    def target_pose(self) -> np.ndarray | None:
        return None if self._target_pose is None else self._target_pose.copy()

    def update(self, action: np.ndarray, measured_pose: np.ndarray) -> TeleopPoseCommand:
        action = np.asarray(action, dtype=np.float64).reshape(6)
        measured = self._pose7(measured_pose)
        if self._target_pose is None:
            self.reset(measured)

        linear_active = np.abs(action[:3]) > self.action_epsilon
        rotation_active = np.abs(action[3:6]) > self.action_epsilon
        active = bool(np.any(linear_active) or np.any(rotation_active))

        if not active:
            kind: Literal["hold", "idle"] = "hold" if self._active else "idle"
            if self._active:
                # Pin exactly where the release was observed.  Do not refresh this on
                # later idle samples: doing so ratchets a moving plant pose forever.
                self._target_pose = measured.copy()
            self._previous_linear_active[:] = False
            self._previous_rotation_active[:] = False
            self._active = False
            return TeleopPoseCommand(kind, self._target_pose.copy())

        if not self._active:
            self._target_pose = measured.copy()
        else:
            changed_linear = linear_active != self._previous_linear_active
            self._target_pose[:3][changed_linear] = measured[:3][changed_linear]
            if np.any(rotation_active != self._previous_rotation_active):
                self._target_pose[3:7] = measured[3:7]

        dxyz = action[:3] * self.linear_scale
        drot = action[3:6] * self.angular_scale
        if self.base_frame_actions:
            if np.any(rotation_active):
                self._target_pose = compose_delta_pose(
                    self._target_pose,
                    delta_xyz=np.zeros(3, dtype=np.float64),
                    delta_rot_xyz=drot,
                )
            self._target_pose[:3] += dxyz
        else:
            self._target_pose = compose_delta_pose(
                self._target_pose,
                delta_xyz=dxyz,
                delta_rot_xyz=drot,
            )

        self._previous_linear_active = linear_active
        self._previous_rotation_active = rotation_active
        self._active = True
        return TeleopPoseCommand("move", self._target_pose.copy())

    @staticmethod
    def _pose7(pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose, dtype=np.float64).reshape(7)
        return pose.copy()
