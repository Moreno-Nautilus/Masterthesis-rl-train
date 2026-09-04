"""Nominal-trajectory progress + residual composition for residual-mode insertion.

Self-contained and FK-free so it is easy to unit-test (tests/test_nominal_residual.py)
and to review (the progress/scaling logic is the highest-risk piece — HIL_RESIDUAL_SPEC.md §4/§5).

Residual semantics (reworked 2026-09-03 after supervisor call)
--------------------------------------------------------------
The nominal is a straight line reset_pose6 -> goal_pose6 along ``insertion_axis``.
Two persistent state variables per episode:

* ``progress`` in [0,1]  -- how far ALONG the insertion axis. Advances each step by
  ``base_rate * forward_scale``. ``base_rate = 1/(N-1)`` matches the plan's slow rate.
  ``forward_scale`` comes from the along-axis component of the residual action:
    scale = clip(1 + k * a_along, min_forward_scale, max_forward_scale)
  so a_along = 0 -> nominal speed (scale 1); a_along < 0 -> slow/halt/back off a little
  (feel contact); clamped so it can never lunge forward fast or yank back hard.
  Contact or L1 (stop-forward) force scale <= 0 region via a hard pause (progress frozen).

* ``lateral_offset`` (3,) base frame -- PERSISTENT accumulated lateral shift. Each step
  the ORTHOGONAL (to-axis) component of the residual translation is ADDED to this offset
  (not applied for one step and reverted). This permanently shifts the whole insertion
  line sideways so, once the operator/policy slides the peg over onto the real hole, it
  STAYS there and the forward push continues down the corrected line. Bounded by a box.

Rotation: OFF by default (locked to the nominal, which is ~constant orientation). A flag
(``rotation_enable``) lets the residual also correct rotation (pipe/screw fallback).

Frames: ``insertion_axis`` and ``lateral_offset`` are base/world translation vectors; the
residual translation here is in that SAME base frame (the env adds dxyz in base frame).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def project_out_axis(vec3: np.ndarray, axis_unit: np.ndarray) -> np.ndarray:
    """Return the component of ``vec3`` orthogonal to ``axis_unit``."""
    vec3 = np.asarray(vec3, dtype=np.float64)
    axis_unit = np.asarray(axis_unit, dtype=np.float64)
    return vec3 - np.dot(vec3, axis_unit) * axis_unit


@dataclass
class ResidualStep:
    """Result of composing one residual step."""
    forward_scale: float          # applied scale on the nominal forward step
    a_along: float                # raw along-axis residual component (diagnostic)
    lateral_offset: np.ndarray    # (3,) current accumulated lateral offset (base frame)
    drot: np.ndarray              # (3,) rotation delta to apply (zeros if rotation disabled)


class NominalResidual:
    """Nominal progress + persistent lateral shift + forward-scaling for ONE insert.

    Parameters
    ----------
    reset_pose6, goal_pose6 : (6,) pose6 [x,y,z,rx,ry,rz]  -- straight-line endpoints (EE).
    insertion_axis : (3,) unit  -- world/base insertion direction.
    nominal_n : int  -- nominal waypoint count (sets base_rate = 1/(n-1)).
    forward_gain : float  -- k mapping along-axis residual to forward-scale change.
    min_forward_scale, max_forward_scale : float  -- clamp on forward_scale (e.g. -0.5, 1.0).
    lateral_gain : float  -- scales how much orthogonal residual accumulates per step.
    lateral_limit : (3,) or float  -- +/- box (m, base frame) bounding lateral_offset.
    rotation_enable : bool  -- if True, residual rotation passes through; else zeroed.
    """

    def __init__(
        self,
        reset_pose6: np.ndarray,
        goal_pose6: np.ndarray,
        insertion_axis: np.ndarray,
        nominal_n: int,
        forward_gain: float = 1.0,
        min_forward_scale: float = -0.5,
        max_forward_scale: float = 1.0,
        lateral_gain: float = 1.0,
        lateral_limit=0.03,
        rotation_enable: bool = False,
    ):
        self.reset_pose6 = np.asarray(reset_pose6, dtype=np.float64).copy()
        self.goal_pose6 = np.asarray(goal_pose6, dtype=np.float64).copy()
        self.axis = np.asarray(insertion_axis, dtype=np.float64).copy()
        self.n = max(int(nominal_n), 2)
        self.base_rate = 1.0 / (self.n - 1)
        self.forward_gain = float(forward_gain)
        self.min_forward_scale = float(min_forward_scale)
        self.max_forward_scale = float(max_forward_scale)
        self.lateral_gain = float(lateral_gain)
        self.lateral_limit = np.asarray(
            lateral_limit if np.ndim(lateral_limit) else [lateral_limit] * 3, dtype=np.float64
        )
        self.rotation_enable = bool(rotation_enable)

        self._progress = 0.0
        self._lateral = np.zeros(3, dtype=np.float64)

    # -- state ------------------------------------------------------------
    def reset(self) -> None:
        self._progress = 0.0
        self._lateral[:] = 0.0

    @property
    def progress(self) -> float:
        return self._progress

    @property
    def at_goal(self) -> bool:
        return self._progress >= 1.0 - 1e-9

    @property
    def lateral_offset(self) -> np.ndarray:
        return self._lateral.copy()

    # -- step -------------------------------------------------------------
    def step(self, dxyz_m: np.ndarray, drot_rad: np.ndarray, *, contact: bool,
             stop_forward: bool, action_along_norm: float = 0.0) -> ResidualStep:
        """Advance the nominal by a scaled forward step and accumulate the lateral shift.

        dxyz_m  : residual translation ALREADY scaled to METRES this step (action*scale).
                  Its ORTHOGONAL component accumulates into the persistent lateral offset (metres).
        action_along_norm : the NORMALIZED [-1,1] along-axis action component (frame: dot of the
                  raw action translation with the insertion axis). Drives the forward-speed scale
                  so full-negative reaches the halt clamp regardless of the metric action scale (#16).
        drot_rad: residual rotation already scaled to radians; passed through iff rotation_enable.
        """
        dxyz_m = np.asarray(dxyz_m, dtype=np.float64)
        a_along_m = float(np.dot(dxyz_m, self.axis))    # metres along axis (for orthogonal split)
        ortho = dxyz_m - a_along_m * self.axis          # metres orthogonal this step

        # forward scale in NORMALIZED units: full negative along-action (-1) -> 1-forward_gain,
        # clamped to [min,max]. So with forward_gain=1.5 a full pull reaches the -0.5 halt clamp.
        forward_scale = float(np.clip(
            1.0 + self.forward_gain * float(action_along_norm),
            self.min_forward_scale, self.max_forward_scale,
        ))
        # hard pause: contact or operator stop-forward freezes forward progress entirely
        # (overrides any positive scale) but NEVER forces a backward move on its own.
        if contact or stop_forward:
            forward_scale = min(forward_scale, 0.0)

        # advance persistent progress (clamped to [0,1]); a negative scale can ease back
        # off the goal slightly to relieve contact, but not below 0.
        self._progress = float(np.clip(
            self._progress + self.base_rate * forward_scale, 0.0, 1.0
        ))

        # accumulate persistent lateral offset (orthogonal, metres), bounded by the box.
        self._lateral = np.clip(
            self._lateral + self.lateral_gain * ortho,
            -self.lateral_limit, self.lateral_limit,
        )

        drot = np.asarray(drot_rad, dtype=np.float64).copy() if self.rotation_enable else np.zeros(3)
        return ResidualStep(
            forward_scale=forward_scale,
            a_along=a_along_m,
            lateral_offset=self._lateral.copy(),
            drot=drot,
        )

    def retract_progress(self, step_m: float) -> None:
        """Ease the nominal progress BACKWARD along the axis by ~step_m metres (contact retract).

        Making the retract part of the nominal state (not a one-off command override) means the
        NEXT step composes from the retracted progress — no re-command of the old forward target,
        so no push/retract oscillation (#13). Lateral offset is preserved."""
        total_len = float(np.linalg.norm(self.goal_pose6[:3] - self.reset_pose6[:3]))
        if total_len < 1e-9:
            return
        dprog = step_m / total_len
        self._progress = float(np.clip(self._progress - dprog, 0.0, 1.0))

    # -- nominal pose (with persistent lateral shift) ---------------------
    def target_pose6(self) -> np.ndarray:
        """Current commanded pose6 = nominal-on-axis(progress) + persistent lateral offset.

        Rotation is the interpolated nominal orientation (residual rotation, if enabled, is
        returned separately as drot and composed by the caller in the tool frame)."""
        f = self._progress
        pose = (1.0 - f) * self.reset_pose6 + f * self.goal_pose6
        pose[:3] = pose[:3] + self._lateral
        return pose

    # -- retract ----------------------------------------------------------
    @property
    def retract_dir(self) -> np.ndarray:
        """Unit direction to back off on contact = -insertion_axis (per-insert)."""
        return -self.axis
