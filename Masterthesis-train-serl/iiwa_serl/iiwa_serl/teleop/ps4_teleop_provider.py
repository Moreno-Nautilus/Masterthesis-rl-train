"""PS4 DualShock 4 teleoperation provider for iiwa + SERL.

Pure pygame — no ROS2 required. Talks to the SERL HTTP robot server directly
for gripper commands and optional force feedback rumble.

Controls
--------
  Left stick X/Y   → linear Y / X  (dy, dx in robot base frame)
  L2 / R2          → linear Z      (down / up)
  Right stick Y/X  → angular pitch / yaw (dry, drz)
  D-pad L/R        → angular roll  (drx), digital ±1
  R1 (hold)        → gates the TELEOP vector: get_action() returns zeros unless held. In HIL this
                     is a human-TAKEOVER switch (the policy/nominal still move without it), NOT a
                     global motion enable. In pure teleop/record it is the effective deadman.
  Cross  (btn 0)   → mark SUCCESS  (operator confirms insertion done)
  Triangle (btn 3) → mark FAILURE  (abandon episode, reset without reward)
  Circle   (btn 1) → open gripper  (HTTP /open_gripper)
  Square   (btn 2) → close gripper (HTTP /close_gripper)

Install deps: pip install pygame requests
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

try:
    import pygame
except ImportError as e:
    raise ImportError("pygame is required for PS4 teleoperation: pip install pygame") from e


# ---------------------------------------------------------------------------
# Abstract base — also imported by record_demo.py
# ---------------------------------------------------------------------------

class TeleopProvider(ABC):
    """Abstract teleop action source for SERL demo recording."""

    @abstractmethod
    def get_action(self) -> np.ndarray:
        """Return 6D delta action [dx, dy, dz, drx, dry, drz] in [-1, 1].
        Zeros mean idle / deadman not held."""
        ...

    @abstractmethod
    def is_success(self) -> bool:
        """True once per operator SUCCESS button press (edge-triggered)."""
        ...

    def is_failure(self) -> bool:
        """True once per operator FAILURE/abort button press (edge-triggered)."""
        return False

    def is_stop_forward(self) -> bool:
        """True WHILE the operator holds the stop-forward button (L1). Level, not edge:
        freezes the residual-mode nominal clock. Default False for providers without it."""
        return False

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# DualShock 4 defaults (Linux SDL2 / hid-sony driver)
# ---------------------------------------------------------------------------

_DS4_DEFAULTS = dict(
    axis_left_x=0,        # left stick horizontal
    axis_left_y=1,        # left stick vertical   (inverted: up = -1)
    axis_l2=2,            # L2 trigger            (rest=-1, pressed=+1)
    axis_right_x=3,       # right stick horizontal
    axis_right_y=4,       # right stick vertical   (inverted: up = -1)
    axis_r2=5,            # R2 trigger            (rest=-1, pressed=+1)
    trigger_rest=-1.0,    # raw axis value when trigger is fully released
    trigger_deadzone=0.08,  # ignore trigger fractions below this (kills resting-trigger Z creep)
    # SDL may report a trigger as 0.0 until it has been moved once.  With rest=-1,
    # that looks half pressed and caused +Z motion on the R1 rising edge.  A trigger
    # is ignored until a genuinely released sample has been observed.
    trigger_release_threshold=-0.75,
    dpad_hat_index=0,     # hat index for the d-pad
    btn_cross=0,          # ✕  → mark success
    btn_circle=1,         # ○  → open gripper
    btn_square=2,         # □  → close gripper
    btn_triangle=3,       # △  → mark failure / abort episode
    btn_r1=5,             # R1 → deadman switch (hold to enable motion)
    btn_l1=4,             # L1 → stop-forward: freeze the nominal clock while held (residual mode)
    deadzone=0.25,   # this DS4's left stick rests at ~-0.06 and noise-spikes to -0.19 (worn/miscentered
                     # stick, measured in teleop_record_diag CSV). 0.15 let that leak past → phantom -X
                     # drift + jitter at rest. 0.25 swallows the biased rest zone with margin.
    operator_position_front=True,   # flip X/Y if operator stands behind robot
)


def _apply_deadzone(value: float, dz: float) -> float:
    if abs(value) < dz:
        return 0.0
    sign = 1.0 if value > 0.0 else -1.0
    return sign * (abs(value) - dz) / (1.0 - dz)


def _trigger_fraction(raw: float, rest: float, dz: float = 0.0) -> float:
    span = 1.0 - rest
    frac = max(0.0, min(1.0, (raw - rest) / span)) if span > 0 else 0.0
    if frac < dz:
        return 0.0
    # rescale so the usable range still spans 0..1 above the deadzone
    return (frac - dz) / (1.0 - dz) if dz < 1.0 else 0.0


# ---------------------------------------------------------------------------
# PS4TeleopProvider
# ---------------------------------------------------------------------------

class PS4TeleopProvider(TeleopProvider):
    """Reads a DualShock 4 via pygame and returns 6D delta actions for SERL.

    Parameters
    ----------
    joystick_index : int
        Which pygame joystick to open (0 = first connected controller).
    server_url : str or None
        SERL HTTP robot server base URL. When set, Circle/Square buttons send
        gripper commands and force feedback rumble is enabled.
    force_feedback : bool
        Poll /getforce and drive rumble motors (requires server_url).
    force_min / force_max : float
        Force magnitude range (N) mapped to 0→1 rumble intensity.
    **mapping
        Override any key in _DS4_DEFAULTS (axis/button indices, deadzone, etc.).
    """

    def __init__(
        self,
        joystick_index: int = 0,
        server_url: Optional[str] = None,
        force_feedback: bool = False,
        force_min: float = 1.0,
        force_max: float = 8.0,
        **mapping,
    ):
        self._cfg = {**_DS4_DEFAULTS, **mapping}
        self._server_url = server_url.rstrip("/") if server_url else None
        self._ff_enabled = force_feedback and server_url is not None
        self._force_min = force_min
        self._force_max = force_max

        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        # SDL2 reads /dev/input/event*, which is often root:input mode rw---- (no user read) unless
        # you're in the `input` group — so pygame sees 0 joysticks even though /dev/input/js0 exists
        # and is world-readable. Default SDL to the legacy js0 interface, which works without the
        # group membership. Override by exporting SDL_JOYSTICK_DEVICE yourself (or `usermod -aG input`).
        if os.path.exists("/dev/input/js0"):
            os.environ.setdefault("SDL_JOYSTICK_DEVICE", "/dev/input/js0")
        pygame.init()
        pygame.joystick.init()

        n = pygame.joystick.get_count()
        if n <= joystick_index:
            raise RuntimeError(
                f"No joystick at index {joystick_index} ({n} detected). "
                "Check that the PS4 controller is connected. If /dev/input/js0 exists but pygame sees "
                "0 joysticks, SDL can't read /dev/input/event* — either `export "
                "SDL_JOYSTICK_DEVICE=/dev/input/js0` or `sudo usermod -aG input $USER` then re-login."
            )
        self._js = pygame.joystick.Joystick(joystick_index)
        self._js.init()
        print(
            f"[PS4] Opened: '{self._js.get_name()}' "
            f"({self._js.get_numaxes()} axes, {self._js.get_numbuttons()} buttons)"
        )

        self._prev_buttons: list[bool] = [False] * self._js.get_numbuttons()
        self._success_flag = False
        self._failure_flag = False
        self._last_ff_time = 0.0
        self._trigger_ready = {"l2": False, "r2": False}
        self._trigger_warning_shown: set[str] = set()
        self._last_snapshot: dict = {}

        if self._ff_enabled:
            try:
                import requests  # noqa: F401
            except ImportError:
                print("[PS4] Warning: 'requests' not installed — force feedback disabled.")
                self._ff_enabled = False

    # ------------------------------------------------------------------
    # TeleopProvider interface
    # ------------------------------------------------------------------

    def get_action(self) -> np.ndarray:
        """Return 6D delta action in [-1, 1]. Zeros while R1 not held."""
        pygame.event.pump()

        # SAFETY: if the controller went away (a dropout can leave a ghost device stuck reporting a
        # pressed button -> deadman false-True -> runaway), FAIL SAFE: publish a neutral snapshot
        # with r1=0 so the deadman reads OFF and the robot freezes. get_init() is False if the
        # joystick was closed/lost (get_attached() doesn't exist in this pygame).
        if not self._js.get_init():
            self._last_snapshot = dict(
                r1=0, raw_lx=0.0, raw_ly=0.0, raw_rx=0.0, raw_ry=0.0,
                raw_l2=0.0, raw_r2=0.0, l2_ready=0, r2_ready=0,
            )
            return np.zeros(6, dtype=np.float32)

        self._update_button_edges()

        cfg = self._cfg
        js = self._js
        dz = cfg["deadzone"]
        rest = cfg["trigger_rest"]

        raw_lx = js.get_axis(cfg["axis_left_x"])
        raw_ly = js.get_axis(cfg["axis_left_y"])
        raw_rx = js.get_axis(cfg["axis_right_x"])
        raw_ry = js.get_axis(cfg["axis_right_y"])
        raw_l2 = js.get_axis(cfg["axis_l2"])
        raw_r2 = js.get_axis(cfg["axis_r2"])
        r1 = int(js.get_button(cfg["btn_r1"]))

        release_threshold = cfg["trigger_release_threshold"]
        if raw_l2 <= release_threshold:
            self._trigger_ready["l2"] = True
        if raw_r2 <= release_threshold:
            self._trigger_ready["r2"] = True

        # This is the exact sample used to create the returned action.  Callers must
        # call get_action() first and debug_snapshot() second; no second event pump is
        # performed, so raw/action diagnostics are frame-aligned.
        self._last_snapshot = dict(
            r1=r1,
            raw_lx=float(raw_lx),
            raw_ly=float(raw_ly),
            raw_rx=float(raw_rx),
            raw_ry=float(raw_ry),
            raw_l2=float(raw_l2),
            raw_r2=float(raw_r2),
            l2_ready=int(self._trigger_ready["l2"]),
            r2_ready=int(self._trigger_ready["r2"]),
        )

        if not r1:
            return np.zeros(6, dtype=np.float32)

        for name, raw in (("l2", raw_l2), ("r2", raw_r2)):
            if not self._trigger_ready[name] and name not in self._trigger_warning_shown:
                print(
                    f"[PS4] {name.upper()} is not calibrated and is being ignored. "
                    "Squeeze it once, then fully release it before using Z motion."
                )
                self._trigger_warning_shown.add(name)

        tdz = cfg["trigger_deadzone"]
        l2 = _trigger_fraction(raw_l2, rest, tdz) if self._trigger_ready["l2"] else 0.0
        r2 = _trigger_fraction(raw_r2, rest, tdz) if self._trigger_ready["r2"] else 0.0

        dx = -_apply_deadzone(raw_ly, dz)   # forward/back: push forward → +x
        dy = _apply_deadzone(raw_lx, dz)    # left/right
        dz_lin = r2 - l2                    # R2=up, L2=down

        # Rotations are applied in the tool frame (see compose_delta_pose). So:
        #   drx / dry = the two tilt axes (right stick)
        #   drz       = spin about the tool Z = joint-7 rotation (d-pad L/R)
        dry = -_apply_deadzone(raw_ry, dz)  # right stick up/down    -> tilt about tool Y
        drx = -_apply_deadzone(raw_rx, dz)  # right stick left/right -> tilt about tool X
        drz = 0.0
        if js.get_numhats() > cfg["dpad_hat_index"]:
            drz = float(js.get_hat(cfg["dpad_hat_index"])[0])  # d-pad L/R -> rotate about joint 7

        if not cfg["operator_position_front"]:
            dx, dy, drx, dry = -dx, -dy, -drx, -dry

        action = np.array([dx, dy, dz_lin, drx, dry, drz], dtype=np.float32)

        self._maybe_send_gripper()
        self._maybe_force_feedback()

        return action

    def debug_snapshot(self) -> dict:
        """Return the raw controller sample used by the latest get_action() call."""
        return self._last_snapshot.copy()

    def is_success(self) -> bool:
        flag = self._success_flag
        self._success_flag = False
        return flag

    def is_failure(self) -> bool:
        flag = self._failure_flag
        self._failure_flag = False
        return flag

    def is_stop_forward(self) -> bool:
        """Level read of L1 (stop-forward). Uses the latched sample from the last
        get_action() pump; callers should call get_action() first each step (the
        intervention wrapper does). No extra event pump here (frame-aligned)."""
        idx = self._cfg["btn_l1"]
        return bool(0 <= idx < self._js.get_numbuttons() and self._js.get_button(idx))

    def close(self) -> None:
        try:
            self._js.stop_rumble()
        except Exception:
            pass
        pygame.joystick.quit()
        pygame.quit()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _update_button_edges(self):
        cfg = self._cfg
        js = self._js
        buttons = [js.get_button(i) for i in range(js.get_numbuttons())]

        def rising(idx):
            return 0 <= idx < len(buttons) and buttons[idx] and not self._prev_buttons[idx]

        if rising(cfg["btn_cross"]):
            self._success_flag = True
        # ABORT = Triangle (primary). Square also aborts ONLY when no server_url is set (HIL/record
        # path), where Square is not used for the gripper — avoids the earlier Square/Triangle doc
        # mismatch (#10) without breaking the gripper mapping when a server_url IS configured.
        if rising(cfg["btn_triangle"]):
            self._failure_flag = True
        if self._server_url is None and rising(cfg["btn_square"]):
            self._failure_flag = True

        self._prev_buttons = buttons

    def _maybe_send_gripper(self):
        if not self._server_url:
            return
        import requests

        cfg = self._cfg
        js = self._js
        buttons = [js.get_button(i) for i in range(js.get_numbuttons())]

        def rising(idx):
            return 0 <= idx < len(buttons) and buttons[idx] and not self._prev_buttons[idx]

        if rising(cfg["btn_circle"]):
            try:
                requests.post(f"{self._server_url}/open_gripper", timeout=0.5)
            except Exception as exc:
                print(f"[PS4] open_gripper failed: {exc}")
        elif rising(cfg["btn_square"]):
            try:
                requests.post(f"{self._server_url}/close_gripper", timeout=0.5)
            except Exception as exc:
                print(f"[PS4] close_gripper failed: {exc}")

    def _maybe_force_feedback(self):
        if not self._ff_enabled:
            return
        now = time.time()
        if now - self._last_ff_time < 0.05:   # poll at ~20 Hz
            return
        self._last_ff_time = now
        import requests

        try:
            resp = requests.post(f"{self._server_url}/getforce", timeout=0.1)
            force = np.array(resp.json()["force"], dtype=np.float64)
            mag = float(np.linalg.norm(force))
            intensity = np.clip(
                (mag - self._force_min) / max(self._force_max - self._force_min, 1e-6), 0.0, 1.0
            )
            if intensity > 1e-3:
                self._js.rumble(intensity, intensity, 100)
            else:
                self._js.stop_rumble()
        except Exception:
            pass  # don't crash training on a stale force reading
