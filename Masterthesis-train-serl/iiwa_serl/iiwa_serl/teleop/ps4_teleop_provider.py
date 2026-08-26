"""PS4 DualShock 4 teleoperation provider for iiwa + SERL.

Pure pygame — no ROS2 required. Talks to the SERL HTTP robot server directly
for gripper commands and optional force feedback rumble.

Controls
--------
  Left stick X/Y   → linear Y / X  (dy, dx in robot base frame)
  L2 / R2          → linear Z      (down / up)
  Right stick Y/X  → angular pitch / yaw (dry, drz)
  D-pad L/R        → angular roll  (drx), digital ±1
  R1 (hold)        → deadman switch — motion published only while held
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
    dpad_hat_index=0,     # hat index for the d-pad
    btn_cross=0,          # ✕  → mark success
    btn_circle=1,         # ○  → open gripper
    btn_square=2,         # □  → close gripper
    btn_triangle=3,       # △  → mark failure / abort episode
    btn_r1=5,             # R1 → deadman switch (hold to enable motion)
    deadzone=0.08,
    operator_position_front=True,   # flip X/Y if operator stands behind robot
)


def _apply_deadzone(value: float, dz: float) -> float:
    if abs(value) < dz:
        return 0.0
    sign = 1.0 if value > 0.0 else -1.0
    return sign * (abs(value) - dz) / (1.0 - dz)


def _trigger_fraction(raw: float, rest: float) -> float:
    span = 1.0 - rest
    return max(0.0, min(1.0, (raw - rest) / span)) if span > 0 else 0.0


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
        pygame.init()
        pygame.joystick.init()

        n = pygame.joystick.get_count()
        if n <= joystick_index:
            raise RuntimeError(
                f"No joystick at index {joystick_index} ({n} detected). "
                "Check that the PS4 controller is connected and that your user "
                "can read /dev/input/js* (see README)."
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
        self._update_button_edges()

        cfg = self._cfg
        js = self._js

        if not js.get_button(cfg["btn_r1"]):
            return np.zeros(6, dtype=np.float32)

        dz = cfg["deadzone"]
        rest = cfg["trigger_rest"]

        raw_lx = js.get_axis(cfg["axis_left_x"])
        raw_ly = js.get_axis(cfg["axis_left_y"])
        raw_rx = js.get_axis(cfg["axis_right_x"])
        raw_ry = js.get_axis(cfg["axis_right_y"])
        l2 = _trigger_fraction(js.get_axis(cfg["axis_l2"]), rest)
        r2 = _trigger_fraction(js.get_axis(cfg["axis_r2"]), rest)

        dx = -_apply_deadzone(raw_ly, dz)   # forward/back: push forward → +x
        dy = _apply_deadzone(raw_lx, dz)    # left/right
        dz_lin = r2 - l2                    # R2=up, L2=down

        dry = -_apply_deadzone(raw_ry, dz)  # pitch
        drz = -_apply_deadzone(raw_rx, dz)  # yaw
        drx = 0.0
        if js.get_numhats() > cfg["dpad_hat_index"]:
            drx = float(js.get_hat(cfg["dpad_hat_index"])[0])  # d-pad roll

        if not cfg["operator_position_front"]:
            dx, dy, drx, dry = -dx, -dy, -drx, -dry

        action = np.array([dx, dy, dz_lin, drx, dry, drz], dtype=np.float32)

        self._maybe_send_gripper()
        self._maybe_force_feedback()

        return action

    def is_success(self) -> bool:
        flag = self._success_flag
        self._success_flag = False
        return flag

    def is_failure(self) -> bool:
        flag = self._failure_flag
        self._failure_flag = False
        return flag

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
        if rising(cfg["btn_triangle"]):
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
