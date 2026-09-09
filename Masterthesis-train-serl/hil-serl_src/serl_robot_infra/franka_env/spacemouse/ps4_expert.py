"""PS4 (DualShock 4) expert — a drop-in replacement for SpaceMouseExpert.

Franka pivot (2026-09-08): teleop device is a PS4 controller, not a 3Dconnexion
SpaceMouse. The HIL-SERL intervention path only touches the expert via ONE contract:

    expert_a, buttons = expert.get_action()   # -> (np.ndarray shape (6,), [left, right])

and intervenes when ``np.linalg.norm(expert_a) > 0.001`` (see
franka_env/envs/wrappers.py::SpacemouseIntervention). So this mirrors
``SpaceMouseExpert`` exactly: a daemon process continuously reading the pad via pygame,
publishing the latest (6-vec, 2 buttons) into shared state; ``get_action()`` returns it.
Swap ``SpaceMouseExpert()`` -> ``PS4Expert()`` and touch nothing else.

Axis / button map is taken verbatim from the student's ROS2 teleop
``ps4_controller_teleop/ps4_publisher.py`` (which drives the same Franka controller):

- Left stick        -> linear x / y   (forward-back / left-right)
- L2 / R2 triggers  -> linear z        (down / up)
- Right stick       -> angular y / z   (pitch / yaw)
- D-pad left/right  -> angular x       (roll, digital)
- **R1 (held)       -> deadman**: output is ZERO unless R1 is held (matches the teleop's
                       enable_button gate — no accidental motion / intervention)
- Square            -> close gripper   (maps to SpacemouseIntervention's `left`)
- Circle            -> open gripper    (maps to SpacemouseIntervention's `right`)

Event buttons (read separately via ``get_events()`` so the SpaceMouse ``get_action()``
contract stays 2-button; consumed by PS4RewardWrapper):

- **X (Cross)       -> SUCCESS**: reward 1, end the episode immediately (seated).
- **Triangle        -> ABORT**:   reward 0, end the episode.
- **Circle          -> REGRASP**: request a regrasp at the next reset (release -> you hand
                       the part into the gripper -> close). Replaces the old F1 keyboard.

(Square stays the gripper-close button in get_action(); with GripperCloseEnv +
single-arm-fixed-gripper the gripper buttons are unused, so Circle is free for regrasp.)

The 6-vec ordering is [x, y, z, roll(ang_x), pitch(ang_y), yaw(ang_z)], the same order
the env's step() consumes (action[:3] -> xyz delta, action[3:6] -> rotvec). This matches
what SpaceMouseExpert emits ([-y, x, z, -roll, -pitch, -yaw] in device frame), because
ps4_publisher already produces the controller-frame linear_x/y/z + angular_x/y/z.
"""

import multiprocessing
import os
import time
import numpy as np
from typing import Tuple

import pygame

# get_action() returns zeros (fail-safe: no intervention) if the reader hasn't published
# within this many seconds — covers a crashed reader or a disconnected controller so a
# stale nonzero action can NEVER be replayed forever.
STALE_TIMEOUT_S = float(os.environ.get("PS4_STALE_TIMEOUT_S", 0.5))
# Reader poll period (Hz). The env steps at ~10Hz; 100Hz reader keeps latency low without
# a busy-spin.
READ_PERIOD_S = 1.0 / float(os.environ.get("PS4_READ_HZ", 100.0))


# Max time to wait for the shared-state lock. The control loop must NEVER block on a child
# that died holding it — on timeout we fail safe (zeros / no events).
LOCK_TIMEOUT_S = float(os.environ.get("PS4_LOCK_TIMEOUT_S", 0.05))


def _now() -> float:
    """Monotonic clock — heartbeat/staleness must not be fooled by system clock changes."""
    return time.monotonic()


# --- axis / button indices (defaults match ps4_publisher.py's declared parameters) ----
# Common Linux SDL2 DualShock 4 layout. Override via the PS4_* env vars if `jstest` on
# the rig shows a different mapping (see FRANKA_RIG_CHECKLIST.md).
AXIS_LEFT_X = int(os.environ.get("PS4_AXIS_LEFT_X", 0))
AXIS_LEFT_Y = int(os.environ.get("PS4_AXIS_LEFT_Y", 1))
AXIS_L2 = int(os.environ.get("PS4_AXIS_L2", 2))
AXIS_RIGHT_X = int(os.environ.get("PS4_AXIS_RIGHT_X", 3))
AXIS_RIGHT_Y = int(os.environ.get("PS4_AXIS_RIGHT_Y", 4))
AXIS_R2 = int(os.environ.get("PS4_AXIS_R2", 5))
TRIGGER_REST = float(os.environ.get("PS4_TRIGGER_REST", -1.0))

BUTTON_GRIPPER_OPEN = int(os.environ.get("PS4_BTN_OPEN", 1))   # Circle
BUTTON_GRIPPER_CLOSE = int(os.environ.get("PS4_BTN_CLOSE", 3))  # Square
BUTTON_ENABLE = int(os.environ.get("PS4_BTN_ENABLE", 5))        # R1 deadman
# Reward / episode-end buttons (SDL DualShock 4: X/Cross=0, Triangle=2).
BUTTON_SUCCESS = int(os.environ.get("PS4_BTN_SUCCESS", 0))      # X -> success, end episode
BUTTON_ABORT = int(os.environ.get("PS4_BTN_ABORT", 2))         # Triangle -> abort, end episode
# Regrasp request (Circle=1). Safe to reuse: with a pregrasped fixed gripper the gripper
# buttons in get_action() are unused. Pressing it flags a regrasp at the next reset.
BUTTON_REGRASP = int(os.environ.get("PS4_BTN_REGRASP", 1))     # Circle -> request regrasp
DPAD_HAT_INDEX = int(os.environ.get("PS4_DPAD_HAT", 0))
# 0.15 chosen at the rig 2026-09-09 (user preference). Measured stick rest-drift was
# 0.0, so this is headroom, not a drift fix: SpacemouseIntervention treats ANY
# ||a||>0.001 as a human intervention that overrides the policy AND is written to
# intervene_action, so a twitchy stick would silently poison the demo/replay buffer.
DEADZONE = float(os.environ.get("PS4_DEADZONE", 0.15))
JOYSTICK_INDEX = int(os.environ.get("PS4_JOYSTICK_INDEX", 0))
# Operator standing in front of the arm flips x/y/roll/pitch, same as the teleop node.
OPERATOR_FRONT = os.environ.get("PS4_OPERATOR_FRONT", "1") not in ("0", "false", "False")


def _apply_deadzone(value, deadzone):
    if abs(value) < deadzone:
        return 0.0
    sign = 1.0 if value > 0.0 else -1.0
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


def _trigger_fraction(raw, rest=TRIGGER_REST):
    span = 1.0 - rest
    if span <= 0.0:
        return 0.0
    return max(0.0, min(1.0, (raw - rest) / span))


# SDL reports 0.0 for an analog trigger that has not been moved since the device was
# opened, and 0.0 is ALSO a legitimate mid-travel value — the two are indistinguishable
# from a single sample. With TRIGGER_REST=-1 an uninitialised trigger therefore reads as
# HALF PRESSED (fraction 0.5), so `linear_z = r2 - l2` is wrong until both have been
# moved once: pressing L2 alone gives z=-0.5, and RELEASING it gives z=+0.5 — a phantom
# UPWARD command (observed on the rig 2026-09-09 as "L2 goes down, then comes back up").
# Fix: require each trigger to have been seen at its true rest value (<= this threshold)
# at least once before z motion is allowed. Until then the reader reports z=0 and
# `triggers_ready` is False so the operator can be prompted.
TRIGGER_READY_MAX = TRIGGER_REST + 0.15


class PS4Expert:
    """PS4 controller interface with the SpaceMouseExpert contract.

    ``get_action() -> (np.ndarray(6,), [left, right])`` where left=close-gripper button
    (Square), right=open-gripper button (Circle). Output 6-vec is ZERO unless R1 (deadman)
    is held.
    """

    def __init__(self):
        # Shared state between the reader process and get_action(). A single Lock guards
        # every read/write so callers see a COHERENT snapshot (action+buttons+events
        # together) and event read-then-clear is atomic.
        self.manager = multiprocessing.Manager()
        self._lock = self.manager.Lock()
        self.latest_data = self.manager.dict()
        self.latest_data["action"] = [0.0] * 6
        self.latest_data["buttons"] = [0, 0]  # [left(close), right(open)]
        self.latest_data["deadman_held"] = False  # is R1 currently held?
        # Latched, edge-triggered events (cleared on consume in get_events()).
        self.latest_data["success"] = False
        self.latest_data["abort"] = False
        self.latest_data["regrasp"] = False
        # Liveness: heartbeat timestamp updated every reader tick; init_error surfaces a
        # child-side joystick-init failure to the parent (else it would silently zero).
        # False until BOTH analog triggers have reported their true rest value once (see
        # TRIGGER_READY_MAX). z motion is suppressed until then; callers can prompt the
        # operator to squeeze+release L2 and R2. Resets with each new reader process.
        self.latest_data["triggers_ready"] = False
        self.latest_data["heartbeat"] = 0.0
        self.latest_data["init_error"] = ""
        self.latest_data["runtime_error"] = ""
        self._warned_dead = False  # one-shot loud warning when operator control is lost

        self.process = multiprocessing.Process(target=self._read_ps4)
        self.process.daemon = True
        self.process.start()

        # Wait briefly for the child to either publish a first heartbeat or report an init
        # error — so a missing/unreadable controller fails LOUDLY here, not silently later.
        deadline = _now() + 5.0
        started = False
        err = ""
        while _now() < deadline:
            err = self.latest_data.get("init_error", "")
            if err:
                break
            if self.latest_data.get("heartbeat", 0.0) > 0.0:
                started = True
                break
            time.sleep(0.05)
        if not started:
            # Tear down the child + Manager before raising, else they leak.
            self.close()
            raise RuntimeError(
                f"PS4Expert reader failed to start: {err}" if err else
                "PS4Expert reader did not produce a heartbeat within 5s — controller "
                "not detected / reader stuck (see FRANKA_RIG_CHECKLIST.md)."
            )

    def _init_joystick(self):
        # Headless-safe (matches ps4_publisher): no display needed.
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() <= JOYSTICK_INDEX:
            raise RuntimeError(
                f"No PS4 joystick at index {JOYSTICK_INDEX} "
                f"({pygame.joystick.get_count()} detected). Check connection/pairing and "
                "read permission on /dev/input/js* (see FRANKA_RIG_CHECKLIST.md)."
            )
        js = pygame.joystick.Joystick(JOYSTICK_INDEX)
        js.init()
        return js

    def _read_ps4(self):
        try:
            js = self._init_joystick()
        except Exception as e:  # report to the parent instead of dying silently
            self.latest_data["init_error"] = repr(e)
            return
        try:
            self._read_loop(js)
        except Exception as e:
            # RUNTIME failure (controller unplugged, pygame/device error). Record it so the
            # parent can report the loss of operator control instead of silently zeroing.
            try:
                self.latest_data["runtime_error"] = repr(e)
            except Exception:
                pass
            return

    def _read_loop(self, js):
        n0 = js.get_numbuttons()
        prev_success = prev_abort = prev_regrasp = False
        # Per-reader-lifecycle trigger readiness (see TRIGGER_READY_MAX). A NEW reader
        # process starts unready again, which is correct: the fault is per-open-device.
        l2_ready = r2_ready = False
        triggers_ready = False
        while True:
            loop_start = _now()
            pygame.event.pump()

            enabled = bool(js.get_button(BUTTON_ENABLE))  # R1 deadman

            # Poll the triggers EVERY loop, deadman or not: readiness must be able to
            # initialise while R1 is released (that is when the operator is told to
            # squeeze/release them), and a trigger only reveals its true rest value by
            # being moved. See TRIGGER_READY_MAX.
            raw_l2 = js.get_axis(AXIS_L2)
            raw_r2 = js.get_axis(AXIS_R2)
            # An untouched trigger reads EXACTLY 0.0 (SDL has never received an event for
            # it); a real trigger at rest reads -1.0 and a real mid-travel value is
            # essentially never a clean 0.0. So treat "not exactly 0.0" as initialised —
            # the operator no longer has to squeeze both triggers at every process start
            # (which meant every single reset during demo recording).
            if raw_l2 != 0.0:
                l2_ready = True
            if raw_r2 != 0.0:
                r2_ready = True
            if raw_l2 <= TRIGGER_READY_MAX:
                l2_ready = True
            if raw_r2 <= TRIGGER_READY_MAX:
                r2_ready = True
            triggers_ready = l2_ready and r2_ready

            if enabled:
                raw_left_x = js.get_axis(AXIS_LEFT_X)
                raw_left_y = js.get_axis(AXIS_LEFT_Y)
                raw_right_x = js.get_axis(AXIS_RIGHT_X)
                raw_right_y = js.get_axis(AXIS_RIGHT_Y)

                linear_x = -_apply_deadzone(raw_left_y, DEADZONE)
                linear_y = _apply_deadzone(raw_left_x, DEADZONE)
                # Until BOTH triggers have reported their rest value once, an unmoved
                # trigger reads as half-pressed and would inject phantom z. Refuse z.
                if triggers_ready:
                    l2 = _trigger_fraction(raw_l2)
                    r2 = _trigger_fraction(raw_r2)
                    linear_z = r2 - l2
                else:
                    linear_z = 0.0

                angular_y = -_apply_deadzone(raw_right_y, DEADZONE)
                angular_z = -_apply_deadzone(raw_right_x, DEADZONE)
                angular_x = 0.0
                if js.get_numhats() > DPAD_HAT_INDEX:
                    angular_x = float(js.get_hat(DPAD_HAT_INDEX)[0])

                # Match ps4_publisher.py EXACTLY: it flips X/Y/roll/pitch when the operator
                # is NOT in front (i.e. `if not operator_position_front`). Same condition here.
                if not OPERATOR_FRONT:
                    linear_x *= -1
                    linear_y *= -1
                    angular_x *= -1
                    angular_y *= -1

                action = [linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]
            else:
                # Deadman not held: zero motion (=> no intervention).
                action = [0.0] * 6

            # Gripper buttons are level-read (SpacemouseIntervention reads them each step).
            n = js.get_numbuttons()
            left = int(BUTTON_GRIPPER_CLOSE < n and js.get_button(BUTTON_GRIPPER_CLOSE))
            right = int(BUTTON_GRIPPER_OPEN < n and js.get_button(BUTTON_GRIPPER_OPEN))

            # Edge-triggered events: latch on the rising edge so a single press counts once.
            succ = bool(BUTTON_SUCCESS < n0 and js.get_button(BUTTON_SUCCESS))
            abrt = bool(BUTTON_ABORT < n0 and js.get_button(BUTTON_ABORT))
            rgrp = bool(BUTTON_REGRASP < n0 and js.get_button(BUTTON_REGRASP))
            new_success = succ and not prev_success
            new_abort = abrt and not prev_abort
            new_regrasp = rgrp and not prev_regrasp
            prev_success, prev_abort, prev_regrasp = succ, abrt, rgrp

            # Publish a COHERENT snapshot under the lock; OR-in event edges so a press is
            # never lost against a concurrent get_events() consume.
            with self._lock:
                self.latest_data["action"] = action
                self.latest_data["buttons"] = [left, right]
                self.latest_data["triggers_ready"] = bool(triggers_ready)
                self.latest_data["deadman_held"] = bool(enabled)
                if new_success:
                    self.latest_data["success"] = True
                if new_abort:
                    self.latest_data["abort"] = True
                if new_regrasp:
                    self.latest_data["regrasp"] = True
                self.latest_data["heartbeat"] = _now()

            # Rate-limit (no busy-spin).
            dt = _now() - loop_start
            if dt < READ_PERIOD_S:
                time.sleep(READ_PERIOD_S - dt)

    def _is_stale(self) -> bool:
        stale = (_now() - self.latest_data.get("heartbeat", 0.0)) > STALE_TIMEOUT_S
        if (stale or not self.process.is_alive()) and not self._warned_dead:
            self._warned_dead = True
            err = (self.latest_data.get("runtime_error", "")
                   or self.latest_data.get("init_error", "") or "no heartbeat")
            print(
                "\n" + "!" * 78 +
                "\n[PS4Expert] OPERATOR CONTROL LOST — the controller reader is dead/stale."
                f"\n[PS4Expert] cause: {err}"
                "\n[PS4Expert] Teleop interventions and X/Triangle/Circle are NO LONGER read."
                "\n[PS4Expert] If a policy is running it KEEPS MOVING — use the E-STOP, then"
                "\n[PS4Expert] stop the actor and reconnect the controller.\n" + "!" * 78,
                flush=True,
            )
        return stale

    def _acquire(self, timeout: float) -> bool:
        """Timed lock acquire. Returns False instead of blocking forever (a child that died
        holding the Manager lock must not freeze the control loop)."""
        try:
            return bool(self._lock.acquire(timeout=timeout))
        except TypeError:  # some proxies don't accept a timeout kwarg
            try:
                return bool(self._lock.acquire(False))
            except Exception:
                return False
        except Exception:
            return False

    def _release(self) -> None:
        try:
            self._lock.release()
        except Exception:
            pass

    def health(self) -> dict:
        """Reader health for preflight / actor-side monitoring."""
        return {
            "alive": bool(self.process.is_alive()),
            "stale": bool((_now() - self.latest_data.get("heartbeat", 0.0)) > STALE_TIMEOUT_S),
            "init_error": self.latest_data.get("init_error", ""),
            "runtime_error": self.latest_data.get("runtime_error", ""),
            "age_s": _now() - self.latest_data.get("heartbeat", 0.0),
        }

    def get_action(self) -> Tuple[np.ndarray, list]:
        """Return the latest (6-vec action, [left, right]) — SpaceMouseExpert contract.

        FAIL-SAFE: if the reader is stale (crashed / controller disconnected), return
        zeros + no buttons so a stale command can never be replayed indefinitely. A loud
        one-shot warning is printed (see _is_stale) because losing the pad while the policy
        drives means losing intervention control.
        """
        if self._is_stale() or not self.process.is_alive():
            return np.zeros(6), [0, 0]
        if not self._acquire(LOCK_TIMEOUT_S):
            return np.zeros(6), [0, 0]   # fail safe rather than block the control loop
        try:
            action = list(self.latest_data["action"])
            buttons = list(self.latest_data["buttons"])
        finally:
            self._release()
        return np.array(action), buttons

    def deadman_held(self) -> bool:
        """True while R1 is held. Fail-safe False if the reader is stale/dead — used by the
        optional bring-up hold-to-move gate (PS4_DEADMAN_GATES_POLICY)."""
        if self._is_stale() or not self.process.is_alive():
            return False
        return bool(self.latest_data.get("deadman_held", False))

    def triggers_ready(self) -> bool:
        """True once BOTH L2 and R2 have reported their true rest value at least once.

        Until then z motion is suppressed: an analog trigger that has not been moved since
        the device was opened reads 0.0, which with TRIGGER_REST=-1 is indistinguishable
        from half-pressed, so `r2 - l2` injects phantom z (rig-observed as "L2 goes down
        then springs back up"). Callers should prompt the operator to squeeze and release
        BOTH triggers. This is READINESS, deliberately separate from liveness/health.
        """
        if self._is_stale() or not self.process.is_alive():
            return False
        return bool(self.latest_data.get("triggers_ready", False))

    def get_events(self) -> dict:
        """Consume the latched events. Returns {"success", "abort", "regrasp"} (bools).

        Atomic read-then-clear under the lock, so an edge arriving concurrently is not
        erased. Each physical press is reported exactly once. If the reader is stale, no
        events (fail-safe). `read_ok` distinguishes a fresh, locked read from a
        timeout/stale-reader fallback; reset must not treat the latter as an empty latch.
        """
        none = {"success": False, "abort": False, "regrasp": False, "read_ok": False}
        if self._is_stale() or not self.process.is_alive():
            # Fail-safe: report nothing, AND best-effort clear the latches so a press
            # captured before the reader died can't fire later. TIMED acquire — a child that
            # died holding the lock must never hang the control loop.
            if self._acquire(LOCK_TIMEOUT_S):
                try:
                    self.latest_data["success"] = False
                    self.latest_data["abort"] = False
                    self.latest_data["regrasp"] = False
                finally:
                    self._release()
            return none

        if not self._acquire(LOCK_TIMEOUT_S):
            # Could not get the lock in time — report no events rather than blocking.
            return none
        try:
            success = bool(self.latest_data["success"])
            abort = bool(self.latest_data["abort"])
            regrasp = bool(self.latest_data["regrasp"])
            if success:
                self.latest_data["success"] = False
            if abort:
                self.latest_data["abort"] = False
            if regrasp:
                self.latest_data["regrasp"] = False
        finally:
            self._release()
        return {"success": success, "abort": abort, "regrasp": regrasp, "read_ok": True}

    def close(self):
        try:
            self.process.terminate()
            self.process.join(timeout=2.0)
            # ESCALATE if SIGTERM was not enough. The reader spends most of its life in
            # time.sleep(), where a terminate() can be missed — orphaned readers then
            # survive Ctrl+C and keep the process alive. Rig-observed 2026-09-09: three
            # jog_and_capture processes at once, all still publishing competing targets
            # to the controller, which looks exactly like "the controller stopped working".
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=2.0)
        except Exception:
            pass
        try:
            self.manager.shutdown()
        except Exception:
            pass
