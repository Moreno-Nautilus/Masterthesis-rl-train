"""Off-robot PS4 (DualShock 4) test — verify controller reading + mapping without the robot.

Two modes:
  --raw     : live-print every axis/button/hat as you move them. Use this to CONFIRM or FIX the
              DS4 axis/button indices in iiwa_serl/teleop/ps4_teleop_provider.py (_DS4_DEFAULTS)
              against your actual controller/driver.
  (default) : run the real PS4TeleopProvider and print the decoded 6-DoF action + success/abort/
              gripper events, exactly as record_demo.py sees them. Verifies the FULL decode path.

SDL gotcha (Linux): SDL2 reads /dev/input/event* which is often root:input rw---- (no user read),
so pygame sees 0 joysticks even though /dev/input/js0 exists. Fix EITHER:
  - export SDL_JOYSTICK_DEVICE=/dev/input/js0     (per-shell, works immediately)
  - sudo usermod -aG input $USER && re-login       (permanent, cleaner)
This script sets SDL_JOYSTICK_DEVICE=/dev/input/js0 automatically if not already set.

Usage:
  python ps4_test.py --raw          # map every control
  python ps4_test.py                # decoded 6-DoF action + events
"""
from __future__ import annotations

import argparse
import os
import time

# Apply the SDL js0 workaround before importing pygame, unless the user overrode it.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_JOYSTICK_DEVICE", "/dev/input/js0")

import numpy as np
import pygame


def raw_mode(index: int):
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() <= index:
        raise SystemExit(
            f"No joystick at index {index}. Detected {pygame.joystick.get_count()}. "
            "Is the DS4 plugged in? Try: export SDL_JOYSTICK_DEVICE=/dev/input/js0"
        )
    js = pygame.joystick.Joystick(index)
    js.init()
    na, nb, nh = js.get_numaxes(), js.get_numbuttons(), js.get_numhats()
    print(f"'{js.get_name()}' — {na} axes, {nb} buttons, {nh} hats")
    print("Move sticks / press buttons. Only CHANGES print. Ctrl-C to stop.\n")
    print("Config expects: L-stick x=0 y=1 | L2=2 | R-stick x=3 y=4 | R2=5 | "
          "✕=btn0 ○=btn1 □=btn2 △=btn3 R1=btn5 | d-pad=hat0\n")

    prev_ax = [0.0] * na
    prev_bt = [0] * nb
    prev_ht = [(0, 0)] * nh
    try:
        while True:
            pygame.event.pump()
            for i in range(na):
                v = round(js.get_axis(i), 2)
                if abs(v - prev_ax[i]) > 0.15:
                    print(f"  axis[{i}] = {v:+.2f}")
                    prev_ax[i] = v
            for i in range(nb):
                v = js.get_button(i)
                if v != prev_bt[i]:
                    print(f"  button[{i}] = {v}   {'(pressed)' if v else '(released)'}")
                    prev_bt[i] = v
            for i in range(nh):
                v = js.get_hat(i)
                if v != prev_ht[i]:
                    print(f"  hat[{i}] = {v}")
                    prev_ht[i] = v
            time.sleep(0.03)
    except KeyboardInterrupt:
        print("\ndone.")


def decoded_mode(index: int):
    # Import here so --raw works even if the package import has an issue.
    from iiwa_serl.teleop import PS4TeleopProvider

    teleop = PS4TeleopProvider(joystick_index=index, server_url=None)
    print("Decoded 6-DoF action [dx, dy, dz, drx, dry, drz] — HOLD R1 (deadman) for nonzero motion.")
    print("✕=success  △=abort  ○=open  □=close.  Ctrl-C to stop.\n")
    try:
        while True:
            a = teleop.get_action()
            s = teleop.is_success()
            f = teleop.is_failure()
            tag = ""
            if s:
                tag += "  <SUCCESS ✕>"
            if f:
                tag += "  <ABORT △>"
            if np.any(np.abs(a) > 1e-3) or tag:
                print(f"  action = [{', '.join(f'{x:+.2f}' for x in a)}]{tag}")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\ndone.")
    finally:
        teleop.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="store_true", help="live-map every axis/button/hat")
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()
    if args.raw:
        raw_mode(args.index)
    else:
        decoded_mode(args.index)


if __name__ == "__main__":
    main()
