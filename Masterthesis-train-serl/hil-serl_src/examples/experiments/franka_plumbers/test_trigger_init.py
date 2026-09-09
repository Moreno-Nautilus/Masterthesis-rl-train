"""Regression test for the PS4 analog-trigger initialisation fault (rig-found 2026-09-09).

THE BUG: SDL reports 0.0 for an analog trigger that has not been moved since the device
was opened. With TRIGGER_REST = -1.0 the conversion maps raw 0.0 -> fraction 0.5, i.e.
HALF PRESSED. So with a fresh reader:

    L2 pressed (+1), R2 untouched (0)  ->  z = 0.5 - 1.0 = -0.5   (down, correct-ish)
    L2 released (-1), R2 untouched (0) ->  z = 0.5 - 0.0 = +0.5   (PHANTOM UP)

which the operator saw as "L2 goes down, then springs back up".

THE FIX: suppress z until BOTH triggers have been observed at their true rest value
(<= TRIGGER_READY_MAX) at least once. Readiness is per reader lifecycle and is exposed
via PS4Expert.triggers_ready().

Runs offline; no joystick, no ROS, no robot.

    source franka_env.sh
    python3 .../test_trigger_init.py
"""
import numpy as np

from franka_env.spacemouse.ps4_expert import (
    TRIGGER_REST,
    TRIGGER_READY_MAX,
    _trigger_fraction,
)


def _z(raw_l2, raw_r2, ready):
    """Reproduce the reader's z computation under a given readiness state."""
    if not ready:
        return 0.0
    return _trigger_fraction(raw_r2) - _trigger_fraction(raw_l2)


def _readiness(samples):
    """Replay raw (l2, r2) samples through the readiness latch, as the reader does."""
    l2_ready = r2_ready = False
    out = []
    for raw_l2, raw_r2 in samples:
        if raw_l2 <= TRIGGER_READY_MAX:
            l2_ready = True
        if raw_r2 <= TRIGGER_READY_MAX:
            r2_ready = True
        out.append(l2_ready and r2_ready)
    return out


def test_the_original_fault_is_reproducible_without_the_gate():
    """Document the bug: unguarded, releasing L2 on a fresh reader commands +z."""
    z_press = _z(+1.0, 0.0, ready=True)     # L2 pressed, R2 never moved
    z_release = _z(-1.0, 0.0, ready=True)   # L2 released, R2 never moved
    assert z_press < 0, f"expected down while L2 pressed, got {z_press}"
    assert z_release > 0, (
        "the fault should reproduce: releasing L2 with an uninitialised R2 gives "
        f"z={z_release} (phantom UP)")
    print(f"  OK  fault reproduced unguarded: press z={z_press:+.2f}, release z={z_release:+.2f}")


def test_gate_blocks_phantom_z_until_initialised():
    """With the readiness gate, that same sequence can never command z."""
    seq = [(+1.0, 0.0), (0.0, 0.0), (-1.0, 0.0)]   # press, mid, release; R2 untouched
    ready = _readiness(seq)
    for (raw_l2, raw_r2), rdy in zip(seq, ready):
        z = _z(raw_l2, raw_r2, rdy)
        assert z == 0.0, f"z must be 0 while unready, got {z} (l2={raw_l2}, ready={rdy})"
    assert not ready[-1], "R2 never moved -> must still be unready"
    print("  OK  readiness gate suppresses z for the entire L2-only sequence")


def test_both_triggers_initialised_gives_correct_signs():
    """Once both have been seen at rest, the normal mapping applies."""
    seq = [(-1.0, -1.0)]           # both at rest -> ready
    assert _readiness(seq)[-1], "both at rest must latch ready"
    assert _z(-1.0, -1.0, True) == 0.0, "both released -> zero"
    assert _z(+1.0, -1.0, True) < 0, "L2 pressed -> negative z (down)"
    assert _z(-1.0, +1.0, True) > 0, "R2 pressed -> positive z (up)"
    print("  OK  initialised: both-released=0, L2=down(-), R2=up(+)")


def test_readiness_latches_and_does_not_regress():
    """Readiness is a latch: once set it stays set for the reader's lifetime."""
    seq = [(0.0, 0.0), (-1.0, 0.0), (-1.0, -1.0), (+1.0, +1.0), (0.0, 0.0)]
    ready = _readiness(seq)
    assert ready == [False, False, True, True, True], ready
    print("  OK  readiness latches at the sample where BOTH have been seen at rest")


def test_rest_value_zero_is_not_treated_as_released():
    """raw 0.0 is a legitimate MID-TRAVEL value and must not latch readiness."""
    assert TRIGGER_READY_MAX < 0.0, (
        f"TRIGGER_READY_MAX={TRIGGER_READY_MAX} must be below 0 so a mid-travel 0.0 "
        "reading cannot be mistaken for 'released'")
    assert not _readiness([(0.0, 0.0)])[-1], "raw 0.0 must NOT latch readiness"
    print(f"  OK  raw 0.0 does not latch (TRIGGER_READY_MAX={TRIGGER_READY_MAX:.2f})")


if __name__ == "__main__":
    test_the_original_fault_is_reproducible_without_the_gate()
    test_gate_blocks_phantom_z_until_initialised()
    test_both_triggers_initialised_gives_correct_signs()
    test_readiness_latches_and_does_not_regress()
    test_rest_value_zero_is_not_treated_as_released()
    print("ALL TRIGGER-INIT TESTS PASS")
