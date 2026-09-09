"""Tests for the ICP core + pin-pose observation.

Uses INDEPENDENTLY sampled clouds (different seed/count than the CAD), partial visibility, noise,
and spin — so the tests exercise the failure modes, not a trivial same-points round-trip. The
point is NOT to prove pin ICP is accurate (it is a weak lateral estimate); it is to prove the
code REPORTS its weakness honestly (coverage flag, spin != tilt, sanity gate not over-trusting).
"""

import numpy as np
import pytest

from cv_insert import icp as I


def _cylinder(n, seed, radius=0.0045, length=0.045):
    """A screw-like shaft along +Z centred at origin, in the part frame (m)."""
    g = np.random.default_rng(seed)
    z = g.uniform(-length / 2, length / 2, n)
    th = g.uniform(0, 2 * np.pi, n)
    return np.c_[radius * np.cos(th), radius * np.sin(th), z]


def test_icp_recovers_known_translation():
    src = _cylinder(1500, 0)
    shift = np.array([0.003, -0.002, 0.0])
    T, rmse, n = I.icp(src, src + shift)
    assert rmse < 1e-3
    assert np.allclose(T[:3, 3], shift, atol=5e-4)


def test_pin_icp_full_coverage_recovers_lateral():
    cad = _cylinder(1500, 0)
    obs = _cylinder(900, 42) + np.array([0.003, 0.0, 0.0])  # independent sample, full ring
    res = I.pin_pose_icp(obs, cad)
    assert res.lateral_shift_mm == pytest.approx(3.0, abs=1.2)
    assert res.lateral_sanity_ok
    assert res.observability["angular_coverage_frac"] > 0.9


def test_pin_icp_partial_view_is_flagged_not_trusted():
    cad = _cylinder(1500, 0)
    obs = _cylinder(900, 7)
    obs = obs[obs[:, 0] > -0.001]              # occlude one side (gripper)
    obs = obs + np.array([0.00361, 0.0, 0.0])  # true lateral 3.61mm
    res = I.pin_pose_icp(obs, cad)
    # coverage must be flagged low and the weak gate must NOT pass on a biased partial view
    assert res.observability["angular_coverage_frac"] < 0.75
    assert not res.lateral_sanity_ok
    assert any("coverage" in w for w in res.warnings)


def test_pin_icp_spin_is_not_reported_as_tilt():
    cad = _cylinder(1500, 0)
    th = np.radians(30)
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    obs = cad @ Rz.T  # pure spin about the shaft axis, zero real tilt
    res = I.pin_pose_icp(obs, cad)
    assert res.tilt_deg < 2.0  # shaft-axis tilt ~0 despite a 30deg spin


def test_pin_icp_axial_always_flagged_unreliable():
    cad = _cylinder(1500, 0)
    res = I.pin_pose_icp(_cylinder(900, 3) + np.array([0.001, 0.0, 0.0]), cad)
    assert res.observability["axial_translation"].startswith("UNRELIABLE")
    assert res.observability["spin_about_axis"].startswith("UNOBSERVABLE")
    assert any("certification" in w.lower() for w in res.warnings)


def test_pin_icp_rejects_sparse_and_implausible():
    cad = _cylinder(1500, 0)
    sparse = _cylinder(20, 1) + np.array([0.002, 0.0, 0.0])
    assert not I.pin_pose_icp(sparse, cad, min_points=150).lateral_sanity_ok
    big = _cylinder(900, 2) + np.array([0.05, 0.0, 0.0])
    r = I.pin_pose_icp(big, cad, max_lateral_mm=15.0)
    assert not r.lateral_sanity_ok
    assert any("implausible" in w or "reject" in w for w in r.warnings)


def test_pin_icp_summary_is_honest_string():
    cad = _cylinder(1500, 0)
    s = I.pin_pose_icp(_cylinder(900, 5) + np.array([0.001, 0.001, 0.0]), cad).summary()
    assert "UNRELIABLE" in s and "not accuracy" in s


def test_part_pose_icp_has_gate():
    cad = _cylinder(2000, 0, radius=0.02, length=0.06)
    T, rmse, n, accepted, warnings = I.part_pose_icp(_cylinder(2000, 9, radius=0.02, length=0.06),
                                                     cad, min_points=100)
    assert isinstance(accepted, bool)
    assert any("certification" in w for w in warnings)


def test_icp_raises_on_tiny_input():
    with pytest.raises(ValueError):
        I.icp(np.zeros((2, 3)), np.zeros((2, 3)))
