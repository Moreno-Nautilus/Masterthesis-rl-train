"""Unit tests for the ROS-free geometry core."""

import numpy as np
import pytest

from cv_insert import geometry as g

INTR = g.CameraIntrinsics(fx=436.31, fy=435.65, cx=418.63, cy=236.51)  # real D405 K_depth


def test_deproject_project_roundtrip():
    p = g.deproject(500.0, 300.0, 0.22, INTR)
    u, v = g.project(p, INTR)
    assert u == pytest.approx(500.0)
    assert v == pytest.approx(300.0)


def test_centre_pixel_is_pure_z_ray():
    c = g.deproject(INTR.cx, INTR.cy, 0.5, INTR)
    assert c[0] == pytest.approx(0.0)
    assert c[1] == pytest.approx(0.0)
    assert c[2] == pytest.approx(0.5)


def test_project_rejects_behind_camera():
    with pytest.raises(ValueError):
        g.project([0.0, 0.0, -0.1], INTR)


def test_invert_T_roundtrip():
    rng = np.random.default_rng(0)
    # random rotation via QR, random translation
    A = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    T = np.eye(4)
    T[:3, :3] = Q
    T[:3, 3] = rng.standard_normal(3)
    Ti = g.invert_T(T)
    assert np.allclose(T @ Ti, np.eye(4), atol=1e-9)


def test_cam_to_base_composition():
    # A point in cam -> base must match applying T_base_cam directly.
    T_base_flange = np.eye(4)
    T_base_flange[:3, 3] = [0.4, 0.1, 0.5]
    T_flange_cam = np.eye(4)
    T_flange_cam[:3, 3] = [-0.05, 0.0, 0.06]
    T_base_cam = T_base_flange @ T_flange_cam
    p_cam = np.array([0.02, -0.01, 0.2])
    p_base = g.cam_point_to_base(p_cam, T_base_cam)
    assert np.allclose(p_base, p_cam + [-0.05, 0.0, 0.06] + [0.4, 0.1, 0.5])


def test_cam_point_to_tool_roundtrip():
    # If tool == cam pose, a cam point maps to itself in tool frame.
    T_base_cam = np.eye(4)
    T_base_cam[:3, 3] = [0.3, 0.0, 0.4]
    p_cam = np.array([0.01, 0.02, 0.15])
    p_tool = g.cam_point_to_tool(p_cam, T_base_cam, T_base_tool=T_base_cam)
    assert np.allclose(p_tool, p_cam, atol=1e-9)


def test_annulus_median_depth_picks_rim_not_hole():
    depth = np.full((100, 100), 0.20, np.float32)
    depth[45:55, 45:55] = 0.0  # a "hole" with no depth in the centre
    d = g.annulus_median_depth(depth, u=50, v=50, r_inner_px=8, r_outer_px=14)
    assert d == pytest.approx(0.20)


def test_annulus_median_depth_none_when_sparse():
    depth = np.zeros((100, 100), np.float32)  # all invalid
    assert g.annulus_median_depth(depth, 50, 50, 8, 14) is None


def test_predicted_radius_scales():
    r_near = g.predicted_radius_px(0.014, 0.15, INTR)
    r_far = g.predicted_radius_px(0.014, 0.30, INTR)
    assert r_near == pytest.approx(2 * r_far, rel=1e-6)
