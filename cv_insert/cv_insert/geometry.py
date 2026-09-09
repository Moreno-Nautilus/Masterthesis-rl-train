"""Pure-numpy camera + frame geometry for the CV insertion patch.

No ROS, no open3d — only numpy (+ cv2 optionally, not required here). Everything is
unit-testable offline against the real `.npz` captures in ``data/`` and, later, the rosbag.

Conventions
-----------
* Camera OPTICAL frame: x right, y down, z forward (OpenCV / RealSense standard).
* Intrinsics: pinhole ``K = [[fx,0,cx],[0,fy,cy],[0,0,1]]``.
* Rigid transforms are 4x4 homogeneous ``T`` with ``p_a = T_a_b @ p_b`` (point in frame b
  expressed in frame a). Matches the capture ``.npz`` keys (``T_base_flange``,
  ``T_flange_cam``, ``T_base_cam`` all compose as ``T_base_cam = T_base_flange @ T_flange_cam``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_K(cls, K) -> "CameraIntrinsics":
        K = np.asarray(K, float).reshape(3, 3)
        return cls(float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], float
        )


def deproject(u: float, v: float, z: float, intr: CameraIntrinsics) -> np.ndarray:
    """Pixel (u,v) + depth z (m) -> 3D point in the camera optical frame (m)."""
    x = (u - intr.cx) * z / intr.fx
    y = (v - intr.cy) * z / intr.fy
    return np.array([x, y, z], float)


def project(p_cam, intr: CameraIntrinsics) -> tuple[float, float]:
    """3D point in the camera optical frame -> pixel (u,v). z>0 required."""
    p = np.asarray(p_cam, float)
    z = p[2]
    if z <= 0:
        raise ValueError("project: point behind camera (z<=0)")
    u = intr.fx * p[0] / z + intr.cx
    v = intr.fy * p[1] / z + intr.cy
    return float(u), float(v)


def transform_point(T, p) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to a 3-vector."""
    T = np.asarray(T, float).reshape(4, 4)
    p = np.asarray(p, float).reshape(3)
    return (T[:3, :3] @ p) + T[:3, 3]


def invert_T(T) -> np.ndarray:
    """Inverse of a rigid 4x4 transform (transpose rotation, back out translation)."""
    T = np.asarray(T, float).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def cam_point_to_base(p_cam, T_base_cam) -> np.ndarray:
    """Camera-frame point -> base frame, given ``T_base_cam`` (= T_base_flange @ T_flange_cam)."""
    return transform_point(T_base_cam, p_cam)


def cam_point_to_tool(p_cam, T_base_cam, T_base_tool) -> np.ndarray:
    """Camera-frame point -> move-arm TOOL frame.

    ``T_base_tool`` is the tool (TCP) pose in base from move-arm FK. Used to express the
    detected hole relative to the tool so the servo can null (hole - known_pin_tip).
    """
    p_base = cam_point_to_base(p_cam, T_base_cam)
    return transform_point(invert_T(T_base_tool), p_base)


def annulus_median_depth(
    depth_m: np.ndarray,
    u: float,
    v: float,
    r_inner_px: float,
    r_outer_px: float,
    *,
    min_valid: int = 8,
) -> float | None:
    """Robust depth at a feature centre: median of valid depths in an annulus around (u,v).

    Depth is holey on rims / dark parts, and the exact centre pixel of a HOLE has no surface
    (it looks through), so we sample a ring on the surrounding rim, not the centre. Returns
    ``None`` if too few valid samples.
    """
    h, w = depth_m.shape[:2]
    y0 = max(0, int(np.floor(v - r_outer_px)))
    y1 = min(h, int(np.ceil(v + r_outer_px)) + 1)
    x0 = max(0, int(np.floor(u - r_outer_px)))
    x1 = min(w, int(np.ceil(u + r_outer_px)) + 1)
    if y1 <= y0 or x1 <= x0:
        return None
    ys, xs = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(xs - u, ys - v)
    ring = (rr >= r_inner_px) & (rr <= r_outer_px)
    d = depth_m[y0:y1, x0:x1][ring]
    d = d[(np.isfinite(d)) & (d > 0)]
    if d.size < min_valid:
        return None
    return float(np.median(d))


def predicted_radius_px(diameter_m: float, depth_m: float, intr: CameraIntrinsics) -> float:
    """Apparent radius (px) of a circular feature of given metric diameter at given depth."""
    f = 0.5 * (intr.fx + intr.fy)
    return 0.5 * diameter_m * f / depth_m
