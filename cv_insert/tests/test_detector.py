"""Detector tests on synthetic frames (deterministic; no rosbag needed).

A synthetic 'hole' = a dark filled circle on a light part, with a depth map that has NO valid
depth inside the hole (looks through) but valid depth on the rim -> exercises FULL detection +
annulus depth. pb_top cutout is expected NONE until implemented from the rosbag.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from cv_insert import cad
from cv_insert import detector as D
from cv_insert import geometry as g

INTR = g.CameraIntrinsics(fx=436.31, fy=435.65, cx=418.63, cy=236.51)


def _synth_round_hole(cx=420, cy=240, r=28, size=(480, 848)):
    rgb = np.full((*size, 3), 200, np.uint8)  # light part
    cv2.circle(rgb, (cx, cy), r + 12, (180, 180, 180), -1)  # part body
    cv2.circle(rgb, (cx, cy), r, (25, 25, 25), -1)  # dark hole
    depth = np.full(size, 0.20, np.float32)  # rim/surface at 20 cm
    yy, xx = np.mgrid[0:size[0], 0:size[1]]
    inside = np.hypot(xx - cx, yy - cy) < r - 2
    depth[inside] = 0.0  # hole looks through -> no depth
    return rgb, depth


def _prior_for_synth():
    # feature diameter that yields r~28px at 0.20m for this INTR:
    # r = 0.5*d*f/z -> d = 2*r*z/f
    f = 0.5 * (INTR.fx + INTR.fy)
    d = 2 * 28 * 0.20 / f
    return cad.FeaturePrior(
        name="synth", kind="round_hole", feature_diameter_m=d,
        part_bbox_m=(0.05, 0.05, 0.05), vision_weight=0.85,
    )


def test_detects_round_hole_full():
    rgb, depth = _synth_round_hole(cx=420, cy=240, r=28)
    det = D.FeatureDetector(_prior_for_synth(), INTR, working_depth_m=0.20).detect(rgb, depth)
    assert det.confidence is D.Confidence.FULL
    assert det.is_full_shape_candidate  # shape gate only, NOT motion authorization
    assert not det.authorize_motion(gate=None)  # fail-closed: no gate => no motion
    assert det.center_px[0] == pytest.approx(420, abs=4)
    assert det.center_px[1] == pytest.approx(240, abs=4)
    # depth sampled on the rim, not the empty hole centre
    assert det.depth_m == pytest.approx(0.20, abs=0.01)
    # deprojected point is roughly on the optical axis at 0.20 m
    assert det.point_cam[2] == pytest.approx(0.20, abs=0.01)


def test_center_deprojects_consistently():
    rgb, depth = _synth_round_hole(cx=500, cy=300, r=24)
    det = D.FeatureDetector(_prior_for_synth(), INTR, working_depth_m=0.20).detect(rgb, depth)
    assert det.is_full_shape_candidate
    u, v = g.project(det.point_cam, INTR)
    assert u == pytest.approx(det.center_px[0], abs=1.0)
    assert v == pytest.approx(det.center_px[1], abs=1.0)


def test_blank_frame_is_none():
    rgb = np.full((480, 848, 3), 200, np.uint8)
    depth = np.full((480, 848), 0.20, np.float32)
    det = D.FeatureDetector(_prior_for_synth(), INTR, working_depth_m=0.20).detect(rgb, depth)
    assert det.confidence is D.Confidence.NONE


class _AcceptGate:
    def accepts(self, det):
        return True


class _RejectGate:
    def accepts(self, det):
        return False


def test_partial_never_authorizes_motion():
    d = D.Detection(D.Confidence.PARTIAL, center_px=(10, 10), depth_m=0.2,
                    point_cam=np.array([0.0, 0.0, 0.2]))
    assert d.has_point
    assert not d.is_full_shape_candidate
    assert not d.authorize_motion(gate=None)
    assert not d.authorize_motion(gate=_AcceptGate())  # even an accepting gate can't lift PARTIAL


def test_none_never_authorizes_motion():
    d = D.Detection(D.Confidence.NONE)
    assert not d.has_point
    assert not d.is_full_shape_candidate
    assert not d.authorize_motion(gate=_AcceptGate())


def test_full_requires_gate_to_authorize():
    d = D.Detection(D.Confidence.FULL, center_px=(10, 10), depth_m=0.2,
                    point_cam=np.array([0.0, 0.0, 0.2]))
    assert d.is_full_shape_candidate
    assert not d.authorize_motion(gate=None)      # fail-closed without a gate
    assert not d.authorize_motion(gate=_RejectGate())  # gate can veto
    assert d.authorize_motion(gate=_AcceptGate())  # only a passing gate authorizes


def test_pb_top_cutout_stub_is_none():
    rgb, depth = _synth_round_hole()
    det = D.FeatureDetector(cad.get_prior("pb_top"), INTR, working_depth_m=0.20).detect(rgb, depth)
    assert det.confidence is D.Confidence.NONE
    assert det.method == "cutout-stub"
