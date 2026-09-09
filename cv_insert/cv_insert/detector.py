"""Per-feature base-hole detector with FULL / PARTIAL / NONE confidence tiers.

This is the perception half of the compliant closed-loop servo (PLAN.md §2, §3). It runs every
control cycle on one wrist (or ZED) RGB-D frame and returns the target hole/cutout centre in the
CAMERA frame plus a confidence tier. The servo composes it to the tool frame and decides how much
correction to apply based on the tier and the feature's ``vision_weight``.

Design intent — degraded / absent vision is the NORMAL case, especially for pb_top:
  * FULL     : clean round hole/bore found -> centre + depth trusted.
  * PARTIAL  : only part of the rim / cutout edge seen -> centre recovered by fitting a KNOWN
               shape (circle/ellipse/CAD edge); returned but flagged for down-weighting.
  * NONE     : nothing reliable -> caller must ABORT after a small budget (NOT descend blindly:
               the sub-mm-class clearance makes a blind push on the grasp prior unsafe).

Only numpy + cv2. The round-hole/round-bore paths are implemented; the pb_top cutout path is a
CAD-edge-fit stub with a clear TODO to be filled from the rosbag once we see the real feature.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np

from . import geometry as g
from .cad import FeaturePrior

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


class Confidence(enum.Enum):
    FULL = "FULL"
    PARTIAL = "PARTIAL"
    NONE = "NONE"


@dataclass
class Detection:
    confidence: Confidence
    center_px: tuple[float, float] | None = None  # (u, v)
    radius_px: float | None = None
    depth_m: float | None = None
    point_cam: np.ndarray | None = None  # (3,) centre in camera optical frame
    score: float = 0.0  # 0..1 detector self-confidence within the tier
    method: str = ""
    debug: dict | None = None

    @property
    def has_point(self) -> bool:
        """A 3D point was produced (any tier). NOT a licence to command motion."""
        return self.point_cam is not None

    @property
    def is_full_shape_candidate(self) -> bool:
        """DESCRIPTIVE ONLY: passed the shape+depth gates at FULL tier. This is NOT motion
        authorization — it does not know whether this is the real hole vs clutter. On the real
        fixture-scan frames, 4/14 screw frames pass this and they are all clutter (one at the
        image corner). Motion authorization requires ``authorize_motion(...)`` with an external
        gate. Named deliberately so ``if det.is_full_shape_candidate: move()`` reads as wrong.
        """
        return self.confidence is Confidence.FULL and self.point_cam is not None

    def authorize_motion(self, gate) -> bool:
        """The ONLY path that may license a correction. FAIL-CLOSED: returns False unless an
        external ``gate`` is supplied AND it accepts this detection.

        ``gate`` must implement ``accepts(detection) -> bool`` and is expected to enforce: a
        tight expected-ROI around the prior hole position, projected-center consistency, temporal
        confirmation across N frames, and a cumulative-correction cap. No gate => no motion. This
        gate does not exist yet (servo layer, pending rosbag) — so today nothing is authorized,
        by design.
        """
        if gate is None:
            return False
        if not self.is_full_shape_candidate:
            return False
        return bool(gate.accepts(self))


def opencv_available() -> bool:
    return cv2 is not None


@dataclass
class DetectorConfig:
    # search-radius tolerance around the predicted apparent radius. WIDE by default because the
    # feature diameter prior is UNVERIFIED (see cad.py) — a too-tight gate can reject the true
    # hole while accepting a larger clutter circle. Tighten ONLY once the real hole size is
    # measured on the rosbag; until then rely on the expected-ROI + confidence gate, not radius.
    radius_tol_frac: float = 0.75
    # circularity gate for FULL (4*pi*A/P^2); a clean circle ~1.0
    min_circularity_full: float = 0.75
    # a looser gate that still counts as PARTIAL (arc / occluded rim)
    min_circularity_partial: float = 0.45
    # HoughCircles fallback params
    hough_dp: float = 1.2
    hough_param1: float = 120.0
    hough_param2: float = 30.0
    # annulus (as fractions of predicted radius) for robust depth sampling
    depth_annulus_inner: float = 1.1
    depth_annulus_outer: float = 1.8


class FeatureDetector:
    """Detect one feature's opening in a single RGB-D frame.

    Parameters
    ----------
    prior : FeaturePrior         the CAD-derived expectations for this insert
    intr  : CameraIntrinsics     the frame's intrinsics
    working_depth_m : float      approx camera->feature distance at preinsert (~0.02-0.4 m);
                                  used to predict apparent radius. Refined online from depth.
    """

    def __init__(self, prior: FeaturePrior, intr: g.CameraIntrinsics, working_depth_m: float,
                 cfg: DetectorConfig | None = None):
        self.prior = prior
        self.intr = intr
        self.working_depth_m = float(working_depth_m)
        self.cfg = cfg or DetectorConfig()

    # -- public API -------------------------------------------------------------------------
    def detect(self, rgb: np.ndarray, depth_m: np.ndarray) -> Detection:
        if cv2 is None:
            return Detection(Confidence.NONE, method="no-cv2")
        if self.prior.kind in ("round_hole", "round_bore"):
            det = self._detect_round(rgb, depth_m)
        elif self.prior.kind == "cutout":
            det = self._detect_cutout(rgb, depth_m)
        else:  # pragma: no cover
            det = Detection(Confidence.NONE, method="unknown-kind")
        if det.center_px is not None and det.depth_m is not None:
            det.point_cam = g.deproject(det.center_px[0], det.center_px[1], det.depth_m, self.intr)
        return det

    # -- round hole / bore ------------------------------------------------------------------
    def _predicted_radius(self, depth_m: float) -> float:
        return g.predicted_radius_px(self.prior.feature_diameter_m, depth_m, self.intr)

    def _detect_round(self, rgb: np.ndarray, depth_m: np.ndarray) -> Detection:
        # LIMITATION (TODO rosbag): this returns the single best-scoring circle, chosen BEFORE any
        # expected-ROI is consulted. An external gate can then only VETO that pick — it cannot
        # promote a lower-ranked real in-ROI hole if a clutter circle outscored it. The proper fix
        # is to return the top-K candidates and let the gate select within the ROI. Until then,
        # the gate must run a tight ROI FIRST (crop) so the detector only ever sees the hole region.
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) if rgb.ndim == 3 else rgb
        gray = cv2.medianBlur(gray, 5)
        r_pred = self._predicted_radius(self.working_depth_m)
        r_lo = max(3.0, r_pred * (1 - self.cfg.radius_tol_frac))
        r_hi = r_pred * (1 + self.cfg.radius_tol_frac)

        # Primary: contour-based dark-blob + circularity (robust, gives PARTIAL too).
        det = self._contour_circle(gray, depth_m, r_lo, r_hi, r_pred)
        if det.confidence is Confidence.FULL:
            return det
        # Fallback: Hough, sized by the predicted radius.
        hough = self._hough_circle(gray, depth_m, r_lo, r_hi)
        # keep the better of the two by score
        return hough if hough.score > det.score else det

    def _contour_circle(self, gray, depth_m, r_lo, r_hi, r_pred) -> Detection:
        # dark interior of a hole -> invert + Otsu
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = Detection(Confidence.NONE, method="contour")
        for c in cnts:
            area = cv2.contourArea(c)
            if area < np.pi * (r_lo * 0.6) ** 2:
                continue
            (u, v), rad = cv2.minEnclosingCircle(c)
            if not (r_lo <= rad <= r_hi):
                continue
            peri = cv2.arcLength(c, True)
            if peri <= 0:
                continue
            circ = 4 * np.pi * area / (peri * peri)
            depth = self._sample_depth(depth_m, u, v, rad)
            if depth is None:
                continue
            if circ >= self.cfg.min_circularity_full:
                conf, score = Confidence.FULL, min(1.0, circ)
            elif circ >= self.cfg.min_circularity_partial:
                conf, score = Confidence.PARTIAL, 0.5 * circ
            else:
                continue
            if score > best.score:
                best = Detection(conf, (u, v), rad, depth, score=score, method="contour",
                                 debug={"circularity": float(circ)})
        return best

    def _hough_circle(self, gray, depth_m, r_lo, r_hi) -> Detection:
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=self.cfg.hough_dp, minDist=gray.shape[0] / 4,
            param1=self.cfg.hough_param1, param2=self.cfg.hough_param2,
            minRadius=int(r_lo), maxRadius=int(r_hi),
        )
        if circles is None:
            return Detection(Confidence.NONE, method="hough")
        u, v, rad = circles[0][0]
        depth = self._sample_depth(depth_m, u, v, rad)
        if depth is None:
            return Detection(Confidence.NONE, method="hough")
        # Hough alone can't prove it's the real hole -> at best PARTIAL confidence.
        return Detection(Confidence.PARTIAL, (float(u), float(v)), float(rad), depth,
                         score=0.4, method="hough")

    # -- cutout (pb_top) --------------------------------------------------------------------
    def _detect_cutout(self, rgb: np.ndarray, depth_m: np.ndarray) -> Detection:
        """CAD-edge-fit of the base top cutout (pb_top).

        TODO(rosbag): implement the real cutout fit once we can see the feature:
          1. segment the coloured base region (saturation mask) to isolate the part;
          2. extract the cutout boundary (edge/contour of the pocket);
          3. fit the KNOWN cutout polygon from pb_base.obj to the visible (possibly partial)
             edge -> recover centre even when occluded (PARTIAL tier);
          4. sample depth on the surrounding rim (annulus_median_depth).
        Until then this returns NONE. Under the fail-closed policy NONE = ABORT (not a blind
        compliance push), so pb_top cannot run until this detector or an independent certified
        prior (planar registration before occlusion, or a datum) exists. (PLAN.md §2/§4.)
        """
        return Detection(Confidence.NONE, method="cutout-stub")

    # -- shared -----------------------------------------------------------------------------
    def _sample_depth(self, depth_m, u, v, rad) -> float | None:
        return g.annulus_median_depth(
            depth_m, u, v,
            r_inner_px=rad * self.cfg.depth_annulus_inner,
            r_outer_px=rad * self.cfg.depth_annulus_outer,
        )
