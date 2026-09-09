"""Minimal OBJ loading + per-feature CAD priors (numpy only, no trimesh/open3d).

The detector uses these priors to (a) size its search (expected feature diameter at the known
working depth), and (b) fit the feature centre from a PARTIAL view by matching a known shape.
We are fitting KNOWN geometry, not doing blind blob detection — that is what makes degraded /
occluded views recoverable (see PLAN.md §2).

CAD source: ``Masterthesis-vision/Data/CAD_Models/{pb_top,pb_base,pb_screw,pb_pipe}.obj``
(units: centimetres). Feature numbers below are measured from those meshes and MUST be
refined against the rosbag once we can see the true mating features (see PLAN.md §6).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# CAD dir — resolve from env, else a repo-relative guess. NOT a hard-coded absolute path.
_THIS = os.path.dirname(os.path.abspath(__file__))
_GUESS = os.path.normpath(os.path.join(_THIS, "..", "..", "..", "Masterthesis-vision",
                                       "Data", "CAD_Models"))
CAD_DIR = os.environ.get("CV_INSERT_CAD_DIR", _GUESS)


def load_obj_vertices(path: str) -> np.ndarray:
    """Load only the ``v`` vertices of an OBJ as an (N,3) float array (cm, as authored)."""
    verts = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
    if not verts:
        raise ValueError(f"no vertices in {path}")
    return np.asarray(verts, float)


@dataclass(frozen=True)
class FeaturePrior:
    """What the detector needs to know about one insert's mating feature.

    All lengths in METRES (converted from the CAD's cm on construction via ``from_cm``).
    """

    name: str
    kind: str  # "round_hole" | "round_bore" | "cutout"
    feature_diameter_m: float  # RECEIVING hole/bore apparent diameter (camera-visible), NOT shaft/OD
    part_bbox_m: tuple[float, float, float]
    vision_weight: float  # PLACEHOLDER: no evidential basis, no consumer yet (see notes)
    verified: bool = False  # True only after measured on rosbag insertion frames
    notes: str = ""

    @classmethod
    def from_cm(cls, name, kind, feature_diameter_cm, part_bbox_cm, vision_weight,
                verified=False, notes=""):
        return cls(
            name=name,
            kind=kind,
            feature_diameter_m=feature_diameter_cm / 100.0,
            part_bbox_m=tuple(b / 100.0 for b in part_bbox_cm),
            vision_weight=vision_weight,
            verified=verified,
            notes=notes,
        )


# Priorities per PLAN.md: pb_top (P1) -> pb_screw (P2) -> pb_pipe (P3). NOTE: the handoff has
# FOUR inserts (parts 3,1,0,4); part 4 is a second screw. Every feature_diameter below is an
# UNVERIFIED placeholder (verified=False) — the receiving-hole size disagrees between CAD bbox
# and independent measurement, so NOTHING here may size a real detection until measured on the
# rosbag. See ADJUDICATION.md (Codex claims #3/#4). vision_weight has no consumer yet.
FEATURE_PRIORS: dict[str, FeaturePrior] = {
    "pb_top": FeaturePrior.from_cm(
        name="pb_top",
        kind="cutout",
        feature_diameter_cm=2.6,  # PLACEHOLDER — a cutout has no single diameter; unused until a fit exists
        part_bbox_cm=(10.0, 4.0, 2.6),
        vision_weight=0.35,
        verified=False,
        notes="Wide cap into base TOP feature. CAD shows the cap starting at the same Z the base "
        "ends and only ~5mm endpoint travel -> may be hover-to-contact PLACEMENT, NOT a deep "
        "pocket with lateral capture. Likely needs planar pose registration BEFORE the cap "
        "occludes it (or a mechanical datum) rather than a circle servo. Confirm on rosbag.",
    ),
    "pb_screw": FeaturePrior.from_cm(
        name="pb_screw",
        kind="round_hole",
        feature_diameter_cm=1.0,  # receiving hole = 10.0mm (CAD ring r=0.500u); shaft 9.5mm CAD /
        part_bbox_cm=(2.88, 2.5, 7.0),  # ~9mm real print -> ~0.25mm RADIAL clearance. verified on hardware.
        vision_weight=0.85,
        verified=False,  # hardware may add a chamfer the CAD lacks -> still confirm on the part
        notes="Slender pin into a round hole; 50mm insert along base -Z (2 screws: parts 1 & 4). "
        "CAD NOMINAL: hole 10.0mm, shaft 9.5mm => 0.25mm RADIAL clearance, no visible CAD chamfer. "
        "PHYSICAL: only the shaft is measured (~9mm x 45mm); the printed HOLE dia + tolerance + "
        "chamfer are UNKNOWN. (If hole stays 10mm and shaft is 9mm that is 0.5mm radial, but do "
        "not assume the hole.) Sub-mm-class clearance either way => 'close enough + compliance' "
        "is NOT assured. (My round-1 ~27mm was the HEAD, not the shaft.)",
    ),
    "pb_pipe": FeaturePrior.from_cm(
        name="pb_pipe",
        kind="round_bore",
        feature_diameter_cm=5.0,  # PLACEHOLDER bore guess; UNVERIFIED
        part_bbox_cm=(4.27, 8.0, 4.95),
        vision_weight=0.7,
        verified=False,
        notes="Cylinder into a round bore, HORIZONTAL insert (~64mm along base +Y, ~68deg off "
        "tool-Z). Oblique view => ELLIPSE fit, not circle. Measure bore dia + obliquity on rosbag.",
    ),
}


def get_prior(name: str) -> FeaturePrior:
    if name not in FEATURE_PRIORS:
        raise KeyError(f"unknown feature '{name}'; known: {sorted(FEATURE_PRIORS)}")
    return FEATURE_PRIORS[name]


def part_cad_path(part_name: str) -> str:
    return os.path.join(CAD_DIR, f"{part_name}.obj")
