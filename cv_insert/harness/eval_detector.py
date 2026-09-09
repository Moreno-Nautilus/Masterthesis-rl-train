#!/usr/bin/env python3
"""Offline detector evaluation harness — Block A of PLAN.md.

Runs the per-feature detector over a set of RGB-D frames and prints a per-frame table:
confidence tier, centre pixel, sampled depth, deprojected point, and (if a ground-truth hole
point is available) the metric error. This is the go/no-go / vision-reliance measurement that
decides how much each feature leans on vision vs compliance.

Two input sources, same code path:
  * ``--npz-dir DIR`` : the real ``frame_*.npz`` + ``color_*.png`` captures already in the repo
                        (each npz carries depth_m, K_depth, and the T_* transforms).
  * ``--rosbag BAG``  : TODO — fill in once the rosbag arrives (topics unknown until then).
                        The frame->detector call is identical; only the reader differs.

Usage:
    PYTHONPATH=.. python3 eval_detector.py --feature pb_screw \
        --npz-dir /tmp/fx/fixture/move_depth_capture
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

# allow running from the harness/ dir without install
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from cv_insert import cad  # noqa: E402
from cv_insert import detector as D  # noqa: E402
from cv_insert import geometry as g  # noqa: E402

try:
    import cv2
except Exception:
    cv2 = None


def _load_npz_frame(npz_path: str):
    """Return (rgb, depth_m, intr, transforms) from a capture .npz (+ sibling color png)."""
    d = np.load(npz_path)  # numeric arrays only; allow_pickle not needed
    depth_m = np.asarray(d["depth_m"], np.float32)
    intr = g.CameraIntrinsics.from_K(d["K_depth"])
    # sibling color image: frame_000.npz -> color_000.png
    base = os.path.basename(npz_path).replace("frame_", "color_").replace(".npz", ".png")
    color_path = os.path.join(os.path.dirname(npz_path), base)
    rgb = cv2.imread(color_path) if (cv2 and os.path.exists(color_path)) else None
    tf = {k: np.asarray(d[k]) for k in d.files if k.startswith("T_")}
    return rgb, depth_m, intr, tf


def _working_depth(depth_m: np.ndarray) -> float:
    """Rough camera->scene distance = median of valid depths (for radius prediction)."""
    v = depth_m[(np.isfinite(depth_m)) & (depth_m > 0)]
    return float(np.median(v)) if v.size else 0.25


def run(feature: str, npz_dir: str, gt_point_base=None):
    prior = cad.get_prior(feature)
    frames = sorted(glob.glob(os.path.join(npz_dir, "frame_*.npz")))
    if not frames:
        print(f"no frame_*.npz in {npz_dir}")
        return
    print(f"# feature={feature} kind={prior.kind} vision_weight={prior.vision_weight} "
          f"feat_dia={prior.feature_diameter_m*1000:.1f}mm  frames={len(frames)}")
    print(f"# {'frame':22s} {'tier':8s} {'center_px':>16s} {'depth_m':>8s} "
          f"{'point_cam(m)':>26s} {'err_mm':>8s}")
    tiers = {"FULL": 0, "PARTIAL": 0, "NONE": 0}
    errs = []
    for f in frames:
        rgb, depth_m, intr, tf = _load_npz_frame(f)
        if rgb is None:
            print(f"  {os.path.basename(f):22s} (no color image)")
            continue
        wd = _working_depth(depth_m)
        det = D.FeatureDetector(prior, intr, working_depth_m=wd).detect(rgb, depth_m)
        tiers[det.confidence.value] += 1
        cpx = f"({det.center_px[0]:.0f},{det.center_px[1]:.0f})" if det.center_px else "-"
        dm = f"{det.depth_m:.3f}" if det.depth_m else "-"
        pc = np.round(det.point_cam, 3).tolist() if det.point_cam is not None else "-"
        err = "-"
        if det.has_point and gt_point_base is not None and "T_base_cam" in tf:
            p_base = g.cam_point_to_base(det.point_cam, tf["T_base_cam"])
            e = float(np.linalg.norm(p_base - np.asarray(gt_point_base))) * 1000
            errs.append(e)
            err = f"{e:.1f}"
        print(f"  {os.path.basename(f):22s} {det.confidence.value:8s} {cpx:>16s} {dm:>8s} "
              f"{str(pc):>26s} {err:>8s}")
    n = sum(tiers.values())
    print(f"\n# tier rate: FULL={tiers['FULL']}/{n} PARTIAL={tiers['PARTIAL']}/{n} "
          f"NONE={tiers['NONE']}/{n}")
    if errs:
        print(f"# error mm: mean={np.mean(errs):.1f} median={np.median(errs):.1f} "
              f"max={np.max(errs):.1f}  (n={len(errs)})")
    else:
        print("# no ground-truth point supplied -> detection-only report "
              "(pass --gt-base X Y Z once the true hole is known)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feature", required=True, choices=sorted(cad.FEATURE_PRIORS))
    ap.add_argument("--npz-dir", help="dir of frame_*.npz + color_*.png captures")
    ap.add_argument("--rosbag", help="TODO: rosbag path (reader not yet implemented)")
    ap.add_argument("--gt-base", nargs=3, type=float, metavar=("X", "Y", "Z"),
                    help="ground-truth hole point in base frame (m) for error scoring")
    args = ap.parse_args()
    if cv2 is None:
        print("cv2 not available", file=sys.stderr)
        sys.exit(2)
    if args.rosbag:
        print("rosbag reader not implemented yet — waiting on the bag (topic names). "
              "Use --npz-dir for now; the detector call is identical.", file=sys.stderr)
        sys.exit(3)
    if not args.npz_dir:
        ap.error("need --npz-dir (or --rosbag once implemented)")
    run(args.feature, args.npz_dir, gt_point_base=args.gt_base)


if __name__ == "__main__":
    main()
