#!/usr/bin/env python3
"""ICP fit: real fixture-scan pointcloud -> fixture CAD, report a 6-DoF residual.

SCOPE / CAVEATS (read before trusting any number):
  * This validates FIXTURE / PICKUP geometry ONLY. It says NOTHING about the held-part poses at
    the assembly location, T_tool_part, grasp slip, or the insertion reset/goal relationship.
  * ``--seed-npz`` reads the capture's ``T_base_board``, which is the MEASURED per-frame board
    pose, NOT the nominal pose from ``nominal_world_pose.json``. So a small residual mostly shows
    self-consistency of the measured board pose + depth scale, not agreement with the plan nominal.
    To test the true nominal, seed from ``nominal_world_pose.json`` instead (not wired here).
  * Method is crude: nonuniform OBJ vertices (not surface-sampled), one-way NN correspondences,
    keep-closest-60% trimming, a 120mm crop that can include table/part clutter. No reciprocal
    matching, multistart, or uncertainty. Treat outputs as INDICATIVE, not ground truth.

Point-to-point ICP, numpy + scipy.cKDTree only (no open3d/trimesh).

Frames:
  * cloud (`*_pointcloud.ply`) is already in the ARM-BASE frame (capture baked in T_base_cam).
  * fixture.obj vertices are authored in CENTIMETRES, in a frame close to the arm base.
We convert the mesh to metres, seed ICP with an initial guess (identity after cm->m, plus an
seeded board offset), then ICP solves the residual rigid transform T that best maps CAD -> cloud.
|T| away from identity == how far the cloud sits from the SEEDED CAD placement (with the default
seed = the MEASURED board pose, this is a self-consistency residual, NOT a plan-nominal check).

Usage:
    python3 icp_fixture_to_cad.py \
        --cloud /tmp/fx/fixture/move_pointcloud.ply \
        --mesh  /tmp/fx/fixture/fixture.obj
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# run standalone from harness/ without install
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from cv_insert.icp import icp, voxel_downsample  # noqa: E402


# ---------- IO ----------
def load_ply_xyz(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header = b""
        while b"end_header" not in header:
            line = f.readline()
            if not line:
                break
            header += line
        lines = header.decode("ascii", "ignore").splitlines()
        n = next(int(l.split()[-1]) for l in lines if l.startswith("element vertex"))
        fmt = next(l for l in lines if l.startswith("format"))
        props = [l.split()[-1] for l in lines if l.startswith("property")]
        if "ascii" in fmt:
            data = np.loadtxt(f, max_rows=n)
            xyz = data[:, :3]
        else:
            dt = []
            for p in props:
                dt.append((p, "<f4") if p in ("x", "y", "z") or "float" else (p, "u1"))
            # robust: assume first 3 float32 are x,y,z, rest uint8 (rgb) — matches these captures
            rec = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
                           + [(p, "u1") for p in props[3:]])
            raw = np.frombuffer(f.read(rec.itemsize * n), dtype=rec, count=n)
            xyz = np.vstack([raw["x"], raw["y"], raw["z"]]).T
    return np.asarray(xyz, float)


def load_obj_xyz(path: str) -> np.ndarray:
    v = [[float(x) for x in l.split()[1:4]] for l in open(path) if l.startswith("v ")]
    return np.asarray(v, float)


# ---------- ICP (shared core lives in cv_insert.icp) ----------
def T_to_rpy_mm(T):
    R = T[:3, :3]
    t = T[:3, 3]
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2]); ry = np.arctan2(-R[2, 0], sy); rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1]); ry = np.arctan2(-R[2, 0], sy); rz = 0.0
    return np.degrees([rx, ry, rz]), t * 1000.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cloud", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--mesh-scale", type=float, default=0.01, help="mesh units->m (cm=0.01)")
    ap.add_argument("--voxel", type=float, default=0.004, help="downsample grid (m)")
    ap.add_argument("--crop-band", type=float, default=0.12,
                    help="keep cloud pts within +-band(m) of the mesh bbox (drops table/arm)")
    ap.add_argument("--seed-npz", default=None,
                    help="a capture frame_*.npz whose T_base_board seeds the mesh->base placement "
                         "(the MEASURED per-frame board pose, NOT the plan nominal). Residual is a "
                         "self-consistency check unless you seed from nominal_world_pose.json.")
    ap.add_argument("--no-centroid-init", action="store_true",
                    help="skip coarse pre-alignment (debug)")
    args = ap.parse_args()

    cloud = load_ply_xyz(args.cloud)
    mesh = load_obj_xyz(args.mesh) * args.mesh_scale

    # Seed the mesh into the base frame at the NOMINAL board pose, then let ICP find the residual.
    # Default seed = captured T_base_board (the MEASURED per-frame board pose, not plan nominal). Fallback =
    # coarse centroid match (translation residual then includes a partial-scan-centroid artifact,
    # so trust the ROTATION + RMSE, not |t|, in that mode).
    if args.seed_npz:
        d = np.load(args.seed_npz)  # numeric arrays only; allow_pickle not needed
        Tbb = np.asarray(d["T_base_board"], float)  # mesh(board frame) -> base
        mesh_seed = (np.c_[mesh, np.ones(len(mesh))] @ Tbb.T)[:, :3]
        seed_mode = "T_base_board (MEASURED per-frame, NOT plan-nominal)"
    elif not args.no_centroid_init:
        near = cloud[(cloud[:, 2] > -0.05) & (cloud[:, 2] < 0.15)]
        mesh_seed = mesh + (near.mean(0) - mesh.mean(0))
        seed_mode = "centroid (|t| = artifact; trust rpy+rmse)"
    else:
        mesh_seed = mesh
        seed_mode = "none"

    # crop cloud to a box around the seeded mesh so ICP registers the fixture, not the whole sweep.
    lo = mesh_seed.min(0) - args.crop_band
    hi = mesh_seed.max(0) + args.crop_band
    m = np.all((cloud >= lo) & (cloud <= hi), axis=1)
    cloud_c = cloud[m]
    mesh = mesh_seed  # ICP now starts from the seeded mesh; residual is the fine correction

    src = voxel_downsample(mesh, args.voxel)
    dst = voxel_downsample(cloud_c, args.voxel)

    print(f"# seed={seed_mode}")
    print(f"# cloud={len(cloud)} cropped={len(cloud_c)} dst_ds={len(dst)}  mesh={len(mesh)} src_ds={len(src)}")
    if len(dst) < 100 or len(src) < 50:
        print("# ERROR: too few points after crop/downsample — check frames/scale/crop-band")
        return

    T, rmse, n = icp(src, dst, init=np.eye(4))
    rpy, t_mm = T_to_rpy_mm(T)
    print(f"# ICP converged: rmse={rmse*1000:.2f} mm  (n_used={n})")
    print(f"# residual CAD->reality:  translation(mm) = [{t_mm[0]:.1f}, {t_mm[1]:.1f}, {t_mm[2]:.1f}]"
          f"   |t| = {np.linalg.norm(t_mm):.1f} mm")
    print(f"#                         rotation(deg rpy) = [{rpy[0]:.2f}, {rpy[1]:.2f}, {rpy[2]:.2f}]")
    print("# interpretation: INDICATIVE fixture-registration residual only (see header caveats). "
          "NOT a validation of held-part / insertion poses, and the seed is the MEASURED board "
          "pose, not the plan nominal. A persistent Z offset consistent across clouds is likely a "
          "CAD-datum (slab-bottom vs surface) convention, not a misregistration.")


if __name__ == "__main__":
    main()
