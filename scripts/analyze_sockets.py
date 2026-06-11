"""Locate the through-holes (candidate insertion sockets) in a receptacle mesh.

Works directly on the source OBJ (authored in centimeters). Takes a planar cross-section
at mid-thickness; the interior loops of that section are the through-holes. Reports each
hole's center (mm, in the centered-mesh frame) and equivalent diameter (mm), flags which
fit a given peg, and saves a top-down visualization PNG.

Usage:
    python scripts/analyze_sockets.py --base cooling_base --peg cooling_screw
    python scripts/analyze_sockets.py --base pb_base --peg pb_screw
"""

import argparse
import os

import numpy as np
import trimesh

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MESH_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "source", "insertion_policy", "insertion_policy", "assets", "meshes")
)


def poly_area_centroid(p: np.ndarray):
    """Signed area and centroid of a closed 2D polygon (shoelace)."""
    x, y = p[:, 0], p[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y2 - x2 * y
    a = cross.sum() / 2.0
    if abs(a) < 1e-9:
        return 0.0, p.mean(axis=0)
    cx = ((x + x2) * cross).sum() / (6 * a)
    cy = ((y + y2) * cross).sum() / (6 * a)
    return a, np.array([cx, cy])


def peg_diameter_mm(name: str) -> float:
    m = trimesh.load(os.path.join(MESH_DIR, f"{name}.obj"), process=False)
    dx, dy, _ = (m.bounds[1] - m.bounds[0]) * 10.0  # cm -> mm
    return float(max(dx, dy))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="cooling_base")
    ap.add_argument("--peg", default="cooling_screw")
    args = ap.parse_args()

    mesh = trimesh.load(os.path.join(MESH_DIR, f"{args.base}.obj"), process=False)
    mesh.apply_scale(10.0)  # cm -> mm
    lo, hi = mesh.bounds
    ext = hi - lo
    zc = (lo[2] + hi[2]) / 2.0
    print(f"{args.base}: extent(mm) = {np.round(ext, 2).tolist()}  (insertion axis = Z, section at z={zc:.1f})")

    section = mesh.section(plane_origin=[0, 0, zc], plane_normal=[0, 0, 1])
    if section is None:
        raise RuntimeError("empty cross-section")
    planar, to_3d = section.to_planar()

    loops = []  # (world_xy_centroid, equiv_diameter_mm, polygon_world_xy, abs_area)
    for poly2d in planar.discrete:
        a, c2d = poly_area_centroid(poly2d)
        area = abs(a)
        # map planar -> world (3D), keep XY
        n = len(poly2d)
        h = np.column_stack([poly2d, np.zeros(n), np.ones(n)])
        world = (to_3d @ h.T).T[:, :2]
        ch = np.array([c2d[0], c2d[1], 0.0, 1.0])
        cworld = (to_3d @ ch)[:2]
        dia = 2.0 * np.sqrt(area / np.pi)
        loops.append([cworld, dia, world, area])

    if not loops:
        raise RuntimeError("no loops found in section")

    # Largest loop = outer plate boundary; the rest are holes.
    loops.sort(key=lambda L: L[3], reverse=True)
    outer = loops[0]
    holes = loops[1:]

    pd = peg_diameter_mm(args.peg)
    print(f"{args.peg}: outer diameter = {pd:.1f} mm")
    print(f"outer boundary: equiv span ~{outer[1]:.1f} mm")
    print(f"Found {len(holes)} interior hole(s):")
    for k, (c, dia, _poly, _area) in enumerate(holes, 1):
        fits = dia >= pd
        print(f"  hole {k}: center=({c[0]:+.1f}, {c[1]:+.1f}) mm  dia~{dia:.1f} mm  [{'FITS PEG' if fits else 'too small'}]")

    # Visualization (top-down).
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(*np.vstack([outer[2], outer[2][:1]]).T, color="0.4", lw=1.5)
    for k, (c, dia, poly, _area) in enumerate(holes, 1):
        col = "tab:green" if dia >= pd else "tab:red"
        ax.plot(*np.vstack([poly, poly[:1]]).T, color=col, lw=2)
        ax.text(c[0], c[1], str(k), color=col, ha="center", va="center", fontsize=12, fontweight="bold")
    ax.set_title(f"{args.base} holes (green=fits {args.peg} {pd:.0f}mm)  frame: centered, mm")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    out = f"/tmp/{args.base}_sockets.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"Saved visualization: {out}")


if __name__ == "__main__":
    main()
