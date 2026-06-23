"""Geometric recoverability check for a peg-in-socket pair.

Answers: can the upstream error distribution (lateral offset + pre-insert tilt + grasp tilt) be
physically SEATED, or are the hard cases (large tilt / large lateral) geometrically impossible
without a lead-in chamfer? Pure mesh analysis (trimesh) on the source OBJs (authored in cm) -- no
Isaac Sim needed.

It reports:
  - socket inner-diameter vs depth (a widening near the mouth = a lead-in chamfer),
  - peg shaft diameter vs height near the tip (a taper = a self-aligning lead-in),
  - the geometric limits: max lateral offset for the tip to clear the mouth, and the max tilt a
    fully-engaged shaft can hold inside the socket (the seated-tilt ceiling).

Usage:
    python scripts/check_recoverability.py --base cooling_base --peg cooling_screw
"""

import argparse
import math
import os

import numpy as np
import trimesh

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


def _section_loops(mesh: trimesh.Trimesh, z: float):
    """List of (world_xy_centroid, area_mm2) for the closed loops of the z-section. No shapely."""
    section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if section is None:
        return []
    planar, to_3d = section.to_planar()
    out = []
    for poly2d in planar.discrete:
        a, c2d = poly_area_centroid(poly2d)
        cworld = (to_3d @ np.array([c2d[0], c2d[1], 0.0, 1.0]))[:2]
        out.append((cworld, abs(a)))
    return out


def hole_diameter_at(mesh: trimesh.Trimesh, z: float, near_xy: np.ndarray, radius_max: float):
    """Equivalent diameter (mm) of the socket hole loop nearest ``near_xy`` (excludes the plate edge)."""
    loops = _section_loops(mesh, z)
    if not loops:
        return None
    # exclude the largest loop (the plate outer boundary); pick the loop nearest the socket center.
    loops_sorted = sorted(loops, key=lambda L: L[1], reverse=True)
    candidates = loops_sorted[1:] if len(loops_sorted) > 1 else loops_sorted
    near = [L for L in candidates if np.linalg.norm(L[0] - near_xy) <= radius_max]
    if not near:
        return None
    c, area = min(near, key=lambda L: np.linalg.norm(L[0] - near_xy))
    return 2.0 * math.sqrt(area / math.pi)


def outer_diameter_at(mesh: trimesh.Trimesh, z: float):
    """Equivalent diameter (mm) of the largest solid loop at height ``z`` (the peg shaft)."""
    loops = _section_loops(mesh, z)
    if not loops:
        return None
    area = max(L[1] for L in loops)
    if area <= 0:
        return None
    return 2.0 * math.sqrt(area / math.pi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="cooling_base")
    ap.add_argument("--peg", default="cooling_screw")
    ap.add_argument("--socket-xy", type=float, nargs=2, default=[30.0, 0.0], help="socket center xy (mm)")
    ap.add_argument("--samples", type=int, default=18)
    args = ap.parse_args()

    base = trimesh.load(os.path.join(MESH_DIR, f"{args.base}.obj"), process=False)
    base.apply_scale(10.0)  # cm -> mm
    peg = trimesh.load(os.path.join(MESH_DIR, f"{args.peg}.obj"), process=False)
    peg.apply_scale(10.0)

    blo, bhi = base.bounds
    plo, phi = peg.bounds
    print(f"{args.base}: extent(mm)={np.round(bhi - blo, 2).tolist()}  z in [{blo[2]:.2f},{bhi[2]:.2f}]")
    print(f"{args.peg}:  extent(mm)={np.round(phi - plo, 2).tolist()}  z in [{plo[2]:.2f},{phi[2]:.2f}]")

    near = np.array(args.socket_xy)

    # --- socket inner diameter vs depth (mouth -> bottom) ---
    print("\nSOCKET inner-diameter vs height z (mm). A larger diameter near the mouth = lead-in chamfer:")
    zs = np.linspace(blo[2] + 0.2, bhi[2] - 0.2, args.samples)
    socket_dias = []
    for z in zs:
        d = hole_diameter_at(base, z, near, radius_max=15.0)
        socket_dias.append(d)
        bar = "" if d is None else f"  dia={d:6.2f}"
        print(f"  z={z:6.2f}{bar}")
    valid = [(z, d) for z, d in zip(zs, socket_dias) if d is not None]

    # --- peg shaft diameter vs height near the tip ---
    print("\nPEG cross-section diameter vs height z (mm), tip first. A shrinking diameter = tip taper/lead-in:")
    zsp = np.linspace(plo[2] + 0.1, plo[2] + 12.0, 12)  # bottom 12mm = the shaft tip region
    shaft_tip_dia = None
    for z in zsp:
        d = outer_diameter_at(peg, z)
        if d is not None and shaft_tip_dia is None and z > plo[2] + 3.0:
            shaft_tip_dia = d
        print(f"  z={z:6.2f}" + ("" if d is None else f"  dia={d:6.2f}"))

    # --- geometric limits ---
    print("\nGEOMETRIC LIMITS:")
    if valid:
        mouth_dia = valid[-1][1]
        body_dia = np.median([d for _, d in valid])
        shaft_dia = shaft_tip_dia or outer_diameter_at(peg, plo[2] + 6.0) or 12.0
        depth = bhi[2] - blo[2]
        clearance = (body_dia - shaft_dia) / 2.0
        chamfer = mouth_dia - body_dia
        # tip clears the mouth while the body is offset: lateral budget ~ (mouth_radius - shaft_radius)
        max_lateral = (mouth_dia - shaft_dia) / 2.0
        # fully-engaged shaft tilt ceiling: shaft can tilt until it jams across the bore
        seated_tilt = math.degrees(math.atan2(2.0 * clearance, depth)) if depth > 0 else 0.0
        print(f"  socket mouth dia   ~ {mouth_dia:.2f} mm")
        print(f"  socket body dia    ~ {body_dia:.2f} mm   (depth {depth:.2f} mm)")
        print(f"  shaft tip dia      ~ {shaft_dia:.2f} mm")
        print(f"  radial clearance   ~ {clearance:.2f} mm")
        print(f"  lead-in chamfer    ~ {chamfer:.2f} mm  ({'PRESENT' if chamfer > 0.3 else 'NONE/negligible'})")
        print(f"  max lateral (tip in mouth) ~ {max_lateral:.2f} mm")
        print(f"  seated-tilt ceiling        ~ {seated_tilt:.1f} deg  (a fully-inserted shaft cannot exceed this)")
        print("\n  => Reset distribution is lateral<=7mm, pre-insert tilt<=25deg (+ grasp<=5deg).")
        print(f"     Lateral {max_lateral:.1f}mm budget vs 7mm ask: {'OK' if max_lateral >= 7 else 'TIGHT/oversampled'}.")
        print(f"     Seated tilt {seated_tilt:.1f}deg vs the <10deg success gate: the policy must straighten")
        print("     the shaft to within the bore before seating; tilt is recovered by sliding the tip in,")
        print("     which NEEDS a mouth chamfer if the initial tilt exceeds the seated ceiling.")
    else:
        print("  socket hole not found at the given --socket-xy; pass the correct socket center.")


if __name__ == "__main__":
    main()
