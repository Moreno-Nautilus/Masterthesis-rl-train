"""ICP core + pin/part-pose observation for the insertion approach (numpy + scipy only).

Two uses, both PREINSERT / one-shot (NOT in the fast servo loop):
  * ``pin_pose_icp`` — register the held pin's cloud to the pin CAD, seeded from the grasp
    nominal, to OBSERVE ``T_tool_part`` (lateral pin position + shaft axis). This is the code
    the plan's §4.5 refers to. It is an INDEPENDENT CHECK, not certification.
  * ``part_pose_icp`` — register a base/part cloud to its CAD as a fallback feature localizer
    when 2D detection is weak.

CRITICAL — observability is reported, not assumed. A shaft-only screw is not fully 6-DoF
observable: spin about its axis and translation ALONG the axis are weak/unobservable without a
visible tip/shoulder/head (which the gripper usually occludes). ``PinPoseResult`` carries an
``observability`` breakdown and a WEAK ``lateral_sanity_ok`` gate (necessary-not-sufficient — a
partial noisy view can still pass it with mm-level error, so it is NOT an accuracy claim); callers
MUST NOT treat a good RMSE or a passing gate as proof of accuracy. ICP fitness cannot certify.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree


# ---------------- ICP core ----------------
def best_fit_transform(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Least-squares rigid T (R,t) mapping paired A -> B (Kabsch)."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cb - R @ ca
    return T


def icp(src: np.ndarray, dst: np.ndarray, *, init: np.ndarray | None = None,
        max_iter: int = 60, tol: float = 1e-6, reject_frac: float = 0.6):
    """Trimmed point-to-point ICP. Returns (T, rmse_m, n_used). T maps src -> dst."""
    if len(src) < 3 or len(dst) < 3:
        raise ValueError("icp: need >=3 points in each cloud")
    T = np.eye(4) if init is None else np.asarray(init, float).copy()
    tree = cKDTree(dst)
    src_h = np.c_[src, np.ones(len(src))]
    prev = np.inf
    keep = np.ones(len(src), bool)
    for _ in range(max_iter):
        cur = (src_h @ T.T)[:, :3]
        d, idx = tree.query(cur, k=1)
        keep = d <= np.quantile(d, reject_frac)
        T = best_fit_transform(cur[keep], dst[idx[keep]]) @ T
        rmse = float(np.sqrt((d[keep] ** 2).mean()))
        if abs(prev - rmse) < tol:
            break
        prev = rmse
    return T, rmse, int(keep.sum())


def voxel_downsample(pts: np.ndarray, voxel_m: float) -> np.ndarray:
    if voxel_m <= 0:
        return pts
    keys = np.floor(pts / voxel_m).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[idx]


# ---------------- pin/part pose observation ----------------
@dataclass
class PinPoseResult:
    T_correction: np.ndarray          # 4x4: maps the SEED (grasp-nominal) pin pose -> observed
    T_tool_part_observed: np.ndarray  # 4x4: observed pin pose in the tool frame
    rmse_m: float
    n_used: int
    lateral_shift_mm: float            # magnitude of the in-plane (perp-to-axis) correction
    axial_shift_mm: float              # along-axis correction (UNRELIABLE — see observability)
    tilt_deg: float                    # SHAFT-AXIS tilt only (spin excluded)
    observability: dict = field(default_factory=dict)
    lateral_sanity_ok: bool = False    # WEAK necessary-not-sufficient gate; NOT an accuracy claim
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        w = ("; ".join(self.warnings)) or "none"
        return (f"pin ICP: rmse={self.rmse_m*1000:.2f}mm n={self.n_used} | "
                f"lateral={self.lateral_shift_mm:.2f}mm axis-tilt={self.tilt_deg:.2f}deg "
                f"axial={self.axial_shift_mm:.2f}mm(UNRELIABLE) | "
                f"lateral_sanity_ok={self.lateral_sanity_ok} (weak, not accuracy) | warnings: {w}")


def _rot_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def pin_pose_icp(
    pin_cloud_tool: np.ndarray,
    pin_cad_points: np.ndarray,
    *,
    shaft_axis_tool: np.ndarray = np.array([0.0, 0.0, 1.0]),
    T_tool_part_seed: np.ndarray | None = None,
    voxel_m: float = 0.002,
    max_lateral_mm: float = 15.0,
    min_points: int = 150,
    rmse_gate_mm: float = 3.0,
) -> PinPoseResult:
    """Observe the held pin's pose in the TOOL frame by ICP-ing its cloud to the pin CAD.

    Parameters
    ----------
    pin_cloud_tool   : (N,3) measured pin points, already expressed in the TOOL frame (m).
    pin_cad_points   : (M,3) pin CAD surface points in the part frame (m), seeded near nominal.
    shaft_axis_tool  : unit shaft axis in the tool frame (default +Z) — used to split the
                       correction into lateral (trustworthy) vs axial (unreliable).
    T_tool_part_seed : 4x4 grasp-nominal pin pose in tool frame (ICP init). Identity if None.

    Returns a ``PinPoseResult`` with the correction split and explicit observability flags.
    Fail-safe: too few points / high RMSE / implausible or partial (low-coverage) view sets
    ``lateral_sanity_ok=False`` + warnings — but passing it is still only a WEAK check, not proof.
    """
    seed = np.eye(4) if T_tool_part_seed is None else np.asarray(T_tool_part_seed, float)
    src = voxel_downsample(np.asarray(pin_cad_points, float), voxel_m)
    dst = voxel_downsample(np.asarray(pin_cloud_tool, float), voxel_m)

    warnings: list[str] = []
    if len(dst) < min_points:
        warnings.append(f"only {len(dst)} pin points (<{min_points}) — cloud too sparse/occluded")

    T, rmse, n = icp(src, dst, init=seed)
    T_corr = T @ np.linalg.inv(seed)  # correction applied on top of the seed

    axis = np.asarray(shaft_axis_tool, float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    t = T_corr[:3, 3]
    axial = float(t @ axis)
    lateral_vec = t - axial * axis
    lateral = float(np.linalg.norm(lateral_vec))
    # tilt = angle the SHAFT AXIS moved, NOT the full rotation angle. Spin about the shaft is
    # unobservable/irrelevant, so we must not report it as tilt: measure the change of the axis
    # direction only (apply the correction's rotation to the axis and take the angle between).
    axis_obs = T_corr[:3, :3] @ axis
    axis_obs /= np.linalg.norm(axis_obs) + 1e-12
    tilt = float(np.degrees(np.arccos(np.clip(axis_obs @ axis, -1, 1))))

    if rmse * 1000 > rmse_gate_mm:
        warnings.append(f"rmse {rmse*1000:.2f}mm > gate {rmse_gate_mm}mm — poor fit")
    if lateral * 1000 > max_lateral_mm:
        warnings.append(f"lateral {lateral*1000:.1f}mm > {max_lateral_mm}mm — implausible, reject")

    # Angular coverage of the observed cloud about the shaft axis: a partial view (gripper
    # occludes one side) makes lateral weakly determined and biased. Estimate it cheaply.
    # build two axes spanning the plane perpendicular to the shaft, measure angular spread there
    tmp = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1, 0])
    e1 = np.cross(axis, tmp); e1 /= np.linalg.norm(e1) + 1e-12
    e2 = np.cross(axis, e1)
    # coverage = fraction of the circle that is NOT one big empty gap (handles angle wrap): sort
    # angles, find the largest gap between consecutive (incl. wrap), coverage = 1 - gap/2pi.
    r = dst - dst.mean(0)
    ang = np.sort(np.arctan2(r @ e2, r @ e1))
    if len(ang) > 3:
        gaps = np.diff(ang)
        wrap = (ang[0] + 2 * np.pi) - ang[-1]
        largest_gap = max(gaps.max(), wrap)
        coverage_frac = float(1.0 - largest_gap / (2 * np.pi))
    else:
        coverage_frac = 0.0
    axial_span_mm = float(np.ptp(dst @ axis)) * 1000
    if coverage_frac < 0.75:
        warnings.append(f"angular coverage {coverage_frac:.0%} < 75% — partial view, lateral biased")

    observability = {
        "lateral": "observable IF angular coverage is good; biased under partial/occluded views",
        "tilt": "weakly observable (short noisy cloud); reported as SHAFT-AXIS tilt only",
        "axial_translation": "UNRELIABLE without a visible tip/shoulder/head",
        "spin_about_axis": "UNOBSERVABLE for a symmetric shaft",
        "angular_coverage_frac": coverage_frac,
        "axial_span_mm": axial_span_mm,
    }
    warnings.append("NOT certification: lateral is a WEAK sanity estimate; axial/spin uncertified; "
                    "verify against repeat-pick statistics + independent metrology before any use")

    # A DELIBERATELY WEAK necessary-not-sufficient gate. Passing it does NOT mean the estimate is
    # accurate (Codex stress test: a partial noisy view gave ~1.8mm error and still passed a
    # count/RMSE-only gate). It only rules out the grossly-bad cases. Real acceptance needs
    # uncertainty + seed-sensitivity + coverage, which this one-shot cannot provide alone.
    lateral_sanity_ok = (
        len(dst) >= min_points
        and rmse * 1000 <= rmse_gate_mm
        and lateral * 1000 <= max_lateral_mm
        and coverage_frac >= 0.75
    )

    return PinPoseResult(
        T_correction=T_corr,
        T_tool_part_observed=T,
        rmse_m=rmse,
        n_used=n,
        lateral_shift_mm=lateral * 1000,
        axial_shift_mm=axial * 1000,
        tilt_deg=tilt,
        observability=observability,
        lateral_sanity_ok=lateral_sanity_ok,
        warnings=warnings,
    )


def part_pose_icp(part_cloud_base: np.ndarray, part_cad_points_base_seed: np.ndarray,
                  *, voxel_m: float = 0.004, min_points: int = 300,
                  rmse_gate_mm: float = 5.0, max_shift_mm: float = 30.0):
    """Fallback base/part localizer: ICP a part cloud to its CAD (both in base frame, seeded).

    Returns (T_correction, rmse_m, n_used, accepted, warnings). ``accepted`` is a WEAK gate (point
    count + RMSE + plausibility bound) — necessary-not-sufficient, same spirit as the pin gate.
    Indicative only; verify against an independent cue before using for weak-2D features (pb_top).
    """
    src = voxel_downsample(np.asarray(part_cad_points_base_seed, float), voxel_m)
    dst = voxel_downsample(np.asarray(part_cloud_base, float), voxel_m)
    warnings: list[str] = []
    if len(dst) < min_points:
        warnings.append(f"only {len(dst)} pts (<{min_points})")
    T, rmse, n = icp(src, dst)
    shift_mm = float(np.linalg.norm(T[:3, 3])) * 1000
    if rmse * 1000 > rmse_gate_mm:
        warnings.append(f"rmse {rmse*1000:.2f}mm > {rmse_gate_mm}mm")
    if shift_mm > max_shift_mm:
        warnings.append(f"shift {shift_mm:.1f}mm > {max_shift_mm}mm — implausible")
    accepted = (len(dst) >= min_points and rmse * 1000 <= rmse_gate_mm
                and shift_mm <= max_shift_mm)
    warnings.append("indicative only; not certification")
    return T, rmse, n, accepted, warnings
