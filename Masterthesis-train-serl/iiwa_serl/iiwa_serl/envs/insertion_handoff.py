"""Adapter: assembly handoff bundle -> our env's per-insert spec.

Wraps the supervisor's dependency-free `load_insertion_handoff.InsertionHandoff`
and converts one insert into the quantities our SERL env needs:

  * `reset_joints`  (7,) rad   -- hard reset target for the inserting arm.
  * `goal_pose6`    (6,)       -- seated EE target in OUR pose6 [x,y,z,rx,ry,rz]
                                  (metres + intrinsic-xyz Euler rad), from the
                                  handoff's 4x4 goal EE pose.
  * `reset_pose6`   (6,)       -- pre-insert EE pose6 (from the 4x4 reset EE pose).
  * `insertion_axis` (3,)      -- UNIT world-frame insertion direction
                                  = normalize(goal_ee_pos - reset_ee_pos).
                                  PER-INSERT: NOT always +Z (e.g. plumbers k=2 is +Y).
                                  Retract direction = -insertion_axis; the residual
                                  ORTHOGONAL-TRANSLATION plane is orthogonal to it.
  * `insertion_len_m` (float)  -- straight-line insertion travel.
  * `nominal_n`     (int)      -- number of nominal insertion waypoints.

Residual DoF split (see HIL_RESIDUAL_SPEC.md, decided 2026-09-03): the nominal
owns ONLY the along-axis forward translation; the residual owns the 2 orthogonal
translations AND all 3 rotations. So only the along-axis translation component of
the residual is projected out; rotations pass through untouched.

The insertion is a near-pure straight-line translation (orientation change
negligible, verified on the real data), so the env interpolates the nominal EE
pose linearly reset_pose6 -> goal_pose6 by t/(N-1) rather than doing FK on the
(N,7) joint trajectory -- keeps the env FK-free.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from iiwa_serl.envs.load_insertion_handoff import InsertionHandoff

# Repo-local copy of the extracted handoff bundles
# (iiwa_serl/iiwa_serl/envs/insertion_handoff.py -> parents[3] == Masterthesis-train-serl).
_DEFAULT_HANDOFF_ROOT = Path(__file__).resolve().parents[3] / "assembly_handoff"

_SHORT_TO_PLANDIR = {
    "plumbers_block": "plumbers_block_new_3",
    "cooling_manifold": "cooling_manifold_aruco_2",
}


def _mat4_to_pose6(T: np.ndarray) -> np.ndarray:
    """4x4 homogeneous (metres, rotation matrix) -> pose6 [x,y,z, rx,ry,rz] (Euler xyz)."""
    T = np.asarray(T, dtype=np.float64)
    pos = T[:3, 3]
    rot = Rotation.from_matrix(T[:3, :3]).as_euler("xyz")
    return np.concatenate([pos, rot])


@dataclass
class InsertSpec:
    assembly: str
    insertion_index: int
    part_id: str
    reset_joints: np.ndarray        # (7,) rad — pre-insert JOINT config (frame-independent)
    goal_joints: np.ndarray         # (7,) rad — seated JOINT config (FK'd at runtime for goal/axis)
    reset_pose6: np.ndarray         # (6,) EE pre-insert (handoff frame — for reference only)
    goal_pose6: np.ndarray          # (6,) EE seated target (handoff frame — for reference only)
    insertion_axis: np.ndarray      # (3,) unit (handoff frame — runtime axis derived via FK)
    insertion_len_m: float
    nominal_n: int


def _bundle_dir(assembly_dir: str | Path | None, assembly: str | None) -> Path:
    if assembly_dir is not None:
        return Path(assembly_dir)
    if assembly is None:
        raise ValueError("Provide either assembly_dir or assembly name.")
    root = Path(os.environ.get("ASSEMBLY_HANDOFF_ROOT", _DEFAULT_HANDOFF_ROOT))
    candidates = []
    if assembly in _SHORT_TO_PLANDIR:
        candidates.append(root / _SHORT_TO_PLANDIR[assembly])
    candidates.append(root / assembly)  # accept a plan-dir name directly too
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"No handoff bundle for '{assembly}' under {root}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def load_insert_spec(
    insertion_index: int,
    assembly: str | None = None,
    assembly_dir: str | Path | None = None,
) -> InsertSpec:
    """Load one insert's spec. `assembly` is 'plumbers_block'|'cooling_manifold'
    (or a plan-dir name); or pass `assembly_dir` to the imitation_handoff dir."""
    bundle = _bundle_dir(assembly_dir, assembly)
    ho = InsertionHandoff(str(bundle))
    ins = ho.insertion(insertion_index)

    reset_pose6 = _mat4_to_pose6(ins.reset_move_ee_pose_m)
    goal_pose6 = _mat4_to_pose6(ins.goal_move_ee_pose_m)

    axis_vec = goal_pose6[:3] - reset_pose6[:3]
    length = float(np.linalg.norm(axis_vec))
    if length < 1e-6:
        # Degenerate (near-zero travel placeholder). Fall back to -Z but the tiny
        # length is the real signal; callers should treat <1mm as suspect.
        axis_unit = np.array([0.0, 0.0, -1.0])
    else:
        axis_unit = axis_vec / length

    return InsertSpec(
        assembly=str(ho.meta.get("assembly", assembly)),
        insertion_index=insertion_index,
        part_id=str(ins.part_id),
        reset_joints=np.asarray(ins.reset_move_arm_q, dtype=np.float64),
        goal_joints=np.asarray(ins.goal_move_arm_q, dtype=np.float64),
        reset_pose6=reset_pose6,
        goal_pose6=goal_pose6,
        insertion_axis=axis_unit,
        insertion_len_m=length,
        nominal_n=int(ins.nominal_insertion_traj_move.shape[0]),
    )
