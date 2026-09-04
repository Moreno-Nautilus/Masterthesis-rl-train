"""Standalone reader for an ``imitation_handoff/`` bundle produced by
``planning/export_insertion_handoff.py``.

Pure ``numpy`` + ``json`` - copy this file into the learning repo, it has **no
Fabrica / trimesh / ikpy dependency**. It just gives typed access to the reset
points and nominal trajectories so a residual-RL / imitation-learning agent can
drive the episode loop.

    from load_insertion_handoff import InsertionHandoff

    ho = InsertionHandoff("logs/kuka_pdz8_cm_aruco/cooling_manifold_aruco_2/imitation_handoff")
    print(ho.insertion_to_part)                     # {0: '2', 1: '6', ...}
    ins = ho.insertion(1)                           # part 6
    q_move0 = ins.reset_move_arm_q                  # (7,) pre-insertion reset, rad
    q_hold0 = ins.reset_hold_arm_q                  # (7,) static holder pose, rad
    ref     = ins.nominal_insertion_traj_move       # (N,7) feed-forward for the residual
    goal    = ins.goal_move_arm_q                   # (7,) target
"""

import json
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class Insertion:
    insertion_index: int
    part_id: str
    segment_index: int
    approach_segment_index: "int | None"
    n_insertion_waypoints: int
    # reset point
    reset_move_arm_q: np.ndarray            # (7,) rad
    reset_hold_arm_q: "np.ndarray | None"   # (7,) rad, static holder pose
    reset_move_gripper_command: "float | None"
    reset_hold_gripper_command: "float | None"
    reset_move_ee_pose_m: np.ndarray        # (4,4)
    reset_hold_ee_pose_m: "np.ndarray | None"
    # targets / references
    goal_move_arm_q: np.ndarray             # (7,) rad
    goal_move_ee_pose_m: np.ndarray         # (4,4)
    nominal_insertion_traj_move: np.ndarray            # (N,7) rad - residual feed-forward
    nominal_approach_traj_move: "np.ndarray | None"    # (M,7) rad - transport into the insertion


class InsertionHandoff:
    def __init__(self, handoff_dir):
        self.dir = handoff_dir
        with open(os.path.join(handoff_dir, "insertion_index.json")) as fp:
            self._index = json.load(fp)
        with open(os.path.join(handoff_dir, "robot_meta.json")) as fp:
            self.robot_meta = json.load(fp)
        self.meta = self._index["meta"]
        self.reset_protocol = self._index["reset_protocol"]
        self.insertion_to_part = {
            int(k): v for k, v in self._index["insertion_to_part"].items()
        }
        self._entries = {e["insertion_index"]: e for e in self._index["insertions"]}

    # -- convenience --------------------------------------------------------
    @property
    def n_insertions(self):
        return len(self._entries)

    @property
    def inserted_parts_in_order(self):
        return self.meta["inserted_parts_in_order"]

    @property
    def holder_part_ids(self):
        return self.meta["holder_part_ids"]

    def part_for_insertion(self, k):
        return self.insertion_to_part[k]

    def insertion_for_part(self, part_id):
        for k, p in self.insertion_to_part.items():
            if p == str(part_id):
                return k
        raise KeyError(f"part {part_id} is not inserted (holder: {self.holder_part_ids})")

    # -- the data ---------------------------------------------------------
    def insertion(self, k):
        e = self._entries[k]
        npz = np.load(os.path.join(self.dir, e["npz"]))
        rp, goal = e["reset_point"], e["insertion_goal"]
        return Insertion(
            insertion_index=k,
            part_id=e["part_id"],
            segment_index=e["segment_index"],
            approach_segment_index=e["approach_segment_index"],
            n_insertion_waypoints=e["n_insertion_waypoints"],
            reset_move_arm_q=np.asarray(rp["move_arm_q"], float),
            reset_hold_arm_q=None if rp["hold_arm_q"] is None
            else np.asarray(rp["hold_arm_q"], float),
            reset_move_gripper_command=rp["move_gripper_command"],
            reset_hold_gripper_command=rp["hold_gripper_command"],
            reset_move_ee_pose_m=np.asarray(rp["move_ee_pose_m"], float),
            reset_hold_ee_pose_m=None if rp["hold_ee_pose_m"] is None
            else np.asarray(rp["hold_ee_pose_m"], float),
            goal_move_arm_q=np.asarray(goal["move_arm_q"], float),
            goal_move_ee_pose_m=np.asarray(goal["move_ee_pose_m"], float),
            nominal_insertion_traj_move=npz["nominal_insertion_traj_move"],
            nominal_approach_traj_move=npz["nominal_approach_traj_move"]
            if "nominal_approach_traj_move" in npz.files else None,
        )

    def all_insertions(self):
        return [self.insertion(k) for k in sorted(self._entries)]

    def nominal_plan(self):
        """The full nominal plan: (segments_json, {traj_key: (n,7) array})."""
        with open(os.path.join(self.dir, "nominal_plan.json")) as fp:
            segs = json.load(fp)["segments"]
        npz = np.load(os.path.join(self.dir, "nominal_plan.npz"))
        return segs, {k: npz[k] for k in npz.files}


if __name__ == "__main__":
    import sys

    ho = InsertionHandoff(sys.argv[1])
    print(f"{ho.meta['assembly']}: {ho.n_insertions} insertions, "
          f"holder {ho.holder_part_ids}, order {ho.inserted_parts_in_order}")
    for ins in ho.all_insertions():
        print(f"  k={ins.insertion_index} part={ins.part_id} "
              f"seg={ins.segment_index} wp={ins.n_insertion_waypoints} "
              f"reset|goal dq_l1={np.abs(ins.goal_move_arm_q - ins.reset_move_arm_q).sum():.3f} rad "
              f"ref{ins.nominal_insertion_traj_move.shape}")
