"""Per-insert HIL-SERL configs for the KUKA cooling_manifold assembly (6 inserts).

Thin subclasses of the shared IiwaInsertBase (all logic there). Inserting arm = RIGHT
(lbr_two): source experiments/iiwa_cooling/env_for_insert.sh <k> before launching.
"""

from __future__ import annotations

from experiments.iiwa_insert_base import IiwaInsertBase


class _CoolingBase(IiwaInsertBase):
    assembly = "cooling_manifold"


class TrainConfigInsert0(_CoolingBase):
    insertion_index = 0   # part 2, -Z, 20mm


class TrainConfigInsert1(_CoolingBase):
    insertion_index = 1   # part 6, -Z, 20mm


class TrainConfigInsert2(_CoolingBase):
    insertion_index = 2   # part 5, -Z, 20mm


class TrainConfigInsert3(_CoolingBase):
    insertion_index = 3   # part 0, -Z, 25mm


class TrainConfigInsert4(_CoolingBase):
    insertion_index = 4   # part 4, -Z, 25mm


class TrainConfigInsert5(_CoolingBase):
    insertion_index = 5   # part 3, -Z, 20mm
