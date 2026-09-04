"""Per-insert HIL-SERL configs for the KUKA plumbers_block assembly (4 inserts).

One TrainConfig per insert (k=0..3). All logic is in the shared IiwaInsertBase; here we
just bind assembly + insertion_index. See iiwa_insert_base.py and HIL_RESIDUAL_SPEC.md.

Inserting arm = RIGHT (lbr_two): launch the T6 server with SERL_ARM_PREFIX=lbr_two and
SERL_RESET_JOINTS for the insert (source experiments/iiwa_plumbers/env_for_insert.sh <k>).
"""

from __future__ import annotations

from experiments.iiwa_insert_base import IiwaInsertBase


class _PlumbersBase(IiwaInsertBase):
    assembly = "plumbers_block"


class TrainConfigInsert0(_PlumbersBase):
    insertion_index = 0   # part 3, -Z, 5mm


class TrainConfigInsert1(_PlumbersBase):
    insertion_index = 1   # part 1, -Z, 50mm


class TrainConfigInsert2(_PlumbersBase):
    insertion_index = 2   # part 0, +Y HORIZONTAL, 64mm


class TrainConfigInsert3(_PlumbersBase):
    insertion_index = 3   # part 4, -Z, 50mm
