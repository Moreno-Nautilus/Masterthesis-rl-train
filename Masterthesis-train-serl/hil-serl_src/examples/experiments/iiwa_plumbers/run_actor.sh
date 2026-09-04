#!/usr/bin/env bash
# KUKA plumbers_block HIL-SERL actor (drives the RIGHT arm). Usage:
#   source experiments/iiwa_plumbers/env_for_insert.sh <k>
#   export SDL_JOYSTICK_DEVICE=/dev/input/js1     # the PS4 controller
#   bash experiments/iiwa_plumbers/run_actor.sh
#
# The actor connects to the learner on :5488 and to the T6 SERL robot server.
# Confirm the T6 server was launched with the SAME SERL_ARM_PREFIX / SERL_RESET_JOINTS
# (source env_for_insert.sh in the T6 terminal too, then restart the server).
# Reset is AUTOMATED to the annotated pre-insert (joint move to reset_move_arm_q) —
# no manual jog. Operator: R1=deadman, L1=stop-forward (freeze nominal), stick=residual,
# X=success, Square/Triangle=abort.
set -euo pipefail
: "${IIWA_EXP_NAME:?source env_for_insert.sh <k> first}"

CKPT="${CKPT:-$(pwd)/checkpoints/${IIWA_EXP_NAME}}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1
export CUDA_ROOT="${CUDA_ROOT:-/usr}"

# #6: resolve train_rlpd.py relative to THIS script, not the caller's cwd.
TRAIN_RLPD="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/train_rlpd.py"
cd "$(dirname "$TRAIN_RLPD")"

python "$TRAIN_RLPD" "$@" \
    --exp_name="${IIWA_EXP_NAME}" \
    --checkpoint_path="${CKPT}" \
    --actor \
    --debug
