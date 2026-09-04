#!/usr/bin/env bash
# KUKA plumbers_block HIL-SERL learner (GPU). Usage:
#   source experiments/iiwa_plumbers/env_for_insert.sh <k>
#   bash experiments/iiwa_plumbers/run_learner.sh --demo_path <merged_demos.pkl>
#
# NOTE (from serl-rlpd-training-live): the learner is the conda `serl` env with
# CUDA_ROOT=/usr; checkpoint_path must be ABSOLUTE and orbax refuses to overwrite
# (rm -rf the checkpoint dir for a fresh run — wipes the replay buffer). --debug
# disables wandb. Wait for "sent initial network to actor" before starting the actor.
set -euo pipefail
: "${IIWA_EXP_NAME:?source env_for_insert.sh <k> first}"

CKPT="${CKPT:-$(pwd)/checkpoints/${IIWA_EXP_NAME}}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3
export CUDA_ROOT="${CUDA_ROOT:-/usr}"

# #6: resolve train_rlpd.py relative to THIS script (examples/experiments/iiwa_plumbers/ -> examples/),
# not the caller's cwd, so it works regardless of where it's launched.
TRAIN_RLPD="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/train_rlpd.py"
cd "$(dirname "$TRAIN_RLPD")"

python "$TRAIN_RLPD" "$@" \
    --exp_name="${IIWA_EXP_NAME}" \
    --checkpoint_path="${CKPT}" \
    --learner \
    --debug
