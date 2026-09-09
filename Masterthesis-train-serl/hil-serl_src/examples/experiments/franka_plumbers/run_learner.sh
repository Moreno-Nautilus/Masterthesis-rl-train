#!/usr/bin/env bash
# Franka plumbers HIL-SERL learner (GPU). Usage:
#   source ../../../../franka_env.sh            # (comment out JAX_PLATFORMS=cpu for GPU)
#   export FRANKA_EXP_NAME=franka_plumbers_insert0
#   bash experiments/franka_plumbers/run_learner.sh --demo_path <merged_demos.pkl>
#
# The learner does NOT need ROS2 or the controller — it builds the env with fake_env=True
# (no PS4 / no camera / no robot). checkpoint_path must be ABSOLUTE; orbax refuses to
# overwrite, so rm -rf the checkpoint dir for a fresh run (wipes the replay buffer).
# --debug disables wandb. Wait for "sent initial network to actor" before the actor.
#
# NEVER auto-launched: the user starts every training run (see FRANKA_RIG_CHECKLIST.md).
set -euo pipefail
: "${FRANKA_EXP_NAME:?set FRANKA_EXP_NAME=franka_plumbers_insertN first}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3
export CUDA_ROOT="${CUDA_ROOT:-/usr}"

# Resolve train_rlpd.py relative to THIS script (experiments/franka_plumbers/ -> examples/).
TRAIN_RLPD="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/train_rlpd.py"
cd "$(dirname "$TRAIN_RLPD")"

# Anchor CKPT to examples/ (script-relative, ABSOLUTE) — must match run_actor.sh exactly.
CKPT="${CKPT:-$(pwd)/checkpoints/${FRANKA_EXP_NAME}}"

# Use the interpreter pinned by franka_env.sh (NOT a bare python3, which an
# active Conda env could hijack and break rclpy).
"${SERL_PYTHON:-/usr/bin/python3.10}" "$TRAIN_RLPD" "$@" \
    --exp_name="${FRANKA_EXP_NAME}" \
    --checkpoint_path="${CKPT}" \
    --learner \
    --debug
