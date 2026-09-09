#!/usr/bin/env bash
# Franka plumbers HIL-SERL actor (FR3, single arm). Usage:
#   source ../../../../franka_env.sh            # SYSTEM py3.10 + serl site-packages + ROS2
#   export FRANKA_EXP_NAME=franka_plumbers_insert0
#   export TELEOP_DEVICE=ps4
#   export SDL_JOYSTICK_DEVICE=/dev/input/js0    # the PS4 controller (adjust via jstest)
#   bash experiments/franka_plumbers/run_actor.sh
#
# The actor connects to the learner and drives the FR3 via the ROS2 backend
# (ROBOT_CLIENT="ros2" -> cartesian_impedance_control). Operator: R1=deadman,
# left stick=XY, L2/R2=Z, right stick=pitch/yaw, D-pad=roll, X=success (ends episode),
# Triangle=abort. Reset pulls up then interpolates to RESET_POSE (no MoveIt).
set -euo pipefail
: "${FRANKA_EXP_NAME:?set FRANKA_EXP_NAME=franka_plumbers_insertN first}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1
export CUDA_ROOT="${CUDA_ROOT:-/usr}"

# Keep JAX on the CPU, but DO NOT hide the GPU: the ZED SDK is CUDA-only and
# CUDA_VISIBLE_DEVICES="" made Camera.open() fail with "NO GPU DETECTED" (rig 2026-09-09).
# JAX_PLATFORMS=cpu keeps the actor's networks off the card (it only does ~10 Hz inference)
# while leaving CUDA available to the camera. MEM_FRACTION above stays small as a backstop.
export JAX_PLATFORMS=cpu
export TELEOP_DEVICE="${TELEOP_DEVICE:-ps4}"

# Resolve train_rlpd.py relative to THIS script (experiments/franka_plumbers/ -> examples/).
TRAIN_RLPD="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/train_rlpd.py"
cd "$(dirname "$TRAIN_RLPD")"

# Anchor CKPT to examples/ (script-relative, ABSOLUTE) — NOT the caller's cwd, so the
# actor and learner always agree on the same checkpoint dir regardless of launch dir.
CKPT="${CKPT:-$(pwd)/checkpoints/${FRANKA_EXP_NAME}}"

# Use the interpreter pinned by franka_env.sh (NOT a bare python3, which an
# active Conda env could hijack and break rclpy).
"${SERL_PYTHON:-/usr/bin/python3.10}" "$TRAIN_RLPD" "$@" \
    --exp_name="${FRANKA_EXP_NAME}" \
    --checkpoint_path="${CKPT}" \
    --actor \
    --debug
