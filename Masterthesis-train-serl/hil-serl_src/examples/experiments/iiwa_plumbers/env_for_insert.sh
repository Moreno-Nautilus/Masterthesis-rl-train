#!/usr/bin/env bash
# Source this to set the per-insert environment for the KUKA plumbers_block HIL-SERL runs.
#   source env_for_insert.sh <k>          # k in 0..3
# Sets:
#   SERL_ARM_PREFIX=lbr_two   -> the RIGHT / inserting arm (plan `move` == robot_num 2)
#   SERL_RESET_JOINTS         -> that insert's reset_move_arm_q (7 rad) for the T6 server
#   RESIDUAL_ENABLE           -> 1 (residual, default) unless already exported to 0 (E2E fallback)
#   IIWA_EXP_NAME             -> the exp_name to pass to train_rlpd.py
#
# These reset-joint vectors are the handoff's reset_move_arm_q, printed by
# load_insert_spec(k, assembly="plumbers_block"). Re-derive with:
#   python -c "from iiwa_serl.envs.insertion_handoff import load_insert_spec as L; \
#     print(','.join(f'{x:.6f}' for x in L($k,assembly='plumbers_block').reset_joints))"

_k="${1:?usage: source env_for_insert.sh <k in 0..3>}"

export SERL_ARM_PREFIX="lbr_two"
export RESIDUAL_ENABLE="${RESIDUAL_ENABLE:-1}"
# RIGHT-arm wrist D405 = realsense_2 (color-only launch). SERL policy/recorder read this topic.
export SERL_WRIST_TOPIC="${SERL_WRIST_TOPIC:-/realsense_2/camera/color/image_raw}"

case "$_k" in
  0) export SERL_RESET_JOINTS="-0.719947,0.959996,-0.616195,-1.513854,0.662503,1.263704,0.006226" ;;  # part3 -Z 5mm
  1) export SERL_RESET_JOINTS="-0.608930,0.839622,-0.739124,-1.583922,0.651323,1.128165,0.010698" ;;  # part1 -Z 50mm
  2) export SERL_RESET_JOINTS="-0.688853,0.968228,-0.089010,-1.363020,-0.210205,1.110175,-0.628006" ;; # part0 +Y 64mm (HORIZONTAL)
  3) export SERL_RESET_JOINTS="-0.596521,1.001240,-0.427133,-1.307039,0.030679,1.634934,-0.468730" ;;  # part4 -Z 50mm
  *) echo "insert k=$_k out of range 0..3" >&2; return 1 2>/dev/null || exit 1 ;;
esac

export IIWA_EXP_NAME="iiwa_plumbers_insert${_k}"
echo "[env_for_insert] k=$_k  arm=$SERL_ARM_PREFIX  residual=$RESIDUAL_ENABLE  exp=$IIWA_EXP_NAME"
echo "[env_for_insert] SERL_RESET_JOINTS=$SERL_RESET_JOINTS"
