#!/usr/bin/env bash
# Per-insert env for KUKA cooling_manifold HIL-SERL runs.
#   source env_for_insert.sh <k>   # k in 0..5
# Sets SERL_ARM_PREFIX=lbr_two (RIGHT/inserting arm), SERL_RESET_JOINTS (this insert's
# reset_move_arm_q), RESIDUAL_ENABLE (default 1), IIWA_EXP_NAME.
# Re-derive a vector with:
#   python -c "from iiwa_serl.envs.insertion_handoff import load_insert_spec as L; \
#     print(','.join(f'{x:.6f}' for x in L(<k>,assembly='cooling_manifold').reset_joints))"

_k="${1:?usage: source env_for_insert.sh <k in 0..5>}"

export SERL_ARM_PREFIX="lbr_two"
export RESIDUAL_ENABLE="${RESIDUAL_ENABLE:-1}"

case "$_k" in
  0) export SERL_RESET_JOINTS="-0.733607,0.898126,-0.380388,-1.581314,0.422101,0.770945,-0.515088" ;;  # part2 -Z 20mm
  1) export SERL_RESET_JOINTS="-1.422653,1.087175,-0.074943,-1.352498,0.654888,1.335552,0.507875" ;;  # part6 -Z 20mm
  2) export SERL_RESET_JOINTS="0.239282,1.479561,-0.738796,-1.579877,-0.414037,1.060443,2.420525" ;;  # part5 -Z 20mm
  3) export SERL_RESET_JOINTS="-0.839716,0.991448,-0.586012,-1.458392,0.785130,1.280328,-0.208228" ;; # part0 -Z 25mm
  4) export SERL_RESET_JOINTS="-0.583741,0.861272,-0.378940,-1.626335,0.051729,1.051907,0.559825" ;;  # part4 -Z 25mm
  5) export SERL_RESET_JOINTS="-0.796184,0.892301,-0.277428,-1.647095,0.095798,1.094047,-0.346820" ;; # part3 -Z 20mm
  *) echo "insert k=$_k out of range 0..5" >&2; return 1 2>/dev/null || exit 1 ;;
esac

export IIWA_EXP_NAME="iiwa_cooling_insert${_k}"
echo "[env_for_insert] k=$_k  arm=$SERL_ARM_PREFIX  residual=$RESIDUAL_ENABLE  exp=$IIWA_EXP_NAME"
echo "[env_for_insert] SERL_RESET_JOINTS=$SERL_RESET_JOINTS"
