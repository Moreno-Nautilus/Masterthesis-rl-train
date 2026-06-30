#!/bin/bash
# Robust multi-eval wrapper. Runs eval_policy.py for each seed (and/or checkpoint) under a per-eval
# TIMEOUT, so an Isaac boot/shutdown HANG (the known rc.36 instability) kills just that eval + frees
# the GPU and the loop moves on -- instead of one hang stalling the whole batch (which cost us a
# 3-seed noise-floor run on 2026-06-30).
#
# Usage:
#   scripts/eval_robust.sh <task> <checkpoint> <seed1,seed2,...> [report_prefix]
# Env knobs:
#   EXTRA_OVERRIDES   extra hydra overrides (e.g. OLD no-aux ckpt under the aux config:
#                     EXTRA_OVERRIDES='~agent.params.network.aux_head agent.params.env.obs_groups.obs=[policy,image]')
#   TIMEOUT=900       seconds per eval before it's treated as hung and killed (default 15 min)
#   NUM_ENVS=128  NUM_EPISODES=512
#   EXPERIENCE=<abs kit>   (defaults to the physx1065 kit)
#
# Example (the noise-floor run that hung today, done robustly):
#   scripts/eval_robust.sh Isaac-Insertion-CoolingPeg-Vision-Direct-v0 \
#     logs/rl_games/Forge/vision_appearance_1/nn/last_Forge_ep_3000_rew_151.50232.pth 42,7,123 deliv \
#     EXTRA_OVERRIDES='~agent.params.network.aux_head agent.params.env.obs_groups.obs=[policy,image]'
set -u
cd /home/moreno/Masterthesis-rl-train

TASK="${1:?task id}"
CKPT="${2:?checkpoint path}"
SEEDS_CSV="${3:?comma-separated seeds}"
PREFIX="${4:-eval}"
PY=/home/moreno/miniconda3/envs/isaaclab/bin/python
EXPERIENCE="${EXPERIENCE:-/home/moreno/Masterthesis-rl-train/apps/isaaclab.python.headless.rendering.physx1065.kit}"
EXTRA="${EXTRA_OVERRIDES:-}"
TIMEOUT="${TIMEOUT:-900}"
NUM_ENVS="${NUM_ENVS:-128}"
NUM_EPISODES="${NUM_EPISODES:-512}"
SUMMARY="/tmp/${PREFIX}_summary.txt"
: > "$SUMMARY"

free_gpu(){
  for gp in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$gp" 2>/dev/null; done
  for s in /dev/shm/carb-*; do [ -e "$s" ] || continue; p=$(echo "$s"|grep -oE '[0-9]+$'); kill -0 "$p" 2>/dev/null || rm -f "$s"; done
}

IFS=',' read -ra SEEDS <<< "$SEEDS_CSV"
for SEED in "${SEEDS[@]}"; do
  REPORT="/tmp/${PREFIX}_s${SEED}.txt"; LOG="/tmp/${PREFIX}_s${SEED}.log"
  echo "[$(date '+%T')] eval seed=$SEED (timeout ${TIMEOUT}s) ..."
  OMNI_KIT_ACCEPT_EULA=YES TORCHDYNAMO_DISABLE=1 PYTHONUNBUFFERED=1 \
    timeout --signal=KILL "$TIMEOUT" "$PY" scripts/eval_policy.py \
      --task "$TASK" --num_envs "$NUM_ENVS" --num_episodes "$NUM_EPISODES" \
      --headless --enable_cameras --experience "$EXPERIENCE" --seed "$SEED" \
      --checkpoint "$CKPT" --report "$REPORT" $EXTRA > "$LOG" 2>&1
  rc=$?
  if [ "$rc" -eq 137 ]; then
    line="seed $SEED: HUNG (killed after ${TIMEOUT}s)"
  else
    res=$(grep -oE "overall success: [0-9.]+%" "$REPORT" 2>/dev/null | head -1)
    line="seed $SEED: ${res:-FAILED (rc=$rc, no report)}"
  fi
  echo "  -> $line"; echo "$line" >> "$SUMMARY"
  free_gpu; sleep 2   # ensure the GPU/shm are clean before the next eval boots
done

echo "=== SUMMARY ($SUMMARY) ==="; cat "$SUMMARY"
