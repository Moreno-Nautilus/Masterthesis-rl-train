#!/bin/bash
# Overnight sequential chain (single GPU): re-baseline the STATE policy on the corrected env, then
# train the VISION fusion policy right after. Both run through the auto-resume watchdog so an
# intermittent PhysX crash mid-night is recovered instead of wasting the slot. Each run stops at its
# --max_iterations (MAX EPOCHS), then the watchdog frees the GPU and the next run starts.
#
#   setsid bash scripts/overnight_chain.sh </dev/null >overnight_chain.log 2>&1 &
set -u
cd /home/moreno/Masterthesis-rl-train
ITERS="${ITERS:-500}"
ts(){ date '+%F %T'; }

echo "[$(ts)] === OVERNIGHT CHAIN START (iters=$ITERS) ==="

echo "[$(ts)] >>> 1/2 state_fixed_1  (state baseline, 128 envs)"
bash scripts/auto_resume_train.sh state_fixed_1 Isaac-Insertion-CoolingPeg-Direct-v0 128 "$ITERS"
echo "[$(ts)] <<< state_fixed_1 finished"
sleep 15  # let the GPU settle before the next launch

echo "[$(ts)] >>> 2/2 vision_fuse_1 (vision fusion, 64 envs)"
bash scripts/auto_resume_train.sh vision_fuse_1 Isaac-Insertion-CoolingPeg-Vision-Direct-v0 64 "$ITERS"
echo "[$(ts)] <<< vision_fuse_1 finished"

echo "[$(ts)] === OVERNIGHT CHAIN DONE ==="
