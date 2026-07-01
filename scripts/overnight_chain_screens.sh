#!/bin/bash
# Overnight chain (single GPU, 2026-06-30): two cooling-vision SCREENS back-to-back, each via the
# auto-resume watchdog (PhysX-crash safe) to its --max_iterations, then eval_robust.sh (Isaac-hang safe)
# @seed 42. See DECISIONS §15.
#   1) vision_holehead_1  hole-only aux (drop grasp head)         64 envs, 160px, 1500 it
#   2) vision_res224_1    no-aux, higher-res 224px                48 envs, 224px, 1200 it
# Launch detached:
#   setsid bash scripts/overnight_chain_screens.sh </dev/null >/tmp/chain_screens.log 2>&1 &
set -u
cd /home/moreno/Masterthesis-rl-train
TASK=Isaac-Insertion-CoolingPeg-Vision-Direct-v0
KIT=/home/moreno/Masterthesis-rl-train/apps/isaaclab.python.headless.rendering.physx1065.kit
export EXPERIENCE="$KIT"
ts(){ date '+%F %T'; }
free_gpu(){ for gp in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$gp" 2>/dev/null; done
           for s in /dev/shm/carb-*; do [ -e "$s" ] || continue; p=$(echo "$s"|grep -oE '[0-9]+$'); kill -0 "$p" 2>/dev/null || rm -f "$s"; done; sleep 8; }
latest_ckpt(){ ls -t "logs/rl_games/Forge/$1/nn/"last_Forge_ep_[0-9]*_rew_[0-9]*.pth 2>/dev/null | head -1; }

HOLE_OV='~agent.params.network.aux_head.targets.grasp'
RES_OV='~agent.params.network.aux_head agent.params.env.obs_groups.obs=[policy,image] env.image_height=224 env.image_width=224 env.tiled_camera.width=224 env.tiled_camera.height=224'

echo "[$(ts)] === SCREEN CHAIN START ==="

# --- 1/2: hole-only (drop grasp head) ------------------------------------------------------------
echo "[$(ts)] >>> 1/2 vision_holehead_1 (hole-only, 64 envs, 1500 it, seed 42)"
SEED_RNG=42 EXTRA_OVERRIDES="$HOLE_OV" bash scripts/auto_resume_train.sh \
  vision_holehead_1 "$TASK" 64 1500
free_gpu
CK1=$(latest_ckpt vision_holehead_1)
echo "[$(ts)] eval vision_holehead_1: $CK1"
EXTRA_OVERRIDES="$HOLE_OV" NUM_ENVS=128 bash scripts/eval_robust.sh "$TASK" "$CK1" 42 holehead
echo "[$(ts)] <<< vision_holehead_1 done (vs no-aux ep1500 = 69.7%)"

# --- 2/2: higher-res 224px (no-aux) --------------------------------------------------------------
echo "[$(ts)] >>> 2/2 vision_res224_1 (no-aux 224px, 48 envs, 1200 it, seed 42)"
SEED_RNG=42 EXTRA_OVERRIDES="$RES_OV" bash scripts/auto_resume_train.sh \
  vision_res224_1 "$TASK" 48 1200
free_gpu
CK2=$(latest_ckpt vision_res224_1)
echo "[$(ts)] eval vision_res224_1: $CK2  (48 envs to fit 224px memory)"
EXTRA_OVERRIDES="$RES_OV" NUM_ENVS=48 bash scripts/eval_robust.sh "$TASK" "$CK2" 42 res224
echo "[$(ts)] <<< vision_res224_1 done (vs no-aux ~72% / ep1200 baseline)"

# --- comparison plot (morning artifact) ----------------------------------------------------------
/home/moreno/miniconda3/envs/isaaclab/bin/python scripts/plot_compare.py \
  --runs vision_appearance_1 vision_holehead_1 vision_res224_1 \
  --out renders/analysis/overnight_screens.png 2>/dev/null || true

echo "[$(ts)] === SCREEN CHAIN DONE ==="
echo "RESULTS: holehead=$(grep -h success /tmp/holehead_summary.txt 2>/dev/null)  res224=$(grep -h success /tmp/res224_summary.txt 2>/dev/null)"
