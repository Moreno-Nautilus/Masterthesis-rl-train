#!/bin/bash
# ============================================================================================
# WEEKEND E2E TRAINING CHAIN — Isaac Sim 5.1 (isaaclab51), RGB visuomotor cooling-peg insertion.
# Runs THREE variants SEQUENTIALLY on one 24 GB GPU (all 48 envs for comparability), each through
# the auto-resume watchdog (crash/stall -> resume from latest checkpoint). Order A -> C -> B so the
# control + velocity variants finish first; EfficientNet soaks up whatever time is left.
#
# COMMON to all runs: 320x180 16:9 RGB, frozen ImageNet encoder, AMP (mixed_precision), torch_compile
# off, kl_threshold 0.02, seed 42, 48 envs, max 2500 epochs. Only ONE thing changes per run (below), so
# results are attributable. NOTE: ~32 s/epoch => 3 x 2500 ~= 67 h; the last run (B) may land a bit short
# of 2500 in a ~66 h weekend -> COMPARE ALL THREE AT THE MIN EPOCH REACHED (fair comparison).
#
# LAUNCH (detached, survives logout):
#     setsid bash scripts/run_weekend_chain.sh >/tmp/weekend_chain.log 2>&1 &
# WATCH:   tail -f /tmp/weekend_chain.log        (chain-level)
#          tail -f /tmp/we_A_resnet18_baseline.log   (per-run stdout)
#          per-run watchdog audit: /tmp/<name>_master.log
# ============================================================================================
set -u
cd /home/moreno/Masterthesis-rl-train

TASK=Isaac-Insertion-CoolingPeg-Iiwa-E2E-Vision-Direct-v0
ENVS=48
MAX=2500
# AMP on + actor count matched to --num_envs. Everything else stays at the cfg default (the validated setup).
COMMON="agent.params.config.mixed_precision=True agent.params.config.num_actors=48"

run(){  # $1 = experiment name   $2 = the ONE variant override (may be empty)
  echo "[$(date '+%F %T')] ===================== START $1 ====================="
  EXTRA_OVERRIDES="$COMMON $2" SEED_RNG=42 bash scripts/auto_resume_train.sh "$1" "$TASK" "$ENVS" "$MAX"
  echo "[$(date '+%F %T')] ===================== END   $1 ====================="
}

# --- A: baseline / control (ResNet-18, baseline 15-dim obs) ----------------------------------
run we_A_resnet18_baseline ""

# --- C: + EE-frame velocity obs (6) -> policy obs 21-dim -------------------------------------
run we_C_resnet18_velocity "env.e2e_use_velocity_obs=True"

# --- B: EfficientNet-B0 encoder (stronger ImageNet features, 1280-d), same 320x180 image -----
run we_B_efficientnetb0 "agent.params.network.cnn.backbone=efficientnet_b0"

echo "[$(date '+%F %T')] ===================== WEEKEND CHAIN COMPLETE ====================="
