#!/bin/bash
# Frame-stack follow-up probe (2026-07-01): the main probe found 224px+frame_stack=3 OOMs at BOTH 40 and 32
# on a transient ~9.2 GB PPO-reshape spike of the 12-ch image. Goal: find a frame-stack config that FITS at 32
# so the holiday family stays unified at 32 (not dragged to 24). Tests: frame_stack=2 (8-ch, smaller spike) and
# the expandable_segments allocator flag on fs3. Same peak-sampling + timeout as probe_max_envs.sh.
#   setsid bash scripts/probe_framestack.sh </dev/null >/tmp/probe_framestack.log 2>&1 &
set -u
cd /home/moreno/Masterthesis-rl-train
PY=/home/moreno/miniconda3/envs/isaaclab/bin/python
KIT=/home/moreno/Masterthesis-rl-train/apps/isaaclab.python.headless.rendering.physx1065.kit
TASK=Isaac-Insertion-CoolingPeg-Vision-Direct-v0
OV224='~agent.params.network.aux_head agent.params.env.obs_groups.obs=[policy,image] env.image_height=224 env.image_width=224 env.tiled_camera.width=224 env.tiled_camera.height=224'
OUT=/tmp/probe_framestack.txt; : > "$OUT"
ts(){ date '+%T'; }

free_gpu(){ for gp in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$gp" 2>/dev/null; done
  for s in /dev/shm/carb-*; do [ -e "$s" ] || continue; p=$(echo "$s"|grep -oE '[0-9]+$'); kill -0 "$p" 2>/dev/null || rm -f "$s"; done; sleep 6; }

probe(){ # name  overrides  num_envs  [alloc_conf]
  local name="$1" ov="$2" ne="$3" alloc="${4:-}"
  local log="/tmp/probe_${name}_${ne}.log"
  echo "[$(ts)] probe $name @ ${ne} env  alloc='${alloc:-default}' ..."
  PYTORCH_CUDA_ALLOC_CONF="$alloc" OMNI_KIT_ACCEPT_EULA=YES TORCHDYNAMO_DISABLE=1 PYTHONUNBUFFERED=1 \
    timeout --signal=KILL 480 "$PY" scripts/train.py --task "$TASK" --num_envs "$ne" \
    --headless --enable_cameras --experience "$KIT" --seed 42 --max_iterations 3 \
    agent.params.config.full_experiment_name=probe_tmp $ov > "$log" 2>&1 &
  local tpid=$! peak=0 m
  while kill -0 "$tpid" 2>/dev/null; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$m" ] && [ "$m" -gt "$peak" ] 2>/dev/null && peak="$m"; sleep 3
  done
  wait "$tpid"; local rc=$? verdict
  if grep -qiE "out of memory|OutOfMemory" "$log"; then verdict="OOM ($(grep -oiE "Tried to allocate [0-9.]+ GiB" "$log" | head -1))"
  elif [ "$rc" -eq 137 ]; then verdict="HANG 480s"
  elif [ "$rc" -ne 0 ]; then verdict="FAIL rc=$rc"
  else verdict="FIT  peak ${peak} MiB"; fi
  printf '%-16s %3s env %-28s : %s\n' "$name" "$ne" "alloc=${alloc:-default}" "$verdict" | tee -a "$OUT"
  free_gpu; rm -rf logs/rl_games/Forge/probe_tmp 2>/dev/null
}

XSEG="expandable_segments:True"
echo "=== FRAME-STACK PROBE $(date '+%F %T') (goal: fit at 32 to keep family unified) ==="
probe fs2_224      "$OV224 env.frame_stack=2" 40
probe fs2_224      "$OV224 env.frame_stack=2" 32
probe fs3_224_xseg "$OV224 env.frame_stack=3" 32 "$XSEG"
probe fs2_224_xseg "$OV224 env.frame_stack=2" 32 "$XSEG"
probe fs3_224      "$OV224 env.frame_stack=3" 24
echo "=== FRAME-STACK PROBE DONE $(date '+%T') ==="; echo "--- summary ---"; cat "$OUT"
