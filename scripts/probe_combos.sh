#!/bin/bash
# 256px + frame-stack combo probe (2026-07-01): the 224px+fs answer comes from probe_framestack.sh; this maps
# the heavier 256px+fs combos so the holiday chain can pick viable env counts. Main probe already found
# 256px+fs3 OOMs at 24 AND 16 -> here we go lower (12/8) + test fs2 (8-ch) + the expandable_segments flag.
#   setsid bash scripts/probe_combos.sh </dev/null >/tmp/probe_combos.log 2>&1 &
set -u
cd /home/moreno/Masterthesis-rl-train
PY=/home/moreno/miniconda3/envs/isaaclab/bin/python
KIT=/home/moreno/Masterthesis-rl-train/apps/isaaclab.python.headless.rendering.physx1065.kit
TASK=Isaac-Insertion-CoolingPeg-Vision-Direct-v0
OV256='~agent.params.network.aux_head agent.params.env.obs_groups.obs=[policy,image] env.image_height=256 env.image_width=256 env.tiled_camera.width=256 env.tiled_camera.height=256'
OUT=/tmp/probe_combos.txt; : > "$OUT"
ts(){ date '+%T'; }
free_gpu(){ for gp in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$gp" 2>/dev/null; done
  for s in /dev/shm/carb-*; do [ -e "$s" ] || continue; p=$(echo "$s"|grep -oE '[0-9]+$'); kill -0 "$p" 2>/dev/null || rm -f "$s"; done; sleep 6; }

probe(){ # name overrides num_envs [alloc]
  local name="$1" ov="$2" ne="$3" alloc="${4:-}" log="/tmp/probe_${name}_${ne}.log"
  echo "[$(ts)] probe $name @ ${ne} env alloc='${alloc:-default}' ..."
  PYTORCH_CUDA_ALLOC_CONF="$alloc" OMNI_KIT_ACCEPT_EULA=YES TORCHDYNAMO_DISABLE=1 PYTHONUNBUFFERED=1 \
    timeout --signal=KILL 480 "$PY" scripts/train.py --task "$TASK" --num_envs "$ne" \
    --headless --enable_cameras --experience "$KIT" --seed 42 --max_iterations 3 \
    agent.params.config.full_experiment_name=probe_tmp $ov > "$log" 2>&1 &
  local tpid=$! peak=0 m
  while kill -0 "$tpid" 2>/dev/null; do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$m" ] && [ "$m" -gt "$peak" ] 2>/dev/null && peak="$m"; sleep 3; done
  wait "$tpid"; local rc=$? verdict
  if grep -qiE "out of memory|OutOfMemory" "$log"; then verdict="OOM ($(grep -oiE 'Tried to allocate [0-9.]+ GiB' "$log"|head -1))"
  elif [ "$rc" -eq 137 ]; then verdict="HANG"; elif [ "$rc" -ne 0 ]; then verdict="FAIL rc=$rc"
  else verdict="FIT  peak ${peak} MiB"; fi
  printf '%-14s %3s env %-26s : %s\n' "$name" "$ne" "alloc=${alloc:-default}" "$verdict" | tee -a "$OUT"
  free_gpu; rm -rf logs/rl_games/Forge/probe_tmp 2>/dev/null
}

XSEG="expandable_segments:True"
echo "=== 256px+FS COMBO PROBE $(date '+%F %T') ==="
probe res256_fs2 "$OV256 env.frame_stack=2" 32
probe res256_fs2 "$OV256 env.frame_stack=2" 24
probe res256_fs2 "$OV256 env.frame_stack=2" 16
probe res256_fs2 "$OV256 env.frame_stack=2" 24 "$XSEG"
probe res256_fs3 "$OV256 env.frame_stack=3" 16 "$XSEG"
probe res256_fs3 "$OV256 env.frame_stack=3" 12
probe res256_fs3 "$OV256 env.frame_stack=3" 8
echo "=== 256px+FS COMBO PROBE DONE $(date '+%T') ==="; echo "--- summary ---"; cat "$OUT"
