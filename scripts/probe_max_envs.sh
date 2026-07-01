#!/bin/bash
# Max-env memory probe (2026-07-01): boot train.py briefly (3 iters) at candidate num_envs for each
# memory-heavy config and record PEAK GPU memory, so the 3-week unattended holiday chain never OOMs and
# we know the common env count for a fair (matched-env) comparison. Torque is memory-free (known 48 OK)
# and plain 224px is known (48 = 21 GB) -> not probed. Each run under a timeout (rc.36 hang-safe).
#   setsid bash scripts/probe_max_envs.sh </dev/null >/tmp/probe_max_envs.log 2>&1 &
set -u
cd /home/moreno/Masterthesis-rl-train
PY=/home/moreno/miniconda3/envs/isaaclab/bin/python
KIT=/home/moreno/Masterthesis-rl-train/apps/isaaclab.python.headless.rendering.physx1065.kit
TASK=Isaac-Insertion-CoolingPeg-Vision-Direct-v0
NOAUX='~agent.params.network.aux_head agent.params.env.obs_groups.obs=[policy,image]'
OV224="$NOAUX env.image_height=224 env.image_width=224 env.tiled_camera.width=224 env.tiled_camera.height=224"
OV256="$NOAUX env.image_height=256 env.image_width=256 env.tiled_camera.width=256 env.tiled_camera.height=256"
OUT=/tmp/probe_max_envs.txt; : > "$OUT"
ts(){ date '+%T'; }

free_gpu(){ for gp in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$gp" 2>/dev/null; done
  for s in /dev/shm/carb-*; do [ -e "$s" ] || continue; p=$(echo "$s"|grep -oE '[0-9]+$'); kill -0 "$p" 2>/dev/null || rm -f "$s"; done; sleep 6; }

probe(){ # name  overrides  num_envs
  local name="$1" ov="$2" ne="$3"
  local log="/tmp/probe_${name}_${ne}.log"
  echo "[$(ts)] probe $name @ ${ne} env ..."
  OMNI_KIT_ACCEPT_EULA=YES TORCHDYNAMO_DISABLE=1 PYTHONUNBUFFERED=1 \
    timeout --signal=KILL 480 "$PY" scripts/train.py --task "$TASK" --num_envs "$ne" \
    --headless --enable_cameras --experience "$KIT" --seed 42 --max_iterations 3 \
    agent.params.config.full_experiment_name=probe_tmp $ov > "$log" 2>&1 &
  local tpid=$! peak=0 m
  while kill -0 "$tpid" 2>/dev/null; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$m" ] && [ "$m" -gt "$peak" ] 2>/dev/null && peak="$m"
    sleep 3
  done
  wait "$tpid"; local rc=$? verdict
  if grep -qiE "out of memory|CUDA error|torch.*alloc.*fail" "$log"; then verdict="OOM (peak ${peak} MiB before fail)"
  elif [ "$rc" -eq 137 ]; then verdict="HANG/killed 480s (peak ${peak})"
  elif [ "$rc" -ne 0 ]; then verdict="FAIL rc=$rc (peak ${peak}); tail: $(tail -1 "$log")"
  else verdict="FIT  peak ${peak} MiB"; fi
  printf '%-14s %3s env : %s\n' "$name" "$ne" "$verdict" | tee -a "$OUT"
  free_gpu; rm -rf logs/rl_games/Forge/probe_tmp 2>/dev/null
}

echo "=== MAX-ENV PROBE $(date '+%F %T') (card = 24 GB; target <=22 GB w/ headroom) ==="
probe res256      "$OV256"                  44
probe res256      "$OV256"                  32
probe fs3_224     "$OV224 env.frame_stack=3" 40
probe fs3_224     "$OV224 env.frame_stack=3" 32
probe res256_fs3  "$OV256 env.frame_stack=3" 24
probe res256_fs3  "$OV256 env.frame_stack=3" 16
echo "=== PROBE DONE $(date '+%T') ==="; echo "--- summary ---"; cat "$OUT"
