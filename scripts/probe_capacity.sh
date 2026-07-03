#!/bin/bash
# Capacity mem probe (2026-07-03): verify the two memory-heavy holiday variants fit before launching the
# unattended chain -- lstm2048 @48env/224px and fc512 @32env/256px. 24GB card, <=22GB target.
#   setsid bash scripts/probe_capacity.sh </dev/null >/tmp/probe_capacity.log 2>&1 &
set -u; cd /home/moreno/Masterthesis-rl-train
PY=/home/moreno/miniconda3/envs/isaaclab/bin/python
KIT=/home/moreno/Masterthesis-rl-train/apps/isaaclab.python.headless.rendering.physx1065.kit
TASK=Isaac-Insertion-CoolingPeg-Vision-Direct-v0
NOAUX='~agent.params.network.aux_head agent.params.env.obs_groups.obs=[policy,image]'
OUT=/tmp/probe_capacity.txt; : > "$OUT"
free_gpu(){ for gp in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$gp" 2>/dev/null; done
  for s in /dev/shm/carb-*; do [ -e "$s" ] || continue; p=$(echo "$s"|grep -oE '[0-9]+$'); kill -0 "$p" 2>/dev/null || rm -f "$s"; done; sleep 6; }
probe(){ local name="$1" ov="$2" ne="$3"
  local log="/tmp/probe_${name}.log"
  echo "[$(date +%T)] probe $name @ ${ne} env ..."
  OMNI_KIT_ACCEPT_EULA=YES TORCHDYNAMO_DISABLE=1 timeout --signal=KILL 480 "$PY" scripts/train.py \
    --task "$TASK" --num_envs "$ne" --headless --enable_cameras --experience "$KIT" --seed 42 \
    --max_iterations 3 agent.params.config.full_experiment_name=probe_tmp $ov >"$log" 2>&1 &
  local tp=$! peak=0 m
  while kill -0 "$tp" 2>/dev/null; do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1); [ -n "$m" ]&&[ "$m" -gt "$peak" ]2>/dev/null&&peak="$m"; sleep 3; done
  wait "$tp"; local rc=$? v
  if grep -qiE "out of memory|OutOfMemory" "$log"; then v="OOM ($(grep -oiE 'Tried to allocate [0-9.]+ GiB' "$log"|head -1))"
  elif [ "$rc" -ne 0 ]; then v="FAIL rc=$rc"; else v="FIT peak ${peak} MiB"; fi
  printf '%-16s %3s env : %s\n' "$name" "$ne" "$v" | tee -a "$OUT"
  free_gpu; rm -rf logs/rl_games/Forge/probe_tmp 2>/dev/null; }
IMG224='env.image_height=224 env.image_width=224 env.tiled_camera.width=224 env.tiled_camera.height=224'
IMG256='env.image_height=256 env.image_width=256 env.tiled_camera.width=256 env.tiled_camera.height=256'
echo "=== CAPACITY MEM PROBE $(date +%T) ==="
probe lstm2048_224 "$NOAUX $IMG224 agent.params.network.rnn.units=2048" 48
probe fc512_256    "$NOAUX $IMG256 agent.params.network.cnn.fc_size=512" 32
echo "=== DONE $(date +%T) ==="; cat "$OUT"
