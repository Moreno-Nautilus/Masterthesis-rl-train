#!/bin/bash
# Auto-resume training watchdog — works around the stochastic PhysX GPU CUDA-700 crash on the
# vision task (rendering + GPU PhysX on this rc.36 Isaac Sim build; ~1h MTBF). The crash HANGS the
# process holding the GPU, so we watch the log for the crash signature OR a progress stall, hard-kill
# the run + any orphaned GPU proc, then relaunch with --checkpoint from the latest saved checkpoint.
# rl_games restores epoch_num from the checkpoint, so training continues toward --max_iterations.
#
# Usage: scripts/auto_resume_train.sh <experiment_name> <task_id> <num_envs> <max_iter> [seed_ckpt]
set -u
cd /home/moreno/Masterthesis-rl-train

NAME="${1:?experiment name}"
TASK="${2:?task id}"
ENVS="${3:?num_envs}"
MAX="${4:?max_iterations}"
SEED="${5:-}"                       # optional: checkpoint to warm-start the first attempt
EXTRA="${EXTRA_OVERRIDES:-}"        # optional extra hydra overrides, e.g. EXTRA_OVERRIDES="env.blank_image=true"
PY=/home/moreno/miniconda3/envs/isaaclab/bin/python
EXPERIENCE="${EXPERIENCE:-/home/moreno/Masterthesis-rl-train/apps/isaaclab.python.headless.rendering.physx1065.kit}"
NN_DIR="logs/rl_games/Forge/${NAME}/nn"
LOG="/tmp/${NAME}.log"              # current attempt's stdout (overwritten each attempt)
MASTER="/tmp/${NAME}_master.log"    # watchdog audit trail (persists)
STALL=600                           # seconds with no new epoch (while training) => treat as a hang
BOOT_GRACE=1200                     # seconds allowed to reach epoch 1 (Isaac boot + first epoch)
MAX_ATTEMPTS=60

log(){ echo "[$(date '+%F %T')] $*" >> "$MASTER"; }

free_gpu(){
  for gp in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    kill -9 "$gp" 2>/dev/null && log "killed leftover GPU pid $gp"
  done
}

EXPERIENCE_ARGS=()
if [ -n "$EXPERIENCE" ]; then
  EXPERIENCE_ARGS=(--experience "$EXPERIENCE")
fi

# Optional rollout-video recording (gym RecordVideo). VIDEO=1 records a third-person clip every
# VIDEO_INTERVAL steps (VIDEO_LENGTH long) to logs/rl_games/Forge/<name>/videos/train. Complements the
# in-env wrist gallery (env.cam_log_interval): that shows what the camera SEES, this shows BEHAVIOUR.
VIDEO_ARGS=()
if [ "${VIDEO:-0}" = "1" ]; then
  VIDEO_ARGS=(--video --video_interval "${VIDEO_INTERVAL:-20000}" --video_length "${VIDEO_LENGTH:-200}")
fi

# Optional fixed RNG seed (reproducibility for the number you'll quote + a clean 2nd-seed comparison).
# Shared across all resume attempts so a mid-run watchdog restart doesn't re-roll the seed.
SEED_ARGS=()
if [ -n "${SEED_RNG:-}" ]; then
  SEED_ARGS=(--seed "$SEED_RNG")
fi

log "=== auto-resume start: name=$NAME task=$TASK envs=$ENVS max=$MAX seed=${SEED:-none} experience=${EXPERIENCE:-default} ==="
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  # pick the newest checkpoint to resume from (own run first, else the seed on attempt 1)
  CKPT=$(ls -t "$NN_DIR"/last_*.pth "$NN_DIR"/Forge.pth 2>/dev/null | head -1)
  [ -z "$CKPT" ] && CKPT="$SEED"
  RESUME=""; [ -n "$CKPT" ] && [ -f "$CKPT" ] && RESUME="--checkpoint $CKPT"
  log "attempt $attempt/$MAX_ATTEMPTS  resume=${CKPT:-FRESH}"

  # TORCHDYNAMO_DISABLE=1: skip torch.compile. Cold-compiling the CNN+LSTM net takes >10min to reach
  # epoch 1 and is re-paid on every resume (SIGKILL never warms the cache) — it broke the watchdog
  # (killed mid-compile every attempt). Eager mode is a bit slower but starts fast and deterministically.
  OMNI_KIT_ACCEPT_EULA=YES TORCHDYNAMO_DISABLE=1 PYTHONUNBUFFERED=1 "$PY" scripts/train.py --task "$TASK" \
    --num_envs "$ENVS" --headless --enable_cameras "${EXPERIENCE_ARGS[@]}" "${VIDEO_ARGS[@]}" \
    "${SEED_ARGS[@]}" --max_iterations "$MAX" \
    agent.params.config.full_experiment_name="$NAME" $RESUME $EXTRA > "$LOG" 2>&1 &
  TPID=$!
  log "launched train pid $TPID"

  # Progress is tracked by the newest mtime under summaries/ + nn/ (TensorBoard events + checkpoints),
  # which advance every epoch REGARDLESS of stdout block-buffering. (Grepping stdout for "epoch:" is
  # unreliable — Isaac block-buffers it, which earlier made the watchdog kill healthy runs.)
  last_mtime=0; last_change=$(date +%s); started=0
  while kill -0 "$TPID" 2>/dev/null; do
    sleep 20
    if grep -qE "PhysX error|CUDA error|error 700|illegal memory access|Traceback \(most recent|Error executing job" "$LOG"; then
      log "attempt $attempt: CRASH/error signature detected -> kill $TPID"
      kill -9 "$TPID" 2>/dev/null; break
    fi
    newest=$(ls -t "logs/rl_games/Forge/${NAME}/summaries/"* "$NN_DIR/"* 2>/dev/null | head -1)
    m=0; [ -n "$newest" ] && m=$(stat -c %Y "$newest" 2>/dev/null || echo 0)
    now=$(date +%s)
    if [ "$m" -gt "$last_mtime" ]; then last_mtime=$m; last_change=$now; started=1; fi
    # generous window to first progress (boot); tight stall once training is underway
    grace=$STALL; [ "$started" -eq 0 ] && grace=$BOOT_GRACE
    if [ $((now - last_change)) -gt "$grace" ]; then
      log "attempt $attempt: STALL (>${grace}s no progress, started=$started) -> kill $TPID"
      kill -9 "$TPID" 2>/dev/null; break
    fi
  done
  wait "$TPID" 2>/dev/null
  free_gpu
  sleep 10

  cur_ep=$(grep -oE "epoch: [0-9]+/" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+"); cur_ep=${cur_ep:-?}
  if grep -q "MAX EPOCHS NUM" "$LOG"; then log "DONE: reached MAX EPOCHS (attempt $attempt)"; break; fi
  log "attempt $attempt ended around epoch $cur_ep; will resume"
done
log "=== auto-resume loop finished ==="
