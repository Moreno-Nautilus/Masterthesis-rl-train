#!/bin/bash
# Generic robust run-chain: trains + evals a LIST of runs back-to-back, unattended-safe (built for the
# 3-week holiday chain; also used to fill the Wed-Fri screen slots). Each train runs under
# auto_resume_train.sh (PhysX-crash watchdog, resumable to max_iters); each eval under eval_robust.sh
# (Isaac boot/shutdown hang timeout). Auto-advances; appends a persistent summary. Per-run markers
# record the iters trained, so the chain is:
#   * restart-safe  -- re-run after a box reboot and it skips/《resumes》 what's already done, and
#   * extend-aware  -- a later list asking for MORE iters (e.g. 224px@32 1200 -> 3000 on holiday)
#                      re-enters that run and auto_resume RESUMES it from its last checkpoint.
#
# Usage:  setsid bash scripts/run_chain.sh <runlist_file> </dev/null >/tmp/<chain>.log 2>&1 &
# Runlist line ('|'-separated, '#' comments + blank lines ok; last field may contain spaces):
#   name | num_envs | max_iters | eval_envs | eval_seeds | extra_overrides
set -u
cd /home/moreno/Masterthesis-rl-train
RUNLIST="${1:?runlist file}"
TASK="${TASK:-Isaac-Insertion-CoolingPeg-Vision-Direct-v0}"
SEED_RNG="${SEED_RNG:-42}"
DRYRUN="${DRYRUN:-0}"                                         # 1 = echo the plan (no GPU) to validate parsing/flow
STATE_DIR="${STATE_DIR:-logs/chain}"; mkdir -p "$STATE_DIR"   # persistent (survives /tmp wipe on reboot)
SUMMARY="${SUMMARY:-$STATE_DIR/summary.txt}"; touch "$SUMMARY"
ts(){ date '+%F %T'; }
log(){ echo "[$(ts)] $*"; }

free_gpu(){ for gp in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$gp" 2>/dev/null; done
  for s in /dev/shm/carb-*; do [ -e "$s" ] || continue; p=$(echo "$s"|grep -oE '[0-9]+$'); kill -0 "$p" 2>/dev/null || rm -f "$s"; done; sleep 8; }
latest_ckpt(){ ls -t "logs/rl_games/Forge/$1/nn/"last_Forge_ep_[0-9]*_rew_[0-9]*.pth 2>/dev/null | head -1; }

log "=== CHAIN START (list=$RUNLIST) ==="
while IFS='|' read -r name envs iters evalenvs seeds ov; do
  name="$(echo "${name:-}" | xargs)"; [ -z "$name" ] && continue
  case "$name" in \#*) continue;; esac
  envs="$(echo "$envs"|xargs)"; iters="$(echo "$iters"|xargs)"; evalenvs="$(echo "$evalenvs"|xargs)"
  seeds="$(echo "$seeds"|xargs)"; ov="$(echo "$ov"|sed 's/^ *//;s/ *$//')"
  marker="$STATE_DIR/${name}.iters"

  if [ -f "$marker" ]; then
    prev="$(cat "$marker" 2>/dev/null || echo 0)"
    if [ "${prev:-0}" -ge "$iters" ] 2>/dev/null; then log ">>> SKIP $name (already @${prev} >= ${iters})"; continue; fi
    log ">>> EXTEND $name (${prev} -> ${iters} it)"
  fi

  if [ "$DRYRUN" = "1" ]; then
    log "[DRY] TRAIN: SEED_RNG=$SEED_RNG bash scripts/auto_resume_train.sh $name $TASK $envs $iters  EXTRA_OVERRIDES='$ov'"
    log "[DRY] EVAL : NUM_ENVS=$evalenvs bash scripts/eval_robust.sh $TASK <latest_ckpt $name> $seeds chain_${name}  EXTRA_OVERRIDES='$ov'"
    echo "$iters" > "$marker"; log "[DRY] marker $name -> $iters"; continue
  fi

  log ">>> TRAIN $name : ${envs} env -> ${iters} it  [ov: ${ov:0:60}...]"
  free_gpu
  SEED_RNG="$SEED_RNG" EXTRA_OVERRIDES="$ov" bash scripts/auto_resume_train.sh "$name" "$TASK" "$envs" "$iters"

  free_gpu
  ck="$(latest_ckpt "$name")"
  if [ -z "$ck" ]; then log "!!! $name: NO checkpoint -> skip eval"; echo "$(ts)  ${name}: NO CKPT (train failed?)" >> "$SUMMARY"; continue; fi
  log ">>> EVAL  $name : $(basename "$ck")  (${evalenvs} env, seeds ${seeds})"
  EXTRA_OVERRIDES="$ov" NUM_ENVS="$evalenvs" bash scripts/eval_robust.sh "$TASK" "$ck" "$seeds" "chain_${name}"
  res="$(grep -h 'success\|HUNG\|FAILED' "/tmp/chain_${name}_summary.txt" 2>/dev/null | tr '\n' ' ')"
  echo "$(ts)  ${name} (${envs}env/${iters}it): ${res:-no report}" >> "$SUMMARY"
  echo "$iters" > "$marker"
  log "<<< DONE $name : ${res}"
done < "$RUNLIST"

log "=== CHAIN DONE ==="; echo "----- SUMMARY ($SUMMARY) -----"; cat "$SUMMARY"
