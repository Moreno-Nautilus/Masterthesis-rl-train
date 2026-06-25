#!/bin/bash
# Hardened-env chain (single GPU): STATE baseline -> VISION fusion -> BLANK-image control, on the
# transfer-valid env (rigid wrist cam, grasp tilt about the pressing axis, 10°/±8mm/25°, cam+proprio
# DR — see DECISIONS.md §12). Each train runs through the auto-resume watchdog (recovers PhysX crashes)
# and stops at its --max_iterations; an EVAL then runs automatically on the resulting checkpoint, so
# the overnight run produces the state-vs-vision(-vs-blank) verdict with no human in the loop.
# Verdict logic: vision>blank≈state ⇒ the image helps; vision≈blank>state ⇒ the net (not the image)
# helps; all≈ ⇒ proprio suffices.
#
# FRESH names (state_hardened_1 / vision_hardened_1) so the watchdog does NOT resume the old look-at
# checkpoints (state_fixed_1 / vision_fuse_1). Override iters/episodes via env: ITERS=1000 EPISODES=512.
#
#   setsid bash scripts/overnight_chain.sh </dev/null >overnight_chain.log 2>&1 &
set -u
cd /home/moreno/Masterthesis-rl-train
ITERS="${ITERS:-1000}"
BLANK_ITERS="${BLANK_ITERS:-300}"
EPISODES="${EPISODES:-512}"
KIT=/home/moreno/Masterthesis-rl-train/apps/isaaclab.python.headless.rendering.physx1065.kit
PY=/home/moreno/miniconda3/envs/isaaclab/bin/python
ts(){ date '+%F %T'; }

free_gpu(){ for gp in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$gp" 2>/dev/null; done; sleep 10; }

# eval_run <name> <task> <num_envs> [extra eval flags...] -> writes /tmp/<name>_eval.txt + a report next to the ckpt.
eval_run(){
  local name="$1" task="$2" envs="$3"; shift 3
  local ckpt="logs/rl_games/Forge/${name}/nn/Forge.pth"
  if [ ! -f "$ckpt" ]; then echo "[$(ts)] WARN: no checkpoint $ckpt — skipping eval"; return; fi
  echo "[$(ts)] eval $name ($EPISODES eps, $envs envs)"
  OMNI_KIT_ACCEPT_EULA=YES "$PY" scripts/eval_policy.py --task "$task" --num_envs "$envs" \
    --num_episodes "$EPISODES" --headless --experience "$KIT" --checkpoint "$ckpt" "$@" \
    --report "/tmp/${name}_eval.txt"
  free_gpu
}

echo "[$(ts)] === HARDENED CHAIN START (iters=$ITERS, eval=$EPISODES eps) ==="

echo "[$(ts)] >>> 1/2 state_hardened_1  (state baseline, 128 envs)"
bash scripts/auto_resume_train.sh state_hardened_1 Isaac-Insertion-CoolingPeg-Direct-v0 128 "$ITERS"
sleep 15
eval_run state_hardened_1 Isaac-Insertion-CoolingPeg-Direct-v0 128
echo "[$(ts)] <<< state_hardened_1 trained + evaluated"

echo "[$(ts)] >>> 2/3 vision_hardened_1 (vision fusion, 64 envs train / 128 eval)"
bash scripts/auto_resume_train.sh vision_hardened_1 Isaac-Insertion-CoolingPeg-Vision-Direct-v0 64 "$ITERS"
sleep 15
eval_run vision_hardened_1 Isaac-Insertion-CoolingPeg-Vision-Direct-v0 128 --enable_cameras
echo "[$(ts)] <<< vision_hardened_1 trained + evaluated"

# Blank-image CONTROL: identical hybrid net (CNN+FC+LSTM), but the image is ZEROED in-env. vision vs
# blank isolates the IMAGE CONTENT (same architecture); blank ≈ vision ⇒ the policy ignored the camera
# (and is the signal to add an auxiliary grasp-pose head later). Shorter run to fill out the night.
echo "[$(ts)] >>> 3/3 blank_hardened_1 (vision arch, image ZEROED — control; ${BLANK_ITERS} it)"
EXTRA_OVERRIDES="env.blank_image=true" bash scripts/auto_resume_train.sh blank_hardened_1 Isaac-Insertion-CoolingPeg-Vision-Direct-v0 64 "$BLANK_ITERS"
sleep 15
eval_run blank_hardened_1 Isaac-Insertion-CoolingPeg-Vision-Direct-v0 128 --enable_cameras env.blank_image=true
echo "[$(ts)] <<< blank_hardened_1 trained + evaluated"

echo "[$(ts)] === HARDENED CHAIN DONE — verdicts in /tmp/{state_hardened_1,vision_hardened_1,blank_hardened_1}_eval.txt ==="
