#!/usr/bin/env bash
set -euo pipefail

python async_drq_iiwa_insert.py "$@" \
  --learner \
  --env="${ENV_NAME:-IiwaInsertReal-Vision-v0}" \
  --checkpoint_path="${CHECKPOINT_PATH:-./checkpoints}"
