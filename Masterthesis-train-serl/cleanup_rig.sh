#!/usr/bin/env bash
# Kill anything still holding the ZED or driving the robot, then report.
#
# WHY: a crashed run (Qt abort, Ctrl+C with a live PS4 reader subprocess) leaves orphans
# alive. They keep /dev/video0 open, so the next run fails with
#     RuntimeError: ZED open failed: CAMERA STREAM FAILED TO START
# and they keep publishing stale targets, so the arm appears unresponsive. Neither is a
# hardware fault. `pkill -f <name>` does NOT reliably match them because the offenders are
# inline `python3 -c "..."` commands, so we kill by WHO HOLDS THE DEVICE instead.
#
#   bash cleanup_rig.sh
set -u

echo "== processes holding the ZED =="
holders="$(fuser /dev/video0 /dev/video1 2>/dev/null | tr -s ' ')"
if [ -n "${holders// /}" ]; then
  echo "  killing:$holders"
  # shellcheck disable=SC2086
  kill -9 $holders 2>/dev/null
  sleep 1
else
  echo "  none"
fi

echo "== rig scripts =="
for pat in jog_and_capture.py jog_log.py record_demos.py gripper.py preflight.py; do
  pids="$(pgrep -f "$pat" 2>/dev/null | tr '\n' ' ')"
  if [ -n "${pids// /}" ]; then
    echo "  killing $pat:$pids"
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null
  fi
done

# inline `python3 -c` runs that mention the experiment (the smoke one-liners)
pids="$(pgrep -f "franka_plumbers_insert" 2>/dev/null | tr '\n' ' ')"
if [ -n "${pids// /}" ]; then
  echo "  killing inline smoke:$pids"
  # shellcheck disable=SC2086
  kill -9 $pids 2>/dev/null
fi

sleep 1
echo
echo "== state =="
if fuser /dev/video0 >/dev/null 2>&1; then
  echo "  /dev/video0 STILL HELD -> $(fuser -v /dev/video0 2>&1 | tail -1)"
else
  echo "  /dev/video0 free"
fi
lsusb 2>/dev/null | grep -i "stereolabs" | sed 's/^/  /' || echo "  ZED NOT ON USB (replug)"
echo "  NOTE: the controller in terminal 1 is untouched by this script."
