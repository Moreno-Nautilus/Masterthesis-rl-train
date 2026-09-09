# shellcheck shell=bash
# Runtime shell for the FRANKA (FR3) HIL-SERL pivot. ONE interpreter that imports both
# ROS2 (rclpy, for Ros2FrankaClient -> cartesian_impedance_control) AND the SERL/JAX
# stack (jax/flax, serl_launcher, franka_env).
#
# WHY NOT `conda activate serl`: ROS2 Humble's rclpy is compiled against SYSTEM python3.10
# and only works under that interpreter. So we run SYSTEM python3.10 and make the SERL
# packages visible by prepending the `serl` conda env's site-packages + the vendored
# HIL-SERL source dirs to PYTHONPATH. System py3.10 and conda py3.10 are ABI-compatible
# (both cp310), so jax/flax/numpy load fine. (Same pattern as serl_env.sh, but Franka:
# it uses hil-serl_src/serl_robot_infra (gymnasium HIL tree) and needs NO LBR deploy ws.)
#
# Usage (actor, on the robot PC — needs the controller running + PS4 + ZED):
#   source franka_env.sh
#   export FRANKA_EXP_NAME=franka_plumbers_insert0
#   export TELEOP_DEVICE=ps4
#   bash hil-serl_src/examples/experiments/franka_plumbers/run_actor.sh
#
# Usage (learner, GPU box — no robot; comment out JAX_PLATFORMS=cpu below for GPU):
#   source franka_env.sh
#   export FRANKA_EXP_NAME=franka_plumbers_insert0
#   bash hil-serl_src/examples/experiments/franka_plumbers/run_learner.sh --demo_path <demos.pkl>
#
# Offline dry-run (home; no ROS2/robot needed for the LEARNER path):
#   source franka_env.sh
#   python3 hil-serl_src/examples/experiments/franka_plumbers/dryrun_learner.py

_ros_distro="${ROS_DISTRO:-humble}"
_repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_serl_site="${SERL_PY_SITE:-/home/moreno/miniconda3/envs/serl/lib/python3.10/site-packages}"

# 1) ROS2 base (rclpy + std_msgs +, at the rig, franka_msgs / messages_fr3 from the
#    franka_ros2_ws overlay). Source the franka_ros2_ws overlay too if it is built.
# shellcheck disable=SC1090
if [ -f "/opt/ros/${_ros_distro}/setup.bash" ]; then
  source "/opt/ros/${_ros_distro}/setup.bash"
else
  echo "[franka_env] NOTE: /opt/ros/${_ros_distro}/setup.bash not found — ROS2 not sourced "\
       "(fine for the offline learner dry-run; required for the actor/robot)." >&2
fi
# The Franka controller workspace overlay (built at the rig per FRANKA_RIG_CHECKLIST.md).
# NOTE: on the home box ~/franka_ros2_ws is the OLD LBR/ZED workspace (misnomer) with NO
# Franka packages — sourcing it is harmless but does NOT provide the FR3 controller. On the
# robot PC, build franka_ros2 v0.1.15 + cartesian_impedance_control here (set FRANKA_ROS2_WS
# if you use a different path).
# Default to ~/fr3_ws — that is where franka_ros2 v2.7.1 + cartesian_impedance_control were
# actually built on 2026-09-09. ~/franka_ros2_ws is the OLD LBR/ZED workspace and has NO
# franka packages: defaulting to it made the actor die with
#   ModuleNotFoundError: No module named 'franka_msgs'
# whenever a shell had not separately sourced fr3_ws. Fall back to the old path only if
# fr3_ws is missing, so nothing else that relied on it breaks.
if [ -z "${FRANKA_ROS2_WS:-}" ] && [ -f "${HOME}/fr3_ws/install/setup.bash" ]; then
  _franka_ws="${HOME}/fr3_ws"
else
  _franka_ws="${FRANKA_ROS2_WS:-${HOME}/franka_ros2_ws}"
fi
if [ -f "${_franka_ws}/install/setup.bash" ]; then
  # shellcheck disable=SC1091
  source "${_franka_ws}/install/setup.bash"
fi

# 2) SERL/JAX stack: the vendored HIL-SERL (Gymnasium) launcher + robot infra + examples,
#    plus the python_compat shim (moves Conda's obsolete typing.py behind the stdlib).
#    ALWAYS put the vendored source dirs on PYTHONPATH (they must be importable even if the
#    conda site-packages path is wrong); append the site-packages only if it exists.
export PYTHONPATH="${_repo_dir}/python_compat:${_repo_dir}/hil-serl_src/serl_launcher:${_repo_dir}/hil-serl_src/serl_robot_infra:${_repo_dir}/hil-serl_src/examples:${PYTHONPATH}"
if [ -d "${_serl_site}" ]; then
  export PYTHONPATH="${PYTHONPATH}:${_serl_site}"
else
  echo "[franka_env] WARNING: serl site-packages not found: ${_serl_site} (set SERL_PY_SITE) "\
       "— jax/flax/etc will be missing; source dirs are still on PYTHONPATH." >&2
fi

# 3) Keep SERL on CPU by DEFAULT (safe at home / while another GPU job runs).
#    FOR A GPU LEARNER RUN: `export FRANKA_USE_GPU=1` before sourcing (preferred), or set
#    JAX_PLATFORMS to a real value (e.g. `export JAX_PLATFORMS=cuda`).
#    NOTE: `export JAX_PLATFORMS=` (empty) does NOT work with ${VAR:-default} — an empty
#    value is treated as unset and would silently fall back to cpu, costing you a whole
#    training run on CPU. We therefore use ${VAR+set} semantics and an explicit GPU flag.
if [ "${FRANKA_USE_GPU:-0}" != "0" ]; then
  unset JAX_PLATFORMS          # let JAX pick the GPU
  echo "[franka_env] FRANKA_USE_GPU=1 -> JAX_PLATFORMS unset (GPU)"
elif [ -n "${JAX_PLATFORMS+set}" ]; then
  export JAX_PLATFORMS="${JAX_PLATFORMS}"   # honor an explicitly set value, empty included
else
  export JAX_PLATFORMS="cpu"
fi
export CUDA_ROOT="${CUDA_ROOT:-/usr}"

# 4) PIN the interpreter. rclpy is built against SYSTEM python3.10; if an active Conda env
#    puts a different python3 first on PATH we would silently recreate the ABI/import
#    problem this shell exists to avoid. Export SERL_PYTHON and VALIDATE it; the launchers
#    use "$SERL_PYTHON", not a bare `python3`.
export SERL_PYTHON="${SERL_PYTHON:-/usr/bin/python3.10}"
if [ ! -x "${SERL_PYTHON}" ]; then
  # fall back to /usr/bin/python3 if 3.10 isn't a separate binary on this box
  if [ -x /usr/bin/python3 ]; then export SERL_PYTHON=/usr/bin/python3; fi
fi
if [ ! -x "${SERL_PYTHON}" ]; then
  echo "[franka_env] ERROR: no system python found (set SERL_PYTHON explicitly)." >&2
else
  _pyver="$("${SERL_PYTHON}" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)"
  if [ "${_pyver}" != "3.10" ]; then
    echo "[franka_env] WARNING: ${SERL_PYTHON} is Python ${_pyver}, expected 3.10 — the "\
         "conda site-packages (cp310) and rclpy may not import. Set SERL_PYTHON." >&2
  fi
  if [ -n "${CONDA_PREFIX:-}" ]; then
    echo "[franka_env] NOTE: a Conda env is active (${CONDA_PREFIX}). We use ${SERL_PYTHON} "\
         "explicitly, so this is OK — but do NOT run bare \`python\`/\`python3\`." >&2
  fi
fi

echo "[franka_env] python=${SERL_PYTHON} (${_pyver:-?})"
echo "[franka_env] ROS=${_ros_distro} | franka_ws=${_franka_ws} | serl_site=${_serl_site} | JAX_PLATFORMS=${JAX_PLATFORMS:-<unset:GPU>}"
echo "[franka_env] verify (learner): python3 -c 'import jax, franka_env, serl_launcher; print(\"OK\")'"
echo "[franka_env] verify (actor):   python3 -c 'import rclpy; print(\"rclpy OK\")'"
