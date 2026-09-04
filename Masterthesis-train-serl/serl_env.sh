# shellcheck shell=bash
# Runtime shell for the SERL server/backend on the REAL robot: ONE interpreter that imports both
# ROS2 (rclpy + the built LBR messages) AND the SERL/JAX stack.
#
# WHY NOT `conda activate serl`: ROS2 Humble's rclpy is compiled against SYSTEM python3.10 and only
# works under that interpreter. Activating the conda env gives you a different python that can't
# import rclpy → LbrIiwaBackend (which needs rclpy + lbr_fri_idl) fails. So, exactly like the deploy
# (deploy_env.sh), we run everything under SYSTEM python3 and make the SERL packages visible by
# prepending the `serl` conda env's site-packages + the editable source dirs to PYTHONPATH.
# System py3.10 and conda py3.10 are ABI-compatible (both cp310), so jax/flax/numpy load fine.
#
# Usage:
#   source serl_env.sh
#   python3 -m iiwa_serl.robot_servers.iiwa_server \
#       --backend="rl_deploy_inference.serl_backend:LbrIiwaBackend"
#
# Then in the SAME kind of shell you can also run record_demo.py / run_learner.sh / run_actor.sh.
# (The learner/actor don't need ROS2, but running them here too keeps one consistent environment.)

_ros_distro="${ROS_DISTRO:-humble}"
_repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_serl_site="${SERL_PY_SITE:-/home/moreno/miniconda3/envs/serl/lib/python3.10/site-packages}"
_deploy_dir="${RL_DEPLOY_DIR:-/home/moreno/Masterthesis-rl-deploy}"

# 1) ROS2 base + the deploy colcon workspace (built messages: lbr_fri_idl, fp_debug_msgs; and the
#    rl_deploy_inference package that provides LbrIiwaBackend).
# shellcheck disable=SC1090
source "/opt/ros/${_ros_distro}/setup.bash"
if [ -f "${_deploy_dir}/install/setup.bash" ]; then
  # shellcheck disable=SC1091
  source "${_deploy_dir}/install/setup.bash"
else
  echo "[serl_env] WARNING: ${_deploy_dir}/install/setup.bash not found — build the deploy ws first." >&2
fi

# 2) SERL/JAX stack: prefer the vendored HIL-SERL Gymnasium launcher. Keep the legacy launcher
#    later on PYTHONPATH for old scripts, but never let its classic-Gym ChunkingWrapper wrap the
#    Gymnasium iiwa environment.
if [ -d "${_serl_site}" ]; then
  # python_compat/sitecustomize.py moves the Conda site directory behind the system
  # stdlib at interpreter startup, preventing its obsolete typing.py from shadowing
  # Python 3.10 while leaving Conda's JAX/Gymnasium packages available.
  export PYTHONPATH="${_repo_dir}/python_compat:${_repo_dir}/iiwa_serl:${_repo_dir}/hil-serl_src/serl_launcher:${_repo_dir}/serl_launcher:${_serl_site}:${PYTHONPATH}"
else
  echo "[serl_env] WARNING: serl site-packages not found: ${_serl_site} (set SERL_PY_SITE)." >&2
fi

# 3) Keep SERL on CPU while a PPO train/deploy is using the GPU. Comment out for a GPU SERL run.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
# Older JAX probes the CUDA toolkit path even when the selected platform is CPU;
# an explicit valid root avoids its broken namespace-package auto-detection.
export CUDA_ROOT="${CUDA_ROOT:-/usr}"

echo "[serl_env] ROS=${_ros_distro} | serl_site=${_serl_site} | JAX_PLATFORMS=${JAX_PLATFORMS}"
echo "[serl_env] verify: python3 -c 'import rclpy, lbr_fri_idl, jax, iiwa_serl; print(\"OK\")'"
