# Insertion Policy — Vision-Based Residual RL

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.2-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)

Master-thesis RL environment for the **last ~1–5 cm of a part insertion** on a **KUKA LBR iiwa**.
The policy is a **pure residual corrector**: the upstream pipeline (pose estimation → grasping →
path planning) hands over a *pre-insert pose* and a *goal (seated) pose*, and the policy only
corrects the accumulated upstream deviation (~3–7 mm laterally and up to **~20–25° in
orientation**). Because the angular error is large, the policy keeps **6-DOF** corrections
(position + orientation) — orientation is *not* dropped from the action. The thesis contribution
is adding a **wrist RGB-D camera** for closed-loop visual feedback on top of the otherwise blind,
state-based insertion policy (Fabrica).

Built as an Isaac Lab extension on top of the **Forge** assembly env (Factory + force sensing +
pose-uncertainty), trained with **PPO (RL-Games)**.

## Status

- ✅ Part meshes (cooling + pb assemblies, 7 parts) converted to USD (meters, SDF collision).
- ✅ **State-obs task** `Isaac-Insertion-CoolingPeg-Direct-v0` (`InsertionEnv`): socket-aware
  multi-socket targeting, realistic pre-insert reset (1–5 cm clearance), pre-insert tilt injection
  (up to ~25°), random grasp misalignment, orientation-aware **multi-keypoint squashing-kernel
  reward** + alignment success (<10°). Trained with PPO/RL-Games → **48% overall** success on the
  tilt/lateral sweep.
- ✅ **Vision task** `Isaac-Insertion-CoolingPeg-Vision-Direct-v0`: wrist **RGB-D** TiledCamera +
  hybrid observation (proprio + image) fed to a custom **CNN + LSTM** rl_games network
  (`insertion_hybrid`), trained end-to-end. Trains stably and the reward improves.
- 🔄 Now: full vision training run + eval sweep vs the state baseline; then path-centric obs,
  domain randomization, and the Franka → iiwa 7 + parallel-gripper swap.

> **Parts note:** the cooling parts are chamfer-free 3D-printed test parts (12 mm shaft into a 14 mm
> socket, 1 mm clearance). Precise lateral alignment — not chamfer-funneling — is the binding
> constraint, which is exactly what the wrist camera is meant to help with. See
> `scripts/check_recoverability.py`.

## Layout

```
source/insertion_policy/insertion_policy/
  tasks/insertion/          # InsertionEnv + cooling task/asset configs (our env)
    agents/                 # rl_games configs + custom insertion_hybrid CNN+LSTM network
  assets/                   # converted USDs (+ source OBJ meshes under meshes/)
scripts/                    # asset conversion, smoke, train, eval, plotting (see below)
```

## Install

Requires [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
(2.3.2 / Isaac Sim 4.5) in a conda env (here: `isaaclab`).

```bash
conda activate isaaclab
python -m pip install -e source/insertion_policy
```

## Running (headless)

All scripts launch Isaac Sim headless. First launch needs the EULA accepted, and Isaac Sim
captures stdout — so the scripts write their results to `/tmp/*_report.txt`.

```bash
conda activate isaaclab
OMNI_KIT_ACCEPT_EULA=YES python scripts/<script>.py
```

| Script | What it does |
|--------|--------------|
| `convert_assets.py`       | OBJ (cm) → USD (m) with SDF collision; `--sdf-resolution/--density/--parts` |
| `add_articulation_root.py`| Bake `ArticulationRootAPI` into part USDs (needed by Forge's Articulation wrapper) |
| `verify_assets.py`        | Check converted USDs: extent (m), SDF, rigid, mass |
| `analyze_sockets.py`      | Locate receptacle sockets (`--base cooling_base --peg cooling_screw`) |
| `check_recoverability.py` | Geometric chamfer/clearance probe: is the error distribution physically seatable? |
| `test_part_physics.py`    | Drop-test: confirm the screw seats in a socket (SDF collision sane) |
| `smoke_env.py`            | Instantiate the env, reset, step; prints obs/reward + reset geometry (dumps wrist frames for the vision task) |
| `train.py`                | Train with RL-Games (registers our tasks + the `insertion_hybrid` net) |
| `eval_policy.py`          | Load a checkpoint, run N episodes, bin success by reset tilt / lateral offset |
| `plot_training.py`        | TensorBoard events → `progress.png` (reward/success curves; CPU-only, safe during training) |
| `list_envs.py`            | List registered Isaac Lab task IDs |

## Training & evaluation

```bash
conda activate isaaclab

# State-obs policy
OMNI_KIT_ACCEPT_EULA=YES python scripts/train.py \
  --task Isaac-Insertion-CoolingPeg-Direct-v0 --num_envs 128 --headless \
  --max_iterations 1500 agent.params.config.full_experiment_name=my_state_run

# Vision policy (wrist RGB-D + CNN). NOTE: requires --enable_cameras.
OMNI_KIT_ACCEPT_EULA=YES python scripts/train.py \
  --task Isaac-Insertion-CoolingPeg-Vision-Direct-v0 --num_envs 128 --headless --enable_cameras \
  --max_iterations 1500 agent.params.config.full_experiment_name=my_vision_run

# Evaluate (add --enable_cameras for the vision task)
OMNI_KIT_ACCEPT_EULA=YES python scripts/eval_policy.py \
  --task Isaac-Insertion-CoolingPeg-Direct-v0 --num_envs 128 --num_episodes 512 --headless \
  --checkpoint logs/rl_games/Forge/<run>/nn/Forge.pth
```

Logs/checkpoints land in `logs/rl_games/Forge/<run>/` (`nn/Forge.pth` = best by mean reward).

> **Vision/GPU note:** the vision task keeps PhysX on Fabric (GPU-stable) but routes only the
> camera's pose view to the USD path, working around a missing `usdrt.hierarchy` in this Isaac Sim
> build. Disabling Fabric globally instead causes PhysX CUDA crashes/hangs — don't.

## Code formatting

```bash
pip install pre-commit
pre-commit run --all-files
```

---

_README last updated: 2026-06-22._
