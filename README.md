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
is adding a **wrist RGB camera** for closed-loop visual feedback on top of the otherwise blind,
state-based insertion policy (Fabrica).

Built as an Isaac Lab extension on top of the **Forge** assembly env (Factory + force sensing +
pose-uncertainty), trained with **PPO (RL-Games)**.

## Status

- ✅ Part meshes (cooling + pb assemblies, 7 parts) converted to USD (meters, SDF collision).
- ✅ Forge fork running: env `Isaac-Insertion-CoolingPeg-Direct-v0` (`InsertionEnv`) — socket-aware
  targeting, multi-socket reset, negative-L2 residual reward. Smoke test passes (Franka still in
  place; iiwa swap pending).
- 🔄 Next: tune/verify reset geometry (pre-insert up to ~5 cm above socket, up to ~20–25°
  misalignment); move to an **orientation-aware reward** (multi-keypoint, not single-point neg-L2);
  swap Franka → iiwa 7 + parallel gripper; then train the state-only residual baseline.

## Layout

```
source/insertion_policy/insertion_policy/
  tasks/insertion/        # InsertionEnv + cooling task/asset configs (our env)
  assets/                 # converted USDs (+ source OBJ meshes under meshes/)
scripts/                  # asset conversion + verification + smoke tests (see below)
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
| `test_part_physics.py`    | Drop-test: confirm the screw seats in a socket (SDF collision sane) |
| `smoke_env.py`            | Instantiate the env, reset, step; prints obs/reward + reset geometry |
| `list_envs.py`            | List registered Isaac Lab task IDs |

## Code formatting

```bash
pip install pre-commit
pre-commit run --all-files
```

---

_README last updated: 2026-06-11._
