# Insertion Policy — Vision-Based Residual RL

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.2-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)

Master-thesis RL environment for the **last ~1–5 cm of a part insertion** on a **KUKA LBR iiwa**.
The policy is a **pure residual corrector**: the upstream pipeline (pose estimation → grasping →
path planning) hands over a *pre-insert pose* and a *goal (seated) pose*, and the policy only
corrects the accumulated upstream deviation (~3–7 mm laterally and up to **~20–25° in
orientation**). Because the angular error is large, the policy keeps **6-DOF** corrections
(position + orientation) — orientation is *not* dropped from the action.

Built as an Isaac Lab extension on top of the **Forge** assembly env (Factory + force sensing +
pose-uncertainty), trained with **PPO (RL-Games)**.

## Status

- ✅ Part meshes (cooling + pb assemblies, 7 parts) converted to USD (meters, SDF collision).
- ✅ **State-obs task** `Isaac-Insertion-CoolingPeg-Direct-v0` (`InsertionEnv`): socket-aware
  multi-socket targeting, realistic pre-insert reset (1–5 cm clearance), pre-insert tilt injection
  (up to ~25°), random grasp misalignment, orientation-aware **multi-keypoint squashing-kernel
  reward** + alignment success (<10°).
- ✅ **Vision task** `Isaac-Insertion-CoolingPeg-Vision-Direct-v0`: wrist **RGB-D** TiledCamera +
  hybrid observation (proprio + image) fed to a custom **CNN + LSTM** rl_games network
  (`insertion_hybrid`), with the CNN features projected to 128 dims before fusing with proprio so
  proprio isn't drowned. Trained end-to-end.
- ✅ **Corrected-env rebaseline (2026-06-25):** two sim bugs found & fixed — the screw was not
  actually gripped (Factory's peg-grasp offset mis-placed our shoulder-origin two-diameter screw, so
  it dangled/slipped), and the 50 g base slid under contact. Fixes: grip the screw **head**; make the
  base **heavy + high-friction** (effectively glued). **Both policies then jumped from ~35% to ~95%**
  (state **94.5%**, vision **95.5%** over 512 eps) → the grasp/base bugs, not task difficulty, were
  the real bottleneck. **All earlier success numbers are superseded.**
- ✅ **Hardened test env (2026-06-25):** the 94.5/95.5 tie above was at 5° grasp + a non-physical
  look-at camera + no DR, so it's the *easy* baseline. Rebuilt the env to be transfer-valid and
  realistic: **rigid** wrist camera bolted to the gripper (fixed pose/roll, no privileged look-at; ~45°
  azimuth so the grasp tilt isn't foreshortened), grasp misalignment raised to **10°** and corrected to
  tip about the **finger pressing axis**, lateral **±8 mm**, plus camera-pose + moderate proprio/force
  **domain randomization** (a fair comparison needs noisy proprio too). See `DECISIONS.md` §12.
- ✅ **Hardened-env verdict (2026-06-26): vision wins.** 512-ep eval @1000 it: **state 66.2% vs vision
  75.6% (+9.4 pp)**, vision flat-robust across tilt/lateral. A matched-300-it control isolates the **image**
  as the cause (`vision@300 50.2%` vs image-zeroed `blank@300 33.8%`, same net + budget). At realistic grasp
  error (10° about the pressing axis) + rigid cam + DR the camera earns its keep → **vision is the
  deliverable**. (Absolutes are below the old 95% because the env is much harder now — by design.)
- ✅ **Appearance DR + final transfer number (2026-06-29):** added the sim-to-real appearance DR — **per-env
  matte materials** (each env its own colour; screw≈base *within* an env so the policy can't lean on a
  colour-contrast cue absent on the same-filament real parts), **dome + directional key light**, **per-env
  photometric aug**, and **D405 sensor noise** (RGB + depth gaussian + dropout). Trained `vision_appearance_1`
  (3000 it, seed 42, crash-free). 512-ep eval, DR-on: **vision 72.7%** (best ckpt ep3000) — **still beats
  state 66.2% (+6.5 pp)** and the flattest across grasp tilt yet (67–77% over every bin incl. >25°). The −3 pp
  vs the old 75.6% is the robustness tax (more DR = harder, by design). Eval kept climbing through ep3000
  (64.6→66.4→72.7) so the full run was justified. See `DECISIONS.md` §14, renders in `renders/appearance_dr/`.
  Best deliverable: `vision_appearance_1/nn/last_Forge_ep_3000` (NB `Forge.pth` = best-by-*reward* ep2700, 70.7%).
- 🔄 Next (see `PLANNING.md`): **push %** — auxiliary grasp-pose head, fc_size sweep, frame-stack, force
  fusion — interleaved with **generalization** (pb_parts); then the iiwa 7 + custom-gripper swap + sim-to-real.

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
2.3.2 with Isaac Sim 5.1 in a conda env (here: `isaaclab51`).

```bash
conda activate isaaclab51
python -m pip install -e source/insertion_policy
```

## Running (headless)

All scripts launch Isaac Sim headless. First launch needs the EULA accepted, and Isaac Sim
captures stdout — so the scripts write their results to `/tmp/*_report.txt`.

```bash
conda activate isaaclab51
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
| `auto_resume_train.sh`    | Watchdog wrapper for `train.py`: uses the stock Sim 5.1 experience and auto-resumes from the latest checkpoint after crashes/stalls until `--max_iterations`. Use this for any unattended run. |
| `overnight_chain.sh`      | Runs two `auto_resume_train.sh` jobs back-to-back on one GPU (e.g. state baseline → vision), freeing the GPU between them. `setsid bash scripts/overnight_chain.sh >log 2>&1 &` |
| `eval_policy.py`          | Load a checkpoint, run N episodes, bin success by reset tilt / lateral offset |
| `render_wrist_rollout.py` | Render a checkpoint rollout as video — `--view wrist` (the RGB-D POV) or `--view scene` (close third-person), following one env |
| `viz_camera.py`           | Dump the wrist-cam POV + third-person shot at reset (zero-action; camera/grasp geometry checks). Prints fingertip-local aim, realized grasp tilt, and the finger pressing axis. Overrides: `--cam_offset --cam_aim --focal --cam_jitter --cam_roll --grasp_deg` (compare mounts / exaggerate the tilt) |
| `plot_training.py`        | TensorBoard events → `progress.png` (reward/success curves; CPU-only, safe during training) |
| `list_envs.py`            | List registered Isaac Lab task IDs |

## Training & evaluation

```bash
conda activate isaaclab51

# State-obs policy (128 envs)
OMNI_KIT_ACCEPT_EULA=YES python scripts/train.py \
  --task Isaac-Insertion-CoolingPeg-Direct-v0 --num_envs 128 --headless \
  --max_iterations 500 agent.params.config.full_experiment_name=my_state_run

# E2E vision policy: wrist RGB, frozen ImageNet ResNet-18, full 320x180 16:9 frames (no crop).
# NOTE: requires --enable_cameras; 32 environments is the tested 24 GB configuration.
# For any unattended run prefer the watchdog (stock Sim 5.1 experience + automatic resume):
SEED_RNG=42 setsid bash scripts/auto_resume_train.sh my_vision_run \
  Isaac-Insertion-CoolingPeg-Iiwa-E2E-Vision-Direct-v0 32 1800 >my_vision_run.log 2>&1 &
# Realistic ranges (grasp 10°, ±8 mm, 25° tilt) + camera/proprio DR + APPEARANCE DR (per-env materials,
# directional light, D405 sensor noise; see DECISIONS.md §12+§14) are now the cfg DEFAULTS. Override per-run
# via EXTRA_OVERRIDES, e.g. EXTRA_OVERRIDES="env.cam_pos_jitter=0.0 env.randomize_part_materials=false".
# Watch what the policy SEES: renders/train_cam/ (periodic gallery, on by default). Add VIDEO=1 for rollout
# clips in logs/.../videos/train/. SEED_RNG pins the RNG for reproducibility.
#
# Sensing/arch toggles (default OFF; add to EXTRA_OVERRIDES). Memory is 24 GB-bound -> use the PROBED env
# counts (scripts/probe_max_envs.sh): 160px=64, 224px=48, 256px OR 224px+frame_stack=32.
#   env.image_height/width + env.tiled_camera.width/height=224|256   # camera resolution (224 > 160, DECISIONS §16)
#   env.use_torque_obs=True                                          # 6-axis F/T: +3 wrist-torque channels (proprio 24->27)
#   env.frame_stack=3                                                # temporal: stack last N frames (image 4->4N ch)
#
# Unattended run CHAINS (train+eval back-to-back, crash/hang-safe, restart-safe): scripts/run_chain.sh <runlist>
#   setsid bash scripts/run_chain.sh scripts/runlist_week.txt </dev/null >/tmp/chain.log 2>&1 &
# Runlist row: name | envs | iters | eval_envs | seeds | overrides . Summary -> logs/chain/summary.txt.

# Evaluate (add --enable_cameras for the vision task)
OMNI_KIT_ACCEPT_EULA=YES python scripts/eval_policy.py \
  --task Isaac-Insertion-CoolingPeg-Direct-v0 --num_envs 128 --num_episodes 512 --headless \
  --checkpoint logs/rl_games/Forge/<run>/nn/Forge.pth
```

Logs/checkpoints land in `logs/rl_games/Forge/<run>/` (`nn/Forge.pth` = best by mean reward). On the
old easy env runs plateaued by ~200–300 it; on the full hardened + appearance-DR env eval success kept
climbing through **~3000 it** (`vision_appearance_1`: 64.6→72.7% from ep1000→3000 — the *training*-success
plateau ~ep1500 was misleading), so deliverable runs warrant ~3000; use **~1500 it only for architecture
screening**, then retrain the winner long. Eval writes a timestamped report next to the checkpoint, and
`Forge.pth` is best-by-*reward* (not always best-by-success — eval the last few checkpoints).

> **Vision/GPU note:** use the stock Isaac Sim 5.1 rendering experience. The custom
> `apps/isaaclab.python.headless.rendering.physx1065.kit` file is retained only for the old Isaac Sim
> 4.5 environment and is not compatible with the upgraded stack.

## Code formatting

```bash
pip install pre-commit
pre-commit run --all-files
```

---

_README last updated: 2026-08-21._
