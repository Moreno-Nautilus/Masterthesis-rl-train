# Walkthrough — Vision-Based Residual RL for Part Insertion

_An overview for the repo: what this project is, how it's built, what we've found, and
where we stand. For install/run commands see [README.md](README.md); for the running design log see
the (local) `DECISIONS.md`._

_Last updated: 2026-06-23._

---

## 1. Goal & scope

Train an RL policy in **NVIDIA Isaac Lab** that performs the **final ~1–5 cm of a part insertion**
on a KUKA LBR iiwa, using a **wrist-mounted RGB-D camera** for closed-loop visual feedback. We
augment the blind, state-based insertion policy of **Fabrica** (Tian et al., 2025); the thesis
contribution is **adding vision**.

**Scope — pure residual insertion.** The upstream pipeline hands over a *pre-insert pose* and a
*goal (seated) pose*; the policy only outputs a **6-DOF residual correction** for the accumulated
upstream deviation — no free-space approach, no search. Orientation is kept in the action
because the angular error is large (up to ~20–25°).

**Where this sits in the larger pipeline** (other students own the rest):
pose estimation (done) → grasping → path planning / part→socket assignment → **insertion (this
project)**. The policy must be **general** across parts/sockets, so the socket distribution defines
the task, not policy logic.

**Uncertainty the policy must correct:** ~3–7 mm lateral, up to ~20–25° orientation, over a ~5 cm
approach, **plus grasp misalignment** (the part is not perfectly aligned in the gripper — an error
the policy cannot observe from proprioception, which is precisely what vision should resolve).

---

## 2. How it's built

- **Base environment:** built on **FORGE** (Noseworthy 2025) → **Factory** (Narang 2022) — force-aware
  assembly with SDF contact physics. Class layering: `DirectRLEnv → FactoryEnv → ForgeEnv →
  InsertionEnv` (our subclass). Algorithm: **PPO (RL-Games)**.
- **Task:** cooling-assembly peg-in-socket. Two Gym tasks share one env class:
  - `Isaac-Insertion-CoolingPeg-Direct-v0` — **state observations**.
  - `Isaac-Insertion-CoolingPeg-Vision-Direct-v0` — **state + wrist RGB-D image**.
- **What `InsertionEnv` adds** (`source/insertion_policy/.../tasks/insertion/`):
  - **Socket-aware, multi-socket targeting** (sockets are off-center on the plate).
  - **Realistic reset:** shaft tip 1–5 cm above the socket opening, ±7 mm lateral.
  - **Pre-insert tilt injection** (up to ~25°, about the shaft tip) — the upstream orientation error.
  - **Grasp misalignment** (≤5°, random, baked into the grasp) — unobservable by proprioception.
  - **Orientation-aware reward:** a multi-keypoint **squashing kernel** (dense, bounded, smooth near
    the goal) + a binary seat bonus + FORGE's success-prediction term.
  - **Success:** centered (<2.5 mm) **and** seated (z within ~0.25× socket depth) **and** aligned
    (shaft axis < 10°).
- **Vision pipeline:** a wrist **RGB-D** `TiledCamera` → a small **CNN**, fused with the proprio
  vector and fed through an **LSTM + MLP** policy (custom rl_games network `insertion_hybrid`). The
  critic is asymmetric (sees the privileged state). Encoder is a small end-to-end CNN — **not**
  DINOv2 (too heavy for the on-policy loop; DINOv2 stays in the offline pose pipeline).

---

## 3. The parts

The cooling parts are **simple 3-D-printed test parts**, not machined hardware:
- **Screw (held):** two stacked cylinders — 20 mm head over a **12 mm** shaft.
- **Base (fixed):** a plate with two **14 mm** sockets.
- **Clearance: 1 mm radial, and there is NO lead-in chamfer** (verified on the source CAD with
  `scripts/check_recoverability.py`). This is by design.

**Consequence:** there is nothing to "funnel" a misaligned peg in. Tilt is recovered *dynamically*
via compliant (impedance) control + contact, but the **binding constraint is lateral precision** —
the shaft tip must be brought to within ~1 mm of the socket centre. That precise alignment, under
unobservable grasp error, is exactly the regime where a camera should help. So the chamfer-free
geometry **strengthens** the vision motivation rather than being a flaw.

---

## 4. Results so far

| Run | Obs | Training | Eval (success) |
|---|---|---|---|
| `squash_state_256` | state | 5000 ep (converges ~1000–1500) | **48% overall** |
| `vision_full_1` | state + 64 px RGB-D | 1500 ep | **13.9% overall** |
| `state_grasp_1` | state + grasp error | 800 ep | _in progress — the fair reference_ |

**State baseline (48%)** — clean monotonic degradation with error magnitude (e.g. 68% at 0–5° tilt →
11% at >25°; 58% at 0–3 mm lateral → 19% at >10 mm). Note this baseline **predates grasp
misalignment**, so it is not a fair comparator for the vision policy (see below).

**Vision (13.9%) — currently *under*performs the state baseline.** This is an honest negative result
that we have diagnosed:
- Reward plateaued at ~56 (the state policy reaches ~127), and success is roughly flat across error
  bins — even an *easy* start (tip within 3 mm) only seats ~18%.
- The network wiring is correct (audited; the CNN receives the image and trains).
- **Most likely root cause — the image is too coarse to resolve the task.** At 64 px the camera sees
  a ~14 cm-wide view ⇒ **~2.2 mm/pixel**, while the task needs ~1 mm precision ⇒ sub-pixel ⇒ the CNN
  *cannot see* the alignment error. So the policy falls back to proprioception, which — with the new
  unobservable grasp error — is *harder* than the original state task. Hence ~14%.

**Fix applied, test queued:** camera bumped to **160 px at a ~6 cm zoomed view ⇒ ~0.37 mm/pixel
(~2.7 px/mm)**, comfortably oversampling the 1 mm threshold (Nyquist). The next experiment is the
**160 px vision run vs. a blank-image control** (identical network, zeroed image) to prove whether a
*resolvable* image beats proprioception.

---

## 5. Key decisions & findings

1. **Reward = squashing kernel, not negative-L2.** Dense, bounded, smooth near the goal; +15 eval
   points over the earlier neg-L2 reward (state policy 33% → 48%).
2. **Grasp misalignment is part of the task.** It is unobservable from proprioception, so it is the
   core justification for vision — and it makes a fair comparison require re-running the state
   baseline *with* it (`state_grasp_1`, in progress).
3. **Chamfer-free physics ⇒ lateral precision is the bottleneck** (Section 3). Not an asset bug.
4. **Camera resolution must satisfy Nyquist for the 1 mm task** (Section 4) — the single most
   important lesson from the first vision run. Framing (mm/pixel on the insertion region), not raw
   sensor megapixels, is what matters.
5. **Input normalization with dict (image+vector) obs is a red herring / pitfall.** Disabling it
   plateaus; enabling it crashes rl_games (TorchScript can't compile its dict normalizer) and, once
   patched to run eager, *destabilizes* training by normalizing pixels. We normalize the image
   in-env and leave proprio handling conservative.

---

## 6. Infrastructure notes (relevant if reproducing)

- **Isaac Lab 2.3.2 / Isaac Sim 4.5.0-rc.36** (a *release candidate*). The RC's `usdrt` lacks the
  `hierarchy` module Isaac Lab's Fabric pose-path expects, so a camera crashes unless worked around.
  We keep PhysX on Fabric (GPU-stable) and route **only the camera's pose view** to the USD path (a
  small monkeypatch in `insertion_env.py`). **The clean fix is to install the final Isaac Sim 4.5.0**
  — recommended before any sim-to-real work.
- **GPU stability:** rendering + GPU physics at 128 envs intermittently hit a PhysX CUDA error;
  **running vision at 64 envs is crash-free** (a full 1500-ep run completed overnight in one go).
  `scripts/auto_resume_train.sh` is a watchdog that checkpoints/resumes through crashes if needed.
- One Isaac Sim job at a time (single GPU). Train detached so runs survive SSH drops.

---

## 7. Current status & next steps

**Status:** the full vision pipeline is implemented and trains stably; the state baseline is strong
(48%); the first 64 px vision policy under-performs and we have a well-supported diagnosis (image
resolution) plus an implemented fix (160 px).

**Immediate next steps (in order):**
1. Finish `state_grasp_1` → the **fair proprio-only reference** under grasp error.
2. Quick render-framing check of the 160 px camera.
3. **160 px vision run + blank-image control** → the decisive test: does a resolvable image beat
   proprioception?
4. If yes → the full **ablation matrix** (state / +vision / +force / +vision+force).

**Then (thesis roadmap):** path-centric observation transform (SE(3)-equivariant generalization);
domain randomization for sim-to-real; KUKA iiwa + parallel-gripper swap (once the robot is in hand);
sim-to-real transfer with PLAI.

---

## 8. Repository map

```
source/insertion_policy/insertion_policy/tasks/insertion/
  insertion_env.py          # env: sockets, reset, tilt, grasp error, reward, camera, vision obs
  cooling_tasks_cfg.py       # task/camera configuration
  assets_cfg.py              # part asset configs
  agents/
    insertion_hybrid_network.py   # custom CNN+LSTM rl_games network (vision)
    rl_games_camera_ppo_cfg.yaml  # vision PPO config
scripts/
  train.py, eval_policy.py, plot_training.py   # train / evaluate / plot
  check_recoverability.py    # geometric chamfer/clearance probe
  auto_resume_train.sh       # crash-resilient training watchdog
  convert_assets.py, smoke_env.py, ...         # assets + sanity tools
logs/rl_games/Forge/<run>/   # checkpoints (nn/), TensorBoard (summaries/), curves, eval reports
```

Key references: Fabrica (Tian 2025), IndustReal (Tang 2023), FORGE (Noseworthy 2025),
Factory (Narang 2022).
