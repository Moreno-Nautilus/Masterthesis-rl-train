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

> **2026-06-25 — two sim bugs invalidated all earlier results.** A wrist-POV rollout review revealed
> the screw was never properly gripped and the base slid on the table. After fixing both, **every
> policy jumped from ~35% to ~95%.** The numbers below are the current, valid ones; the old broken-env
> numbers (state 36.5–48%, vision 13.9–34.8%) are kept only for history at the end of this section.

| Run | Obs | Training | Eval (success) |
|---|---|---|---|
| `state_fixed_1` | proprioception only | 500 ep | **94.5%** |
| `vision_fuse_1` | proprio + **160 px** RGB-D, fused (FC→128) | 500 ep | **95.5%** |

**The grasp/base bugs were the real bottleneck — not task difficulty.** Two issues, both in the
provisional Franka setup:
- **The screw was not actually gripped.** Factory's peg-grasp offset (`relative_z = height −
  fingerpad_length`) assumes the part's origin is at its *base*. Our screw's origin is at the
  head/shaft *shoulder* and it's two-diameter (20 mm head / 12 mm shaft), so the screw was placed ~one
  shaft-length too low: the gripper closed on empty air and the screw merely *dangled* below the
  fingers (held by friction, never re-pinned), free to wobble and slip. Fix: grip the **head** as the
  graspable cylinder.
- **The 50 g base slid** under the jamming forces of a chamfer-free 1 mm-clearance insertion. Fix:
  make it **heavy (10 kg) + high-friction** — effectively glued, while still randomized into place each
  reset (a kinematic body breaks the articulation wrapper; `fix_root_link` would kill the placement
  randomization).

With a screw that's actually held and a base that doesn't move, the insertion task is **highly
solvable**: the proprioception-only policy reaches **94.5%**, robust across the whole tilt/lateral range
(≈94% even at 15–25° tilt and 7–10 mm lateral), dropping only at the extremes.

**Vision ties proprioception — it does not beat it.** `vision_fuse_1` = **95.5%** vs **94.5%** is a tie
within noise (binomial SE ≈ ±1 % over 512 episodes, single seed each). The fusion FC (project the 3136
CNN features → 128 before concatenating with the 24 proprio dims) works as intended, but **at the
current 5° grasp misalignment there's simply no headroom left for vision to add** — proprioception with
a firm grasp already absorbs a 5° in-jaw tilt. The unobservable error vision was built to handle is, at
5°, too small to matter.

**What this means / next.** The decision-relevant question is now whether vision helps at the *realistic*
grasp uncertainty (~10°, vs the 5° we trained at): re-run the state-vs-vision pair at
`grasp_misalign_max_deg = 10` (and maybe 15). If proprioception *drops* while vision *holds*, the camera
is genuinely needed; if both stay ~95%, the blind policy suffices. This directly answers "do we need
vision at all," and takes priority over fusion micro-tuning (the earlier "fusion is the bottleneck"
hypothesis was wrong — the broken env was).

<details><summary><b>Superseded (broken-env) results, for history</b></summary>

| Run | Obs | Eval |
|---|---|---|
| `squash_state_256` | state, no grasp error | 48% |
| `state_grasp_1` | state + grasp error | 36.5% |
| `vision_full_1` | 64 px RGB-D + grasp error | 13.9% (image too coarse) |
| `vision_160_1` | 160 px RGB-D + grasp error | 34.8% |
| `vision_blank_1` | 160 px arch, image zeroed | 8.8% |

These trained with the dangling grasp + sliding base. The 64 px → 160 px resolution fix (Nyquist for
the 1 mm task) was real and still applies. The "fusion drowns proprio / vision is image-only" diagnosis
from the blank-image control motivated the fusion FC we kept — but the dominant limiter turned out to be
the env bugs, not fusion.
</details>

---

## 5. Key decisions & findings

1. **Validate the grasp before trusting any number.** The single biggest lesson: a custom part with a
   non-base origin silently broke Factory's grasp formula, so the screw was never really held — and it
   capped *every* result at ~35% until found. Watch a rollout; don't trust success curves alone.
2. **Reward = squashing kernel, not negative-L2.** Dense, bounded, smooth near the goal (a clear gain
   over the earlier neg-L2 reward on the state policy).
3. **Grasp misalignment is part of the task** and unobservable from proprioception — the core
   justification for vision. But at **5° it's too small to matter** once the grasp is firm (state and
   vision both ~95%); the realistic ~10° is the test that decides whether vision is needed.
4. **Chamfer-free physics ⇒ lateral precision is the bottleneck** (Section 3). Not an asset bug. (It
   also made the *base* slide until pinned by mass+friction.)
5. **Camera resolution must satisfy Nyquist for the 1 mm task** (Section 4) — 64 px (~2.2 mm/px) was
   sub-pixel and useless; 160 px (~2.7 px/mm) is resolvable. Framing (mm/pixel on the insertion
   region), not raw megapixels, is what matters.
6. **Fuse image + proprio at comparable width.** The hybrid net concatenated 3136 CNN dims with 24
   proprio dims (proprio drowned); a `Linear(3136→128)+ReLU` before the concat fixes the imbalance.
   (Helpful in principle, but not what was capping results — the env bugs were.)
7. **Input normalization with dict (image+vector) obs is a pitfall.** Enabling it crashes rl_games
   (TorchScript can't compile its dict normalizer); patched to run eager. We normalize the image
   in-env.

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

**Status:** with the grasp + base bugs fixed, the corrected env is **largely solved at 5° grasp
misalignment** — proprioception **94.5%**, vision **95.5%** (a tie). The vision pipeline (160 px RGB-D,
CNN+LSTM, fusion FC) trains stably end-to-end. The open question is no longer "does vision work" but
"is vision *needed*."

**Immediate next steps (in order):**
1. **Grasp-error comparison at the realistic 10° (and 15°):** re-run state vs vision at
   `env.grasp_misalign_max_deg=10`. If proprioception drops while vision holds → vision is needed; if
   both stay ~95% → the blind policy suffices. This is the decision-relevant test.
2. **Rigid (physical) camera mount** — replace the non-physical look-at (which centres the true socket)
   with a fixed wrist mount, for sim-to-real validity (and possibly a better signal).
3. **Domain randomization on** (appearance + camera-pose jitter) so the CNN survives the real D405 feed.
4. If vision proves its worth → the **ablation matrix** (state / +vision / +force / +vision+force).

**Then:** KUKA iiwa + custom parallel-gripper swap (once the gripper is built — the grasp offset will
be re-tuned for it); sim-to-real transfer.

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
