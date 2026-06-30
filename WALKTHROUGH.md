# Walkthrough — Vision-Based Residual RL for Part Insertion

_An overview for the repo: what this project is, how it's built, what we've found, and
where we stand. For install/run commands see [README.md](README.md); for the running design log see
the (local) `DECISIONS.md`._

_Last updated: 2026-06-29._

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
  - **Realistic reset:** shaft tip 1–5 cm above the socket opening, **±8 mm** lateral.
  - **Pre-insert tilt injection** (up to ~25°, about the shaft tip) — the upstream orientation error.
  - **Grasp misalignment** (up to **10°**, baked into the grasp, about the **finger pressing axis** —
    the only way a cylinder can tip between flat pads) — unobservable by proprioception.
  - **Domain randomization:** camera-pose jitter (5 mm eye + 2° roll), moderate proprio/force
    observation noise (0.5 mm / 0.5° / larger F/T), plus appearance DR for the wrist RGB-D stream
    (per-env matte materials, dome + directional key light, photometric aug, and D405-style RGB/depth
    noise). The actor sees the noisy signal; the critic keeps privileged clean state.
  - **Orientation-aware reward:** a multi-keypoint **squashing kernel** (dense, bounded, smooth near
    the goal) + a binary seat bonus + FORGE's success-prediction term.
  - **Success:** centered (<2.5 mm) **and** seated (z within ~0.25× socket depth) **and** aligned
    (shaft axis < 10°).
- **Vision pipeline:** a wrist **RGB-D** `TiledCamera` → a small **CNN**, fused with the proprio
  vector and fed through an **LSTM + MLP** policy (custom rl_games network `insertion_hybrid`). The
  critic is asymmetric (sees the privileged state). Encoder is a small end-to-end CNN — **not**
  DINOv2 (too heavy for the on-policy loop; DINOv2 stays in the offline pose pipeline).
  - **Rigid wrist mount (sim-to-real-valid):** the camera is bolted to the gripper — a fixed pose
    (position + pointing + roll) in the fingertip frame, never re-aimed at the true socket. Earlier it
    used a privileged look-at that centred the socket and pinned roll to world-up; that was a transfer
    killer and is fixed. Mounted at ~45° azimuth so the shaft's grasp tilt is visible (a side mount
    foreshortens it; a top-down mount lets the gripper occlude the hole). Provisional until the custom
    gripper, then re-derived.

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

> **2026-06-29 — APPEARANCE DR + final transfer number: vision holds 72.7%.** Added the deferred
> appearance DR (the sim-to-real piece) onto the hardened env and trained the vision policy there — this
> is now **the env we quote the final number on**. Four cfg-gated layers: **per-env matte materials**
> (each env its own colour; screw≈base *within* an env so the policy can't lean on a colour-contrast cue
> that won't exist on the same-filament real parts), **dome + directional key light** (moving
> shadows/specular a dome can't give), **per-env photometric aug**, and **D405 sensor noise** (RGB +
> depth gaussian + dropout holes). `vision_appearance_1` (3000 it, seed 42) finished crash-free.
>
> | policy | env | success (512 ep, DR-on) |
> |---|---|---|
> | state (proprio) | hardened | 66.2% |
> | vision | hardened (no appearance DR) | 75.6% |
> | **vision** | **+ appearance DR (final env), best ckpt ep3000** | **72.7%** |
>
> **72.7% still beats state 66.2% (+6.5 pp)** and is the **flattest across grasp tilt yet** (67–77% over
> *every* bin incl. 72% past 25°) — the camera reading the unobservable tilt. The −3 pp vs the old 75.6%
> is the appearance-DR/sensor-noise **robustness tax**, paid on purpose (north star = transfer, not sim %).
> **Eval kept climbing** 64.6%(ep1000)→66.4%(ep2000)→72.7%(ep3000), so the full 3000 it was justified for the
> deliverable (the *training*-success plateau ~ep1500 was a red herring — exploration variance); use short
> ~1500-it runs only for architecture *screening*, then retrain the winner long. NB `Forge.pth` (best by
> reward, ep2700, 70.7%) ≠ best by success (ep3000). See `DECISIONS.md` §14, renders in `renders/appearance_dr/`.

> **2026-06-26 — HARDENED-ENV VERDICT: vision wins.** On the transfer-valid env (rigid wrist cam, grasp
> **10° about the pressing axis**, **±8 mm**, 25° tilt, camera + proprio/force **DR**), 512-ep eval @1000 it:
>
> | policy | @300 it | @1000 it |
> |---|---|---|
> | state (proprio) | 58.6% | **66.2%** |
> | vision (proprio + RGB-D) | 50.2% | **75.6%** |
> | blank (image zeroed) | **33.8%** | — |
>
> **Vision 75.6% > state 66.2% (+9.4 pp)**, flat-robust across all tilt/lateral bins. A **matched-300-it
> control isolates the image** as the cause: `vision@300 50.2%` vs `blank@300 33.8%` (+16.4 pp, **same net +
> budget, only image on/off**) — so it's the image content, not the architecture. (The hybrid net is slow
> to train — both vision and blank trail the lean state net at 300 it, then vision overtakes by 1000 — which
> is why blank's number is low: undertraining, not a proprio ceiling.) Unlike the 5° **"tie" below** — now
> the *easy, superseded* baseline (5° grasp, non-physical look-at cam, no DR) — at realistic grasp error the
> camera earns its keep. Absolutes are lower than 95% because DR + 10° + rigid cam is much harder (and
> correctly so; see the realistic target). See `DECISIONS.md` §13. This result motivated the appearance-DR
> transfer run summarized above.

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

**What this meant / what happened next.** The decision-relevant question was whether vision helps at the
*realistic* grasp uncertainty (~10°, vs the 5° easy baseline). That test is now done in the hardened env:
proprioception dropped to **66.2%** while vision reached **75.6%**, then vision held **72.7%** after
appearance DR. So the camera is genuinely needed under the realistic transfer-targeted assumptions; the
earlier "fusion is the bottleneck" hypothesis was wrong — the broken env was.

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

**Status (2026-06-29):** Plan A step 1 is **done** — appearance DR built and the vision policy trained on
it (`vision_appearance_1`, 3000 it, crash-free) and eval'd to **72.7%** on the full transfer env, **still
beating state 66.2% (+6.5 pp)** and the flattest across grasp tilt yet. This is the **final transfer-targeted
env** and the current best deliverable is `vision_appearance_1/nn/last_Forge_ep_3000` (eval climbed through
ep3000; `Forge.pth` best-by-reward is ep2700/70.7%, slightly behind). (Earlier: the 2026-06-26 hardened-env
verdict `vision 75.6%` vs `state 66.2%` — pre-appearance-DR; and the superseded 94.5/95.5% "tie" at the easy
5° baseline.) Deliverable runs warrant ~3000 it (eval kept improving); use **~1500 it only for architecture
screening**, then retrain the winner long.

**Immediate next steps (see `PLANNING.md` for the full week plan + schedule):**
1. **Push the sim % (architecture) — lead with an auxiliary head** predicting the held/grasp pose from the
   image (forces the CNN to encode the unobservable grasp tilt, may speed convergence), then fc_size sweep /
   frame-stack / force-tactile fusion. Tuned on *this* final env.
2. **Generalization** — multi-socket is already live; bring in **pb_parts** (second, non-cylindrical part)
   so the policy isn't cooling_screw-only. Interleave with (1) to keep the single GPU busy.
3. **Sim-to-real (gripper-gated):** KUKA iiwa + custom parallel-gripper swap (re-derive grasp offset *and*
   camera mount azimuth), mount the real D405, retrain on the new-gripper env, deploy/fine-tune. Install
   final Isaac Sim 4.5.0 first. The gripper file may arrive early next week → it preempts the GPU queue.

*Optional rigor (engineering, so nice-to-have): a 2nd training seed of vision.*

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
