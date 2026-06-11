# Design Decisions — RL Insertion Policy

Living log of design choices, **why** we made them, and **what to revisit later**.
Many choices here are *provisional* because the real robot isn't physically available yet
(arrives ~mid-June 2026). When it arrives, revisit everything tagged **[REVISIT @ robot]**.

---

## 0. Scope (per supervisor, 2026-06-10): **pure residual insertion RL**

**Decision:** The policy does *only* residual correction. Both the **pre-insert pose** and the
**goal (seated) pose** are GIVEN by the upstream pipeline (grasping + path-planning). The policy
output is a small **residual delta** on top of the nominal pre-insert→goal straight-line motion;
it only has to account for the **deviation** (accumulated upstream error). No searching, no
free-space approach — just the last-cm correction.

**Why:** Supervisor confirmed this is the intended scope; it matches Fabrica's residual-action
design and keeps the learning problem small (correct ~3–7mm / few-deg error, nothing more).

**Implications:**
- Action space = residual position delta (orientation eliminated by path-centric transform). The
  nominal trajectory is fed as the baseline; PLAI applies actions as `s_des += action`.
- Reward = negative L2 to the given goal (Fabrica), not Factory's free-form keypoint search.
- **Domain randomization must be centered on the *expected upstream error distribution*** —
  randomize the initial deviation in the **vicinity of the real accumulated error** (vision ~5mm
  + grasp + path-plan → ~3–7mm position, few deg orientation). Not arbitrary large noise.
  → tune `fixed_asset_init_pos_noise` / `obs_rand` / held-asset noise to that band (see §6).

---

## 1. Base environment: **Forge** (`Isaac-Forge-PegInsert-Direct-v0`)

**Decision:** Fork Forge (not plain Factory) into our `insertion_policy` extension.

**Why:**
- Forge = Factory + force sensing (`get_link_incoming_joint_force()` at a `force_sensor`
  body) + observation pose-noise on the fingertip + contact-force penalties + success
  prediction. Two of those (force, pose-noise) are core pillars of this thesis.
- Force obs maps directly onto the iiwa's 7 joint torque sensors → our EE-wrench stretch goal.
- Observation pose-noise directly models the accumulated 3–7mm upstream error our policy corrects.

**Alternatives considered:**
- *Factory*: simpler, but no force/noise — we'd reimplement what Forge already has.
- *AutoMate*: built for generalizing across 100+ parts via specialist→generalist distillation.
  Closest to our "generalize across sockets" aim but much heavier. **[REVISIT]** borrow its
  distillation ideas if our path-centric + multi-socket training generalizes poorly.

**What we add on top (not in Forge):** wrist RGB camera + CNN encoder (our contribution),
path-centric coordinate transform + residual actions (from Fabrica), PLAI for sim-to-real
(from IndustReal).

---

## 2. Robot: **KUKA iiwa 7** (provisional, public/bundled asset)

**Decision (provisional):** Use the iiwa **7** arm from Isaac's bundled
`{ISAACLAB_NUCLEUS_DIR}/Robots/KukaAllegro/kuka.usd` (joints `iiwa7_joint_1..7`) for now,
swapping out Factory/Forge's Franka Panda.

**Why:**
- Real robot model unconfirmed (7 R800 vs 14 R820) and not physically available for ~1 week.
- The bundled `kuka.usd` is the only ready-made iiwa USD on hand → fastest path to a working sim.

**[REVISIT @ robot]:**
- Confirm exact model: **iiwa 7 R800 vs iiwa 14 R820**. If 14, link lengths / joint limits /
  dynamics differ → need the iiwa14 USD.
- For sim-to-real, replace this asset with the URDF from the **lab's own ROS `*_description`
  package** (matches the real arm + flange + gripper exactly), converted via `convert_urdf.py`.
- The bundled asset ships with an **Allegro hand** we don't want (see §3).

---

## 3. Gripper: **generic parallel gripper** (provisional) — NOT the Franka Hand

**Decision (provisional):** Strip the Allegro hand from the bundled iiwa and attach a generic
**parallel** gripper. Leading candidate: **Robotiq 2F-85** (common, well-supported URDF). User
confirmed (2026-06-10) the real gripper is **definitely NOT a Franka Hand**.

Note: in the step-(a) milestone we keep Forge's Franka arm+hand to validate the *parts* first,
so the gripper choice only bites in step (b) (robot swap). The franka_gripper-style params in
the vision configs (`gripper_epsilon_inner/outer`, `grasp_width/force/speed`) are therefore NOT
evidence of a Franka Hand — likely just a reused ROS action interface.

**Why a parallel gripper:** insertion needs a simple parallel grip on the screw, not a dexterous
hand; Robotiq 2F-85 has a standard URDF and is widely used in Isaac.

**[REVISIT @ robot]:**
- Confirm the real gripper model (Robotiq 2F-85/140? Schunk? lab-custom?). Finger geometry
  changes how the 20mm screw head is held → affects grasp pose and held-asset offset.
- Align the in-sim grasp pose with the **grasping student's** actual grasp.

---

## 4. Insertion task definition (cooling assembly first)

**Decision:**
- First pair: **cooling_screw → cooling_base**. (cooling_f converted but not yet a target.)
- **Multi-socket from the start**: cooling_base has **2 sockets at (±30, 0)mm** on its raised
  80×80 platform (the 4 corner holes at ±50,±50 are mounting holes, NOT targets).
- **Straight linear insertion** along −Z (no threading).
- **Success = screw head bottoms flush on the platform top** (shaft fully seated); letting it
  drop the last ~1mm under gravity is acceptable.

**Why:** Policy is a *generalized last-cm* corrector; the path-planning student ("master")
chooses the part→hole assignment and delivers to pre-insert. Sockets define the **training
scene distribution**, not policy logic. Generalization comes from path-centric canonicalization
+ training across sockets/parts + pose-error randomization.

**[REVISIT]:** Analyze pb_base sockets (`scripts/analyze_sockets.py --base pb_base`) and add the
pb assembly to the training distribution. Confirm exact seated z and clearance against real parts.

**Status / known issue (2026-06-10):**
- Step (a) DONE: forked env `Isaac-Insertion-CoolingPeg-Direct-v0` registers, loads the cooling
  USDs as Articulations, and runs Forge's reset/step/reward loop (smoke test passes, Franka still
  in place). Required baking `ArticulationRootAPI` into the part USDs
  (`scripts/add_articulation_root.py`) and reusing Factory's `peg_insert` behavior-selector.
- **CRITICAL NEXT FIX — off-center sockets:** Factory assumes the socket sits at the fixed-asset
  XY center; our cooling_base sockets are at x=±30mm. The env currently aims at the platform
  center (no hole). Fix by subclassing `ForgeEnv` and overriding the socket-target computation to
  use a per-env socket offset (∈ {(+30,0),(−30,0)} → multi-socket), rather than re-centering the
  USD. This override is also where the **residual-action reward (neg-L2 to given goal, §0)** and
  path-centric transform will live.

---

## 5. Asset conversion (done — Todo 1)

**Decisions:** cm→m scale (×0.01); **SDF collision** (res 256) for accurate concave socket
contact; **rigid + mass + collision** props; mass estimated from mesh volume @ 1.0 g/cm³;
**`make_instanceable=False`**.

**Why instanceable=False:** per-env visual/texture **domain randomization** for the wrist
camera needs distinct prims; instancing shares geometry and fights that. Meshes are tiny
(≤6.4k faces) so the perf cost is minor.

**[REVISIT]:** if many-env throughput becomes a bottleneck, reconsider instancing the
*collision* geometry while keeping per-env visual materials. Re-estimate masses if real part
weights are measured.

---

## Sequencing note

To isolate risk, Todo 2 proceeds as: (a) fork Forge + swap in cooling parts/sockets **keeping
the Franka** → verify scene loads + smoke-trains; then (b) swap Franka → iiwa7 + gripper →
verify again. That way a breakage is unambiguously a *parts* issue or a *robot* issue.
