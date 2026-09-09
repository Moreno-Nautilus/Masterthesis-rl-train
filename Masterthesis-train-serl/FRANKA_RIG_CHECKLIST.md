# Franka FR3 HIL-SERL — Rig Checklist (3 days)

Companion to `FRANKA_PIVOT_PLAN.md`. Everything below runs on the **robot PC** (ROS2
Humble + FCI + RT kernel). **CONTROL SAFETY IS #1: no autonomous motion until compliance
+ PS4 deadman + E-stop are hand-verified on the real FR3.**

Legend: **`TODO(rig)`** = a value/name to confirm on the robot PC (mostly `ros2 topic list`
/ `ros2 action list`) and then fill into the code where the matching `TODO(rig)` marker is.

---

## Day 1 — CONTROL FIRST

### 1. Franka control stack — CHECK FIRST, rebuild only if missing
The robot PC very likely ALREADY has the controller built + running (it's the student's
proven setup). If so, DO NOT rebuild — just find its topics (step 3) and point our client
at them. Check first:
```bash
ros2 pkg list | grep -iE "cartesian_impedance_control|messages_fr3|franka"
ls ~/franka_ros2_ws/install/setup.bash 2>/dev/null   # or wherever their ws lives
ros2 topic list | grep cartesian_position_controller  # controller already up?
```
If present → skip to step 2 (source their ws). Only if ABSENT, clone + build from the
supervisor PDFs (Important_Info.pdf names them):
- `github.com/frankarobotics/franka_ros2` (branch **v0.1.15**)
- `github.com/CurdinDeplazes/cartesian_impedance_control` + its `messages_fr3`

#### (fallback) Rebuild `franka_ros2_ws` v0.1.15 (from Leo's install PDF)
```bash
sudo apt install ros-humble-desktop ros-dev-tools python3-vcstool
mkdir -p ~/franka_ros2_ws/src && cd ~/franka_ros2_ws
git clone -b v0.1.15 https://github.com/frankarobotics/franka_ros2 src
# hand-write src/franka.repos pinning: franka_description 0.4.0, libfranka 0.13.2
vcs import src --recursive --input src/franka.repos
rosdep install --from-paths src --ignore-src -r -y
# patch franka_ign_ros2_control/CMakeLists.txt: Franka 0.13.3 -> 0.13.2
# apply the GitHub-issue-160 3-file additions
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```
Then clone + build into the SAME ws and rebuild:
- `git clone https://github.com/CurdinDeplazes/cartesian_impedance_control`
- `git clone <messages_fr3>` (the controller's custom msgs)
- the student's `ps4_controller_teleop` (already present in this repo tree; copy in if needed)

RT kernel + NVIDIA per the docker-realtime note. **FCI**: static net `172.16.0.x` on the C2
shop-floor port; in Desk → unlock joints + **Activate FCI**.

### 2. Bring up the controller + validate compliance BY HAND
```bash
source ~/franka_ros2_ws/install/setup.bash
ros2 launch cartesian_impedance_control <bringup>.launch.py   # TODO(rig): exact launch file
```
- **Push the arm by hand** → it must yield (compliant). If it fights back / is stiff, STOP —
  do not proceed to autonomy.
- Confirm the **E-stop** cuts motion.
- Confirm the **PS4 R1 deadman** for BRING-UP: launch with `PS4_DEADMAN_GATES_POLICY=1` so
  motion is commanded only while R1 is held; release → arm holds. (This gate is bring-up
  ONLY — see the R1 note under "Notes / invariants". For real HIL training it is OFF and the
  POLICY drives autonomously; the compliant controller + E-stop are the safety layer then.)

### 3. Confirm the ROS2 interface names, then fill the `TODO(rig)` markers
```bash
ros2 topic list      # expect /cartesian_position_controller/commands + a robot-state topic
ros2 action list     # expect /fr3_gripper/{move,grasp} + an error-recovery action
ros2 topic info -v /cartesian_position_controller/commands   # confirm Float64MultiArray
ros2 topic echo <robot_state_topic> --once                   # confirm wrench + pose fields
```
Fill these into `franka_env/envs/robot_client.py` (`Ros2FrankaClient`, every `TODO(rig)`):
- [ ] `CMD_POSE_TOPIC` — should already be `/cartesian_position_controller/commands`.
- [ ] `ROBOT_STATE_TOPIC` + msg type → wire the subscription + `_on_state` → `get_state()`
      dict. NOTE the real msg does NOT map 1:1: `o_f_ext_hat_k` is a NESTED WrenchStamped
      (`.wrench.force.{x,y,z}` / `.wrench.torque.{x,y,z}`) — NOT a flat 6-vec to slice;
      the JACOBIAN and GRIPPER WIDTH are NOT in FrankaRobotState (separate model/gripper
      sources), and it exposes DESIRED twists, so measured `tcp_vel` must come from J·dq.
      Normalize gripper width to [0,1] (the env compares it against 0.85).
      **This is the one method that MUST be finished before any env step.**
- [ ] `UPDATE_PARAM_SERVICE` — how stiffness is set (rcl param vs service); map
      COMPLIANCE/PRECISION.
- [ ] gripper `Move`/`Grasp` actions (`GRIPPER_MOVE_ACTION`/`GRIPPER_GRASP_ACTION`).
- [ ] `ERROR_RECOVERY_ACTION` (recover / clearerr equivalent).
- [ ] `joint_reset()` — nominal-q move (joint controller switch, or MoveIt day-1 only).

### 4. PS4 controller check (`jstest` / pygame)
```bash
ls /dev/input/js*                       # note the device (js0/js1)
jstest /dev/input/js0                    # read axis + button indices
```
If the SDL layout differs from the defaults, override via env vars (no code edit needed):
```bash
export PS4_AXIS_LEFT_X=0 PS4_AXIS_LEFT_Y=1 PS4_AXIS_L2=2 \
       PS4_AXIS_RIGHT_X=3 PS4_AXIS_RIGHT_Y=4 PS4_AXIS_R2=5 \
       PS4_BTN_ENABLE=5 PS4_BTN_CLOSE=3 PS4_BTN_OPEN=1 \
       PS4_BTN_SUCCESS=0 PS4_BTN_ABORT=2       # X=success, Triangle=abort
export SDL_JOYSTICK_DEVICE=/dev/input/js0
```
Operator map: **R1**=deadman/intervene, **L stick**=XY, **L2/R2**=Z(down/up), **R stick**=
pitch/yaw, **D-pad L/R**=roll, **X**=success(ends episode), **Triangle**=abort,
**Circle**=REGRASP request (applied at the next reset; replaces the old F1 key). Square is the
(unused) gripper button when pregrasped.

### 4b. TELEOP WORKS END-TO-END — validate BEFORE any autonomy
Before smoking the RL env, confirm the PS4 actually drives the arm well through OUR stack
(this is separate from the raw `jstest` read in step 4 — it exercises PS4Expert ->
Ros2FrankaClient -> controller):
```bash
source franka_env.sh
export TELEOP_DEVICE=ps4 SDL_JOYSTICK_DEVICE=/dev/input/js0
"${SERL_PYTHON}" -c "
from franka_env.spacemouse.ps4_expert import PS4Expert
import numpy as np, time
e = PS4Expert()
for _ in range(200):
    a, btns = e.get_action(); ev = e.get_events()
    print(np.round(a,2), btns, ev)   # hold R1 + move sticks -> nonzero; X/Triangle -> events
    time.sleep(0.05)
"
```
Check ALL of: R1 deadman gates motion (zero when released, nonzero when held); each axis
moves the RIGHT DoF in the RIGHT direction; deadzone feels right; X and Triangle each fire
exactly ONCE per press. THEN drive the actual arm (via the env smoke in step 6) and confirm
it moves smoothly + compliantly and stops on deadman release. Only proceed once teleop is
trustworthy — everything downstream (demos, interventions, reset capture) depends on it.

### 5. ZED Mini check
```bash
"${SERL_PYTHON}" -c "import pyzed.sl as sl; print(sl.Camera().open(sl.InitParameters()))"
```
`ZEDCapture` uses the LEFT eye (RGB), HD720, resized to 128×128 by the env. Single camera →
serial can stay `None` in the config.

### 6. Smoke the env on the real robot (control loop, human reward)
> ⚠️ **DO STEP 7 (capture poses) FIRST.** `RESET_POSE` and `ABS_POSE_LIMIT_*` ship as ZEROS.
> `env.reset()` drives the arm to `RESET_POSE`, so smoking before the poses are pasted in
> would command it toward the base-frame ORIGIN and clip teleop into a zero-sized safety box.
> Capture (step 7) → paste into `EnvConfigInsert0` → then run this smoke.
>
> ⚠️ **R1 does NOT gate reset/regrasp motion** — `PS4_DEADMAN_GATES_POLICY` only gates
> `step()`. Reset/regrasp move the arm regardless. Hand on the E-STOP for the first reset.

```bash
source franka_env.sh
export FRANKA_EXP_NAME=franka_plumbers_insert0 TELEOP_DEVICE=ps4 SDL_JOYSTICK_DEVICE=/dev/input/js0
export PS4_DEADMAN_GATES_POLICY=1
"${SERL_PYTHON}" -c "
from experiments.mappings import CONFIG_MAPPING
env = CONFIG_MAPPING['franka_plumbers_insert0']().get_environment(fake_env=False)
o,_ = env.reset(); print('reset ok', list(o.keys()))
import numpy as np
for k in range(20):
    o,r,d,t,i = env.step(np.zeros(6)); print(k,'rew',r,'done',d,'succeed',i.get('succeed'))
    if d:   # HONOR termination (X / Triangle / timeout); don't keep commanding motion
        print('episode ended ->', 'SUCCESS' if i.get('succeed') else 'no success'); break
"
```
Verify: reset pulls up + moves to RESET_POSE; teleop moves the arm under R1; **X ends the
episode with reward 1**, **Triangle ends with reward 0**, else it ends at the step limit
(reward 0). The wrist image + wrench appear in the obs.

### 7. Capture the per-insert poses (jog + read back)
Do this BEFORE constructing/resetting the env. Jog using the controller's already-validated
teleop or hand guidance, then read the live state with `ros2 topic echo`. Capture a pose
aligned with the socket, with the socket centered in the wrist view. Convert the measured
quaternion to XYZ Euler radians: config poses are `[x,y,z,roll,pitch,yaw]`, not quaternions.
Repeat per insert. Do not use an env with placeholder poses to capture its own reset pose.
- [ ] **TARGET_POSE** — part fully seated (also used for the safety box + optional threshold).
- [ ] **RESET_POSE** — fixed pre-insert pose, ALIGNED with the socket + a few cm along the
      approach axis (ABOVE the base for a vertical insert; BESIDE it for the horizontal
      pipe insert) so the wrist camera already sees the target.
- [ ] **ABS_POSE_LIMIT_LOW/HIGH** — physically verified bounds covering reset+target AND
      the full 4 cm reset / 8 cm regrasp extraction from expected starting poses. A clipped
      retract now stops reset; never widen bounds without checking physical clearance.
- [ ] **IMAGE_CROP["wrist"]** — crop so the socket + part tip fill the frame.
- [ ] **RETRACT_DIR** — unit vector OUT of the socket:
      `normalize(RESET_POSE[:3] - TARGET_POSE[:3])` (pre-insert MINUS seated; the
      reverse points INTO the hole).
      Default (0,0,1) is ONLY valid for a VERTICAL insert. The HORIZONTAL pipe insert
      MUST override it, else the pre-reset retract shoves the part into the socket wall.
Fill these into `experiments/franka_plumbers/config.py` → `EnvConfigInsert0` (etc.).

Reset noise (DECIDED): TRANSLATIONAL ONLY — `RANDOM_XY_RANGE = 0.01` (±1cm XY),
`RANDOM_RZ_RANGE = 0.0` (rotation variation comes from the human placing the part
differently in the hand each grasp). Start `RANDOM_RESET = False` (fixed reset) day 1;
flip to `True` once insert 0 is learning.

FIXED vs MOVING base: the task assumes the base is glued at a FIXED spot (re-capture
RESET_POSE if you re-fixture between sessions). Obs/actions are in RelativeFrame, but that
does NOT make a moving base free — the socket must still land in ~the same camera view, so
a moving base needs its pose measured each episode (ArUco/vision) to recompute RESET_POSE.
Hook is stubbed: `FrankaPlumbersEnv.set_reset_from_base_pose()`. Do fixed-spot FIRST.

### 8. Record ~20 demos, then train
```bash
# Demos: stock examples/record_demos.py (VERIFIED compatible with our PS4/reward wrappers).
# Circle = regrasp (release -> hand part in -> close); X = mark seated -> banks the demo;
# Triangle/timeout -> discarded. Only successful (X) trajectories are saved.
(
cd hil-serl_src/examples || exit 1
"${SERL_PYTHON}" record_demos.py --exp_name=franka_plumbers_insert0 --successes_needed=20
)
# -> examples/demo_data/franka_plumbers_insert0_20_demos_<ts>.pkl (use as --demo_path below)

# LEARNER (GPU) — the USER launches this; do NOT auto-start:
cd ~/Masterthesis-rl-train/Masterthesis-train-serl
export FRANKA_USE_GPU=1
source franka_env.sh
unset PS4_DEADMAN_GATES_POLICY
export FRANKA_EXP_NAME=franka_plumbers_insert0
# Keep existing checkpoints; choose a new absolute CKPT in BOTH terminals for a fresh run.
# Follow TOMORROW_COMMANDS.md step 8 for the GPU check and separate actor-terminal setup.
bash hil-serl_src/examples/experiments/franka_plumbers/run_learner.sh --demo_path "/absolute/path/to/demos.pkl"
```
HIL cadence: intervene often early (explore ~20–30 steps then correct), taper as success
climbs; interventions route to BOTH buffers, policy transitions to RL only (upstream native).

---

## Day 2 — inserts 0→converging, then 1–2
- Tune insert 0 to converging (adjust ACTION_SCALE, compliance stiffness, crop). Turn
  `RANDOM_RESET` on for grasp/pose variation once it learns.
- Copy the captured poses into `EnvConfigInsert1/2`; record + train each.

## Day 3 — insert 3 + polish + eval
- Insert 3; then eval:
```bash
bash .../run_actor.sh --eval_checkpoint_step <N> --eval_n_trajs 20
```

---

## Notes / invariants
- **R1 semantics (IMPORTANT):** stock HIL-SERL is SHARED AUTONOMY — the POLICY drives the
  arm on its own and the human holds R1 + stick to CORRECT; releasing R1 hands control back
  to the policy (the arm keeps moving). The policy MUST be able to act autonomously or it
  can't learn, so **HIL TRAINING runs with the deadman gate OFF** (default). For DAY-1
  BRING-UP / teleop checks (no policy yet, or you want hold-to-move safety), set
  `PS4_DEADMAN_GATES_POLICY=1` → no motion unless R1 is held. The real emergency stop is the
  E-STOP + the compliant impedance controller, NOT R1. (This corrects the earlier wrong
  "release R1 = arm stops" claim, which only holds under the bring-up gate.)
- **NEVER auto-launch a training run** — the user starts every learner/actor run.
- **The user does all git.** Leave changes in the working tree.
- The **HTTP backend stays behavior-identical** (`ROBOT_CLIENT="http"`); the FR3 uses
  `ROBOT_CLIENT="ros2"`. The learner builds with `fake_env=True` and constructs NO robot
  client (verified offline).
- `Ros2FrankaClient` is FAIL-CLOSED: `get_state()` AND every control method
  (`update_param`, `joint_reset`, `open/close_gripper`, `close_gripper_slow`, `set_load`)
  raise `NotImplementedError` until wired at the rig (step 3). This is intentional — a
  half-wired backend must NOT move the arm with compliance-switching / recovery / gripper
  as silent no-ops. Implement each and it stops raising. `recover()` is the ONE exception:
  it is a cheap no-op (called every step; must NOT be a per-step blocking action call).
  Order to wire: get_state → update_param (compliance) → gripper → joint_reset → set_load.
- Offline validation already PASSED at home (env build + ResNet-10 load + SACAgent +
  forward pass): `"${SERL_PYTHON}" hil-serl_src/examples/experiments/franka_plumbers/dryrun_learner.py`.
