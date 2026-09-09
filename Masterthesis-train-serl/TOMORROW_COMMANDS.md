# Tomorrow — one-page command sheet (Franka plumbers, insert 0)

Run everything after `source franka_env.sh`. Order matters: **control-first, then teleop,
then camera, then env smoke, then demos, then train.** Reserve the first 60–90 min for a
strict hardware smoke — if it fails, DO NOT launch the learner yet.

## 0. Environment
```bash
cd ~/Masterthesis-rl-train/Masterthesis-train-serl
source franka_env.sh
"${SERL_PYTHON}" -c "import jax, franka_env, serl_launcher; print('env OK')"
```

## 1. Interface discovery (send these to whoever's at the rig if you're remote)
```bash
ros2 topic list
ros2 action list
ros2 interface show franka_msgs/msg/FrankaRobotState
ros2 topic echo <robot_state_topic> --once      # one sample -> confirm wrench/pose/joints
ros2 topic info -v /cartesian_position_controller/commands   # expect Float64MultiArray
```
Fill the confirmed names into `franka_env/envs/robot_client.py` (every `TODO(rig)`), then:

## 2. Preflight self-check (before ANY motion)
```bash
"${SERL_PYTHON}" hil-serl_src/examples/experiments/franka_plumbers/preflight.py
# PASS on ROS topics/actions + PS4 heartbeat + ZED frame. FAIL => fix before proceeding.
```
This screens obvious stubs and interfaces, not backend behavior. Before motion, verify
live state frames/units/freshness, compliance and the physical E-stop using the checklist.

## 3. PS4 test (raw + through our stack)
```bash
ls /dev/input/js*; jstest /dev/input/js0        # confirm axis/button indices
export TELEOP_DEVICE=ps4 SDL_JOYSTICK_DEVICE=/dev/input/js0
"${SERL_PYTHON}" -c "
from franka_env.spacemouse.ps4_expert import PS4Expert
import time
e=PS4Expert()
for _ in range(200):
    print(e.get_action(), e.get_events(), 'R1=',e.deadman_held()); time.sleep(0.05)
"
# Verify: R1 gates motion; axis directions/signs correct; X/Triangle/Circle fire once each.
```

## 4. ZED test
```bash
"${SERL_PYTHON}" -c "import pyzed.sl as sl; c=sl.Camera(); print(c.open(sl.InitParameters()))"
```

## 5. CAPTURE POSES FIRST — **do NOT call env.reset() yet**
> ⚠️ **ORDER MATTERS.** `RESET_POSE` and `ABS_POSE_LIMIT_*` start as **zeros**. `env.reset()`
> commands the arm to `RESET_POSE`, so resetting before you fill them would drive toward the
> base-frame ORIGIN and clip all teleop into a zero-sized safety box. Capture first, paste,
> **then** smoke.
>
> ⚠️ **R1 does NOT gate reset/regrasp motion** — `PS4_DEADMAN_GATES_POLICY` only gates
> `step()`. Reset and regrasp move the arm regardless. Keep a hand on the E-STOP the first
> time you reset, especially while poses are still placeholders.

Read poses WITHOUT constructing the env (no reset, no motion):
```bash
source franka_env.sh
# Jog the arm by hand / with the controller's own teleop, then read the live TCP pose:
ros2 topic echo <robot_state_topic> --once        # o_t_ee -> xyz + quat
```
Capture, for insert 0:
- **RESET_POSE** — pre-insert: socket centered in wrist view, aligned along the approach axis.
- **TARGET_POSE** — part fully seated.
- **ABS_POSE_LIMIT_LOW/HIGH** — physically verified box covering reset+target AND full
  4 cm reset / 8 cm regrasp extraction from expected starting poses (NOT zeros).
  A clipped retract now raises instead of continuing with insufficient clearance.
  Do not blindly widen the box to silence that error; verify the entire extraction path.
- **IMAGE_CROP["wrist"]** — socket + part tip fill the frame.
- **RETRACT_DIR** — unit vector pointing OUT of the socket:
  `normalize(RESET_POSE[:3] - TARGET_POSE[:3])` (pre-insert MINUS seated; the reverse
  points INTO the hole and would drive the part further in).
  Default `(0,0,1)` is correct only for a VERTICAL insert; the **horizontal pipe insert MUST
  override it**, else the retract shoves the part sideways into the socket wall.

Paste all of them into `experiments/franka_plumbers/config.py` → `EnvConfigInsert0`.
Config poses are six values `[x,y,z,roll,pitch,yaw]` (metres, XYZ Euler radians): convert
the measured quaternion before pasting it, rather than copying a seven-element pose.

## 6. Env smoke on the real robot (ONLY after step 5 is pasted in)
```bash
export FRANKA_EXP_NAME=franka_plumbers_insert0 TELEOP_DEVICE=ps4 SDL_JOYSTICK_DEVICE=/dev/input/js0
# BRING-UP: hold-to-move deadman ON so releasing R1 holds the arm during step() while you
# learn the feel. (It does NOT gate reset/regrasp — hand on the E-stop.)
export PS4_DEADMAN_GATES_POLICY=1
"${SERL_PYTHON}" -c "
from experiments.mappings import CONFIG_MAPPING
import numpy as np
cfg = CONFIG_MAPPING['franka_plumbers_insert0']()
assert np.any(cfg.env_config_cls.RESET_POSE), 'RESET_POSE still zeros — do step 5 first!'
assert np.any(cfg.env_config_cls.ABS_POSE_LIMIT_HIGH), 'safety box still zeros — step 5 first!'
env = cfg.get_environment(fake_env=False)
o,_ = env.reset(); print('reset ok', list(o.keys()))
for k in range(30):
    o,r,d,t,i = env.step(np.zeros(6)); print(k, 'rew',r, 'done',d, 'succeed',i.get('succeed'))
    if d:   # HONOR termination (X / Triangle / timeout) — do not keep commanding motion
        print('episode ended ->', 'SUCCESS' if i.get('succeed') else 'no success'); break
"
# Verify: retract along RETRACT_DIR -> RESET_POSE, ~1.5s settle, then loop; X=reward1+end,
# Triangle=0+end, Circle=regrasp at next reset; wrist image + wrench in obs.
```

<details><summary>old inline capture note (superseded by step 5)</summary>

```
#   RESET_POSE, TARGET_POSE, ABS_POSE_LIMIT_LOW/HIGH, IMAGE_CROP["wrist"], RETRACT_DIR
```
</details>

## 7. Record ~20 demos (stock record_demos.py — verified compatible with our wrappers)
```bash
export FRANKA_EXP_NAME=franka_plumbers_insert0 TELEOP_DEVICE=ps4 SDL_JOYSTICK_DEVICE=/dev/input/js0
(
cd hil-serl_src/examples || exit 1
"${SERL_PYTHON}" record_demos.py --exp_name=franka_plumbers_insert0 --successes_needed=20
)
# Teleop the insert (hold R1 + sticks). Press X when SEATED -> that trajectory is banked.
# Triangle/timeout -> discarded, try again. Circle -> regrasp at next reset. Collects until
# 20 successes, saves to examples/demo_data/franka_plumbers_insert0_20_demos_<ts>.pkl
# -> pass that path as --demo_path to run_learner.sh in step 8.
```
Note: only X-marked (successful) trajectories are saved — that's intended (demos = good
examples). intervene_action (your R1+stick corrections) is captured automatically.

## 8. Train — LEARNER first, then ACTOR (turn the deadman gate OFF for training)
Learner terminal (replace the demo filename with the actual absolute path):
```bash
cd ~/Masterthesis-rl-train/Masterthesis-train-serl
export FRANKA_USE_GPU=1
source franka_env.sh
unset PS4_DEADMAN_GATES_POLICY      # training = stock shared autonomy (policy drives)
export FRANKA_EXP_NAME=franka_plumbers_insert0

# Keep existing checkpoints. For a fresh run, choose a new absolute CKPT path and export
# the SAME value in BOTH terminals. Default: examples/checkpoints/<exp>.
"${SERL_PYTHON}" -c "import jax; print(jax.devices()); assert jax.default_backend() == 'gpu', 'GPU unavailable — stop before training'" && \
bash hil-serl_src/examples/experiments/franka_plumbers/run_learner.sh --demo_path "/absolute/path/to/demos.pkl"
```
Wait for "sent initial network to actor", then in the actor terminal:
```bash
cd ~/Masterthesis-rl-train/Masterthesis-train-serl
export FRANKA_USE_GPU=0 JAX_PLATFORMS=cpu
source franka_env.sh
unset PS4_DEADMAN_GATES_POLICY
export FRANKA_EXP_NAME=franka_plumbers_insert0 TELEOP_DEVICE=ps4 SDL_JOYSTICK_DEVICE=/dev/input/js0
bash hil-serl_src/examples/experiments/franka_plumbers/run_actor.sh
```
Operator during training: **R1**=hold to correct with sticks, **X**=success(ends), **Triangle**
=abort, **Circle**=regrasp next reset. Expect the arm to move autonomously after each reset.

## Offline sanity (already green at home)
```bash
"${SERL_PYTHON}" hil-serl_src/examples/experiments/franka_plumbers/dryrun_learner.py
"${SERL_PYTHON}" hil-serl_src/examples/experiments/franka_plumbers/test_robot_client_contract.py
"${SERL_PYTHON}" hil-serl_src/examples/experiments/franka_plumbers/test_demo_recording.py
"${SERL_PYTHON}" hil-serl_src/examples/experiments/franka_plumbers/test_rig_safety.py
```
