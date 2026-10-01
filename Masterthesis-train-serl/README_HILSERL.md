# HIL-SERL on the Franka FR3 — record demos & train a policy

**This is the entry point for the real-robot RL half of the thesis.** It is written so that
someone on a **new machine**, who knows ROS2 and robotics but has never seen this repo, can
go from a bare checkout to a trained insertion policy.

What it produces: a vision-based insertion policy that drives a Franka FR3 to push a part
into a socket, trained **on the real robot** from 20 human demonstrations plus online human
corrections. Four such policies exist in this repo (see
[RESULTS_FRANKA_HILSERL.md](RESULTS_FRANKA_HILSERL.md)); **all four reach 15/15 autonomous**
insertions, each trained in 1–3 hours of robot time.

> **The single most important thing on this page:** you need **two terminals** — a
> **learner** (GPU, trains) and an **actor** (robot, collects) — and they must agree on the
> *same experiment name* and the *same checkpoint directory*. Everything else is detail.

---

## Table of contents

1. [What HIL-SERL is and how this implementation works](#1-what-hil-serl-is)
2. [Repository layout](#2-repository-layout)
3. [Install on a new machine](#3-install-on-a-new-machine)
4. [The runtime shell (`franka_env.sh`) — why it exists](#4-the-runtime-shell)
5. [Hardware bring-up and preflight](#5-hardware-bring-up-and-preflight)
6. [Defining a new task (capturing poses, reset noise)](#6-defining-a-new-task)
7. [Recording demos](#7-recording-demos)
8. [Training](#8-training)
9. [Operating during training — the practical guide](#9-operating-the-pad-during-training)
10. [Monitoring: how to tell if it is working](#10-monitoring)
11. [Evaluating a trained policy](#11-evaluating-a-trained-policy)
12. [Config reference — every knob that matters](#12-config-reference)
13. [Troubleshooting](#13-troubleshooting)
14. [Known limitations](#14-known-limitations)
15. [Session quick reference](#15-session-quick-reference)

---

## 1. What HIL-SERL is

HIL-SERL (Human-in-the-Loop Sample-Efficient Robotic Reinforcement Learning,
[rail-berkeley/hil-serl](https://github.com/rail-berkeley/hil-serl)) is **RLPD** (RL with
Prior Data) running **on a real robot**, with a human holding a game controller who can take
over at any moment.

### The loop

```
                 ┌──────────────── learner (GPU box / same PC) ────────────────┐
                 │  RLPD / SAC  •  50 gradient steps per actor step            │
                 │  samples 50/50 from: demo buffer  ‖  online buffer          │
                 │  saves checkpoint_N every 1000 steps                        │
                 └───────▲───────────────────────────────────┬─────────────────┘
                         │ transitions                       │ network weights
                         │ (over agentlace, TCP :5588)       ▼
                 ┌───────┴──────────────── actor (robot PC) ──────────────────┐
                 │  ZED wrist image + proprio ──► policy ──► 6-DoF delta pose │
                 │  ──► cartesian_impedance_control ──► FR3                   │
                 │                                                            │
                 │  HUMAN holds R1 + sticks  ──► overrides the policy action  │
                 │  HUMAN presses X          ──► reward 1, episode ends       │
                 └────────────────────────────────────────────────────────────┘
```

Three things make this sample-efficient enough to run on real hardware:

| Ingredient | What it does | Where it lives |
|---|---|---|
| **Prior data (RLPD)** | 20 human demos go in a separate buffer; every training batch is half demo, half online. The policy never starts from pure noise. | `train_rlpd.py`, `--demo_path` |
| **Human interventions** | When the policy does something dumb, you hold R1 and steer. Your correction is used as the action for that step and is routed into the **demo** buffer, so it carries the same weight as a demonstration. | `SpacemouseIntervention` wrapper |
| **Human reward** | No learned reward classifier, no engineered reward. You press **X** when the part is seated → reward 1 and the episode ends. Everything else is reward 0. | `PS4RewardWrapper` |

The human is therefore both the **exploration policy** and the **reward function**. This is
why intervention rate is the progress metric: when you stop needing to intervene, the policy
has learned the task.

### What the policy actually sees and does

| | |
|---|---|
| **Observation** | One 128×128 RGB wrist image (ZED Mini, left eye) + proprioception: `tcp_pose`, `tcp_vel`, `tcp_force`, `tcp_torque`, `gripper_pose` |
| **Image encoder** | **Frozen** pretrained ResNet-10 (`resnet10_params.pkl`). Only the MLP head trains. |
| **Action** | 6-DoF **relative** delta pose (Δx Δy Δz Δroll Δpitch Δyaw), scaled by `ACTION_SCALE`, at ~10 Hz. The gripper is **not** in the action space — the part is pre-grasped. |
| **Frame** | `RelativeFrame` wrapper: everything is expressed relative to the reset pose, so absolute table position is invisible to the state branch. |
| **Episode** | Ends on **X** (success, reward 1), **Triangle** (abort, reward 0), or `MAX_EPISODE_LENGTH` steps (reward 0). |

### This implementation's deviations from upstream

This is a vendored copy of upstream HIL-SERL with four targeted changes:

1. **ROS2 backend.** Upstream talks to a Flask server (`franka_server.py`) over HTTP. Here a
   `RobotClient` seam ([robot_client.py](hil-serl_src/serl_robot_infra/franka_env/envs/robot_client.py))
   lets the identical env drive **either** the stock HTTP server (`ROBOT_CLIENT="http"`,
   byte-identical to upstream) **or** `cartesian_impedance_control` over `franka_ros2`
   (`ROBOT_CLIENT="ros2"`, what we use). Nothing else in the stack knows the difference.
2. **PS4 pad instead of a SpaceMouse.** `PS4Expert` is a drop-in for `SpacemouseExpert`,
   selected by `TELEOP_DEVICE=ps4`.
3. **ZED Mini instead of a RealSense.** `ZEDCapture` is drop-in compatible with `RSCapture`,
   selected by `CAMERA_TYPE="zed"`.
4. **Human reward instead of a classifier.** No `train_reward_classifier.py` step.

---

## 2. Repository layout

```
Masterthesis-train-serl/
├── README_HILSERL.md          ← you are here
├── franka_env.sh              ← the runtime shell. ALWAYS source this first.
├── RESULTS_FRANKA_HILSERL.md  ← per-insert training cost, convergence shapes
│
├── hil-serl_src/              ← vendored HIL-SERL (edit ONLY here)
│   ├── serl_launcher/         ← the RL side: RLPD/SAC agent, encoders, replay buffer
│   ├── serl_robot_infra/
│   │   └── franka_env/
│   │       ├── envs/
│   │       │   ├── franka_env.py     ← the gym env: step/reset, safety box, guards
│   │       │   ├── robot_client.py   ← HTTP ‖ ROS2 backend seam
│   │       │   ├── relative_env.py   ← reset-relative frame wrapper
│   │       │   └── wrappers.py       ← intervention + PS4 reward wrappers
│   │       ├── camera/zed_capture.py
│   │       └── spacemouse/ps4_expert.py
│   └── examples/
│       ├── train_rlpd.py             ← THE training entry point (learner AND actor)
│       ├── record_demos.py           ← THE demo recording entry point
│       ├── experiments/
│       │   ├── mappings.py           ← exp_name → config class registry
│       │   └── franka_plumbers/      ← OUR task package
│       │       ├── config.py         ← poses, safety box, guards, hyperparameters
│       │       ├── wrapper.py        ← reset / retract / regrasp behaviour
│       │       ├── run_learner.sh    ← launcher: learner
│       │       ├── run_actor.sh      ← launcher: actor
│       │       ├── jog_and_capture.py← pose capture tool (no env, no placeholders)
│       │       ├── preflight.py      ← pre-motion self-check
│       │       └── gripper.py        ← open/close the hand from the CLI
│       ├── demo_data/         ← recorded demo .pkl files land here
│       └── checkpoints/       ← default checkpoint root
│
├── demo_data/                 ← the demos actually used for the trained policies
└── checkpoints/               ← the trained policies (large; see .gitignore)
```

**Edit `hil-serl_src/` only.** The sibling trees (`serl_launcher/`, `serl_robot_infra/`,
`examples/`, `iiwa_serl/`) are an older KUKA-era SERL checkout kept for reference. They are
not what `franka_env.sh` puts on the path.

---

## 3. Install on a new machine

### 3.1 Clone the repo (including the submodule)

**Install Git LFS first** — the trained policy checkpoints are stored with it, and without
LFS the clone either fails or leaves you with pointer stubs instead of weights:

```bash
sudo apt install git-lfs && git lfs install
git clone --recurse-submodules <this-repo-url> Masterthesis-rl-train
cd Masterthesis-rl-train
```

> Only the **checkpoints** need LFS. To skip the multi-GB download and train from scratch
> yourself, clone with `GIT_LFS_SKIP_SMUDGE=1 git clone ...` — everything in this guide works
> without them; you just cannot evaluate the pre-trained policies (§11).

Already cloned without `--recurse-submodules`?

```bash
git submodule update --init
```

There is **one** submodule: `ps4_controller_teleop/` — a third-party ROS2 PS4 teleop package
([avanetta/ps4_controller_teleop](https://github.com/avanetta/ps4_controller_teleop)).

> ### ⚠️ The submodule fetch will fail for most people — and that is OK
>
> `avanetta/ps4_controller_teleop` is a **private** GitHub repository belonging to another
> student. Unless your GitHub account has been granted access, the submodule step fails with:
>
> ```
> fatal: could not read Username for 'https://github.com'
> fatal: clone of '.../ps4_controller_teleop.git' into submodule path '...' failed
> ```
>
> **This does not block you. You do not need this package to record demos or train.** The
> HIL-SERL stack reads the PS4 pad in-process via `ps4_expert.py` and never touches it
> (§3.5). Continue the install; `ps4_controller_teleop/` stays an empty directory.
>
> To clone the rest of the repo without the submodule stopping you, just omit
> `--recurse-submodules`. If you *do* need the package, ask the author for access (or for a
> tarball) — then `git submodule update --init`.

### 3.2 What the machine needs

| | Requirement | Note |
|---|---|---|
| OS | Ubuntu 22.04 | |
| ROS | ROS2 **Humble** | `rclpy` is built against **system** python3.10 — this matters, see §4 |
| Kernel | **RT kernel** (`PREEMPT_RT`) | required by `libfranka` / FCI |
| GPU | NVIDIA, CUDA 12 | needed by the learner (JAX) **and** by the ZED SDK |
| Python | system **3.10** + Miniconda | both; see §4 |
| Robot | Franka FR3 with **FCI unlocked** | static net `172.16.0.x`, Desk → unlock joints → Activate FCI |
| Camera | ZED Mini + **ZED SDK** with python bindings (`pyzed`) | CUDA-only |
| Input | PS4 DualShock 4 controller | USB or BT |

### 3.3 Build the Franka control stack

The policy does **not** command joints — it commands a Cartesian reference pose to a
**compliant impedance controller**, which is also the safety layer. Build it first:

```bash
sudo apt install ros-humble-desktop ros-dev-tools python3-vcstool
mkdir -p ~/fr3_ws/src && cd ~/fr3_ws

# franka_ros2 (this rig ran v2.7.1 with libfranka 0.20.4)
git clone https://github.com/frankarobotics/franka_ros2 src/franka_ros2
vcs import src --recursive --input src/franka_ros2/franka.repos

# the compliant Cartesian controller + its custom messages
git clone https://github.com/CurdinDeplazes/cartesian_impedance_control src/cartesian_impedance_control
git clone https://github.com/CurdinDeplazes/messages_fr3 src/messages_fr3

rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

`franka_env.sh` auto-sources `~/fr3_ws/install/setup.bash`. If you build elsewhere, export
`FRANKA_ROS2_WS=/your/path` before sourcing.

> **Stiffness is compile-time.** `cartesian_impedance_control` does **not** expose stiffness
> as ROS parameters — the gains are constants in its header. `COMPLIANCE_PARAM` /
> `PRECISION_PARAM` in our config are therefore **inert on the ROS2 path**. To change
> stiffness you must edit the controller and rebuild. The gains this rig ran:
> **K = 1500 translational, 100/100/40 rotational.**

### 3.4 Build the Python environment

JAX/flax live in a **conda env**; ROS2's `rclpy` lives in **system python3.10**. §4 explains
why both are needed at once. Create the conda env:

```bash
cd ~/Masterthesis-rl-train/Masterthesis-train-serl

conda create -n serl python=3.10 -y
conda activate serl

pip install -e hil-serl_src/serl_launcher
pip install -r hil-serl_src/serl_launcher/requirements.txt

# PIN the JAX ecosystem — upstream's >= pins resolve to JAX 0.6 / flax 0.10,
# which break serl_launcher (jax.tree_map was removed in 0.6).
pip install -r iiwa_serl/requirements-serl-cpu.txt

pip install gymnasium pygame

# CRITICAL: serl_launcher's setup.py pulls a bogus `typing` backport that SHADOWS the
# stdlib. Harmless inside conda, FATAL under system python via PYTHONPATH (our pattern).
pip uninstall -y typing
```

Then swap JAX for the **GPU** build (the learner needs it):

```bash
pip install "jax[cuda12]==0.4.35"
```

The verified working set (`pip list`): `jax 0.4.35`, `jaxlib 0.4.35`, `flax 0.8.5`,
`optax 0.2.3`, `orbax-checkpoint 0.5.20`, `numpy 1.26.4`, `tensorflow 2.15.1`,
`protobuf 4.25.9`.

If your conda env is not at `/home/<you>/miniconda3/envs/serl`, export
`SERL_PY_SITE=/path/to/envs/serl/lib/python3.10/site-packages` before sourcing
`franka_env.sh`.

### 3.5 The `ps4_controller_teleop` submodule — do you need it?

**For HIL-SERL: no.** This is the easiest thing to get confused by, so to be explicit:

| | |
|---|---|
| **What it is** | A standalone **ROS2 package** that teleoperates the Franka with a PS4 pad, publishing `geometry_msgs/Twist` → an integrating `cartesian_bridge` → the impedance controller. Written by a fellow student ([avanetta/ps4_controller_teleop](https://github.com/avanetta/ps4_controller_teleop)), vendored here as a git submodule. It also drives the pad's **rumble motors** as haptic force feedback. |
| **What HIL-SERL actually uses** | **Not this package.** The training stack reads the pad *in-process* through [`ps4_expert.py`](hil-serl_src/serl_robot_infra/franka_env/spacemouse/ps4_expert.py), a `SpacemouseExpert` drop-in that talks to pygame directly and returns a 6-vec action. No ROS2 nodes, no topics. |
| **Why it is still here** | It is the **reference implementation** `ps4_expert.py` was written against (same controller, same conventions), it is a useful **standalone teleop** for jogging the arm outside the RL loop, and the repo adds two local files for the KUKA-era SERL HTTP path (`iiwa_serl_bridge.py`, `config/iiwa_serl_config.yaml`). |

So you can train without ever building it — and since the upstream repo is **private**
(§3.1), most people will not be able to fetch it at all. Build it only if you have access
*and* want ROS2-native teleop or the rumble feedback:

```bash
cp -r ps4_controller_teleop ~/fr3_ws/src/ && cd ~/fr3_ws && colcon build --symlink-install
```

> **Note on the two local files.** `ps4_controller_teleop/ps4_controller_teleop/iiwa_serl_bridge.py`
> and `config/iiwa_serl_config.yaml` are **our** additions, untracked inside the submodule
> (they belong to the older KUKA/SERL-HTTP path, not the Franka one). A submodule only
> tracks the upstream commit, so these files are **not carried by a clone** — if you need
> them, copy them in by hand. They are not required for Franka HIL-SERL.

### 3.6 The pretrained encoder (downloads itself)

The policy's vision encoder is a **frozen pretrained ResNet-10**. The weights
(`resnet10_params.pkl`, ~21 MB) are **not in this repo** — they are fetched automatically on
first use from the upstream SERL release and cached in `~/.serl/`:

```
Downloading file from https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl
Download complete!
```

Nothing to do — but the **first** learner start (or `dryrun_learner.py`) needs network access.
On an offline rig, pre-seed the cache by copying the file to `~/.serl/resnet10_params.pkl`,
or fetch it manually:

```bash
mkdir -p ~/.serl && curl -L -o ~/.serl/resnet10_params.pkl \
  https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl
```

### 3.7 Verify the install (no robot needed)

```bash
cd ~/Masterthesis-rl-train/Masterthesis-train-serl
source franka_env.sh

"${SERL_PYTHON}" -c "import jax, franka_env, serl_launcher; print('env OK')"
"${SERL_PYTHON}" hil-serl_src/examples/experiments/franka_plumbers/dryrun_learner.py
```

`dryrun_learner.py` builds the env with `fake_env=True`, loads the ResNet-10 encoder,
constructs the SAC agent and runs a forward pass. **If this passes, the whole learner path
works** — no robot, no camera, no ROS2 required. Run it before you ever go to the rig.

Three more offline tests exist and should all pass:

```bash
"${SERL_PYTHON}" hil-serl_src/examples/experiments/franka_plumbers/test_robot_client_contract.py
"${SERL_PYTHON}" hil-serl_src/examples/experiments/franka_plumbers/test_demo_recording.py
"${SERL_PYTHON}" hil-serl_src/examples/experiments/franka_plumbers/test_rig_safety.py
```

---

## 4. The runtime shell

**Every command in this README starts with `source franka_env.sh`.** Here is why it exists,
because it is the single most confusing part of the setup.

`rclpy` is compiled against **system** python3.10 and only imports under that interpreter.
JAX/flax/serl_launcher are installed in the **conda** `serl` env. The actor needs both in one
process. So instead of activating conda, `franka_env.sh`:

1. sources `/opt/ros/humble/setup.bash` and `~/fr3_ws/install/setup.bash`;
2. prepends the vendored source dirs **and** the conda env's `site-packages` to `PYTHONPATH`;
3. pins `SERL_PYTHON=/usr/bin/python3.10` and validates it;
4. defaults JAX to **CPU** (`JAX_PLATFORMS=cpu`) unless you ask for GPU.

System py3.10 and conda py3.10 are both `cp310`, so the wheels load fine.

> **Never run a bare `python`/`python3` in this project.** Always `"${SERL_PYTHON}"`. An
> active conda env on `PATH` will hijack a bare `python3` and break `rclpy`.
>
> **GPU:** `export FRANKA_USE_GPU=1` **before** sourcing. Do *not* try
> `export JAX_PLATFORMS=` — an empty value is treated as unset and silently falls back to
> CPU, which costs you a whole training run.

Also on the path: `python_compat/`, a shim that pushes conda's obsolete `typing.py` behind
the stdlib (the counterpart to the `pip uninstall -y typing` above).

---

## 5. Hardware bring-up and preflight

> ### ⚠️ Safety
> - The **E-stop** and the **compliant controller** are the real safety layer — not any
>   software gate.
> - **R1 does NOT gate reset or regrasp motion.** `PS4_DEADMAN_GATES_POLICY` only gates
>   `step()`. Reset moves the arm regardless. **Hand on the E-stop for the first reset of
>   every session.**
> - Never call `env.reset()` with placeholder (zero) poses — it will command the arm toward
>   the base-frame **origin**. Capture poses first (§6).

### 5.1 Start the controller

```bash
source /opt/ros/humble/setup.bash
source ~/fr3_ws/install/setup.bash
ros2 launch cartesian_impedance_control cartesian_impedance_controller.launch.py
```

**Push the arm by hand.** It must yield. If it is stiff, stop and fix the controller before
going further.

### 5.2 Home the gripper — EVERY SESSION

```bash
ros2 run franka_gripper franka_gripper_node --ros-args -p robot_ip:=172.16.0.2
# then, in the Desk UI or via the homing action, run a gripper HOMING
```

> **This is not optional.** A drifted gripper zero reports a fully-open hand as width
> `0.434`, which looks exactly like a grip-force problem and will send you chasing a
> non-existent bug. Home the hand at the start of every session.

Check it any time with:

```bash
source franka_env.sh
"${SERL_PYTHON}" hil-serl_src/examples/experiments/franka_plumbers/gripper.py status
"${SERL_PYTHON}" hil-serl_src/examples/experiments/franka_plumbers/gripper.py open
"${SERL_PYTHON}" hil-serl_src/examples/experiments/franka_plumbers/gripper.py close
```

### 5.3 Confirm the ROS2 interfaces

The backend expects these names ([robot_client.py:217](hil-serl_src/serl_robot_infra/franka_env/envs/robot_client.py#L217)):

| Purpose | Name | Type |
|---|---|---|
| Command pose | `/cartesian_impedance_controller/reference_pose` | `geometry_msgs/Pose` |
| Robot state | `/franka_robot_state_broadcaster/robot_state` | `franka_msgs/FrankaRobotState` |
| Controller node | `/cartesian_impedance_controller` | |
| Gripper move | `/franka_gripper/move` | `franka_msgs/action/Move` |
| Gripper grasp | `/franka_gripper/grasp` | `franka_msgs/action/Grasp` |
| Error recovery | `/action_server/error_recovery` | |

```bash
ros2 topic list; ros2 action list
ros2 topic echo /franka_robot_state_broadcaster/robot_state --once
```

If your stack uses different names, edit the class constants in `robot_client.py`.

### 5.4 PS4 controller

```bash
ls /dev/input/js*
jstest /dev/input/js0        # note the axis and button indices
```

Default SDL2 DualShock 4 mapping, each overridable by an env var — **no code edit needed**:

```bash
export SDL_JOYSTICK_DEVICE=/dev/input/js0
export PS4_AXIS_LEFT_X=0 PS4_AXIS_LEFT_Y=1 PS4_AXIS_L2=2 \
       PS4_AXIS_RIGHT_X=3 PS4_AXIS_RIGHT_Y=4 PS4_AXIS_R2=5 \
       PS4_BTN_ENABLE=5 PS4_BTN_SUCCESS=0 PS4_BTN_ABORT=2 \
       PS4_BTN_REGRASP=1 PS4_BTN_CLOSE=3
# other knobs: PS4_DEADZONE (0.15), PS4_READ_HZ (100), PS4_OPERATOR_FRONT (1)
```

Test the pad through our stack (not just `jstest`):

```bash
source franka_env.sh
export TELEOP_DEVICE=ps4 SDL_JOYSTICK_DEVICE=/dev/input/js0
"${SERL_PYTHON}" -c "
from franka_env.spacemouse.ps4_expert import PS4Expert
import numpy as np, time
e = PS4Expert()
for _ in range(200):
    a, btns = e.get_action()
    print(np.round(a,2), btns, e.get_events(), 'R1=', e.deadman_held()); time.sleep(0.05)
"
```

Verify: output is **zero unless R1 is held**; each axis moves the right DoF in the right
direction; X / Triangle / Circle each fire **exactly once** per press.

> ### ⚠️ Squeeze L2 and R2 fully, once, at the start of every session
>
> **Before any Z motion will work, pull both analog triggers all the way in and release
> them.** Do this every time a PS4 reader starts — that means before demos, before the
> actor, before `jog_and_capture.py`.
>
> SDL reports `0.0` for an analog trigger that has not been moved since the device was
> opened, but the true resting value is `-1.0`. So an untouched trigger reads as **half
> pressed**, and the operator sees *"L2 moves down, then the arm springs back up"* — a
> phantom +Z command from the trigger you never touched.
>
> The fix is a readiness latch: Z motion is **suppressed entirely** until **both** triggers
> have been observed at their true rest value at least once. Squeezing each one and letting
> it go is what arms it. Check it directly:
>
> ```bash
> "${SERL_PYTHON}" -c "
> from franka_env.spacemouse.ps4_expert import PS4Expert
> import time
> e = PS4Expert()
> while not e.triggers_ready():
>     print('squeeze L2 and R2 fully, then release...'); time.sleep(0.5)
> print('triggers ready — Z is armed')
> "
> ```
>
> So: **if Z does nothing, you have not squeezed the triggers yet.** That is the latch doing
> its job, not a broken pad. (Regression test:
> `test_trigger_init.py`.)

### 5.5 ZED Mini

```bash
"${SERL_PYTHON}" -c "import pyzed.sl as sl; print(sl.Camera().open(sl.InitParameters()))"
```

### 5.6 Preflight

```bash
source franka_env.sh
"${SERL_PYTHON}" hil-serl_src/examples/experiments/franka_plumbers/preflight.py
```

Prints PASS/FAIL for ROS topics & actions, the PS4 heartbeat, and a ZED frame. It commands
no motion. **Do not proceed to §6 until it is green.**

---

## 6. Defining a new task

Each insert is its own config class and its own policy. **One model per insert** — they are
not interchangeable.

### 6.1 Register the experiment

Add a config pair in [config.py](hil-serl_src/examples/experiments/franka_plumbers/config.py)
— an `EnvConfigInsertN(FrankaPlumbersConfigBase)` holding the poses, and a
`TrainConfigInsertN(FrankaPlumbersTrainBase)` pointing at it:

```python
class EnvConfigInsert4(FrankaPlumbersConfigBase):
    RESET_POSE = np.array([...])     # filled in §6.2
    TARGET_POSE = np.array([...])
    RETRACT_DIR = (0.0, 0.0, 1.0)
    ABS_POSE_LIMIT_LOW = np.array([...])
    ABS_POSE_LIMIT_HIGH = np.array([...])
    LOCK_ORIENTATION = True
    RANDOM_RESET = True

class TrainConfigInsert4(FrankaPlumbersTrainBase):
    env_config_cls = EnvConfigInsert4
```

Then register the name in
[mappings.py](hil-serl_src/examples/experiments/mappings.py):

```python
"franka_plumbers_insert4": _FrankaPlumbers4,
```

That string is what you pass as `--exp_name` / `FRANKA_EXP_NAME` everywhere.

### 6.2 Capture the poses — measure them, never type them

You need two poses per insert: the **goal** (part seated in the hole) and the **reset**
(a few cm back along the approach axis). The reliable way to get them is to **put the part in
the hole by hand and read the pose off the robot** — never to type numbers in.

#### Method A (recommended): hand-guide the arm, then read the pose

The FR3 has a guiding mode: press and hold the **two buttons on the robot's wrist** (the
enabling/guiding grips) and the arm goes gravity-compensated so you can **push it around by
hand**. Alternatively put it in guiding mode from the Desk UI. Then:

1. **Grasp the part** in the Franka Hand (close the gripper on it — §5.2).
2. **Hand-guide the arm until the part is fully seated in the hole.** Push it until you feel
   real contact, not hovering — insert 0's goal was captured with ~9 N of contact force.
3. **Read the pose back** without touching the env:
   ```bash
   source franka_env.sh
   "${SERL_PYTHON}" hil-serl_src/examples/experiments/franka_plumbers/read_pose.py
   ```
   or straight off the topic:
   ```bash
   ros2 topic echo /franka_robot_state_broadcaster/robot_state --once
   ```
   That is your **`TARGET_POSE`**.
4. **Derive `RESET_POSE` from it** — do not capture it separately by hand. Take the goal pose
   and **back off a few centimetres along the approach axis only**, keeping x/y and the
   orientation identical:

   | Insert | Goal → reset |
   |---|---|
   | vertical −Z (inserts 0, 2, 3) | **z + 6…10 cm** (insert 0 used +10.3 cm, inserts 2/3 +6.5 cm) |
   | horizontal −Y (insert 1) | **y + 10 cm**, z and x unchanged |

   Why derive rather than capture: it makes the nominal approach a **pure single-axis
   translation** with no lateral component, so the policy only has to learn the correction,
   not the whole trajectory. Captured-by-hand reset poses always have a few mm of lateral
   offset baked in.
5. Sanity-check the reset pose by eye: the socket should be **centered in the wrist camera
   view** and the part should be clearly clear of the hole.

> **How far back?** Far enough that the part is clear and the socket is in frame; close
> enough that the policy can reach the hole inside `MAX_EPISODE_LENGTH`. 6–10 cm worked for
> every insert here. Further back means more wandering to learn.

#### Method B: jog with the pad

`jog_and_capture.py` talks **straight to the robot client** (`get_state` + `send_pos`) and
never constructs the env, so it cannot command a placeholder pose. Motion only while R1
is held; per-step motion is clamped to 1 mm / 0.17° and the target can never run ahead of the
measured TCP. Use it when hand-guiding is awkward (tight clearance, or you want fine
sub-millimetre adjustment).

```bash
source /opt/ros/humble/setup.bash && source ~/fr3_ws/install/setup.bash
source franka_env.sh
export TELEOP_DEVICE=ps4 SDL_JOYSTICK_DEVICE=/dev/input/js0
"${SERL_PYTHON}" hil-serl_src/examples/experiments/franka_plumbers/jog_and_capture.py
```

| Button | Captures |
|---|---|
| **X** | current pose as `RESET_POSE` (pre-insert) |
| **Triangle** | current pose as `TARGET_POSE` (part fully seated) |
| **Circle** | print the config block for everything captured so far |

Paste the printed block into your `EnvConfigInsertN`.

**What to capture:**

- **`TARGET_POSE`** — part **fully seated**. Push until you see real contact force (~9 N on
  insert 0), so it is genuinely seated and not hovering.
- **`RESET_POSE`** — the fixed pre-insert pose, **derived** from `TARGET_POSE` by backing off
  along the approach axis (step 4 above), not captured separately.
- **`ABS_POSE_LIMIT_LOW` / `HIGH`** — the safety box. **Six** values:
  `[x, y, z, roll, pitch, yaw]`. `franka_env` slices `[:3]` for the xyz box and `[3:]` for
  the rpy box — a 3-element array leaves the rpy box empty and `clip_safety_box` throws
  `IndexError` on the first step.
  The box must cover reset **and** target **and** the full **4 cm reset / 8 cm regrasp
  retraction**, plus room for human corrections. Insert 0 settled on **±15 cm in xy** after
  a tighter box silently clipped the operator's teleop mid-insert.
  **Verify the whole box is physically clear by hand before the first reset.**
- **`RETRACT_DIR`** — unit vector pointing **OUT** of the socket:
  `normalize(RESET_POSE[:3] - TARGET_POSE[:3])` — pre-insert **minus** seated.
  The reverse points *into* the hole. The default `(0,0,1)` is correct **only for a vertical
  insert**; a horizontal insert (insert 1 uses `(0,-1,0)`) **must** override it or the
  pre-reset retract shoves the part sideways into the socket wall.
- **`IMAGE_CROP["wrist"]`** — a lambda cropping the frame so the socket and part tip fill it.

> Poses are **six values `[x, y, z, roll, pitch, yaw]`** in metres and XYZ-Euler **radians**.
> If you read a pose off a ROS topic you will get a quaternion — convert it. Pasting a
> 7-element pose will not work.

### 6.3 How the reset noise works

Every episode starts at `RESET_POSE` **plus a random offset**, so the policy never learns one
single starting point. This is what makes the policy tolerant of the part sitting slightly
differently in the gripper each time.

```
                 RANDOM_XY_RANGE = 0.01
            ◄──────────── ±1 cm ────────────►
            ┌───────────────────────────────┐
            │  ·     ·        ·      ·      │   each episode: one uniform
            │      ·      ✛       ·    ·    │   sample from this square
            │   ·      ·     ·        ·     │   ✛ = RESET_POSE
            │        ·    ·       ·         │
            └───────────────────────────────┘
                              │
                              │  pure −Z (or −Y) approach
                              ▼
                        ╔═══════════╗
                        ║   socket  ║
                        ╚═══════════╝
```

What is sampled, per episode, in `go_to_reset()`
([wrapper.py](hil-serl_src/examples/experiments/franka_plumbers/wrapper.py)):

| Knob | Value used | What it does |
|---|---|---|
| `RANDOM_RESET` | `True` | Master switch. `False` → every episode starts at exactly `RESET_POSE`. |
| `RANDOM_XY_RANGE` | `0.01` | Uniform ±1 cm added to **x and y** independently |
| `RANDOM_Z_RANGE` | `0.0` | Uniform ±N added to **z**; off by default. When set, the result is **clipped to the safety-box floor** so jitter can never start you below the box. |
| `RANDOM_RZ_RANGE` | `0.0` | Uniform ±N added to **yaw** only |

Three things to understand about it:

1. **It is translation-only by design.** `RANDOM_RZ_RANGE = 0.0` is deliberate, not an
   oversight — **rotational variation comes for free from the human**, because you hand the
   part into the gripper slightly differently on every regrasp. You get orientation diversity
   without the reset ever commanding a rotation.

2. **Turn it on from the very first demo.** `RANDOM_RESET = True` *while recording*, so the
   20 demos already cover the ±1 cm of start variation the policy will meet during training.
   If you record all demos from one exact pose and then enable noise for training, the demo
   buffer does not match what the policy sees and you have made the problem harder for no
   reason.

3. **±1 cm is the tested value.** It is big enough that the policy cannot just memorise one
   trajectory, small enough that the socket stays in the wrist camera's view. Much more and
   the hole starts leaving the frame; the policy then has to search before it can align,
   which costs training time.

> **Reset noise is not the same as fixture tolerance.** ±1 cm of *start pose* jitter does not
> make the policy robust to the *fixture* moving — see §14, where a 276 mm fixture move left
> the policy completely lost zero-shot.

### 6.4 Smoke the env on the real robot

**Only after the poses are pasted in.**

```bash
source franka_env.sh
export FRANKA_EXP_NAME=franka_plumbers_insert4
export TELEOP_DEVICE=ps4 SDL_JOYSTICK_DEVICE=/dev/input/js0
export PS4_DEADMAN_GATES_POLICY=1       # BRING-UP ONLY: hold-to-move

"${SERL_PYTHON}" -c "
from experiments.mappings import CONFIG_MAPPING
import numpy as np
cfg = CONFIG_MAPPING['franka_plumbers_insert4']()
assert np.any(cfg.env_config_cls.RESET_POSE), 'RESET_POSE still zeros — do §6.2 first!'
assert np.any(cfg.env_config_cls.ABS_POSE_LIMIT_HIGH), 'safety box still zeros!'
env = cfg.get_environment(fake_env=False)
o,_ = env.reset(); print('reset ok', list(o.keys()))
for k in range(30):
    o,r,d,t,i = env.step(np.zeros(6))
    print(k, 'rew', r, 'done', d, 'succeed', i.get('succeed'))
    if d:
        print('episode ended ->', 'SUCCESS' if i.get('succeed') else 'no success'); break
"
```

Verify: the arm retracts along `RETRACT_DIR`, interpolates to `RESET_POSE`, waits ~1.5 s;
teleop moves it under R1; **X** ends the episode with reward 1; **Triangle** ends with 0;
the wrist image and wrench appear in the obs keys.

---

## 7. Recording demos

**20 successful demos is the standard** — that is what every trained policy in this repo used.

```bash
cd ~/Masterthesis-rl-train/Masterthesis-train-serl
source franka_env.sh
export FRANKA_EXP_NAME=franka_plumbers_insert4
export TELEOP_DEVICE=ps4 SDL_JOYSTICK_DEVICE=/dev/input/js0

cd hil-serl_src/examples
"${SERL_PYTHON}" record_demos.py --exp_name=franka_plumbers_insert4 --successes_needed=20
```

**What happens.** `record_demos.py` steps the env with a **zero** action every tick. The
intervention wrapper replaces it with *your* R1+stick action and writes it into
`info["intervene_action"]`, which is what gets recorded. So you are simply teleoperating; the
recorder banks what you do.

> ### ⚠️ You must HOLD R1 the entire time you are moving the robot
>
> **R1 is the deadman. Nothing moves unless it is held.** During demo recording there is no
> policy — *you* are the only thing driving the arm — so if you let go of R1 the arm just
> stops and your sticks do nothing. The most common first-session confusion is "the pad is
> broken"; it is almost always a released R1 or uncalibrated triggers (§5.4).
>
> So the grip is: **R1 held down with your right index finger for the whole episode**, sticks
> with your thumbs, L2/R2 for Z. Only release R1 when you want the arm to stop.

**Per episode:**

1. The arm resets to `RESET_POSE` (± the reset noise) and waits ~1.5 s.
2. **Hold R1** and teleop the part into the socket.
3. Press **X** the moment it is seated → the trajectory is **banked** and counts toward 20.
4. **Triangle** or a timeout → the trajectory is **discarded**; try again.
5. Press **Circle** if you need to re-grasp. At the next reset the arm retracts 8 cm, opens
   the hand, and prompts you at the terminal (`Press enter…`) to place the part and close.

Only X-marked trajectories are saved.

### Your demos do NOT need to be clean

This matters more than anything else about recording, because it is where people waste hours:

> **You do not need smooth, expert teleoperation.** Wobbly, hesitant, all-over-the-place demos
> **still work**. If you overshoot, back up and come in again — that is a fine demo. If you
> scrub around the hole for a while before it drops in, that is a fine demo. Bank it and move
> on.

What actually matters is that the episode **ends in a real insertion** that you marked with X.
A messy successful demo is worth far more than a clean one you abandoned.

The trade-off is **time, not feasibility**: clean, tight, direct demos are genuinely
*better* — they converge substantially faster (see below) — but messy demos converge too, just
slower. So:

- **Do** aim for direct motions and re-record a demo that went truly wild.
- **Do not** re-record a demo because it was a bit shaky, and do not spend a session practising
  teleop before you start. Just record 20 and start training.

### Demo quality drives training cost

**Output:**

```
hil-serl_src/examples/demo_data/franka_plumbers_insert4_20_demos_<timestamp>.pkl
```

Note that **absolute path** — you pass it to the learner.

> **The original demo files are not in the repo.** They are 130–440 MB each (~709 MB total)
> and are gitignored, so `demo_data/` will be empty on a fresh clone. This is not a problem:
> **you record your own**, and you have to anyway — demos are tied to *your* fixture position,
> *your* camera mount and *your* captured poses, so another rig's demos would not transfer.
> Ask for them only if you want to inspect what a demo set looks like.

The headline finding of this work
([RESULTS_FRANKA_HILSERL.md](RESULTS_FRANKA_HILSERL.md)) is that **mean demo episode length
predicts training cost**:

| | insert 1 | insert 2 | insert 3 |
|---|---|---|---|
| demo episode length | 230 steps | 67 steps | 77 steps |
| online transitions to converge | 44 500 | 11 500 | 9 001 |
| learner steps | 192 000 | 67 000 | 50 000 |

So: **record tight, direct demos.** Do not wander. If your demos average 200+ steps, expect
roughly 4× the robot time of a task you can demonstrate in 70. Check your average before
training:

```bash
"${SERL_PYTHON}" -c "
import pickle, sys
t = pickle.load(open(sys.argv[1],'rb'))
n = sum(x['dones'] for x in t)
print(f'{len(t)} transitions, {n} episodes, mean length {len(t)/max(n,1):.0f} steps')
" demo_data/franka_plumbers_insert4_20_demos_<ts>.pkl
```

One more decision worth copying: turn `RANDOM_RESET = True` **from the first demo**, so the
demos already cover the ±1 cm of start variation the policy will meet at training time.

---

## 8. Training

Two terminals. **Learner first**, then actor.

> **The user launches training runs.** Never auto-start one.

### Terminal 1 — learner (GPU)

```bash
cd ~/Masterthesis-rl-train/Masterthesis-train-serl
export FRANKA_USE_GPU=1                 # BEFORE sourcing
source franka_env.sh
unset PS4_DEADMAN_GATES_POLICY          # training = shared autonomy; policy must drive
export FRANKA_EXP_NAME=franka_plumbers_insert4

# verify the GPU is actually there — do not burn a run on CPU
"${SERL_PYTHON}" -c "import jax; print(jax.devices()); assert jax.default_backend()=='gpu'"

bash hil-serl_src/examples/experiments/franka_plumbers/run_learner.sh \
     --demo_path "/absolute/path/to/franka_plumbers_insert4_20_demos_<ts>.pkl"
```

Wait for **`sent initial network to actor`** before starting the actor.

### Terminal 2 — actor (robot)

```bash
cd ~/Masterthesis-rl-train/Masterthesis-train-serl
export FRANKA_USE_GPU=0
source franka_env.sh
unset PS4_DEADMAN_GATES_POLICY
export FRANKA_EXP_NAME=franka_plumbers_insert4
export TELEOP_DEVICE=ps4 SDL_JOYSTICK_DEVICE=/dev/input/js0

bash hil-serl_src/examples/experiments/franka_plumbers/run_actor.sh
```

The arm will now **move on its own** after each reset. That is correct and necessary — a
policy that cannot act cannot learn.

### What the launchers do

Both resolve `train_rlpd.py` relative to themselves and anchor the checkpoint dir to
`examples/checkpoints/$FRANKA_EXP_NAME` (**absolute**, independent of your cwd), so learner
and actor always agree. The learner builds the env with `fake_env=True` — **no robot, no
camera, no ROS2** — so it can run on a different machine. The actor keeps JAX on **CPU**
(`JAX_PLATFORMS=cpu`) while leaving CUDA visible, because the ZED SDK needs the GPU and
`CUDA_VISIBLE_DEVICES=""` makes `Camera.open()` fail with "NO GPU DETECTED".

### Fresh run vs. resume

Orbax **refuses to overwrite** an existing checkpoint. The learner also auto-resumes the
replay buffer from `<ckpt>/buffer/` and `<ckpt>/demo_buffer/`.

- **Resume:** just relaunch both with the same `CKPT`.
- **Fresh run:** export a **new absolute `CKPT`** in **both** terminals:
  ```bash
  export CKPT=$HOME/.../examples/checkpoints/insert4_run2
  ```
  (Or `rm -rf` the old dir — but that wipes the replay buffer and the policy. Checkpoint dirs
  run **10–25 GB** per insert; keep an eye on disk.)

- **Warm start from another policy** (fine-tuning): point `CKPT` at a copy of the old
  checkpoint dir. Be warned — see §14; a warm start can look *worse* than from-scratch for
  thousands of transitions.

### Learner hyperparameters

In `FrankaPlumbersTrainBase` ([config.py:179](hil-serl_src/examples/experiments/franka_plumbers/config.py#L179)):

| Setting | Value | Why |
|---|---|---|
| `steps_per_update` | 50 | 50 gradient steps per actor step — this is what makes it sample-efficient |
| `replay_buffer_capacity` | 20000 | The buffer is **preallocated**. Upstream's 200 000 × 128×128×3 ≈ 20 GB in one allocation **OOM-killed the learner at startup**. 20 000 ≈ 2 GB and is ample. |
| `checkpoint_period` | 1000 | ~30 s of learner progress at risk |
| `buffer_period` | 500 | online transitions → disk; 500 steps of operator time at risk |
| `encoder_type` | `resnet-pretrained` | frozen ResNet-10 |
| `setup_mode` | `single-arm-fixed-gripper` | gripper not in the action space |

`--debug` (set by both launchers) disables wandb. Drop it and set `WANDB_API_KEY` for
online logging.

---

## 9. Operating the pad during training

| Control | Action |
|---|---|
| **R1** (hold) | **Intervene** — your stick input overrides the policy. Release → the policy drives again. |
| **Left stick** | X / Y translation |
| **L2 / R2** | Z down / up |
| **Right stick** | pitch / yaw |
| **D-pad L/R** | roll |
| **X** | **Success** → reward 1, episode ends |
| **Triangle** | **Abort** → reward 0, episode ends |
| **Circle** | **Regrasp** at the next reset (retract, open, hand the part in, close) |

> **R1 is not an emergency stop.** In stock HIL-SERL the policy drives autonomously and R1
> lets you *correct*; releasing it hands control **back to the policy** and the arm keeps
> moving. The E-stop and the compliant controller are the safety layer.
>
> `PS4_DEADMAN_GATES_POLICY=1` turns R1 into a true hold-to-move gate — but it gates only
> `step()`, **not reset/regrasp**, and it must be **OFF during training** or the policy can
> never act.

### Why your interventions matter so much

Interventions go into **both** buffers; pure policy transitions go to the online buffer only.
So every correction you make is weighted like a demonstration. You are not just *preventing
failures* — you are **writing training data in real time**. That is the whole reason this
converges in hours instead of weeks, and it is why *how* you intervene matters as much as
*how often*.

The single governing principle:

> ## 🔑 Get successful insertions on the board early.
> The policy learns from reward, and reward only exists when the part seats. An episode that
> ends in success — **even if you did 95 % of the work** — is worth more than a clean,
> hands-off failure. Early on, your job is not to evaluate the policy. Your job is to
> **manufacture successes.**

### The intervention schedule

This is a feel thing and nobody hits it exactly — but this is the shape that worked for all
four inserts:

```
intervention
    100% │████████████▓▓▓▓▓▓▓▓▓▒▒▒▒▒▒░░░░
         │            ╲                      ╲
     50% │  PHASE 1    ╲   PHASE 2            ╲  PHASE 3
         │  "show it"   ╲  "hand over"         ╲ "get out of the way"
      0% │               ╲                      ╲────────────────────
         └──────────────────────────────────────────────────────────▶
          ep 1-3      ep ~5-10      ep ~20-30         breakthrough
         near-100%    heavy          tapering          hands off
```

#### Phase 1 — episodes 1–3: near-100 % intervention

**Drive almost the whole insertion yourself.** The freshly-initialised policy is random; left
alone it will wander, and you will bank nothing but failures. Treat these first few episodes
as extra demos recorded through the live loop.

#### Phase 2 — roughly episodes 5–30: heavy, but start handing over

Still intervening a lot, but begin letting the policy do parts of the motion. **It is normal
and fine to keep intervening heavily for 20–30 episodes before it starts breaking through on
its own.** Do not read a long heavy-intervention stretch as failure.

The most useful technique in this phase:

> ### 💡 Drive it to the hole, then let go
> Intervene to bring the part **down to just above the hole** — the hard, boring traverse —
> then **release R1 and let the policy find the hole from there.**
>
> This is the highest-value thing you can do. It concentrates the policy's own exploration
> on the only part that actually needs learning (the final alignment), while still producing
> a reward at the end. You get successful insertions on the board *and* genuine
> self-discovered behaviour, instead of having to choose.

#### Phase 3 — after the breakthrough: get out of the way

Once it is seating the part on its own, **stop helping.** Intervene only to prevent a jam or
damage. Your remaining job is pressing **X**. Every unnecessary intervention now is a worse
action than the policy's own, injected into the demo buffer at demonstration weight.

### When to press Triangle (abort)

The abort policy **inverts** over the run, and getting this backwards is a common mistake:

| Phase | What to do | Why |
|---|---|---|
| **Early** | **Be patient. Let it wander.** Give it the time to explore, and prefer *rescuing* the episode with an intervention over aborting it. | Early aborts bank nothing. A rescued episode banks a success. |
| **Later** | **Abort as soon as you can see it is off.** The moment it is clearly heading nowhere, hit Triangle and reset. | Once the policy is mostly competent, a doomed episode is just wasted robot time — and 300 steps of wandering is 30 s you could spend on a fresh attempt. |

Rule of thumb: **early, your instinct should be "can I save this?"; later, it should be "is
this going anywhere?"**

### Things that cost us time — learn them free

- **Do not stop at the first 0 %-intervention dump.** Both converged runs were called done
  prematurely and both improved substantially afterwards (§10).
- **A long flat stretch is normal.** Insert 1 sat flat for **25 000 transitions** before it
  climbed. Nothing is wrong.
- **Watch the grasp, not just the policy.** If the part slips in the hand, the policy's whole
  visual mapping is off. Hit **Circle** and regrasp rather than fighting it.
- **Press X the instant it seats**, not after admiring it. The reward marks the step, so a
  late X appends steps of already-successful hovering to the trajectory.
- **Press X honestly.** You are the reward function — rewarding a near-miss teaches the policy
  that a near-miss is the goal.

---

## 10. Monitoring

Progress is read off the **buffer dumps** in the checkpoint directory:

```bash
watch -n 30 'ls -t <CKPT>/demo_buffer/ | head -5; echo ---; ls -t <CKPT>/buffer/ | head -5'
```

**Signals, in order of usefulness:**

1. **Intervention rate falling** — the real signal. Measured from `demo_buffer` dump growth
   (interventions land there), not the online buffer.
2. **Episode length falling** — the policy is reaching the goal faster. A converged policy
   **beats the human demos**: insert 2 ended at 32.5 steps vs. 67 in its demos.
3. **Successes per 500-step dump rising** — noisiest; single dumps swing wildly.

### Expect a long flat phase, then a sharp climb

```
successes
per 500     │                                        ╭────────
            │                                    ╭───╯
            │                                ╭───╯
            │────────────────────────────────╯
            └──────────────────────────────────────────────────▶
             0        10k       20k       30k      40k   transitions
                   ← "nothing is happening" →   ← breakthrough →
```

- **Insert 1** sat flat at ~1 success / 500 steps for **25 000** transitions, then climbed
  from ~27 000 to 9 successes / 500 at 0 % intervention.
- **Insert 2** went 66 % intervention at step 500 → 25 % by 3 000 → **0 % by 5 000**, then
  roughly **doubled** its success rate again between 6 000 and 11 500.

> **Do not stop on a flat stretch, and do not stop at the first 0 %-intervention dump.**
> Both runs were called converged prematurely and both improved substantially afterwards.

**Budget** (from [RESULTS_FRANKA_HILSERL.md](RESULTS_FRANKA_HILSERL.md)): 9 000–45 000 online
transitions, i.e. roughly **1–3 hours of robot time** per insert, depending on demo length.

---

## 11. Evaluating a trained policy

Hands-off eval: run the **actor alone** (no learner) with an eval checkpoint.

```bash
cd ~/Masterthesis-rl-train/Masterthesis-train-serl
source franka_env.sh
export FRANKA_EXP_NAME=franka_plumbers_insert4
export TELEOP_DEVICE=ps4 SDL_JOYSTICK_DEVICE=/dev/input/js0

bash hil-serl_src/examples/experiments/franka_plumbers/run_actor.sh \
     --eval_checkpoint_step 136000 \
     --eval_n_trajs 15
```

It restores `checkpoint_<step>`, runs N episodes, and prints the success rate and mean time
to success. **You still press X when seated** — that is the reward function. 15 trajectories
is what the reference results used.

> **Keep your hands off R1 during eval.** This is the one phase where intervening invalidates
> the number: a "success" you steered is not a measurement. Press X when it seats, press
> Triangle when it clearly fails, and otherwise do not touch the sticks. Expect to re-grasp
> (Circle) between episodes — that is setup, not intervention.

Reference results (4 policies, all from 20 demos) — **every insert evaluated at 15/15
autonomous successes, 0 % intervention**:

| Insert | Task | Transitions | Learner steps | Eval |
|---|---|---|---|---|
| 0 | plumbers screw, vertical −Z | 29 001 | 136 000 | **15/15** |
| 1 | pb_pipe, horizontal −Y | 44 500 | 192 000 | **15/15** |
| 2 | cooling base, vertical −Z | 11 500 | 67 000 | **15/15** |
| 3 | cooling base 2nd part, vertical −Z | 9 001 | 50 000 | **15/15** |

---

## 12. Config reference

Everything below lives in
[config.py](hil-serl_src/examples/experiments/franka_plumbers/config.py). The values are not
defaults — each one was tuned on the real robot, and several encode a failure that cost a
training run.

### Per-insert (you must set these)

| Knob | What it is |
|---|---|
| `RESET_POSE` | Pre-insert pose, 6-vec `[x,y,z,r,p,y]` (m, rad) |
| `TARGET_POSE` | Seated pose |
| `ABS_POSE_LIMIT_LOW/HIGH` | Safety box, **6 values each** |
| `RETRACT_DIR` | Unit vector OUT of the socket |
| `IMAGE_CROP["wrist"]` | Crop lambda |

### Motion guards (the hard-won ones)

| Knob | Value | Why it is that value |
|---|---|---|
| `ACTION_SCALE` | `(0.01, 0.04, 1)` | 1 cm / 0.04 rad per unit action. Cutting it to 0.004 m made the arm so slow the operator could not traverse a few cm in an episode — it *felt* like a hard axis limit but was just speed. |
| `MAX_Z_DOWN_STEP` | `0.004` | Caps the **downward z** delta only (4 mm/step = 4 cm/s at 10 Hz). The policy learned from demos where z was at full stick 41 % of the time, so it slammed down and tripped the joint-torque reflex — but halving `ACTION_SCALE` globally made everything else uselessly slow. Full speed in x/y and for lifting; gentle into contact. |
| `Z_FORCE_LIMIT_N` | `15.0` | Downward-z motion is refused once \|Fz\| exceeds this. Stops the part being driven harder into an already-seated socket. Upward motion always allowed. |
| `LOCK_ORIENTATION` | `True` | Hard-locks tool attitude to `RESET_POSE`'s orientation and **ignores rotation actions**. For a vertical descent, orientation is not a needed DoF, and locking it removes translation/rotation coupling entirely rather than trying to cancel it. Accepts `True` (always), `False` (never), or an **int N** = *warmup lock*: locked for the first N episodes, then released — locked while the policy is still random (when it tumbles the wrist into the torque limit), released so it can still learn the rotation corrections the demos contain. **Insert 1 (horizontal) uses `6`** — it needs continuous counter-steer, which is exactly why it cost 4× the training data. |
| `MAX_EPISODE_LENGTH` | `300` | 30 s at 10 Hz. The stock 100 (10 s) is not enough to reposition *and* insert. |
| `POST_RESET_WAIT_S` | `1.5` | A beat after reset to get ready / hit Circle. |
| `RETRACT_DIST` | `0.04` | Pre-reset retract along `RETRACT_DIR`. |
| `REGRASP_RETRACT_DIST` | `0.08` | Extra clearance for the hand-off. |

### Reset noise

| Knob | Value | Note |
|---|---|---|
| `RANDOM_RESET` | `True` | On from the first demo, so demos cover the variation the policy will face |
| `RANDOM_XY_RANGE` | `0.01` | ±1 cm in X/Y |
| `RANDOM_Z_RANGE` | `0.0` | Off by default; clipped to the box floor when set |
| `RANDOM_RZ_RANGE` | `0.0` | No yaw jitter — rotation variation comes free from the human grasping the part differently each time |

### Environment variables

| Variable | Purpose |
|---|---|
| `FRANKA_EXP_NAME` | Experiment name; **must match** in both terminals |
| `CKPT` | Checkpoint dir override; **must match** in both terminals |
| `FRANKA_USE_GPU` | `1` before sourcing → JAX on GPU (learner) |
| `TELEOP_DEVICE` | `ps4` |
| `SDL_JOYSTICK_DEVICE` | `/dev/input/js0` |
| `PS4_DEADMAN_GATES_POLICY` | `1` = hold-to-move. **Bring-up only — unset for training.** |
| `FRANKA_DISPLAY_IMAGE` | `0`/`1`; auto-detects `$DISPLAY`. Must be `0` over SSH or Qt aborts the process. |
| `FRANKA_ROS2_WS` | Controller workspace (default `~/fr3_ws`) |
| `SERL_PY_SITE` | conda `serl` site-packages path |
| `SERL_PYTHON` | Pinned interpreter (default `/usr/bin/python3.10`) |

---

## 13. Troubleshooting

**`ModuleNotFoundError: No module named 'franka_msgs'`**
`franka_env.sh` sourced the wrong workspace. It defaults to `~/fr3_ws`; set
`FRANKA_ROS2_WS` to wherever you actually built `franka_ros2`.

**Learner is OOM-killed at startup**
The replay buffer is preallocated. Lower `replay_buffer_capacity` in
`FrankaPlumbersTrainBase`. (20 000 ≈ 2 GB with 128×128×3 images; upstream's 200 000 ≈ 20 GB.)

**JAX silently runs on CPU**
`export JAX_PLATFORMS=` (empty) does **not** mean "GPU" — an empty value is treated as unset
and falls back to CPU. Use `export FRANKA_USE_GPU=1` **before** sourcing, and always check
`jax.default_backend()` before launching.

**ZED fails with "NO GPU DETECTED"**
Something set `CUDA_VISIBLE_DEVICES=""`. The ZED SDK is CUDA-only. Keep CUDA visible and use
`JAX_PLATFORMS=cpu` to keep JAX off the card instead — that is what `run_actor.sh` does.

**The process aborts a few steps in: "could not connect to display" / core dumped**
The camera preview uses `cv2.imshow`, which needs X. Over SSH set `FRANKA_DISPLAY_IMAGE=0`.

**`IndexError` in `clip_safety_box` on the first step**
`ABS_POSE_LIMIT_*` has 3 elements, not 6. `franka_env` slices `[3:]` for the rpy box.

**Reset refuses: "Current pose is invalid/outside the safety box"**
The arm is parked outside the box — often by less than a millimetre, because an impedance
controller settles where forces balance, not where you commanded. There is 2 cm of slack; a
gross excursion still aborts. Jog back toward `RESET_POSE`, and recheck that the box has room
for where the arm actually sits at rest.

**"No room to retract: only N mm available"**
The arm is jammed against the box edge. A retract that *fits* is shortened automatically and
a missing retract is skipped (reset proceeds — `interpolate_move` clips every waypoint). If
you still see this, jog back toward `RESET_POSE`.

**Joint torque limit trips on contact**
Lower `MAX_Z_DOWN_STEP` (not `ACTION_SCALE` globally — that cripples lateral motion). The
reflex fires on a transient faster than a 10 Hz guard can catch, so the only reliable fix is
not to generate the impact. Also check `Z_FORCE_LIMIT_N`.

**Gripper reports 0.434 when open / "grip force problem"**
Home the hand (§5.2). A drifted zero fakes exactly this.

**Compliance / stiffness changes have no effect**
Expected on the ROS2 path. `cartesian_impedance_control` does not expose stiffness as ROS
parameters — edit the controller's header and rebuild.

**`mappings.py` prints "franka_plumbers configs unavailable"**
Missing dep (`franka_env` / `pygame`) — benign in a non-Franka env. A *real* bug in the
config prints a full traceback instead; that one you must fix.

**`ps4_controller_teleop/` is empty, or the clone fails on it**
It is a git submodule pointing at a **private** repo. If you have access, run
`git submodule update --init`; otherwise clone **without** `--recurse-submodules` and ignore
it. You do **not** need it to record demos or train — see §3.5.

**Orbax refuses to write the checkpoint**
It will not overwrite. Use a new `CKPT` (in **both** terminals) or remove the old directory.

**Actor and learner do not connect**
They talk over TCP **:5588** (request/reply) and **:5589** (weight broadcast). Same-machine is
the default (`--ip localhost`); across machines pass `--ip <learner>` to the actor and open
both ports.

---

## 14. Known limitations

**One policy per insert.** Nothing is shared between tasks. Four inserts = four policies,
four demo sets, four training runs.

**The fixture must not move.** The base is assumed glued at a fixed spot. `RelativeFrame`
absorbs *translation* (proprioception is reset-relative), but **not rotation** — the wrist
camera is rigidly mounted, so rotating the fixture rotates every image relative to training.
When the cooling-base fixture was moved 276 mm and ~90° of yaw, the policy was **completely
lost zero-shot** and never seated the part. `set_reset_from_base_pose()` is stubbed in
`wrapper.py` as the hook for measuring the base each episode (ArUco/vision); it is
deliberately unimplemented.

**Warm starts are not free.** Fine-tuning the moved-fixture policy took 7 500 transitions vs.
11 500 from scratch — a ~35 % win — but it went *backwards* for the first ~5 000, with
intervention climbing 12 % → 47 %, while the from-scratch run was already at 0 %. A
pre-trained policy does not start from ignorance; it starts from **confident wrongness** and
must unlearn before it can relearn. Budget for that, and do not read the early regression as
failure.

**Human reward means a human must be present** for the whole run, including eval.

**`joint_reset()` is a no-op** on the ROS2 backend — the stack has no joint-space controller
loaded. A wedged arm is recovered by hand.

**Stiffness is compile-time** on the ROS2 backend (see §3.3).

---

## 15. Session quick reference

Tape this to the rig.

### Every session, before anything else

```
□  Home the gripper                      (§5.2 — a drifted zero fakes a grip-force bug)
□  Controller up; push the arm by hand → it YIELDS
□  Squeeze L2 and R2 fully, release      (§5.4 — else Z is dead or phantom-up)
□  preflight.py is green                 (§5.6)
□  Hand on the E-STOP for the first reset
```

### Recording demos

```
source franka_env.sh
export FRANKA_EXP_NAME=<exp> TELEOP_DEVICE=ps4 SDL_JOYSTICK_DEVICE=/dev/input/js0
cd hil-serl_src/examples
"${SERL_PYTHON}" record_demos.py --exp_name=<exp> --successes_needed=20
```
**HOLD R1 the whole time.** X = seated (banks it) · Triangle = discard · Circle = regrasp.
Messy demos are fine. 20 successes, then stop.

### Training — two terminals

```bash
# T1 learner
export FRANKA_USE_GPU=1; source franka_env.sh; unset PS4_DEADMAN_GATES_POLICY
export FRANKA_EXP_NAME=<exp>
bash hil-serl_src/examples/experiments/franka_plumbers/run_learner.sh --demo_path <abs.pkl>
#   ...wait for "sent initial network to actor"

# T2 actor
export FRANKA_USE_GPU=0; source franka_env.sh; unset PS4_DEADMAN_GATES_POLICY
export FRANKA_EXP_NAME=<exp> TELEOP_DEVICE=ps4 SDL_JOYSTICK_DEVICE=/dev/input/js0
bash hil-serl_src/examples/experiments/franka_plumbers/run_actor.sh
```

### Operating cheat-sheet

| Phase | Intervene | Abort | Mindset |
|---|---|---|---|
| **Ep 1–3** | ~100 % — drive it yourself | almost never | manufacture successes |
| **Ep 5–30** | heavy; *drive to the hole, then let go* | rarely — rescue instead | hand over gradually |
| **After breakthrough** | only to prevent a jam | **as soon as it's off** | get out of the way |

Expect a **long flat stretch** (insert 1: 25 000 transitions) then a sharp climb.
**Do not stop at the first 0 % dump.** Budget 9k–45k transitions ≈ 1–3 h.

### Eval

```bash
bash .../run_actor.sh --eval_checkpoint_step <N> --eval_n_trajs 15
```
**Hands off R1.** X when seated, Triangle when it fails. Nothing else.

---

## Further reading

| Document | Contents |
|---|---|
| [RESULTS_FRANKA_HILSERL.md](RESULTS_FRANKA_HILSERL.md) | Per-insert training cost, convergence shapes, the moved-fixture robustness study |
| [rail-berkeley/hil-serl](https://github.com/rail-berkeley/hil-serl) | Upstream HIL-SERL |
| [frankarobotics/franka_ros2](https://github.com/frankarobotics/franka_ros2) | FR3 ROS2 driver |
| [CurdinDeplazes/cartesian_impedance_control](https://github.com/CurdinDeplazes/cartesian_impedance_control) | The compliant Cartesian controller this stack commands |

**This README plus the code comments are the documentation.** The rig notes that preceded it
(day-by-day bring-up logs, command sheets, the KUKA-era residual-SERL design docs) have been
removed — everything from them that still applies was folded into the sections above, and the
rest described a robot and a training scheme this repo no longer uses.
