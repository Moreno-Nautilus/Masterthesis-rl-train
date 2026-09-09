# Franka HIL-SERL Pivot — FINAL Plan (2026-09-08, home-office prep day) — v3

Salvage pivot: KUKA cell out of rig time → run **HIL-SERL** on the **Franka Research 3 (FR3)** for the
**4 plumbers inserts**. Part **bases glued to the table** (no holder arm); single FR3 does the insert;
human jogs to a fixed pre-insert pose and hands the part into the Franka Hand (grasp varies by hand).

Today = code-only prep (no robot, no ZED at home). Then 3 days on-robot: bringup → teleop → record demos
→ train → debug. **Principle: change HIL-SERL as LITTLE as possible; swap only leaf nodes (robot-comm,
teleop device, camera).**

---

## 0. Locked decisions (all resolved)

- **Robot** = FR3, **Franka Hand** gripper, pregrasped (`GripperCloseEnv`, `single-arm-fixed-gripper`).
- **Control stack** = **ROS2 Humble + `franka_ros2` (v0.1.15) + FCI + RT kernel** (the robot PC's stack).
- **Controller** = the student's **`github.com/CurdinDeplazes/cartesian_impedance_control`** (+
  `messages_fr3`). It IS a franka_ros2 Cartesian **impedance** controller (compliant — mandatory for
  insertion), proven on this arm; his PS4 teleop already drives its topic. NOT ROS1 `serl_franka_
  controllers`, NOT a generic example.
- **Scheme** = **stock E2E** (free 6-DoF Cartesian delta from a fixed `RESET_POSE`), NOT KUKA residual.
- **Codebase** = vendored `hil-serl_src/` (already the clean `rail-berkeley/hil-serl` clone; keep it).
- **Teleop** = **PS4**, in-process `PS4Expert` drop-in for `SpaceMouseExpert` (identical
  `get_action()->(6-vec,buttons)`); axis map from the student's `ps4_publisher.py`. Swap the expert only.
- **Camera** = **ZED Mini wrist only**; `ZEDCapture` mirroring `RSCapture.read()->(ok,bgr)`/`.close()`.
- **Reward** = human button (X=success/terminate, other=abort).
- **MoveIt** = NOT in the RL loop (see §3a). Optional day-1 convenience only.
- **Target** = all 4 plumbers inserts; `franka_plumbers` experiment → inserts 1/2/3 are a config copy.

---

## 1. The ONE structural change: a `RobotClient` seam (ROS2)

HIL-SERL's `franka_env.py` centralizes robot I/O in ~8 calls (`pose`, `getstate`, `clearerr`,
`update_param`, gripper, `set_load`, `jointreset`). Wrap them behind a thin client with two backends:
- **`HttpFrankaClient`** — stock `requests.post` behavior, verbatim (reference; must stay byte-identical).
- **`Ros2FrankaClient`** — rclpy node that talks to `cartesian_impedance_control`:
  - **send_pose** → publish `/cartesian_position_controller/commands` (`Float64MultiArray
    [x,y,z,qx,qy,qz,qw]`), same topic his teleop/bridge uses.
  - **get_state** → read TCP pose/vel + **estimated external wrench** from franka_ros2 / the controller's
    state (`O_F_ext_hat_K`/`K_F_ext_hat_K`, `tau_ext_hat_filtered`) — the wrench HIL-SERL's obs needs;
    NO separate F/T sensor.
  - **update_param(stiffness)** → set the controller's impedance stiffness params (FR3 range: transl
    10–3000 N/m, rot 1–300 Nm/rad) → our `COMPLIANCE_PARAM`↔`PRECISION_PARAM` swap.
  - **send_gripper** → Franka Hand `Move`/`Grasp` action (the `/fr3_gripper` servers his gripper_client
    uses). **recover** → franka error-recovery action.

`FrankaEnv.__init__` picks the backend via a config flag. **Env/wrappers/agent/PS4Expert/ZEDCapture/
plumbers-configs are IDENTICAL either way** — only `Ros2FrankaClient`'s topic/action names are stack-
specific. This is the KUKA-adaptation pattern minimized to one leaf.

## 1a. How stock teleop wires in (mirror EXACTLY — verified)

In-process, **no ROS**: `SpaceMouseExpert.get_action()->(np.array(6),buttons)`;
`SpacemouseIntervention.action()` REPLACES policy action when `‖expert_a‖>0.001` and sets
`info["intervene_action"]` (what HIL-SERL trains on). ⇒ `PS4Expert` = same contract (pygame daemon;
L-stick XY, L2/R2=Z, R-stick pitch/yaw, dpad roll, **R1 deadman gates output to zero**); swap
`SpaceMouseExpert()`→`PS4Expert()` behind `TELEOP_DEVICE=ps4|spacemouse`. Touch nothing else.

## 1b. Stock env/obs/action facts

Wrappers: `GripperCloseEnv`→`SpacemouseIntervention(→PS4)`→`RelativeFrame`→`Quat2EulerWrapper`→
`SERLObsWrapper`→`ChunkingWrapper`→`RecordEpisodeStatistics`. obs `Dict{state,wrist}`; state=TCP
pose(euler)+vel+**wrench**; image cropped→128×128. action=6-DoF delta in **RelativeFrame**;
`ACTION_SCALE=(0.01m,0.06rad,1)`. reset/grasp: `RAMEnv` already does pull-up→release→`input("place
part")`→close→`RESET_POSE`; `RANDOM_RESET` XY±2cm/Rz = grasp variation. agent=`drq`,
encoder=`resnet-pretrained` (frozen ResNet-10), discount 0.97, batch 256.

---

## 2. Today's build (code only — dependency order). CONTROL SAFETY IS #1.

1. **Env sanity** — offline via the `serl_env.sh` pattern (SYSTEM python3.10 + serl conda site-packages
   on PYTHONPATH; NOT `conda activate`). Confirm `jax`, `franka_env`, `serl_launcher` import. `pyzed`
   absent at home → ZEDCapture import-guarded.
2. **`RobotClient` seam** — refactor the ~8 robot calls behind a client; `HttpFrankaClient` (verbatim
   stock) + `Ros2FrankaClient` stub with the topic/action names above. Prove HTTP path unchanged.
3. **`PS4Expert`** (`franka_env/spacemouse/ps4_expert.py`) — pygame daemon; identical `get_action()`;
   R1 deadman; `TELEOP_DEVICE` switch.
4. **`ZEDCapture`** (`franka_env/camera/zed_capture.py`) — `RSCapture`-compatible; left-eye RGB; import-
   guarded. `CAMERA_TYPE=realsense|zed` switch; plumbers uses one `wrist`.
5. **`franka_plumbers` experiment** — copy `ram_insertion/{config,wrapper,run_actor,run_learner}` →
   `experiments/franka_plumbers/`; register `franka_plumbers_insert0..3` in `mappings.py`. One `wrist`;
   `image_keys=["wrist"]`; proprio incl. wrench; `single-arm-fixed-gripper`. Poses/crops = placeholders,
   filled at rig by jogging.
6. **Offline dry-run** — `fake_env=True` learner init: env builds, ResNet10 loads, SACAgent constructs,
   one forward pass.
7. **`FRANKA_RIG_CHECKLIST.md`** — incl. the `franka_ros2_ws` v0.1.15 rebuild (below), controller launch,
   FCI/RT/network bringup, control-first validation, jog-capture-poses, record + train commands.

## 2a. Day-1 `franka_ros2_ws` rebuild (from Leo's install PDF — robot PC, not today)

`ros-humble-desktop` + dev-tools; `git clone -b v0.1.15 frankarobotics/franka_ros2 src`; hand-write
`franka.repos` (franka_description **0.4.0**, libfranka **0.13.2**); `vcs import --recursive`; rosdep;
patch `franka_ign_ros2_control/CMakeLists.txt` (Franka 0.13.3→0.13.2) + the GitHub-issue-160 3-file
additions; `colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release`. Then clone +
build `CurdinDeplazes/cartesian_impedance_control` + `messages_fr3` + the student's `ps4_controller_
teleop` into the same ws. RT kernel + NVIDIA per the 2b-t/docker-realtime note. FCI: static net
172.16.0.x on the C2 shop-floor port, unlock joints + Activate FCI in Desk.

---

## 3. Rig days 1–3

- **Day 1 — CONTROL FIRST.** Rebuild/verify `franka_ros2_ws` + controller (§2a); bring up
  `cartesian_impedance_control`; validate compliance (push the arm by hand) + PS4 deadman + E-stop
  BEFORE autonomy. Point `Ros2FrankaClient` at the topics; smoke the env on the real robot. Then jog to
  capture `TARGET/RESET/GRASP` + crop for insert 0; record ~20 demos; start training (`RANDOM_RESET`
  off first, on later).
- **Day 2** — tune insert 0 to converging; copy config for inserts 1–2; record + train.
- **Day 3** — insert 3 + polish; eval numbers (`--eval_checkpoint_step`, `--eval_n_trajs`).

## 3a. MoveIt — explicitly NOT in the loop

Policy → 6-DoF Cartesian delta at ~10Hz → impedance controller over FCI. Episode reset to `RESET_POSE`
uses the controller's interpolate-move (stock `RAMEnv.go_to_reset`), NOT MoveIt. MoveIt is only an
optional day-1 convenience to hand-jog in RViz and read off pose numbers; the PS4 + state readback do
the same. No MoveIt dependency in the trained pipeline.

## 4. Hard rules

- USER does ALL git + launches every GPU/training run. Prep+validate, then WAIT for explicit "go".
- Never auto-start training. Do not re-architect HIL-SERL's RL/env core.
- Mirror the stock teleop/robot path EXACTLY; the HTTP backend stays behavior-identical.
- **Control safety > everything**: no autonomous motion until compliance + deadman + E-stop are
  hand-verified on the real FR3.
