"""Record teleoperated demonstrations for iiwa peg insertion (SERL demo buffer).

Saves a .pkl of transitions that async_drq_iiwa_insert.py loads via --demo_path.
Uses a PS4 DualShock 4 controller; all motion gated behind the R1 deadman switch.

Prerequisites
-------------
1. SERL HTTP robot server running:
       python -m iiwa_serl.robot_servers.iiwa_server --backend="your_module:RealIiwaBackend"
2. PS4 controller paired (USB or Bluetooth, check /dev/input/js0).
3. pip install pygame requests

Controls
--------
  Left stick / L2+R2 / right stick / d-pad  → move EE (see ps4_teleop_provider.py)
  R1 (hold)     → deadman switch
  Cross  (✕)    → confirm SUCCESS (saves episode + resets)
  Triangle (△)  → abort episode without reward (resets without saving)
  Circle  (○)   → open gripper
  Square  (□)   → close gripper

Usage
-----
  # Dry-run (mock backend, no real robot):
  python record_demo.py --fake_env --n_demos 3

  # Real robot:
  python record_demo.py --n_demos 20 --server_url http://127.0.0.1:5000
"""

import argparse
import copy
import datetime
import os
import pickle as pkl
import time

import numpy as np
from tqdm import tqdm

# NOTE: import `gym` (not gymnasium) to match the registry iiwa_serl registers into
# and the registry the training script (async_drq_iiwa_insert.py) uses. Mixing the two
# causes NameNotFound at gym.make.
import gym

import iiwa_serl  # noqa: F401 — registers environments
from iiwa_serl.teleop import PS4TeleopProvider
from serl_launcher.wrappers.chunking import ChunkingWrapper


def record_demos(
    env_id: str,
    n_demos: int,
    out_dir: str,
    fake_env: bool,
    server_url: str,
    force_feedback: bool,
    joystick_index: int,
    z_down_limit_m: float,
    z_up_limit_m: float,
    xy_limit_m: float,
    z_step_m: float,
    rot_step_rad: float,
    tilt_limit_rad: float,
    spin_limit_rad: float,
    max_camera_stale_fraction: float,
):
    env = gym.make(env_id, fake_env=fake_env)
    # Manual pre-insert workflow: each episode starts wherever the operator jogged the peg
    # (pre-insert pose), with a workspace box RELATIVE to that start. Enable on the live config.
    cfg = env.unwrapped.config
    cfg.manual_reset = True
    cfg.relative_pose_limit = True
    cfg.reward_mode = "manual"        # success decided by the operator's Cross button
    cfg.base_frame_actions = True     # record-phase driving matches the base-frame teleop jog feel
    cfg.integrate_action_target = True  # accumulate motion, but brake to a fixed joint hold on release
    cfg.reuse_last_camera_frame = True  # one dropped frame must not kill a recording session
    cfg.debug_step = False            # console spam OFF — we log to CSV instead (see diag_csv below)
    if min(
        z_down_limit_m,
        z_up_limit_m,
        xy_limit_m,
        z_step_m,
        rot_step_rad,
        tilt_limit_rad,
        spin_limit_rad,
    ) <= 0.0:
        raise ValueError("teleop step sizes and workspace limits must be positive")
    if not 0.0 <= max_camera_stale_fraction <= 1.0:
        raise ValueError("max_camera_stale_fraction must be between 0 and 1")
    # Demo-only overrides.  Keep global/training safety defaults unchanged.
    # Full trigger was producing up to 9.6 mm of measured Z travel per 100 ms;
    # 4 mm target increments make the approach/braking substantially finer.
    cfg.action_scale_xyz_mult = np.asarray(
        cfg.action_scale_xyz_mult, dtype=np.float64
    ).copy()
    cfg.action_scale_xyz_mult[2] = float(z_step_m) / float(cfg.action_scale_xyz_m)
    cfg.action_scale_rot_rad = float(rot_step_rad)
    cfg.rel_limit_low = np.asarray(cfg.rel_limit_low, dtype=np.float64).copy()
    cfg.rel_limit_high = np.asarray(cfg.rel_limit_high, dtype=np.float64).copy()
    cfg.rel_limit_low[:2] = -float(xy_limit_m)
    cfg.rel_limit_high[:2] = float(xy_limit_m)
    cfg.rel_limit_low[2] = -float(z_down_limit_m)
    cfg.rel_limit_high[2] = float(z_up_limit_m)
    cfg.rel_limit_low[3:5] = -float(tilt_limit_rad)
    cfg.rel_limit_high[3:5] = float(tilt_limit_rad)
    cfg.rel_limit_low[5] = -float(spin_limit_rad)
    cfg.rel_limit_high[5] = float(spin_limit_rad)
    # No step-clock cutoff: the operator ends every demo with ✕/△.
    cfg.max_episode_length = 10**9
    _e = env
    while _e is not None:
        if _e.__class__.__name__ == "TimeLimit":
            _e._max_episode_steps = 10**9
            break
        _e = getattr(_e, "env", None)
    # Demos MUST carry the same obs shape as the learner's replay buffer, i.e. image
    # observations need the temporal axis (B,T,H,W,C). The training script wraps the
    # env with ChunkingWrapper(obs_horizon=1); apply the identical wrapper here so the
    # saved demo transitions load into the demo buffer without a shape mismatch.
    if any(k != "state" for k in env.observation_space.spaces.keys()):
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    teleop = PS4TeleopProvider(
        joystick_index=joystick_index,
        server_url=None if fake_env else server_url,
        force_feedback=force_feedback and not fake_env,
    )

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, f"iiwa_insert_{n_demos}_demos_{uuid}.pkl")

    all_transitions: list[dict] = []
    ep_transitions: list[dict] = []
    success_count = 0
    total_episodes = 0
    pbar = tqdm(total=n_demos, desc="successes recorded")

    print("\n[DEMO] Controls: R1=enable  ✕=success  △=abort  ○=open □=close gripper")
    print(f"[DEMO] Target: {n_demos} successful demos → {file_path}\n")
    print(
        "[DEMO] Relative workspace from pre-insert: "
        f"XY=±{xy_limit_m*1000:.0f} mm, "
        f"Z=[-{z_down_limit_m*1000:.0f}, +{z_up_limit_m*1000:.0f}] mm; "
        f"full-trigger Z step={z_step_m*1000:.1f} mm; "
        f"rotation step={rot_step_rad:.3f} rad, tilt=±{tilt_limit_rad:.2f} rad"
    )

    # Jog controls for the pre-insert positioning phase — same tuned feel as teleop_drive.py:
    # base-frame linear, latched reference pose (no sag leak), tool-frame rotation.
    from iiwa_serl.teleop.pose_target import TeleopPoseTarget
    client = env.unwrapped.client
    JOG_LIN = np.array([0.004, 0.004, z_step_m], dtype=np.float64)  # m per step
    JOG_ROT = rot_step_rad
    JOG_DT = 1.0 / float(cfg.hz)

    # --- Diagnostics CSV: RAW inputs + timing per step, for BOTH jog and record phases,
    # so we can separate "stick noisy" / "command dropped" / "robot drifted" / "loop stalled".
    import csv
    diag_dir = os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "diagnostics")
    )
    os.makedirs(diag_dir, exist_ok=True)
    diag_path = os.path.join(diag_dir, f"teleop_record_diag_{uuid}.csv")
    diag_f = open(diag_path, "w", newline="")
    diag_w = csv.writer(diag_f)
    diag_w.writerow([
        "t_wall", "dt_step", "phase", "ep",
        "r1", "raw_lx", "raw_ly", "raw_rx", "raw_ry", "raw_l2", "raw_r2",
        "ax", "ay", "az", "arx", "ary", "arz",
        "meas_x", "meas_y", "meas_z", "tgt_x", "tgt_y", "tgt_z", "gap_mm",
        "post_x", "post_y", "post_z", "motion_mm", "cmd_kind",
        "l2_ready", "r2_ready", "camera_stale", "workspace_clipped",
    ])
    print(f"[DEMO] Diagnostics CSV → {diag_path}")
    print(
        f"[DEMO] Teleop command rate: {cfg.hz} Hz. Run command_upsampler with "
        f"--interpolate --input-hz {cfg.hz}."
    )
    _t0 = time.monotonic()
    _phase_prev = {"jog": None, "record": None}

    def log_diag(
        phase, ep, snap, action, measured, target, post, command_kind,
        camera_stale=False, workspace_clipped=False,
    ):
        now = time.monotonic()
        previous = _phase_prev[phase]
        dt_step = 0.0 if previous is None else now - previous
        _phase_prev[phase] = now
        measured = np.asarray(measured, dtype=float)
        target = np.asarray(target, dtype=float)
        post = np.asarray(post, dtype=float)
        diag_w.writerow([
            round(now - _t0, 4), round(dt_step, 4), phase, ep,
            snap.get("r1", ""), snap.get("raw_lx", ""), snap.get("raw_ly", ""),
            snap.get("raw_rx", ""), snap.get("raw_ry", ""),
            snap.get("raw_l2", ""), snap.get("raw_r2", ""),
            *[round(float(a), 4) for a in action],
            *[round(float(v), 5) for v in measured[:3]],
            *[round(float(v), 5) for v in target[:3]],
            round(float(np.linalg.norm(target[:3] - post[:3])) * 1000.0, 2),
            *[round(float(v), 5) for v in post[:3]],
            round(float(np.linalg.norm(post[:3] - measured[:3])) * 1000.0, 2),
            command_kind,
            snap.get("l2_ready", ""), snap.get("r2_ready", ""), int(bool(camera_stale)),
            int(bool(workspace_clipped)),
        ])
        # Keep the trace usable even if ROS, the camera process, or the recorder
        # exits abruptly.  At 10 Hz the flush overhead is negligible.
        diag_f.flush()

    def arm_episode():
        # Drive the peg to the PRE-INSERT pose with the good teleop feel, then ✕ to arm.
        # env.reset() (manual_reset) then captures the current pose as the episode origin.
        print("\n[DEMO] JOG peg to PRE-INSERT pose (hold R1 + sticks). Press ✕ when ready...")
        # Cancel any retained target from the preceding record/jog phase before
        # accepting another input.  This bypasses Cartesian IK and nullspace bias.
        client.hold_position()
        time.sleep(JOG_DT)
        initial = client.get_state().pose.copy()
        target = TeleopPoseTarget(
            np.full(3, JOG_LIN, dtype=np.float64),
            JOG_ROT,
            base_frame_actions=True,
        )
        target.reset(initial)
        _phase_prev["jog"] = None
        while True:
            loop_start = time.monotonic()
            action = teleop.get_action()
            snap = teleop.debug_snapshot() if hasattr(teleop, "debug_snapshot") else {}
            measured = client.get_state().pose.copy()
            armed = teleop.is_success()
            if armed:
                # Cross may be pressed while a motion input is still held.  Brake
                # first so env.reset() captures a stationary pre-insert origin.
                action = np.zeros(6, dtype=np.float32)
                client.hold_position()
                command_kind = "hold"
                commanded = measured.copy()
            else:
                command = target.update(action, measured)
                command_kind = command.kind
                commanded = command.target_pose
                if command.kind == "move":
                    client.move_pose(commanded)
                elif command.kind == "hold":
                    client.hold_position()
            time.sleep(max(0.0, JOG_DT - (time.monotonic() - loop_start)))
            post = client.get_state().pose.copy()
            log_diag("jog", "", snap, action, measured, commanded, post, command_kind)
            if armed:
                break
        print("[DEMO] Armed — recording. ✕=success  △=abort")
        _phase_prev["record"] = None
        return env.reset()

    obs = None
    workspace_limit_active = False
    ep_camera_stale_steps = 0
    camera_stale_warned = False

    try:
        # Keep initial bringup/reset failures inside the cleanup boundary so the
        # diagnostic rows collected before the failure are still flushed/closed.
        obs, _ = arm_episode()
        while success_count < n_demos:
            action = teleop.get_action()
            _snap = teleop.debug_snapshot() if hasattr(teleop, "debug_snapshot") else {}
            manual_success = teleop.is_success()
            manual_abort = teleop.is_failure()
            if manual_success or manual_abort:
                # Stop on the button edge before taking the terminal observation.
                action = np.zeros(6, dtype=np.float32)
            next_obs, rew, terminated, truncated, info = env.step(action)
            measured = info["measured_pose_before"]
            commanded = info["commanded_pose"]
            post = info["measured_pose_after"]
            log_diag(
                "record", total_episodes, _snap, action, measured, commanded, post,
                info["command_kind"], info.get("camera_stale", False),
                info.get("workspace_clipped", False),
            )
            camera_stale = bool(info.get("camera_stale", False))
            ep_camera_stale_steps += int(camera_stale)
            if camera_stale and not camera_stale_warned:
                print(
                    "[DEMO] Camera did not deliver a fresh frame. "
                    "This attempt will be discarded if stale frames exceed "
                    f"{max_camera_stale_fraction:.0%}."
                )
                camera_stale_warned = True
            clipped = bool(info.get("workspace_clipped", False))
            if clipped and not workspace_limit_active:
                print("[DEMO] Relative workspace limit reached; outward motion is being clipped.")
            workspace_limit_active = clipped

            if manual_success:
                rew = 1.0
                terminated = True
            elif manual_abort:
                terminated = True
                rew = 0.0

            ep_transitions.append(
                copy.deepcopy(
                    dict(
                        observations=obs,
                        actions=action,
                        next_observations=next_obs,
                        rewards=float(rew),
                        masks=1.0 - float(terminated),
                        dones=terminated,
                    )
                )
            )
            obs = next_obs

            if terminated or truncated:
                camera_stale_fraction = ep_camera_stale_steps / max(1, len(ep_transitions))
                camera_bad = camera_stale_fraction > max_camera_stale_fraction
                episode_success = rew > 0.5 and not camera_bad
                total_episodes += 1

                if episode_success:
                    all_transitions.extend(ep_transitions)
                    success_count += 1
                    pbar.update(1)
                    status = "SUCCESS"
                else:
                    if rew > 0.5 and camera_bad:
                        status = "CAMERA-STALE DISCARD"
                    else:
                        status = "abort" if manual_abort else "timeout"

                print(
                    f"  [{status}] ep={total_episodes}  "
                    f"successes={success_count}/{n_demos}  "
                    f"steps={len(ep_transitions)}  "
                    f"camera_stale={camera_stale_fraction:.1%}"
                )
                ep_transitions = []
                workspace_limit_active = False
                ep_camera_stale_steps = 0
                camera_stale_warned = False
                if success_count >= n_demos:
                    break
                obs, _ = arm_episode()

    except KeyboardInterrupt:
        print("\n[DEMO] Interrupted — saving partial demos.")

    finally:
        with open(file_path, "wb") as f:
            pkl.dump(all_transitions, f)
        print(
            f"\n[DEMO] Saved {len(all_transitions)} transitions "
            f"({success_count} successes) → {file_path}"
        )
        try:
            diag_f.close()
            print(f"[DEMO] Diagnostics CSV saved → {diag_path}")
        except Exception:
            pass
        env.close()
        teleop.close()
        pbar.close()


def main():
    parser = argparse.ArgumentParser(description="Record PS4-teleoperated iiwa insertion demos.")
    parser.add_argument("--env", default="IiwaInsertReal-Vision-v0", help="gym env id")
    parser.add_argument("--n_demos", type=int, default=20, help="number of successful demos to collect")
    parser.add_argument(
        "--out_dir",
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "demos"),
        help="directory to save the .pkl file",
    )
    parser.add_argument("--server_url", default="http://127.0.0.1:5000", help="SERL HTTP robot server")
    parser.add_argument("--fake_env", action="store_true", help="dry-run with mock backend (no real robot)")
    parser.add_argument("--force_feedback", action="store_true", help="rumble controller on contact force")
    parser.add_argument("--joystick_index", type=int, default=0)
    parser.add_argument(
        "--z_down_limit_m",
        type=float,
        default=0.12,
        help="maximum downward base-Z travel from the captured pre-insert pose (default: 0.12 m)",
    )
    parser.add_argument(
        "--z_up_limit_m",
        type=float,
        default=0.04,
        help="maximum upward base-Z travel from the captured pre-insert pose (default: 0.04 m)",
    )
    parser.add_argument(
        "--xy_limit_m",
        type=float,
        default=0.04,
        help="symmetric X/Y travel from the captured pre-insert pose (default: 0.04 m)",
    )
    parser.add_argument(
        "--z_step_m",
        type=float,
        default=0.004,
        help="Z target increment at full L2/R2 per 10 Hz step (default: 0.004 m)",
    )
    parser.add_argument(
        "--rot_step_rad",
        type=float,
        default=0.03,
        help="right-stick/d-pad rotation increment per 10 Hz step (default: 0.03 rad)",
    )
    parser.add_argument(
        "--tilt_limit_rad",
        type=float,
        default=0.35,
        help="symmetric tool-X/Y tilt limit from pre-insert (default: 0.35 rad)",
    )
    parser.add_argument(
        "--spin_limit_rad",
        type=float,
        default=0.50,
        help="symmetric tool-Z spin limit from pre-insert (default: 0.50 rad)",
    )
    parser.add_argument(
        "--max_camera_stale_fraction",
        type=float,
        default=0.05,
        help="discard successful attempts above this stale-frame fraction (default: 0.05)",
    )
    args = parser.parse_args()

    if args.fake_env:
        print("[DEMO] Running in fake_env mode — mock backend, zero-action dry-run")

    record_demos(
        env_id=args.env,
        n_demos=args.n_demos,
        out_dir=args.out_dir,
        fake_env=args.fake_env,
        server_url=args.server_url,
        force_feedback=args.force_feedback,
        joystick_index=args.joystick_index,
        z_down_limit_m=args.z_down_limit_m,
        z_up_limit_m=args.z_up_limit_m,
        xy_limit_m=args.xy_limit_m,
        z_step_m=args.z_step_m,
        rot_step_rad=args.rot_step_rad,
        tilt_limit_rad=args.tilt_limit_rad,
        spin_limit_rad=args.spin_limit_rad,
        max_camera_stale_fraction=args.max_camera_stale_fraction,
    )


if __name__ == "__main__":
    main()
