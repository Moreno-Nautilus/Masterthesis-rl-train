"""Record RESIDUAL teleop demonstrations for KUKA HIL-SERL insertion.

Separate from the Codex-hardened `record_demo.py` (which is E2E / free-delta and is
NOT modified). This recorder runs the SAME env in RESIDUAL mode: a slow nominal
insertion trajectory auto-plays and the operator adds a residual (lateral + rotation,
orthogonal to the per-insert insertion axis) with the PS4 stick. Each logged
transition's `action` is the residual — exactly what the residual policy will output —
so these demos drop straight into the HIL-SERL demo buffer (same transition schema as
record_demo.py).

Why this is safe / minimal: the hard part (nominal clock, along-axis projection,
per-insert axis, contact/L1 pause, auto joint reset to reset_move_arm_q) all lives in
the env (base_real_env + nominal_residual, unit-tested). This script is just the
operator loop: reset -> {teleop residual -> env.step -> log} -> X success / Square abort.

Controls (see ps4_teleop_provider.py):
  R1 hold      deadman (any motion)
  L1 hold      stop-forward: freeze the nominal clock to slide/align on a rim
  sticks/L2R2  residual nudge (env projects out the along-axis translation)
  X (Cross)    mark SUCCESS -> reward 1, end episode
  Square/Tri   abort -> reward 0, end episode

Usage:
  source ../experiments/... env; or set env vars, then:
  python record_demo_residual.py --insert 1 --assembly plumbers_block --n_demos 20 \
      --server_url http://127.0.0.1:5000 --out_dir demos

The T6 SERL server must be up with SERL_ARM_PREFIX=lbr_two and SERL_RESET_JOINTS set to
this insert's reset_move_arm_q (source experiments/iiwa_plumbers/env_for_insert.sh <k>).
"""

from __future__ import annotations

import argparse
import copy
import os
import pickle as pkl
import time
import uuid

import numpy as np

import gymnasium as gym  # noqa: F401  (#5: gymnasium HIL stack)

from iiwa_serl.teleop import PS4TeleopProvider


def _train_config_for(assembly: str, insert: int):
    """Return the SAME per-insert TrainConfig the training loop uses (via CONFIG_MAPPING).

    #1/#6/#20 fix: recording must NOT rebuild a parallel config — it MUST use the identical
    IiwaInsertBase config so goal_joints/reset_joints_target (the FK frame path), action scales,
    residual flags, and cameras are byte-identical to training. Requires the examples/ dir on the
    path (as train_rlpd runs); add it if launching this script standalone."""
    import sys, os as _os
    hil_root = _os.path.abspath(
        _os.path.join(_os.path.dirname(__file__), "..", "..", "hil-serl_src")
    )
    # Force the Gymnasium HIL launcher ahead of the repo's legacy classic-Gym
    # launcher. The two packages share a name, so ordinary import fallback is unsafe.
    for source_dir in (_os.path.join(hil_root, "examples"), _os.path.join(hil_root, "serl_launcher")):
        if source_dir not in sys.path:
            sys.path.insert(0, source_dir)
    from experiments.mappings import CONFIG_MAPPING
    exp_name = f"iiwa_{'plumbers' if assembly=='plumbers_block' else 'cooling'}_insert{insert}"
    if exp_name not in CONFIG_MAPPING:
        raise KeyError(f"{exp_name} not in CONFIG_MAPPING (have {sorted(CONFIG_MAPPING)})")
    return CONFIG_MAPPING[exp_name]()


def _build_env(insert: int, assembly: str, server_url: str, fake_env: bool):
    """Build the env from the shared training config (identical to training), then inject the
    HTTP server_url so the recorder talks to the live robot."""
    train_cfg = _train_config_for(assembly, insert)
    iiwa_cfg = train_cfg._make_iiwa_config()
    iiwa_cfg.server_url = server_url
    from iiwa_serl.envs.iiwa_insert_env import KukaIiwaInsertionEnv
    return KukaIiwaInsertionEnv(config=iiwa_cfg, include_image=True, fake_env=fake_env)


def record(insert, assembly, n_demos, server_url, out_dir, fake_env):
    env = _build_env(insert, assembly, server_url, fake_env)
    # #12: wrap with the SAME ChunkingWrapper the training env uses, so saved obs carry the T=1
    # temporal axis on images and load into the memory-efficient buffer without corruption.
    from serl_launcher.wrappers.chunking import ChunkingWrapper
    if any(k != "state" for k in env.observation_space.spaces.keys()):
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    base_env = env.unwrapped
    # Push L1 stop-forward into the env each step (same mechanism as PS4Intervention).
    teleop = PS4TeleopProvider(joystick_index=0, server_url=None)

    os.makedirs(out_dir, exist_ok=True)
    tag = f"{assembly}_insert{insert}"
    uid = time.strftime("%Y-%m-%d_%H-%M-%S")
    file_path = os.path.join(out_dir, f"residual_demos_{tag}_{n_demos}_{uid}.pkl")

    all_transitions: list[dict] = []
    ep_transitions: list[dict] = []
    success_count = 0
    total_episodes = 0
    recording_active = False

    # --- opt-in per-step diagnostic log (DEBUG_LOG=1): what you press vs what the robot does.
    _dbg = os.environ.get("DEBUG_LOG", "0") in ("1", "true", "True")
    _dbg_csv = None
    _dbg_step = 0
    if _dbg:
        _dbg_path = os.path.join(out_dir, f"debug_{tag}_{uid}.csv")
        _dbg_csv = open(_dbg_path, "w")
        _dbg_csv.write("step,ep,r1,l1,raw_ly,raw_lx,raw_r2,raw_l2,r2_ready,l2_ready,"
                       "act0,act1,act2,contact,progress,paused,shielded,"
                       "cmd_z,meas_x,meas_y,meas_z\n")
        print(f"[REC][DEBUG] logging per-step to {_dbg_path}")

    print(f"[REC] residual demos: {tag}  n={n_demos}  "
          f"axis={base_env.config.insertion_axis.round(2)}  speed={base_env.config.nominal_speed_mm_s}mm/s")
    print("[REC] R1=deadman  L1=stop-forward  stick=residual  X=success  Square/Tri=abort")

    try:
        obs, _ = env.reset()   # AUTOMATED joint reset to reset_move_arm_q (no manual jog)
        while success_count < n_demos:
            action = teleop.get_action()
            # Frame-aligned button reads (single pump inside get_action, like the actor).
            snap = (
                teleop.debug_snapshot()
                if hasattr(teleop, "debug_snapshot")
                else {"r1": int(np.linalg.norm(action) > 1e-6)}
            )
            deadman_held = bool(snap.get("r1", 0))
            success_pressed = teleop.is_success()
            manual_success = bool(success_pressed and deadman_held)
            manual_abort = teleop.is_failure()
            # R1 is both the motion deadman and the recording level gate. Releasing it
            # pauses the episode without adding zero/hold transitions to the demo.
            base_env.set_deadman(deadman_held)
            if deadman_held != recording_active:
                print("[REC] recording ON (R1 held)" if deadman_held else "[REC] recording PAUSED (R1 released)")
                recording_active = deadman_held
            # #11: on a terminal button, zero the action AND freeze forward so the terminal step
            # commands a hold (no last-instant push into the recorded transition).
            if manual_success or manual_abort:
                action = np.zeros(6, dtype=np.float32)
                base_env.set_stop_forward(True)
            else:
                base_env.set_stop_forward(teleop.is_stop_forward())  # L1 -> freeze nominal clock

            next_obs, rew, terminated, truncated, info = env.step(action)

            if _dbg:
                try:
                    meas = base_env.client.get_state().pose[:3]
                except Exception:
                    meas = (float("nan"),) * 3
                prog = info.get("nominal_progress", float("nan"))
                paused = int(bool(info.get("nominal_paused", False)))
                shielded = int(bool(info.get("shielded", False)))
                contact = int(paused and not bool(teleop.is_stop_forward()))  # paused w/o L1 ~ contact
                l1 = int(bool(base_env._stop_forward))
                # commanded Z = interpolate(reset_z, goal_z, progress) — what the nominal is asking for
                try:
                    rz = float(base_env.config.nominal_reset_pose6[2])
                    gz = float(base_env.config.nominal_goal_pose6[2])
                    cmd_z = (1.0 - prog) * rz + prog * gz
                except Exception:
                    cmd_z = float("nan")
                row = (f"{_dbg_step},{total_episodes},{snap.get('r1',0)},{l1},"
                       f"{snap.get('raw_ly',0):.3f},{snap.get('raw_lx',0):.3f},"
                       f"{snap.get('raw_r2',0):.3f},{snap.get('raw_l2',0):.3f},"
                       f"{snap.get('r2_ready',0)},{snap.get('l2_ready',0)},"
                       f"{action[0]:.3f},{action[1]:.3f},{action[2]:.3f},"
                       f"{contact},{prog:.3f},{paused},{shielded},"
                       f"{cmd_z:.4f},{meas[0]:.4f},{meas[1]:.4f},{meas[2]:.4f}")
                _dbg_csv.write(row + "\n"); _dbg_csv.flush()
                if _dbg_step % 5 == 0:
                    print(f"[DBG] R1={snap.get('r1',0)} L1={l1} "
                          f"lx={snap.get('raw_lx',0):+.2f} ly={snap.get('raw_ly',0):+.2f} "
                          f"act={action[:3].round(2)} "
                          f"prog={prog:.2f} paused={paused} "
                          f"cmdZ={cmd_z:.4f} measX={meas[0]:.4f} measY={meas[1]:.4f} measZ={meas[2]:.4f}")
                _dbg_step += 1

            if manual_success:
                rew, terminated = 1.0, True
            elif manual_abort:
                rew, terminated = 0.0, True

            # #5: dones marks the EPISODE boundary (terminated OR truncated) so the image buffer
            # doesn't stack the last pre-reset frame into the next episode; masks = 1 - terminated
            # (bootstrap only on a true time-limit, not on task success).
            # #7: strip the record-only ZED 'scene' key before the buffer transition (training obs
            # is {state,wrist}; scene is saved SEPARATELY for backup).
            def _policy_obs(o):
                return {k: v for k, v in o.items() if k != "scene"}
            def _scene(o):
                return o.get("scene")
            if deadman_held:
                ep_transitions.append(copy.deepcopy(dict(
                    observations=_policy_obs(obs),
                    actions=action,
                    next_observations=_policy_obs(next_obs),
                    rewards=float(rew),
                    masks=1.0 - float(terminated),
                    dones=bool(terminated or truncated),
                    scene=_scene(obs),                    # backup-only, ignored by the learner
                    shielded=bool(info.get("shielded", False)),  # #8: retract/L1 flag
                )))
            obs = next_obs

            if terminated or truncated:
                total_episodes += 1
                if manual_success:
                    all_transitions.extend(ep_transitions)
                    success_count += 1
                    print(f"[REC] SUCCESS {success_count}/{n_demos} "
                          f"(steps={len(ep_transitions)}, "
                          f"nominal_progress={info.get('nominal_progress'):.2f})")
                else:
                    print(f"[REC] discarded episode (abort/timeout, steps={len(ep_transitions)})")
                ep_transitions = []
                recording_active = False
                obs, _ = env.reset()
    finally:
        if _dbg_csv is not None:
            _dbg_csv.close()
        with open(file_path, "wb") as f:
            pkl.dump(all_transitions, f)
        print(f"\n[REC] Saved {len(all_transitions)} transitions "
              f"({success_count} successful demos) -> {file_path}")
        teleop.close()
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--insert", type=int, required=True, help="insert index (plumbers: 0..3)")
    p.add_argument("--assembly", default="plumbers_block")
    p.add_argument("--n_demos", type=int, default=20)
    p.add_argument("--server_url", default="http://127.0.0.1:5000")
    p.add_argument("--out_dir", default="demos")
    p.add_argument("--fake_env", action="store_true", help="mock backend (offline dry-run)")
    args = p.parse_args()
    record(args.insert, args.assembly, args.n_demos, args.server_url, args.out_dir, args.fake_env)


if __name__ == "__main__":
    main()
