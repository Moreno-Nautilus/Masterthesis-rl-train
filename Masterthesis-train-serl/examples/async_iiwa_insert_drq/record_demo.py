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
):
    env = gym.make(env_id, fake_env=fake_env)
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

    obs, _ = env.reset()

    try:
        while success_count < n_demos:
            action = teleop.get_action()
            next_obs, rew, terminated, truncated, info = env.step(action)

            manual_success = teleop.is_success()
            manual_abort = teleop.is_failure()

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
                episode_success = rew > 0.5
                total_episodes += 1

                if episode_success:
                    all_transitions.extend(ep_transitions)
                    success_count += 1
                    pbar.update(1)
                    status = "SUCCESS"
                else:
                    status = "abort" if manual_abort else "timeout"

                print(
                    f"  [{status}] ep={total_episodes}  "
                    f"successes={success_count}/{n_demos}  "
                    f"steps={len(ep_transitions)}"
                )
                ep_transitions = []
                obs, _ = env.reset()

    except KeyboardInterrupt:
        print("\n[DEMO] Interrupted — saving partial demos.")

    finally:
        with open(file_path, "wb") as f:
            pkl.dump(all_transitions, f)
        print(
            f"\n[DEMO] Saved {len(all_transitions)} transitions "
            f"({success_count} successes) → {file_path}"
        )
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
    )


if __name__ == "__main__":
    main()
