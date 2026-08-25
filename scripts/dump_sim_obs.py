# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Dump ONE sim observation (policy + image) as an npz, for deploy obs-parity checks.

Produces the SAME npz format the deploy node writes via `obs_dump_path`
(keys: ``policy`` (21,), ``image`` (H, W, C)), so it can be compared with:

    ros2 run rl_deploy_inference obs_parity --sim /tmp/sim_obs.npz --deploy /tmp/deploy_obs.npz

This is a pure ENV-observation snapshot: it builds the env, resets, steps a few frames with ZERO
actions (to warm the camera pipeline), and saves env-index 0's raw obs -- the policy vector + the
normalized RGB-D image, exactly what the deploy pipeline reconstructs. No policy/checkpoint is loaded
(the obs the env EMITS is independent of the trained weights), so there is no rl_games model to
size-match.

The env's obs composition is config-driven, so we force the two flags that define the DEPLOY obs
contract (they are off in the task default but were ON for w2_estimator_192):
  * e2e_use_proprio_obs = True  -> policy = [fingertip-socket(3), quat(4), linvel(3), angvel(3),
                                             force(3), prev_action(5)] = 21   (else just force+act = 8)
  * e2e_keep_aux_label  = True  -> also emits the training-only aux_label(4)  (ignored by obs_parity)

Sim and real scenes can't be matched pixel-for-pixel, so parity is about structure/units/
normalization/ranges (channel means, depth scaling, policy-vector layout), not exact pixel equality.

Launch (GPU/Isaac; same launcher as eval_robust.sh -- isaaclab conda python + the physx1065 kit app;
enable_cameras is forced on since the obs contains the wrist image):

    cd ~/Masterthesis-rl-train
    OMNI_KIT_ACCEPT_EULA=YES TORCHDYNAMO_DISABLE=1 PYTHONUNBUFFERED=1 \
      /home/moreno/miniconda3/envs/isaaclab/bin/python scripts/dump_sim_obs.py \
        --task Isaac-Insertion-CoolingPeg-Iiwa-E2E-Vision-Direct-v0 \
        --headless --enable_cameras \
        --experience /home/moreno/Masterthesis-rl-train/apps/isaaclab.python.headless.rendering.physx1065.kit \
        --num_envs 1 --warmup 2 --out /tmp/sim_obs.npz
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump one sim observation (policy+image) for deploy parity.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (1 is enough).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point", help="RL agent config entry point (unused; kept for hydra parity).")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--warmup", type=int, default=2, help="Zero-action env steps before dumping (warms the camera pipeline; keep small to stay near the pre-insert start).")
parser.add_argument("--env_index", type=int, default=0, help="Which env's obs to dump.")
parser.add_argument("--out", type=str, default="/tmp/sim_obs.npz", help="Output npz path.")
parser.add_argument(
    "--env_yaml",
    type=str,
    default=None,
    help="Path to the run's saved params/env.yaml. When given, the obs-composition flags "
    "(e2e_use_proprio_obs / e2e_goal_obs / e2e_use_velocity_obs / e2e_keep_aux_label / "
    "e2e_force_in_tcp_frame / image_* / frame_stack) are copied from it so the dumped obs matches "
    "THAT checkpoint's contract exactly. Without it, the legacy w2_estimator_192 contract is forced.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# The E2E obs contains the wrist RGB-D image, so cameras must render.
args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import random

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg

import isaaclab_tasks  # noqa: F401
import insertion_policy.tasks  # noqa: F401  (registers the insertion tasks)
from isaaclab_tasks.utils.hydra import hydra_task_config


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _obs_dict(ret):
    """Isaac reset/step returns (obs, ...); obs is the {'policy','image','aux_label',...} group dict."""
    obs = ret[0] if isinstance(ret, tuple) else ret
    if isinstance(obs, dict) and "obs" in obs and isinstance(obs["obs"], dict):
        obs = obs["obs"]
    return obs


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else getattr(env_cfg, "seed", 42)

    # Obs composition is config-driven. Prefer copying the exact flags from the run's saved env.yaml
    # (so the dump matches THAT checkpoint's obs contract); otherwise fall back to the legacy
    # w2_estimator_192 contract (21-d proprio policy + training-only aux_label).
    if args_cli.env_yaml:
        import yaml

        with open(args_cli.env_yaml) as _f:
            _saved = yaml.unsafe_load(_f)
        # These are the only keys that change the EMITTED obs (policy-vector layout + image tensor).
        _obs_keys = (
            "e2e_use_proprio_obs",
            "e2e_goal_obs",
            "e2e_use_velocity_obs",
            "e2e_keep_aux_label",
            "e2e_force_in_tcp_frame",
            "image_height",
            "image_width",
            "image_channels",
            "frame_stack",
        )
        for _k in _obs_keys:
            if _k in _saved:
                setattr(env_cfg, _k, _saved[_k])
                print(f"[dump_sim_obs] env.yaml override: {_k} = {_saved[_k]}")
    else:
        env_cfg.e2e_use_proprio_obs = True
        env_cfg.e2e_keep_aux_label = True

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    u = env.unwrapped
    act_dim = int(getattr(u.cfg, "action_space", 5))
    device = u.device

    obs = _obs_dict(env.reset())
    zero = torch.zeros((u.num_envs, act_dim), device=device)
    for _ in range(max(0, int(args_cli.warmup))):
        obs = _obs_dict(env.step(zero))

    if not (isinstance(obs, dict) and "policy" in obs and "image" in obs):
        raise SystemExit(f"Unexpected obs structure (need dict with 'policy'+'image'): keys={list(obs) if isinstance(obs, dict) else type(obs)}")

    i = int(args_cli.env_index)
    policy = _to_np(obs["policy"])[i]      # (21,)
    image = _to_np(obs["image"])[i]        # (H, W, C)
    save = {"policy": policy.astype(np.float32), "image": image.astype(np.float32)}
    if "aux_label" in obs:                  # informative only; obs_parity ignores it
        save["aux_label"] = _to_np(obs["aux_label"])[i].astype(np.float32)
    np.savez(args_cli.out, **save)

    print(f"\n[dump_sim_obs] wrote {args_cli.out}")
    print(f"  policy {policy.shape}: {np.array2string(policy, precision=4, max_line_width=200)}")
    print(f"  image  {image.shape}  dtype={image.dtype}")
    for c in range(image.shape[-1]):
        ch = image[..., c]
        print(f"    ch{c}: min={ch.min():.4f} max={ch.max():.4f} mean={ch.mean():.4f}")
    if "aux_label" in save:
        print(f"  aux_label {save['aux_label'].shape}: {np.array2string(save['aux_label'], precision=4)}  (training-only; not in deploy)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
