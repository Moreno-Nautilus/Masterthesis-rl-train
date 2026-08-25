# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Render N separate rollout clips (one per env) from a SINGLE Isaac session.

render_wrist_rollout.py follows ONE env per launch; inspecting 10 "random goes" that way needs 10
launches. This runs the trained policy once over `num_envs` parallel envs and dumps the first
`n_clips` of them as separate videos, so 10 goes = 1 launch. Scene (third-person) view by default
(best for judging reset placement + seating); `--view wrist` dumps the RGB|depth POV instead.

    OMNI_KIT_ACCEPT_EULA=YES python scripts/render_rollout_multi.py \
        --task Isaac-Insertion-CoolingPeg-Iiwa-E2E-Vision-Direct-v0 --num_envs 16 --n_clips 10 \
        --checkpoint logs/rl_games/Forge/e2e_socket_grasp/nn/last_Forge_ep_1000_rew_97.57502.pth \
        --frames 160 --view scene --out_dir diagnostics/renders --prefix socket_go \
        --experience /home/moreno/Masterthesis-rl-train/apps/isaaclab.python.headless.rendering.physx1065.kit \
        env.e2e_use_proprio_obs=True env.e2e_socket_anchored_action=True env.weekend_tilt_deg=18.0
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render N per-env rollout clips from one policy launch.")
parser.add_argument("--task", type=str, default="Isaac-Insertion-CoolingPeg-Iiwa-E2E-Vision-Direct-v0")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth).")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--n_clips", type=int, default=10, help="How many envs (clips) to record.")
parser.add_argument("--view", type=str, default="scene", choices=["wrist", "scene"])
parser.add_argument("--stills", action="store_true", help="save a PNG of the settled frame per clip (no video)")
parser.add_argument("--frames", type=int, default=160, help="Number of frames (sim steps) to record.")
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--out_dir", type=str, default="diagnostics/renders")
parser.add_argument("--prefix", type=str, default="go")
parser.add_argument("--seed", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import os

import gymnasium as gym
import numpy as np
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import insertion_policy.tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


def _aim_scene_camera(u):
    """Aim each env's third-person camera at its own socket+gripper region (close 3/4 view)."""
    target = 0.5 * (u.fixed_pos_obs_frame + u.fingertip_midpoint_pos) + u.scene.env_origins
    eye = target + torch.tensor([0.16, 0.13, 0.14], device=u.device)
    u._scene_camera.set_world_poses_from_view(eye, target)


def _scene_frame(u, env_idx):
    rgb = u._scene_camera.data.output["rgb"][env_idx, :, :, :3].clamp(0, 255).to(torch.uint8)
    return rgb.cpu().numpy()


def _wrist_frame(u, env_idx, far):
    out = u._tiled_camera.data.output
    rgb = out["rgb"][env_idx, :, :, :3].clamp(0, 255).to(torch.uint8).cpu().numpy()
    if "depth" not in out:  # RGB-only config (current pdz/e2e setup) -> just the RGB POV
        return rgb
    depth = out["depth"][env_idx].clone().float()
    depth[~torch.isfinite(depth)] = 0.0
    depth = (depth.clamp(0.0, far) / far * 255.0).to(torch.uint8).cpu().numpy()
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = np.repeat(depth[:, :, None], 3, axis=2)
    sep = np.full((rgb.shape[0], 4, 3), 255, dtype=np.uint8)
    return np.concatenate([rgb, sep, depth], axis=1)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.seed is not None:
        agent_cfg["params"]["seed"] = args_cli.seed
    env_cfg.seed = agent_cfg["params"]["seed"]

    resume_path = retrieve_file_path(args_cli.checkpoint)

    if args_cli.view == "scene":
        env_cfg.scene_camera = TiledCameraCfg(
            prim_path="/World/envs/env_.*/scene_cam",
            offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 5.0)
            ),
            width=360,
            height=360,
        )

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    u = env.unwrapped
    _tc = getattr(u.cfg, "tiled_camera", None)
    far = float(_tc.spawn.clipping_range[1]) if _tc is not None else 5.0
    n_clips = max(1, min(args_cli.n_clips, u.num_envs))

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    clips = [[] for _ in range(n_clips)]
    print(f"[INFO] recording {args_cli.frames} {args_cli.view} frames for {n_clips} envs ...")
    for i in range(args_cli.frames):
        with torch.inference_mode():
            if args_cli.view == "scene":
                _aim_scene_camera(u)
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            obs, _, dones, _ = env.step(actions)
            for e in range(n_clips):
                clips[e].append(_scene_frame(u, e) if args_cli.view == "scene" else _wrist_frame(u, e, far))
            if agent.is_rnn and agent.states is not None:
                d = dones.nonzero(as_tuple=False).squeeze(-1)
                if len(d) > 0:
                    for s in agent.states:
                        s[:, d, :] = 0.0

    env.close()

    os.makedirs(args_cli.out_dir, exist_ok=True)
    import imageio.v2 as imageio

    if args_cli.stills:  # PNG of the last (settled) frame per clip -- faster than encoding video
        for e in range(n_clips):
            out = os.path.join(args_cli.out_dir, f"{args_cli.prefix}_{e:02d}.png")
            imageio.imwrite(out, clips[e][-1])
            print(f"[OK] wrote {out}")
    else:
        for e in range(n_clips):
            out = os.path.join(args_cli.out_dir, f"{args_cli.prefix}_{e:02d}.mp4")
            imageio.mimwrite(out, clips[e], fps=args_cli.fps, macro_block_size=None)
            print(f"[OK] wrote {out}  ({len(clips[e])} frames)")


if __name__ == "__main__":
    main()
    simulation_app.close()
