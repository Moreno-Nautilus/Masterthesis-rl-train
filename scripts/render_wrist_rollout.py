# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Render the WRIST CAMERA POV of a single env as a video while a trained policy runs.

Unlike ``eval_policy.py --video`` (which records the third-person viewport), this dumps what the
wrist RGB-D camera actually SEES each step -- RGB and depth side by side -- for one chosen env, so
you can watch the part enter the socket from the policy's own eye. Output: an mp4 (falls back to a
numbered PNG sequence if no ffmpeg writer is available).

    OMNI_KIT_ACCEPT_EULA=YES python scripts/render_wrist_rollout.py \
        --task Isaac-Insertion-CoolingPeg-Vision-Direct-v0 --num_envs 16 \
        --checkpoint logs/rl_games/Forge/vision_160_1/nn/Forge.pth \
        --frames 300 --env 0 --out /tmp/wrist_rollout.mp4 \
        --experience /home/moreno/Masterthesis-rl-train/apps/isaaclab.python.headless.rendering.physx1065.kit
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render the wrist-cam POV of a trained policy rollout.")
parser.add_argument("--task", type=str, default="Isaac-Insertion-CoolingPeg-Vision-Direct-v0")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth).")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--env", type=int, default=0, help="Which env to follow.")
parser.add_argument("--view", type=str, default="wrist", choices=["wrist", "scene"],
                    help="wrist = the wrist RGB-D POV; scene = a close third-person view of the env.")
parser.add_argument("--frames", type=int, default=300, help="Number of frames (sim steps) to record.")
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--out", type=str, default="/tmp/wrist_rollout.mp4")
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


def _scene_frame(u, env_idx):
    """One (H, W, 3) uint8 third-person frame of env `env_idx`."""
    rgb = u._scene_camera.data.output["rgb"][env_idx, :, :, :3].clamp(0, 255).to(torch.uint8)
    return rgb.cpu().numpy()


def _aim_scene_camera(u):
    """Aim the third-person camera at each env's socket+gripper region (close 3/4 view)."""
    target = 0.5 * (u.fixed_pos_obs_frame + u.fingertip_midpoint_pos) + u.scene.env_origins
    eye = target + torch.tensor([0.16, 0.13, 0.14], device=u.device)
    u._scene_camera.set_world_poses_from_view(eye, target)


def _to_frame(u, env_idx, far):
    """Build one uint8 frame: raw RGB, plus depth side-by-side IF the camera outputs depth.

    RGB-only tasks (e.g. pdz_v3) have no 'depth' key -> return just the RGB frame.
    """
    out = u._tiled_camera.data.output
    rgb = out["rgb"][env_idx, :, :, :3].clamp(0, 255).to(torch.uint8).cpu().numpy()

    if "depth" not in out:
        return rgb  # RGB-only policy view

    depth = out["depth"][env_idx].clone().float()
    depth[~torch.isfinite(depth)] = 0.0
    depth = (depth.clamp(0.0, far) / far * 255.0).to(torch.uint8).cpu().numpy()
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = np.repeat(depth[:, :, None], 3, axis=2)  # gray -> 3ch

    sep = np.full((rgb.shape[0], 4, 3), 255, dtype=np.uint8)  # white divider
    return np.concatenate([rgb, sep, depth], axis=1)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.seed is not None:
        agent_cfg["params"]["seed"] = args_cli.seed
    env_cfg.seed = agent_cfg["params"]["seed"]

    resume_path = retrieve_file_path(args_cli.checkpoint)

    # Add a close third-person camera for the "scene" view (the env builds it from cfg.scene_camera).
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
    # far-clip only needed for wrist-DEPTH normalization; the state task has no wrist cam (scene view only).
    _tc = getattr(u.cfg, "tiled_camera", None)
    far = float(_tc.spawn.clipping_range[1]) if _tc is not None else 5.0
    env_idx = max(0, min(args_cli.env, u.num_envs - 1))

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    frames = []
    print(f"[INFO] recording {args_cli.frames} {args_cli.view} frames of env {env_idx} ...")
    for i in range(args_cli.frames):
        with torch.inference_mode():
            if args_cli.view == "scene":
                _aim_scene_camera(u)  # aim before stepping so the in-step render uses this pose
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            obs, _, dones, _ = env.step(actions)
            frames.append(_scene_frame(u, env_idx) if args_cli.view == "scene" else _to_frame(u, env_idx, far))
            if agent.is_rnn and agent.states is not None:
                d = dones.nonzero(as_tuple=False).squeeze(-1)
                if len(d) > 0:
                    for s in agent.states:
                        s[:, d, :] = 0.0

    env.close()

    # Write video (mp4 via imageio-ffmpeg; fall back to PNG sequence).
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)) or ".", exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.mimwrite(args_cli.out, frames, fps=args_cli.fps, macro_block_size=None)
        print(f"[OK] wrote {args_cli.out} ({len(frames)} frames, {args_cli.fps} fps)  [left=RGB | right=depth]")
    except Exception as exc:
        seq_dir = os.path.splitext(args_cli.out)[0] + "_frames"
        os.makedirs(seq_dir, exist_ok=True)
        try:
            import imageio.v2 as imageio

            for i, f in enumerate(frames):
                imageio.imwrite(os.path.join(seq_dir, f"frame_{i:04d}.png"), f)
        except Exception:
            np.save(os.path.splitext(args_cli.out)[0] + "_frames.npy", np.stack(frames))
        print(f"[WARN] mp4 write failed ({exc}); wrote frames to {seq_dir}")
        print(f"       ffmpeg -framerate {args_cli.fps} -i {seq_dir}/frame_%04d.png -pix_fmt yuv420p {args_cli.out}")


if __name__ == "__main__":
    main()
    simulation_app.close()
