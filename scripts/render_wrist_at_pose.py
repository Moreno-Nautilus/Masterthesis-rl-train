"""Render the wrist-cam RGB at a SPECIFIC joint pose (no policy rollout).

Boots the vision env, writes a fixed 7-DoF iiwa joint pose (e.g. copied from a real
rosbag), steps once so the wrist camera pose + render update, and saves the 320x180
wrist RGB. For sim-vs-real comparison against the deploy policy_image_320x180.png.

    OMNI_KIT_ACCEPT_EULA=YES python scripts/render_wrist_at_pose.py \
        --task Isaac-Insertion-CoolingPeg-Iiwa-E2E-Vision-Direct-v0 \
        --q 0.041484 1.089750 -0.155479 -1.112798 0.133833 1.033430 1.371645 \
        --out ~/Masterthesis-rl-deploy/sim_wrist_at_pose.png \
        --experience ~/Masterthesis-rl-train/apps/isaaclab.python.headless.rendering.physx1065.kit
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Insertion-CoolingPeg-Iiwa-E2E-Vision-Direct-v0")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
parser.add_argument("--q", type=float, nargs=7, required=True, help="7 iiwa joint angles [rad] A1..A7")
parser.add_argument("--out", type=str, default="/tmp/sim_wrist_at_pose.png")
parser.add_argument("--settle", type=int, default=4, help="sim steps to settle before capture")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import numpy as np
import torch
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import insertion_policy.tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

try:
    import cv2
except Exception:
    cv2 = None


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: dict):
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 0

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    u = env.unwrapped
    u.reset()

    q = torch.tensor(args_cli.q, device=u.device, dtype=torch.float32).unsqueeze(0)
    # write the 7 arm joints (leave any gripper joints at their reset value)
    n = u._robot.num_joints
    full = u._robot.data.joint_pos.clone()
    full[:, :7] = q
    u._robot.write_joint_state_to_sim(full, torch.zeros_like(full))
    u._robot.set_joint_position_target(full)

    for _ in range(max(1, args_cli.settle)):
        u.scene.write_data_to_sim()
        u.sim.step()
        u.scene.update(dt=u.physics_dt)
    # make sure the wrist camera rigidly follows the (now updated) fingertip
    if hasattr(u, "_update_camera_pose"):
        u._update_camera_pose()
    u.sim.render()
    u._tiled_camera.update(dt=0.0)

    out = u._tiled_camera.data.output
    rgb = out["rgb"][0, :, :, :3].clamp(0, 255).to(torch.uint8).cpu().numpy()
    path = os.path.expanduser(args_cli.out)
    if cv2 is not None:
        cv2.imwrite(path, rgb[:, :, ::-1])  # RGB -> BGR
    else:
        from PIL import Image
        Image.fromarray(rgb).save(path)
    print(f"WROTE {path}  shape={rgb.shape}  (wrist RGB at q={args_cli.q})")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
