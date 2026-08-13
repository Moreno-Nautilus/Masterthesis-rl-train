# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Render rollout clips with the EXPLICIT-ESTIMATOR's predicted hole marked on the 3rd-person view.

For each frame we take the aux head's predicted tip->socket gap (the value fed into the policy),
DENORMALIZE it with the model's aux_label RunningMeanStd, turn it into a world position, and project
it onto the scene camera:
  * GREEN ring  = TRUE socket opening (u._socket_opening_pos())
  * RED  ring   = ESTIMATOR's predicted hole
  * yellow line = error between them
So you literally watch the red estimate snap onto the green socket as the tip approaches. Requires the
estimator task + env.e2e_keep_aux_label=True (so the aux head + its label are present).

    OMNI_KIT_ACCEPT_EULA=YES python scripts/render_estimator_overlay.py \
        --task Isaac-Insertion-CoolingPeg-Iiwa-E2E-Vision-Estimator-Direct-v0 --num_envs 8 --n_clips 4 \
        --checkpoint logs/rl_games/Forge/w2_estimator_192/nn/last_Forge_ep_2000_rew_162.13815.pth \
        --frames 160 --out_dir diagnostics/renders --prefix w2_estimator \
        --experience /home/moreno/Masterthesis-rl-train/apps/isaaclab.python.headless.rendering.physx1065.kit \
        env.e2e_use_proprio_obs=True env.e2e_socket_anchored_action=True env.e2e_keep_aux_label=True \
        env.weekend_tilt_deg=20.0 env.tilt_curriculum_steps=128000 env.fixed_asset_pos_obs_noise_curriculum_steps=128000 \
        env.use_gravity_comp=True env.e2e_reward_center_weight=20.0 \
        env.wrist_cam_offset_pos=[0.05567,0.00900,-0.07472] env.wrist_cam_offset_quat=[-0.18301,0.68301,-0.68301,0.18301]
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render rollout clips with the estimator's predicted hole marked.")
parser.add_argument("--task", type=str, default="Isaac-Insertion-CoolingPeg-Iiwa-E2E-Vision-Estimator-Direct-v0")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the estimator checkpoint (.pth).")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--n_clips", type=int, default=4, help="How many envs (clips) to record.")
parser.add_argument("--frames", type=int, default=160, help="Number of frames (sim steps) to record.")
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--out_dir", type=str, default="diagnostics/renders")
parser.add_argument("--prefix", type=str, default="estimator")
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
from isaaclab.utils.math import quat_apply, quat_rotate_inverse
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import insertion_policy.tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


def _aim_scene_camera(u):
    target = 0.5 * (u.fixed_pos_obs_frame + u.fingertip_midpoint_pos) + u.scene.env_origins
    eye = target + torch.tensor([0.16, 0.13, 0.14], device=u.device)
    u._scene_camera.set_world_poses_from_view(eye, target)


def _project(cam, p_world, e):
    """Project a world point (3,) onto scene-camera env e -> (px, py, z_forward). ROS optical frame."""
    K = cam.data.intrinsic_matrices[e]
    pos = cam.data.pos_w[e]
    quat = cam.data.quat_w_ros[e]
    p_cam = quat_rotate_inverse(quat.unsqueeze(0), (p_world - pos).unsqueeze(0))[0]  # into optical frame
    z = float(p_cam[2].clamp(min=1e-4))
    px = float(K[0, 0] * p_cam[0] / z + K[0, 2])
    py = float(K[1, 1] * p_cam[1] / z + K[1, 2])
    return px, py, float(p_cam[2])


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.seed is not None:
        agent_cfg["params"]["seed"] = args_cli.seed
    env_cfg.seed = agent_cfg["params"]["seed"]
    resume_path = retrieve_file_path(args_cli.checkpoint)

    env_cfg.scene_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/scene_cam",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 5.0)
        ),
        width=480,
        height=480,
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
    net = agent.model.a2c_network
    # aux_label RunningMeanStd slice [1:4] -> denormalize the predicted hole gap back to meters.
    aux_mean = torch.zeros(3, device=u.device)
    aux_std = torch.ones(3, device=u.device)
    try:
        aux = agent.model.running_mean_std.running_mean_std["aux_label"]
        aux_mean = aux.running_mean.float()[1:4]
        aux_std = torch.sqrt(aux.running_var.float()[1:4] + 1e-5)
        print("[INFO] using aux_label RMS to denormalize the estimator prediction")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] no aux_label RMS ({exc}); prediction treated as raw meters")

    n_clips = max(1, min(args_cli.n_clips, u.num_envs))
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    import cv2
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    # In-scene 3D markers at the TRUE and PREDICTED hole WORLD positions. Isaac renders them at the right
    # place in 3D, so there is NO manual camera projection (my pinhole projection was landing at image
    # center -- wrong optical-frame convention). cyan sphere = true socket opening, magenta sphere = the
    # estimator's predicted hole; the following 3/4 scene camera captures both.
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/est_holes",
        markers={
            "true": sim_utils.SphereCfg(radius=0.005, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0))),
            "pred": sim_utils.SphereCfg(radius=0.0045, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0))),
        },
    )
    markers = VisualizationMarkers(marker_cfg)

    def _project_to_cam(cam, p_world, e):
        """Project a world point onto camera ``cam`` for env e -> (px, py, z_forward) in ROS optical frame."""
        K = cam.data.intrinsic_matrices[e]
        pos = cam.data.pos_w[e]
        quat = cam.data.quat_w_ros[e]
        p_cam = quat_rotate_inverse(quat.unsqueeze(0), (p_world - pos).unsqueeze(0))[0]
        z = float(p_cam[2].clamp(min=1e-4))
        return float(K[0, 0] * p_cam[0] / z + K[0, 2]), float(K[1, 1] * p_cam[1] / z + K[1, 2]), float(p_cam[2])

    clips = [[] for _ in range(n_clips)]
    wrist_clips = [[] for _ in range(n_clips)]  # policy-SEEN wrist view w/ true(green)+pred(red) hole
    print(f"[INFO] recording {args_cli.frames} scene frames w/ 3D hole markers for {n_clips} envs ...")
    for _fi in range(args_cli.frames):
        with torch.inference_mode():
            _aim_scene_camera(u)
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)  # fills _estimator_pred

            # world hole positions (env-local + env origin, matching _aim_scene_camera's convention)
            org = u.scene.env_origins
            tip = u._held_base_pose()[0] + org
            true_hole = u._socket_opening_pos() + org
            pred = getattr(net, "_estimator_pred", None)
            if pred is not None:
                gap_w = quat_apply(u.fingertip_midpoint_quat, pred.float() * aux_std + aux_mean)
                pred_hole = tip + gap_w
                trans = torch.cat([true_hole, pred_hole], dim=0)
                idx = torch.cat([torch.zeros(u.num_envs, dtype=torch.long, device=u.device),
                                 torch.ones(u.num_envs, dtype=torch.long, device=u.device)])
                err_mm = torch.linalg.vector_norm(pred_hole - true_hole, dim=1) * 1000.0
            else:
                trans, idx, err_mm = true_hole, torch.zeros(u.num_envs, dtype=torch.long, device=u.device), None
            markers.visualize(translations=trans, marker_indices=idx)

            obs, _, dones, _ = env.step(actions)  # renders the scene INCLUDING the markers

            for e in range(n_clips):
                frame = u._scene_camera.data.output["rgb"][e, :, :, :3].clamp(0, 255).to(torch.uint8).cpu().numpy()
                frame = np.ascontiguousarray(frame)
                H = frame.shape[0]
                if err_mm is not None:
                    cv2.putText(frame, f"estimator error: {err_mm[e]:4.0f} mm", (8, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "cyan = TRUE hole", (8, H - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "magenta = ESTIMATE", (8, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
                clips[e].append(frame)

            # WRIST (policy-seen) view: project the TRUE (green) + PREDICTED (red) hole onto the exact wrist
            # RGB the policy consumes, so you see pred-vs-GT on the corrupted-depth-era image, not just the
            # clean 3rd-person scene. Uses the raw wrist RGB (pre-normalization) for legibility.
            wcam = getattr(u, "_tiled_camera", None)
            if wcam is not None:
                for e in range(n_clips):
                    wf = wcam.data.output["rgb"][e, :, :, :3].clamp(0, 255).to(torch.uint8).cpu().numpy()
                    wf = np.ascontiguousarray(wf)
                    Hs, Ws = wf.shape[:2]
                    tu, tv, tz = _project_to_cam(wcam, true_hole[e], e)
                    if tz > 1e-4 and 0 <= tu < Ws and 0 <= tv < Hs:
                        cv2.circle(wf, (int(tu), int(tv)), 6, (0, 255, 0), 2, cv2.LINE_AA)
                    if pred is not None:
                        pu, pv, pz = _project_to_cam(wcam, pred_hole[e], e)
                        if pz > 1e-4 and 0 <= pu < Ws and 0 <= pv < Hs:
                            cv2.circle(wf, (int(pu), int(pv)), 5, (0, 0, 255), 2, cv2.LINE_AA)
                    wrist_clips[e].append(wf)

            if agent.is_rnn and agent.states is not None:
                d = dones.nonzero(as_tuple=False).squeeze(-1)
                if len(d) > 0:
                    for s in agent.states:
                        s[:, d, :] = 0.0

    env.close()
    os.makedirs(args_cli.out_dir, exist_ok=True)
    import imageio.v2 as imageio

    for e in range(n_clips):
        out = os.path.join(args_cli.out_dir, f"{args_cli.prefix}_{e:02d}.mp4")
        imageio.mimwrite(out, clips[e], fps=args_cli.fps, macro_block_size=None)
        print(f"[OK] wrote {out}  ({len(clips[e])} frames)")
        if wrist_clips[e]:
            wout = os.path.join(args_cli.out_dir, f"{args_cli.prefix}_{e:02d}_wrist.mp4")
            imageio.mimwrite(wout, wrist_clips[e], fps=args_cli.fps, macro_block_size=None)
            print(f"[OK] wrote {wout}  ({len(wrist_clips[e])} frames)  green=TRUE red=ESTIMATE")


if __name__ == "__main__":
    main()
    simulation_app.close()
