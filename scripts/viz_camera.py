"""Visualize the wrist camera: dump what it SEES (RGB + depth) and a third-person view of HOW it is
MOUNTED. Adds a temporary external "scene" camera (cfg.scene_camera) that looks at env_0's workspace.

Outputs to /tmp:  wrist_rgb.png, wrist_depth.png (the wrist cam POV), scene_view.png (third-person).
Also prints the camera/gripper/socket geometry.

    OMNI_KIT_ACCEPT_EULA=YES python scripts/viz_camera.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Insertion-CoolingPeg-Vision-Direct-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=6)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import traceback

import torch
import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg, save_images_to_file

REPORT = []


def emit(line):
    REPORT.append(str(line))


def main():
    import isaaclab_tasks  # noqa: F401
    import insertion_policy.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    # Dump the wrist-cam POV frames, and add a third-person scene camera.
    env_cfg.write_image_to_file = True
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

    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    env.reset()

    n_act = u.cfg.action_space
    zero = torch.zeros((args_cli.num_envs, n_act), device=u.device)

    # Position the third-person camera to look at env_0's workspace (from above/side/front).
    for i in range(args_cli.steps):
        env.step(zero)
        # aim at a point between socket and gripper so the mount (cam-on-wrist + part) is in frame
        target = 0.5 * (u.fixed_pos_obs_frame.clone() + u.fingertip_midpoint_pos + u.scene.env_origins)
        eye = target + torch.tensor([0.16, 0.13, 0.14], device=u.device)  # ~0.25m, 3/4 close view
        u._scene_camera.set_world_poses_from_view(eye, target)

    # one more step so the scene cam renders at its set pose
    env.step(zero)

    # ---- save all renders ----
    scene_rgb = u._scene_camera.data.output["rgb"][..., :3].float() / 255.0
    save_images_to_file(scene_rgb, "/tmp/scene_view.png")
    emit("wrote /tmp/scene_view.png (third-person), /tmp/wrist_rgb.png + /tmp/wrist_depth.png (wrist POV)")

    # ---- geometry readout (env 0) ----
    o = u.scene.env_origins[0]
    cam = u._tiled_camera.data.pos_w[0] - o
    ft = u.fingertip_midpoint_pos[0]
    hb, _ = u._held_base_pose()
    shaft_tip = hb[0]
    socket = u.fixed_pos_obs_frame[0] - o
    dist = torch.linalg.norm((u._tiled_camera.data.pos_w[0]) - u.fixed_pos_obs_frame[0]).item()
    emit(f"GEOM env0 (rel env origin): wrist_cam={[round(x,3) for x in cam.tolist()]}")
    emit(f"  fingertip={[round(x,3) for x in ft.tolist()]}  shaft_tip={[round(x,3) for x in shaft_tip.tolist()]}")
    emit(f"  socket_opening={[round(x,3) for x in socket.tolist()]}")
    emit(f"  wrist_cam -> socket distance = {dist*100:.1f} cm")
    emit(f"  wrist obs image shape = {tuple(u._get_camera_image().shape)}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        emit("FAILED\n" + traceback.format_exc())
    finally:
        with open("/tmp/viz_camera_report.txt", "w") as f:
            f.write("\n".join(REPORT) + "\n")
        simulation_app.close()
