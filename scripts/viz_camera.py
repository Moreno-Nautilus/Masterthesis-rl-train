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
parser.add_argument("--cam_jitter", type=float, default=None,
                    help="Override env.cam_pos_jitter (m, stddev). 0.0 = clean nominal framing.")
parser.add_argument("--cam_offset", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="Override wrist_cam_offset_pos (eye, fingertip frame) for comparing mounts.")
parser.add_argument("--cam_aim", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="Override wrist_cam_look_target_pos (aim point, fingertip frame).")
parser.add_argument("--focal", type=float, default=None, help="Override tiled_camera focal_length (mm).")
parser.add_argument("--grasp_deg", type=float, default=None,
                    help="Override grasp_misalign_max_deg (exaggerate the tilt to visualize the cue).")
parser.add_argument("--cam_roll", type=float, default=None,
                    help="Override cam_rot_jitter_deg (camera roll DR stddev, deg). 0.0 = none.")
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
    if args_cli.cam_jitter is not None:
        env_cfg.cam_pos_jitter = args_cli.cam_jitter
    if args_cli.cam_roll is not None:
        env_cfg.cam_rot_jitter_deg = args_cli.cam_roll
    if args_cli.cam_offset is not None:
        env_cfg.wrist_cam_offset_pos = tuple(args_cli.cam_offset)
    if args_cli.cam_aim is not None:
        env_cfg.wrist_cam_look_target_pos = tuple(args_cli.cam_aim)
    if args_cli.focal is not None:
        env_cfg.tiled_camera.spawn.focal_length = args_cli.focal
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
    if args_cli.grasp_deg is not None:
        u.cfg_task.grasp_misalign_max_deg = args_cli.grasp_deg  # exaggerate the tilt for visualization
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

    # ---- nominal aim point in the FINGERTIP-LOCAL frame (for the rigid mount) ----
    # Express socket-opening and shaft-tip relative to the fingertip, rotated into the fingertip
    # frame. Mean over envs washes out the per-episode lateral/tilt error -> the nominal aim
    # direction the rigid camera should point along. std shows the per-episode spread.
    import isaacsim.core.utils.torch as tu
    ft_pos = u.fingertip_midpoint_pos                      # env-local
    ft_quat = u.fingertip_midpoint_quat
    ft_quat_inv = tu.quat_conjugate(ft_quat)
    socket_w = u.fixed_pos_obs_frame                       # env-local
    shaft_w, _ = u._held_base_pose()                       # env-local
    socket_local = tu.quat_apply(ft_quat_inv, socket_w - ft_pos)
    shaft_local = tu.quat_apply(ft_quat_inv, shaft_w - ft_pos)
    cam_local = tu.quat_apply(ft_quat_inv, (u._tiled_camera.data.pos_w - u.scene.env_origins) - ft_pos)
    emit(f"FINGERTIP-LOCAL (mean over {u.num_envs} envs +/- std):")
    emit(f"  socket_opening_local mean={[round(x,4) for x in socket_local.mean(0).tolist()]} "
         f"std={[round(x,4) for x in socket_local.std(0).tolist()]}")
    emit(f"  shaft_tip_local      mean={[round(x,4) for x in shaft_local.mean(0).tolist()]} "
         f"std={[round(x,4) for x in shaft_local.std(0).tolist()]}")
    emit(f"  cam_eye_local        mean={[round(x,4) for x in cam_local.mean(0).tolist()]}")

    # ---- grasp misalignment realized + finger pressing axis (the physical grasp-tilt axis) ----
    zaxis = torch.zeros_like(ft_pos); zoff = zaxis.clone(); zoff[:, 2] = 1.0
    held_axis = tu.quat_apply(u._held_base_pose()[1], zoff)   # screw shaft axis (world)
    grip_axis = tu.quat_apply(ft_quat, zoff)                  # gripper approach axis (world)
    cosang = (held_axis * grip_axis).sum(-1).clamp(-1.0, 1.0)
    mis_deg = torch.rad2deg(torch.arccos(cosang.abs()))       # shaft-vs-gripper tilt (abs: ignore z sign)
    emit(f"GRASP MISALIGN realized (shaft vs gripper axis, deg): mean={mis_deg.mean():.2f} "
         f"std={mis_deg.std():.2f} min={mis_deg.min():.2f} max={mis_deg.max():.2f} "
         f"(cfg max={u.cfg_task.grasp_misalign_max_deg})")
    # Finger pressing/closing axis (between the two pads) in the fingertip frame = the physical axis
    # the screw is allowed to tilt about. Tells us which local axis grasp misalignment should use.
    lp = u._robot.data.body_pos_w[:, u.left_finger_body_idx]
    rp = u._robot.data.body_pos_w[:, u.right_finger_body_idx]
    press_local = tu.quat_apply(ft_quat_inv, rp - lp)
    press_local = press_local / press_local.norm(dim=-1, keepdim=True)
    emit(f"FINGER PRESSING AXIS in fingertip frame (mean unit vec): "
         f"{[round(x,3) for x in press_local.mean(0).tolist()]}")
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
