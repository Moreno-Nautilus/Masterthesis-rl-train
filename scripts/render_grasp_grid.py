"""Verify the iiwa grasp by rendering reset grasps as a tiled grid. Two modes:

  * default: N randomized reset grasps (grasp-tilt + pre-insert-tilt ON) at the cfg's franka_fingerpad_length.
  * --sweep lo,hi: DEPTH SWEEP -- clean vertical screw, each COLUMN a different franka_fingerpad_length
    (labelled), so you can pick the value where the pads clamp the head with the shaft protruding. Smaller
    value -> screw further OUT of the jaws; larger -> retracted back in.

Each env's per-env scene camera is aimed at its own gripper, so the whole grid comes from ONE render.

    OMNI_KIT_ACCEPT_EULA=YES python scripts/render_grasp_grid.py --sweep 0.005,0.018 --num_envs 40 --cols 8 \
        --out diagnostics/grasp_sweep.png --experience <physx1065.kit>
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Insertion-CoolingPeg-Iiwa-E2E-Vision-Direct-v0")
parser.add_argument("--num_envs", type=int, default=50)
parser.add_argument("--out", type=str, default="diagnostics/grasp_grid.png")
parser.add_argument("--cols", type=int, default=10)
parser.add_argument("--settle", type=int, default=6, help="sim steps after reset before capture")
parser.add_argument("--sweep", type=str, default=None, help="'lo,hi' -> per-column franka_fingerpad_length sweep")
parser.add_argument(
    "--report_contacts",
    action="store_true",
    help="print an approximate per-env pad/head clearance report using the gripper collision-mesh face offsets",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as tu
from isaaclab.sensors import TiledCameraCfg

import isaaclab_tasks  # noqa: F401
import insertion_policy.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    N = args_cli.num_envs
    cols = args_cli.cols
    sweep = None
    if args_cli.sweep:
        lo, hi = (float(x) for x in args_cli.sweep.split(","))
        sweep = (lo, hi)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=N)

    # Clean grey geometry (DR OFF).
    if hasattr(env_cfg, "randomize_part_materials"):
        env_cfg.randomize_part_materials = False
    for a in ["photo_brightness", "photo_contrast", "photo_gain_rgb", "photo_gamma", "rgb_noise_std",
              "depth_noise_std", "depth_dropout_prob", "cam_pos_jitter", "cam_rot_jitter_deg", "cam_log_interval"]:
        if hasattr(env_cfg, a):
            setattr(env_cfg, a, 0.0)
    if hasattr(env_cfg, "light_intensity_range"):
        env_cfg.light_intensity_range = (3000.0, 3000.0)
    if hasattr(env_cfg, "key_light_intensity_range"):
        env_cfg.key_light_intensity_range = (2000.0, 2000.0)
    if sweep is not None:
        # isolate DEPTH: vertical screw, no reset noise, so tiles differ ONLY by the swept fingerpad (column).
        env_cfg.task.grasp_misalign_max_deg = 0.0
        env_cfg.task.pre_insert_tilt_max_deg = 0.0
        env_cfg.task.fixed_asset_init_pos_noise = [0.0, 0.0, 0.0]
        env_cfg.task.hand_init_pos_noise = [0.0, 0.0, 0.0]
        env_cfg.task.hand_init_orn_noise = [0.0, 0.0, 0.0]

    env_cfg.scene_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/scene_cam",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 8.0)
        ),
        width=320,
        height=320,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    dev = u.device
    env.reset()
    zero = torch.zeros((u.num_envs, u.cfg.action_space), device=dev)
    for _ in range(args_cli.settle):
        env.step(zero)

    if args_cli.report_contacts:
        li = u._robot.body_names.index("left_finger_link")
        ri = u._robot.body_names.index("right_finger_link")
        gbi = u._robot.body_names.index("gripper_base_link")
        lfj = u._robot.joint_names.index("left_finger_joint")
        rfj = u._robot.joint_names.index("right_finger_joint")
        gb_pos = u._robot.data.body_pos_w[:, gbi]
        gb_qi = tu.quat_conjugate(u._robot.data.body_quat_w[:, gbi])
        lf = tu.quat_apply(gb_qi, u._robot.data.body_pos_w[:, li] - gb_pos)
        rf = tu.quat_apply(gb_qi, u._robot.data.body_pos_w[:, ri] - gb_pos)
        held = tu.quat_apply(gb_qi, u.held_pos + u.scene.env_origins - gb_pos)

        # The custom gripper collision meshes have nonzero face offsets from their link origins; q=0 still
        # leaves the two inner faces about 4 mm apart. These constants come from the exported USD mesh bounds.
        face_offset = 0.042
        radius = 0.5 * float(u.cfg_task.held_asset_cfg.diameter)
        left_face_y = lf[:, 1] - face_offset
        right_face_y = rf[:, 1] + face_offset
        left_clearance = (held[:, 1] - radius) - left_face_y
        right_clearance = right_face_y - (held[:, 1] + radius)
        max_gap = torch.maximum(left_clearance, right_clearance).max()
        max_pen = torch.minimum(left_clearance, right_clearance).min()
        bad_gap = ((left_clearance > 0.001) | (right_clearance > 0.001)).nonzero(as_tuple=False).flatten()
        bad_pen = ((left_clearance < -0.001) | (right_clearance < -0.001)).nonzero(as_tuple=False).flatten()
        worst_pen_val, worst_pen_idx = torch.minimum(left_clearance, right_clearance).sort()
        worst_gap_val, worst_gap_idx = torch.maximum(left_clearance, right_clearance).sort(descending=True)
        report_lines = [
            "[contacts] clearance mm: "
            f"left min/mean/max={left_clearance.min().item()*1000:.2f}/"
            f"{left_clearance.mean().item()*1000:.2f}/{left_clearance.max().item()*1000:.2f}, "
            f"right min/mean/max={right_clearance.min().item()*1000:.2f}/"
            f"{right_clearance.mean().item()*1000:.2f}/{right_clearance.max().item()*1000:.2f}; "
            f"max_gap={max_gap.item()*1000:.2f}mm max_pen={max_pen.item()*1000:.2f}mm "
            f"envs_gap>1mm={bad_gap[:20].tolist()} envs_pen>1mm={bad_pen[:20].tolist()}",
            "[contacts] joint q mm: "
            f"left={u._robot.data.joint_pos[:, lfj].mean().item()*1000:.2f}, "
            f"right={u._robot.data.joint_pos[:, rfj].mean().item()*1000:.2f}; "
            f"screw_y mean={held[:, 1].mean().item()*1000:.2f}mm",
        ]
        report_lines.append("[contacts] worst penetration envs:")
        for idx in worst_pen_idx[:10].tolist():
            report_lines.append(
                f"  env {idx:03d}: left={left_clearance[idx].item()*1000:+.2f}mm "
                f"right={right_clearance[idx].item()*1000:+.2f}mm "
                f"lf_y={lf[idx, 1].item()*1000:+.2f} rf_y={rf[idx, 1].item()*1000:+.2f} "
                f"screw_y={held[idx, 1].item()*1000:+.2f} "
                f"q=({u._robot.data.joint_pos[idx, lfj].item()*1000:.2f},"
                f"{u._robot.data.joint_pos[idx, rfj].item()*1000:.2f})"
            )
        report_lines.append("[contacts] worst gap envs:")
        for idx in worst_gap_idx[:10].tolist():
            report_lines.append(
                f"  env {idx:03d}: left={left_clearance[idx].item()*1000:+.2f}mm "
                f"right={right_clearance[idx].item()*1000:+.2f}mm "
                f"lf_y={lf[idx, 1].item()*1000:+.2f} rf_y={rf[idx, 1].item()*1000:+.2f} "
                f"screw_y={held[idx, 1].item()*1000:+.2f} "
                f"q=({u._robot.data.joint_pos[idx, lfj].item()*1000:.2f},"
                f"{u._robot.data.joint_pos[idx, rfj].item()*1000:.2f})"
            )
        for line in report_lines:
            print(line, flush=True)
        report_path = os.path.splitext(args_cli.out)[0] + ".contacts.txt"
        os.makedirs(os.path.dirname(os.path.abspath(report_path)) or ".", exist_ok=True)
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines) + "\n")

    col_vals = None
    if sweep is not None:
        lo, hi = sweep
        colidx = (torch.arange(N, device=dev) % cols).float()
        fps = lo + (hi - lo) * colidx / max(1, cols - 1)
        hh = u.cfg_task.held_asset_cfg.height - u.cfg_task.screw_shaft_length  # head height (shoulder origin)
        off = torch.zeros((N, 3), device=dev)
        off[:, 2] = hh - fps  # held-asset z-offset in the fingertip frame (the grasp formula, per env)
        # Keep the screw's CORRECT orientation from the reset grasp (using the fingertip quat directly flips it,
        # since it drops held_asset_relative_quat). We only vary the DEPTH (z-offset) per column.
        held_quat = u._held_asset.data.root_state_w[:, 3:7].clone()
        # Pin the screw at the swept depth for a few steps so it holds against the closed jaws for the render.
        for _ in range(4):
            ftp, ftq = u.fingertip_midpoint_pos, u.fingertip_midpoint_quat
            hp = ftp + tu.quat_apply(ftq, off)
            st = u._held_asset.data.root_state_w.clone()
            st[:, 0:3] = hp + u.scene.env_origins
            st[:, 3:7] = held_quat
            st[:, 7:13] = 0.0
            u._held_asset.write_root_state_to_sim(st)
            env.step(zero)
        col_vals = [lo + (hi - lo) * c / max(1, cols - 1) for c in range(cols)]

    o = u.scene.env_origins
    ft = u.fingertip_midpoint_pos + o
    center = ft + torch.tensor([0.0, 0.0, -0.018], device=dev)
    eyes = center + torch.tensor([0.075, 0.05, 0.022], device=dev)
    u._scene_camera.set_world_poses_from_view(eyes, center)
    for _ in range(2):
        if sweep is not None:  # keep the screw pinned (correct orientation) while re-aiming
            hp = u.fingertip_midpoint_pos + tu.quat_apply(u.fingertip_midpoint_quat, off)
            st = u._held_asset.data.root_state_w.clone()
            st[:, 0:3] = hp + o
            st[:, 3:7] = held_quat
            st[:, 7:13] = 0.0
            u._held_asset.write_root_state_to_sim(st)
        env.step(zero)

    imgs = u._scene_camera.data.output["rgb"][:, :, :, :3].clamp(0, 255).to(torch.uint8).cpu().numpy()

    rows = (N + cols - 1) // cols
    h, w = imgs.shape[1], imgs.shape[2]
    grid = np.full((rows * h, cols * w, 3), 25, np.uint8)
    for i in range(N):
        r, c = divmod(i, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = imgs[i]

    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(grid)
        d = ImageDraw.Draw(im)
        if col_vals is not None:
            for c in range(cols):
                d.text((c * w + 6, 6), f"fp={col_vals[c]:.4f}", fill=(255, 255, 0))
        im.save(args_cli.out)
    except Exception:
        import imageio.v2 as imageio
        imageio.imwrite(args_cli.out, grid)
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)) or ".", exist_ok=True)
    print(f"[OK] wrote {args_cli.out}  ({N} grasps, {rows}x{cols})" + (f"  sweep {sweep}" if sweep else ""))

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
