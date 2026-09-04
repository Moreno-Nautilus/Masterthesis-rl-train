"""Render the wrist-cam RGB at a SPECIFIC joint pose, using a RUN's SAVED env.yaml.

This is the sim2real-correct pose render. Unlike render_wrist_at_pose.py (which builds
the env from HYDRA DEFAULTS -> wrong base placement / DR / FOV vs the trained checkpoint,
giving an INVALID sim-vs-real comparison), this loads the exact env config the checkpoint
was trained with (logs/.../<run>/params/env.yaml), writes a fixed 7-DoF iiwa joint pose
(copied from a real rosbag / the deploy log), steps to settle, and saves the 320x180 wrist
RGB. Compare it directly against the real deploy policy_image_320x180.png to MEASURE the
sim<->real wrist-camera viewpoint offset before retuning wrist_cam_offset_pos/look_target.

    OMNI_KIT_ACCEPT_EULA=YES python scripts/render_wrist_at_pose_from_run.py \
        --task Isaac-Insertion-CoolingPeg-Iiwa-E2E-Vision-Direct-v0 \
        --env_yaml logs/rl_games/Forge/pdz_v3_fulltilt_20260826/params/env.yaml \
        --q 0.041484 1.089750 -0.155479 -1.112798 0.133833 1.033430 1.371645 \
        --out diagnostics/renders/sim_wrist_at_pose.png \
        --experience ~/Masterthesis-rl-train/apps/isaaclab.python.headless.rendering.physx1065.kit

Optional overrides (to A/B a camera offset WITHOUT editing the cfg / retraining):
    --cam_pos 0.009 -0.05056 -0.07257   --cam_look 0.0 0.0 0.07
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Insertion-CoolingPeg-Iiwa-E2E-Vision-Direct-v0")
parser.add_argument("--env_yaml", type=str, default=None,
                    help="Path to the RUN's saved params/env.yaml. Omit to use the CURRENT code cfg defaults "
                         "(e.g. to preview new DR that isn't in the saved yaml).")
parser.add_argument("--q", type=float, nargs=7, required=True, help="7 iiwa joint angles [rad] A1..A7")
parser.add_argument("--cam_pos", type=float, nargs=3, default=None,
                    help="Override wrist_cam_offset_pos (eye, fingertip frame) to A/B a mount.")
parser.add_argument("--cam_look", type=float, nargs=3, default=None,
                    help="Override wrist_cam_look_target_pos (aim, fingertip frame).")
parser.add_argument("--out", type=str, default="diagnostics/renders/sim_wrist_at_pose.png")
parser.add_argument("--settle", type=int, default=4, help="sim steps to settle before capture")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import torch
import gymnasium as gym
import yaml

import isaaclab_tasks  # noqa: F401
import insertion_policy.tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab.utils import update_class_from_dict

try:
    import cv2
except Exception:
    cv2 = None


def main():
    # Start from the registered cfg CLASS, then overlay the run's saved env.yaml so the env is byte-for-byte
    # the trained config (base placement, FOV, DR, camera offset) -- NOT hydra defaults. __post_init__ has
    # already run on the registered cfg; update_class_from_dict overwrites the resolved leaf values with the
    # saved ones (env.yaml is the POST-resolution snapshot), which is exactly what the checkpoint saw.
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    if args_cli.env_yaml:
        with open(os.path.expanduser(args_cli.env_yaml)) as f:
            saved = yaml.unsafe_load(f)
        # Drop keys we set explicitly below / that trip update_class_from_dict's strict type check (the saved
        # seed is an int but the cfg field is typed Optional[None]); they don't affect the rendered view.
        for k in ("seed", "sim"):
            saved.pop(k, None)
        update_class_from_dict(env_cfg, saved)

    env_cfg.scene.num_envs = 1
    env_cfg.seed = 0
    # Optional camera A/B (bypass a retrain to eyeball a candidate mount offset).
    if args_cli.cam_pos is not None:
        env_cfg.wrist_cam_offset_pos = tuple(args_cli.cam_pos)
    if args_cli.cam_look is not None:
        env_cfg.wrist_cam_look_target_pos = tuple(args_cli.cam_look)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    u = env.unwrapped
    u.reset()

    q = torch.tensor(args_cli.q, device=u.device, dtype=torch.float32).unsqueeze(0)
    full = u._robot.data.joint_pos.clone()
    full[:, :7] = q  # leave gripper joints at their reset value
    u._robot.write_joint_state_to_sim(full, torch.zeros_like(full))
    u._robot.set_joint_position_target(full)

    for _ in range(max(1, args_cli.settle)):
        u.scene.write_data_to_sim()
        u.sim.step()
        u.scene.update(dt=u.physics_dt)
    if hasattr(u, "_update_camera_pose"):
        u._update_camera_pose()  # rigidly re-attach the wrist cam to the updated fingertip
    u.sim.render()
    u._tiled_camera.update(dt=0.0)

    # --- DEBUG geometry: where is the camera vs the screw vs the base? (so we can see WHY things are/aren't
    # in frame). All world coords minus env origin so they're comparable to the cfg placement ranges. ---
    try:
        from isaaclab.utils.math import quat_apply_inverse
        eo = u.scene.env_origins[0]
        ft_w = u.fingertip_midpoint_pos[0]
        ft_q = u.fingertip_midpoint_quat[0]
        tip_w = u._held_base_pose()[0][0]
        fix_w = u.fixed_pos[0]
        # Express screw-tip and base in the FINGERTIP frame (this is the frame wrist_cam_offset_pos /
        # wrist_cam_look_target_pos live in). To frame the screw+base like the real view, the look-target
        # should point AT this cluster and the eye sit back from it.
        tip_ft = quat_apply_inverse(ft_q.unsqueeze(0), (tip_w - ft_w).unsqueeze(0))[0].tolist()
        fix_ft = quat_apply_inverse(ft_q.unsqueeze(0), (fix_w - ft_w).unsqueeze(0))[0].tolist()
        print(f"FTF screw_tip_in_fingertip = [{tip_ft[0]:.4f}, {tip_ft[1]:.4f}, {tip_ft[2]:.4f}]")
        print(f"FTF base_in_fingertip      = [{fix_ft[0]:.4f}, {fix_ft[1]:.4f}, {fix_ft[2]:.4f}]")
        cam_p = (u._tiled_camera.data.pos_w[0] - eo).tolist()
        ft_p = (ft_w - eo).tolist()
        tip_p = (tip_w - eo).tolist()
        fix_p = (fix_w - eo).tolist()
        dbg = (
            f"DBG cam_pos_w   = [{cam_p[0]:.3f}, {cam_p[1]:.3f}, {cam_p[2]:.3f}]\n"
            f"DBG fingertip   = [{ft_p[0]:.3f}, {ft_p[1]:.3f}, {ft_p[2]:.3f}]\n"
            f"DBG screw_tip   = [{tip_p[0]:.3f}, {tip_p[1]:.3f}, {tip_p[2]:.3f}]\n"
            f"DBG base(socket)= [{fix_p[0]:.3f}, {fix_p[1]:.3f}, {fix_p[2]:.3f}]\n"
        )
        dbg += (
            f"FTF screw_tip_in_fingertip = [{tip_ft[0]:.4f}, {tip_ft[1]:.4f}, {tip_ft[2]:.4f}]\n"
            f"FTF base_in_fingertip      = [{fix_ft[0]:.4f}, {fix_ft[1]:.4f}, {fix_ft[2]:.4f}]\n"
        )
        # Write to a sidecar file next to the image (app stdout is swallowed by the omni logger).
        with open(os.path.splitext(os.path.expanduser(args_cli.out))[0] + "_geom.txt", "w") as _fh:
            _fh.write(dbg)
    except Exception as _e:  # noqa: BLE001
        print(f"DBG geometry print failed: {_e}")

    rgb = u._tiled_camera.data.output["rgb"][0, :, :, :3].clamp(0, 255).to(torch.uint8).cpu().numpy()
    path = os.path.expanduser(args_cli.out)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if cv2 is not None:
        cv2.imwrite(path, rgb[:, :, ::-1])
    else:
        from PIL import Image
        Image.fromarray(rgb).save(path)
    print(f"WROTE {path}  shape={rgb.shape}")
    print(f"  cam_offset_pos={tuple(env_cfg.wrist_cam_offset_pos)}  look_target={tuple(env_cfg.wrist_cam_look_target_pos)}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
