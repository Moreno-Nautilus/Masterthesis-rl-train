"""Validate the auxiliary-head LABELS (grasp tilt + hole gap) before committing a long train run.

The aux heads are only as good as their targets. A sign error, wrong frame, or bad scale in
``InsertionEnv._get_aux_label`` would still produce a clean-looking decreasing loss (the head just
regresses whatever it's given) and silently waste a ~12 h run. This script boots the vision env,
resets a few times, and for every env checks the emitted ``aux_label`` against an INDEPENDENT
ground-truth computation + physical-range expectations:

  label[0]   = signed grasp tilt (rad)  -> must be in +/- grasp_misalign_max_deg; compare the COMMANDED
               angle to the REALIZED shaft-vs-gripper angle (the gravity-sag gap = label noise).
  label[1:4] = shaft-tip -> socket-opening gap in the FINGERTIP frame (m) -> magnitude cm-scale; its
               norm should match the true tip<->opening distance; +Z (camera looks down the shaft).

It also saves a wrist-cam MONTAGE with the per-env grasp angle burned on, so you can eyeball that a
screw that looks tilted in the image carries a matching label.

    OMNI_KIT_ACCEPT_EULA=YES python scripts/check_aux_labels.py --num_envs 16 --resets 3 \
        --experience /home/moreno/Masterthesis-rl-train/apps/isaaclab.python.headless.rendering.physx1065.kit
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sanity-check the aux-head labels + render a wrist montage.")
parser.add_argument("--task", type=str, default="Isaac-Insertion-CoolingPeg-Vision-Direct-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--resets", type=int, default=3, help="How many reset cycles to sample.")
parser.add_argument("--out", type=str, default="renders/aux_check/aux_labels.png")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import os

import numpy as np
import torch

import isaacsim.core.utils.torch as torch_utils
import gymnasium as gym


def _project(P_world, cam_pos, cam_quat_ros, K):
    """World point (3,) -> (u_px, v_px) in the camera image, or None if behind the camera."""
    rel = (P_world - cam_pos).unsqueeze(0)
    pc = torch_utils.quat_rotate_inverse(cam_quat_ros.unsqueeze(0), rel)[0]  # optical frame: x-right,y-down,z-fwd
    z = pc[2]
    if z <= 1e-4:
        return None
    return (float(K[0, 0] * pc[0] / z + K[0, 2]), float(K[1, 1] * pc[1] / z + K[1, 2]))


def _grid(frames):
    """Tile a list of (H, W, 3) uint8 frames into a near-square grid."""
    n = len(frames)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    h, w, _ = frames[0].shape
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = f
    return canvas


def main():
    import isaaclab_tasks  # noqa: F401
    import insertion_policy.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    max_deg = float(getattr(u.cfg_task, "grasp_misalign_max_deg", 0.0))
    print(f"[INFO] task={args_cli.task} num_envs={args_cli.num_envs} grasp_misalign_max_deg={max_deg}")

    montage_frames = None
    fails = []

    def check(tag):
        nonlocal montage_frames
        obs, _ = env.reset()
        assert "aux_label" in obs, "env produced no 'aux_label' group"
        lbl = obs["aux_label"].float()
        if not torch.isfinite(lbl).all():
            fails.append(f"{tag}: aux_label has NaN/inf")
            return

        theta = lbl[:, 0]            # commanded grasp tilt (rad)
        gap = lbl[:, 1:4]            # tip->opening gap in fingertip frame (m)

        # --- independent ground truth -------------------------------------------------
        # gap recomputed from privileged poses (the label SHOULD equal this).
        tip, _ = u._held_base_pose()
        gap_gt = torch_utils.quat_rotate_inverse(u.fingertip_midpoint_quat, u._socket_opening_pos() - tip)
        gap_err_mm = torch.linalg.vector_norm(gap - gap_gt, dim=1).max().item() * 1000.0

        # realized grasp tilt = angle(shaft axis, gripper approach axis) — what the CAMERA sees,
        # vs the commanded theta (they diverge under gravity sag).
        zc = torch.tensor([0.0, 0.0, 1.0], device=u.device).repeat(u.num_envs, 1)
        _, held_q = u._held_base_pose()
        shaft_axis = torch_utils.quat_apply(held_q, zc)
        grip_axis = torch_utils.quat_apply(u.fingertip_midpoint_quat, zc)
        # At a perfect grasp the screw's local +z points OPPOSITE the gripper approach (+z), i.e. the
        # axes are anti-parallel (~180deg). The physical misalignment is the deviation from anti-parallel
        # = 180 - angle(shaft, grip). So this should sit near |commanded theta|, modulo gravity sag.
        realized = 180.0 - torch.rad2deg(torch.acos(
            torch.clamp((shaft_axis * grip_axis).sum(1), -1.0, 1.0)))

        gap_norm_mm = torch.linalg.vector_norm(gap, dim=1) * 1000.0
        theta_deg = torch.rad2deg(theta)

        # --- range / semantic assertions ----------------------------------------------
        if theta_deg.abs().max().item() > max_deg + 0.6:
            fails.append(f"{tag}: |theta| max {theta_deg.abs().max():.2f}deg exceeds cap {max_deg}")
        if gap_err_mm > 0.5:
            fails.append(f"{tag}: gap label != recomputed GT (max {gap_err_mm:.2f}mm)")
        if not (5.0 < gap_norm_mm.mean().item() < 90.0):
            fails.append(f"{tag}: gap magnitude mean {gap_norm_mm.mean():.1f}mm outside cm-scale window")

        print(f"\n=== {tag} ===")
        print(f"  theta(commanded) deg  min/mean/max = "
              f"{theta_deg.min():+.2f}/{theta_deg.abs().mean():.2f}/{theta_deg.max():+.2f}  (cap +/-{max_deg})")
        print(f"  tilt(realized)   deg  min/mean/max = "
              f"{realized.min():.2f}/{realized.mean():.2f}/{realized.max():.2f}  (camera-visible; sag drift vs commanded)")
        print(f"  gap norm         mm   min/mean/max = "
              f"{gap_norm_mm.min():.1f}/{gap_norm_mm.mean():.1f}/{gap_norm_mm.max():.1f}")
        print(f"  gap z-comp(fingertip) mm mean = {(gap[:,2]*1000).mean():.1f}  (down-shaft toward socket)")
        print(f"  gap label vs recomputed GT: max err = {gap_err_mm:.4f} mm  (expect ~0)")

        # --- wrist montage with the LABEL drawn on the image (first reset only) --------
        # Red line = realized shaft axis (head->tip). Green line = the SAME shaft with the grasp tilt
        # removed (rotate -theta about the finger pressing axis) = where an untilted grasp would point;
        # the visible angle between red and green IS the grasp-theta label. Cyan crosshair = the true
        # socket opening (the "hole finding" target). Lets you EYEBALL that the label matches the image.
        if montage_frames is None and getattr(u, "_tiled_camera", None) is not None:
            try:
                from PIL import Image, ImageDraw
                cam = u._tiled_camera
                rgb = cam.data.output["rgb"][..., :3].clamp(0, 255).to(torch.uint8).cpu().numpy()
                K_all = cam.data.intrinsic_matrices
                org = u.scene.env_origins
                # cam.data.pos_w is stale here (usdrt/fabric pose gotcha -> returns the env origin), so
                # RECONSTRUCT the true camera world pose exactly as InsertionEnv._update_camera_pose does:
                # rigid offset from the fingertip + per-episode jitter/roll. Built in OpenGL convention,
                # then converted to the ROS optical frame (180deg about X) that the pinhole K expects.
                ftp, ftq = u.fingertip_midpoint_pos, u.fingertip_midpoint_quat
                cpos_all = ftp + torch_utils.quat_apply(ftq, u._cam_offset_pos + u._cam_jitter) + org
                roll_q = torch_utils.quat_from_angle_axis(u._cam_roll_jitter, u._cam_roll_axis)
                cam_gl = torch_utils.quat_mul(torch_utils.quat_mul(ftq, u._cam_offset_quat), roll_q)
                qx180 = torch.tensor([0.0, 1.0, 0.0, 0.0], device=u.device).repeat(u.num_envs, 1)
                cq_all = torch_utils.quat_mul(cam_gl, qx180)  # OpenGL -> ROS optical (z-forward, y-down)
                head_w = u.held_pos + org                     # screw origin (head/shoulder) = shaft top
                tip_w = tip + org                             # shaft tip (realized)
                sock_w = u._socket_opening_pos() + org        # true socket opening
                shaft_len = torch.linalg.vector_norm(tip_w - head_w, dim=1, keepdim=True)
                shaft_dir = (tip_w - head_w) / shaft_len.clamp_min(1e-6)
                press_axis = torch_utils.quat_apply(u.fingertip_midpoint_quat,
                                                    torch.tensor([0.0, 1.0, 0.0], device=u.device).repeat(u.num_envs, 1))
                untilt_q = torch_utils.quat_from_angle_axis(-theta, press_axis)   # remove the grasp tilt
                nom_tip_w = head_w + torch_utils.quat_apply(untilt_q, shaft_dir) * shaft_len

                S, R = 240, 240 / cam.image_height if hasattr(cam, "image_height") else 240 / rgb.shape[1]
                frames = []
                for i in range(u.num_envs):
                    im = Image.fromarray(rgb[i]).resize((S, S), Image.NEAREST)
                    d = ImageDraw.Draw(im)
                    K, cp, cq = K_all[i], cpos_all[i], cq_all[i]
                    ph = _project(head_w[i], cp, cq, K)
                    pt = _project(tip_w[i], cp, cq, K)
                    pn = _project(nom_tip_w[i], cp, cq, K)
                    ps = _project(sock_w[i], cp, cq, K)
                    sc = lambda p: (p[0] * R, p[1] * R)
                    if ph and pn:
                        d.line([sc(ph), sc(pn)], fill=(0, 255, 0), width=2)      # nominal/untilted (green)
                    if ph and pt:
                        d.line([sc(ph), sc(pt)], fill=(255, 40, 40), width=3)    # realized shaft (red)
                    if ps:
                        x, y = sc(ps)
                        d.ellipse([x - 7, y - 7, x + 7, y + 7], outline=(0, 230, 255), width=2)  # hole (cyan)
                        d.line([(x - 10, y), (x + 10, y)], fill=(0, 230, 255), width=1)
                        d.line([(x, y - 10), (x, y + 10)], fill=(0, 230, 255), width=1)
                    d.text((4, 3), f"grasp {theta_deg[i]:+.1f}", fill=(255, 255, 0))
                    d.text((4, S - 16), f"gap {gap_norm_mm[i]:.0f}mm", fill=(0, 230, 255))
                    frames.append(np.asarray(im))
                montage_frames = frames
            except Exception as exc:  # noqa: BLE001
                import traceback
                print(f"  [warn] montage overlay failed: {exc}\n{traceback.format_exc()}")

    for r in range(args_cli.resets):
        check(f"reset {r+1}/{args_cli.resets}")

    if montage_frames is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)) or ".", exist_ok=True)
        try:
            import imageio.v2 as imageio
            imageio.imwrite(args_cli.out, _grid(montage_frames))
            print(f"\n[OK] wrote wrist montage -> {args_cli.out}")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] montage write failed: {exc}")

    print("\n" + ("=" * 60))
    if fails:
        print("AUX-LABEL CHECK: FAIL")
        for f in fails:
            print("  -", f)
    else:
        print("AUX-LABEL CHECK: PASS  (labels finite, in-range, match ground truth)")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
