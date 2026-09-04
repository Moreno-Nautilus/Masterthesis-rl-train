"""Render the pb_pipe HORIZONTAL insertion scene for visual verification (grasp + base placement).

Produces 3 zoomed-out views of ONE env at reset (DR off, grey geometry):
  * grasp_front : looking along -X (down the intended horizontal insertion axis) at the grasped pipe.
  * grasp_side  : looking along -Y at the gripper+pipe, so the 90deg (pipe long axis horizontal) is obvious.
  * scene_iso   : an isometric wide shot showing the base on the table + the gripper above it.

NOT training. Reset-only capture. Run:
  OMNI_KIT_ACCEPT_EULA=YES ~/isaaclab/isaaclab.sh -p scripts/render_pb_pipe_scene.py
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Insertion-PbPipe-Iiwa-E2E-Vision-Direct-v0")
parser.add_argument("--out_dir", type=str, default="diagnostics/renders")
parser.add_argument("--settle", type=int, default=8)
parser.add_argument("--num_envs", type=int, default=1, help="parallel envs to create; stills always capture env 0")
parser.add_argument("--fixture_y", type=float, default=None, help="optional fixed world-Y fixture position (m)")
parser.add_argument("--orbit_frames", type=int, default=72, help="frames for the orbit video (0 = skip)")
parser.add_argument("--fps", type=int, default=24)
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
from isaaclab.sensors import TiledCameraCfg

import isaaclab_tasks  # noqa: F401
import insertion_policy.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    os.makedirs(args_cli.out_dir, exist_ok=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    # DR OFF -> clean grey geometry.
    if hasattr(env_cfg, "randomize_part_materials"):
        env_cfg.randomize_part_materials = False
    for a in ["photo_brightness", "photo_contrast", "photo_gain_rgb", "photo_gamma", "rgb_noise_std",
              "cam_pos_jitter", "cam_rot_jitter_deg"]:
        if hasattr(env_cfg, a):
            setattr(env_cfg, a, 0.0)
    # Freeze the reset so the capture is the nominal pose (no tilt/anchor/yaw noise obscuring the geometry).
    env_cfg.task.pre_insert_tilt_max_deg = 0.0
    if hasattr(env_cfg.task, "goal_anchor_injection"):
        env_cfg.task.goal_anchor_injection = False
    if hasattr(env_cfg.task, "wrist_yaw_max_deg"):
        env_cfg.task.wrist_yaw_max_deg = 0.0
    env_cfg.task.hand_init_pos_noise = [0.0, 0.0, 0.0]
    env_cfg.task.hand_init_orn_noise = [0.0, 0.0, 0.0]
    if hasattr(env_cfg.task, "grasp_misalign_max_deg"):
        env_cfg.task.grasp_misalign_max_deg = 0.0
    if hasattr(env_cfg.task, "grasp_misalign_secondary_deg"):
        env_cfg.task.grasp_misalign_secondary_deg = 0.0
    if args_cli.fixture_y is not None:
        x_lo, x_hi, _, _ = env_cfg.task.fixed_asset_xy_ranges
        env_cfg.task.fixed_asset_xy_ranges = [x_lo, x_hi, args_cli.fixture_y, args_cli.fixture_y]

    env_cfg.scene_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/scene_cam",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 12.0)
        ),
        width=640,
        height=480,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    dev = u.device
    env.reset()
    zero = torch.zeros((u.num_envs, u.cfg.action_space), device=dev)
    for _ in range(args_cli.settle):
        env.step(zero)

    o = u.scene.env_origins[0]
    tip = u.fingertip_midpoint_pos[0] + o          # grasp point (fingertip pos is env-relative)
    # fixed_pos is the env-relative socket/base position the env already computed; add the env origin.
    seat = (u.fixed_pos[0] if hasattr(u, "fixed_pos") else u._fixed_asset.data.root_pos_w[0] - o) + o
    base_root = u._fixed_asset.data.root_pos_w[0]
    print(f"[geom] fingertip(world)={tip.tolist()}  seat(world)={seat.tolist()}  "
          f"base_root(world)={base_root.tolist()}", flush=True)

    # DECISIVE grasp check: where is the pipe RELATIVE to the fingertip (TCP)? Express the held pose in the
    # fingertip frame so occlusion in the render doesn't matter. rel_pos = R(tip)^-1 . (held - tip).
    import isaacsim.core.utils.torch as _tu
    held_w = u._held_asset.data.root_pos_w[0]        # world (includes env origin)
    held_ref_w = u._held_base_pose()[0][0] + o
    reset_target_w = u._socket_opening_pos()[0] + o
    print(
        f"[reset] held midpoint(world)={held_ref_w.tolist()}  target(world)={reset_target_w.tolist()}  "
        f"seat-minus-held(mm)={((seat - held_ref_w) * 1000.0).tolist()}",
        flush=True,
    )
    ftq = u.fingertip_midpoint_quat[0]
    rel = _tu.quat_apply(_tu.quat_conjugate(ftq).unsqueeze(0), (held_w - tip).unsqueeze(0))[0]
    # Pipe long axis in world = held_quat rotates mesh-Y; express it in the fingertip frame too.
    hq = u._held_asset.data.root_quat_w[0]
    long_axis_w = _tu.quat_apply(hq.unsqueeze(0), torch.tensor([[0.0, 1.0, 0.0]], device=dev))[0]
    long_axis_ft = _tu.quat_apply(_tu.quat_conjugate(ftq).unsqueeze(0), long_axis_w.unsqueeze(0))[0]
    print(f"[grasp] held pos in FINGERTIP frame (m) = {[round(v,4) for v in rel.tolist()]}  "
          f"(x=along-approach-out, y=pressing, z=?)", flush=True)
    print(f"[grasp] pipe LONG axis in fingertip frame = {[round(v,3) for v in long_axis_ft.tolist()]}  "
          f"(want ~horizontal: big |x| or |y|, small |z|); world = {[round(v,3) for v in long_axis_w.tolist()]}",
          flush=True)
    # GRIPPER (fingertip) orientation: is it tool-DOWN? fingertip-Z should point world -Z ([0,0,-1]) for a
    # straight-down hand. A tilt here is why the pipe isn't level.
    ft_z_w = _tu.quat_apply(ftq.unsqueeze(0), torch.tensor([[0.0, 0.0, 1.0]], device=dev))[0]
    ft_y_w = _tu.quat_apply(ftq.unsqueeze(0), torch.tensor([[0.0, 1.0, 0.0]], device=dev))[0]
    import math as _mm
    tilt_from_down = _mm.degrees(_mm.acos(max(-1.0, min(1.0, float(-ft_z_w[2])))))
    pipe_tilt = _mm.degrees(_mm.asin(max(-1.0, min(1.0, abs(float(long_axis_w[2]))))))
    print(f"[hand] fingertip-Z -> world {[round(v,3) for v in ft_z_w.tolist()]} (want [0,0,-1]); "
          f"tilt-from-straight-down = {tilt_from_down:.1f} deg", flush=True)
    print(f"[hand] pipe tilt from level = {pipe_tilt:.1f} deg (want ~0)", flush=True)
    # BASE orientation: mesh-X is the 16cm base-long axis, mesh-Y is the central halfpipe channel, and mesh-Z
    # is height. The task's intended world mapping is X->Y, Y->X, Z->Z.
    bq = u._fixed_asset.data.root_quat_w[0]
    for nm, ax in [("long(X)", [1.0, 0, 0]), ("bore(Y)", [0, 1.0, 0]), ("height(Z)", [0, 0, 1.0])]:
        w = _tu.quat_apply(bq.unsqueeze(0), torch.tensor([ax], device=dev))[0]
        print(f"[base] mesh-{nm} -> world {[round(v,3) for v in w.tolist()]}", flush=True)

    W, H = 640, 480
    _cam_state = {}

    def _project(world_pts):
        eye = _cam_state["eye"]; look = _cam_state["look"]
        fwd = look - eye; fwd = fwd / fwd.norm()
        wup = torch.tensor([0.0, 0.0, 1.0], device=dev)
        right = torch.cross(fwd, wup); right = right / right.norm()
        up = torch.cross(right, fwd)
        rel = world_pts - eye.unsqueeze(0)
        z = (rel * fwd).sum(1)                # depth along view dir
        x = (rel * right).sum(1)
        y = (rel * up).sum(1)
        fl = 18.0; ap = 20.955                 # PinholeCameraCfg focal_length + horizontal_aperture (mm)
        fx = W * fl / ap
        valid = z > 1e-3
        u_px = fx * (x / z.clamp(min=1e-3)) + W / 2
        v_px = -fx * (y / z.clamp(min=1e-3)) + H / 2   # image y grows downward
        return torch.stack([u_px, v_px], dim=1), valid

    def _draw_axes(draw, origin_w, length=0.06, labels=True):
        """Draw world X(red)/Y(green)/Z(blue) axes at origin_w onto a PIL ImageDraw."""
        ends = torch.stack([
            origin_w,
            origin_w + torch.tensor([length, 0, 0], device=dev),
            origin_w + torch.tensor([0, length, 0], device=dev),
            origin_w + torch.tensor([0, 0, length], device=dev),
        ], dim=0)
        px, valid = _project(ends)
        pxn = px.cpu().numpy()
        print(f"[axes] origin px={pxn[0].round(1).tolist()} valid={valid.tolist()} "
              f"Xend={pxn[1].round(1).tolist()} Yend={pxn[2].round(1).tolist()} Zend={pxn[3].round(1).tolist()}",
              flush=True)
        cols = [(255, 60, 60), (60, 220, 60), (80, 120, 255)]
        names = ["X", "Y", "Z"]
        o = tuple(float(v) for v in pxn[0])
        for i in range(3):
            e = tuple(float(v) for v in pxn[i + 1])
            draw.line([o, e], fill=cols[i], width=4)
            if labels:
                draw.text((e[0] + 3, e[1] - 6), names[i], fill=cols[i])

    def shot(name, eye_off, look, hold_fn=None, axes_at=None):
        eye = look + torch.tensor(eye_off, device=dev)
        _cam_state["eye"] = eye; _cam_state["look"] = look
        u._scene_camera.set_world_poses_from_view(eye.unsqueeze(0), look.unsqueeze(0))
        for _ in range(2):
            if hold_fn is not None:
                hold_fn(0)
            env.step(zero)
        img = u._scene_camera.data.output["rgb"][0, :, :, :3].clamp(0, 255).to(torch.uint8).cpu().numpy()
        from PIL import Image, ImageDraw
        im = Image.fromarray(img)
        if axes_at is not None:
            d = ImageDraw.Draw(im)
            for org in axes_at:
                _draw_axes(d, org)
        path = os.path.join(args_cli.out_dir, f"pb_pipe_{name}.png")
        im.save(path)
        print(f"[OK] wrote {path}", flush=True)

    # The pipe center sits ~at the grip point (near one end); aim at the fingertip. Top view = straight down
    # (shows the long axis in-plane + which end points at the bore). A high-and-back 3/4 top view avoids the
    # arm-mesh occlusion the pure nadir shot hit.
    shot("grasp_side", (0.0, -0.40, 0.06), tip)     # side -> pipe-horizontal is obvious
    shot("grasp_top", (0.12, 0.0, 0.45), tip)       # 3/4 top-down -> long axis + grip-near-end visible
    shot("grasp_topfar", (0.25, 0.02, 0.30), tip)   # shallower, further -> whole pipe + base in one frame
    # Wide isometric of the whole cell: base on the table + gripper above.
    mid = 0.5 * (tip + seat)
    shot("scene_iso", (-0.5, -0.5, 0.4), mid)
    # Dedicated reset views WITH WORLD AXES drawn at the gripper (fingertip) AND the base bore centre, so the
    # start pose relative to the halfpipe is unambiguous. X=red, Y=green, Z=blue. Captured after env.reset(),
    # before the render-only goal pose is injected.
    axes_pts = [tip, seat]
    shot("reset_side", (0.0, -0.42, 0.12), mid, axes_at=axes_pts)
    shot("reset_iso", (0.38, -0.42, 0.30), mid, axes_at=axes_pts)
    shot("reset_top", (0.02, 0.0, 0.5), mid, axes_at=axes_pts)

    import math as _m

    def orbit(name, center, hold_fn=None, r=0.32, zc=0.14):
        """Circle `center` over orbit_frames, writing an mp4. hold_fn(k) runs each frame (e.g. pin the pipe)."""
        n_frames = args_cli.orbit_frames
        if n_frames <= 0:
            return
        frames = []
        for k in range(n_frames):
            if hold_fn is not None:
                hold_fn(k)
            a = 2.0 * _m.pi * k / n_frames
            eye = center + torch.tensor([r * _m.cos(a), r * _m.sin(a), zc], device=dev)
            u._scene_camera.set_world_poses_from_view(eye.unsqueeze(0), center.unsqueeze(0))
            env.step(zero)
            frames.append(u._scene_camera.data.output["rgb"][0, :, :, :3].clamp(0, 255).to(torch.uint8).cpu().numpy())
        import imageio.v2 as imageio
        try:
            vpath = os.path.join(args_cli.out_dir, f"pb_pipe_{name}.mp4")
            imageio.mimwrite(vpath, frames, fps=args_cli.fps, macro_block_size=None)
        except Exception as e:
            vpath = os.path.join(args_cli.out_dir, f"pb_pipe_{name}.gif")
            imageio.mimwrite(vpath, frames, duration=1.0 / args_cli.fps)
            print(f"  (mp4 failed: {e})", flush=True)
        print(f"[OK] wrote {vpath}  ({len(frames)} frames)", flush=True)

    # (1) GRASP orbit: the settled reset scene (horizontal pipe, off-centre grip).
    orbit("grasp_orbit", tip)

    # (2) GOAL-POSE orbit: place the pipe SEATED -- its MIDPOINT at the halfpipe centre (fixed_pos), long axis
    # along the channel -- and solve the arm to retain the exact reset grasp transform at that target. The held
    # pipe remains explicitly pinned for the visualisation so the render shows the exact analytical goal.
    # Disable only the PIPE colliders after the reset captures: this is a render-only goal illustration,
    # and the exact-fit pipe/socket surfaces otherwise push the welded arm away during IK. Fixture and robot
    # collisions remain active, so a zero residual still verifies that the angled gripper clears the base.
    import omni.usd
    from pxr import UsdPhysics
    goal_stage = omni.usd.get_context().get_stage()
    held_prefix = "/World/envs/env_0/HeldAsset"
    for prim in goal_stage.Traverse():
        if prim.GetPath().pathString.startswith(held_prefix) and prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
    u.step_sim_no_action()
    u._weld_held = False  # stop the FixedJoint sync from yanking the pipe off the goal
    bore_center = u.fixed_pos[0] + o   # world bore-station centre (= pipe-midpoint goal)
    # Goal orientation: align the pipe LONG axis to the halfpipe channel. Channel axis (world) = base_quat . task axis.
    bq = u._fixed_asset.data.root_quat_w[0]
    channel_axis_w = _tu.quat_apply(bq.unsqueeze(0), u._socket_axis_local[0].unsqueeze(0))[0]
    channel_axis_w = channel_axis_w / channel_axis_w.norm()
    # A through-pipe is axially sign-symmetric. Match the RESET pipe direction instead of forcing the
    # equivalent opposite direction, which would make the welded gripper execute a needless 180deg wrist
    # flip at the goal and lose the authored angled grasp.
    current_pipe_quat = u._held_asset.data.root_quat_w[0]
    current_pipe_axis_w = _tu.quat_apply(current_pipe_quat.unsqueeze(0), u._held_axis_local[0].unsqueeze(0))[0]
    if bool(getattr(u.cfg_task, "axis_alignment_abs", False)) and torch.dot(
        current_pipe_axis_w, channel_axis_w
    ) < 0.0:
        channel_axis_w = -channel_axis_w
    # The pipe's long axis in its OWN root frame is mesh-Y (unchanged by the grasp spawn rot at the ROOT level,
    # since that rot IS the pipe root orientation). Build a quat whose Y column = bore_axis, keeping Z ~ up.
    up = torch.tensor([0.0, 0.0, 1.0], device=dev)
    x_col = torch.cross(channel_axis_w, up, dim=0); x_col = x_col / x_col.norm()
    z_col = torch.cross(x_col, channel_axis_w, dim=0); z_col = z_col / z_col.norm()
    R = torch.stack([x_col, channel_axis_w, z_col], dim=1)  # columns = pipe-mesh X,Y,Z in world
    # rotation-matrix -> quat (w,x,y,z)
    import isaacsim.core.utils.numpy.rotations as _rot  # noqa
    def mat2quat(m):
        t = m.trace()
        w = torch.sqrt(torch.clamp(1 + t, min=1e-8)) / 2
        x = (m[2, 1] - m[1, 2]) / (4 * w)
        y = (m[0, 2] - m[2, 0]) / (4 * w)
        z = (m[1, 0] - m[0, 1]) / (4 * w)
        return torch.stack([w, x, y, z])
    goal_quat = mat2quat(R)

    # Use the exact authored TCP->pipe fixed-joint transform, shared with the reset code, to make the goal
    # render show the arm holding the pipe rather than a visually similar but frame-mismatched pose.
    tcp_to_pipe_quat = u._weld_frame0_quat[0]
    tcp_to_pipe_pos = u._weld_frame0_pos[0]
    goal_tip_quat = _tu.quat_mul(goal_quat, _tu.quat_conjugate(tcp_to_pipe_quat))
    goal_tip_pos = bore_center - _tu.quat_apply(goal_tip_quat.unsqueeze(0), tcp_to_pipe_pos.unsqueeze(0))[0]
    goal_env_ids = torch.tensor([0], device=dev, dtype=torch.long)
    saved_goal_ik_budget = getattr(u, "_reset_ik_budget", None)
    if hasattr(u, "_reset_ik_budget"):
        u._reset_ik_budget = None
    try:
        goal_ik_iters = min(12, max(3, int(getattr(u.cfg_task, "reset_ref_ik_iters", 12) or 12)))
        for _ in range(goal_ik_iters):
            u.set_pos_inverse_kinematics(goal_tip_pos.unsqueeze(0), goal_tip_quat.unsqueeze(0), goal_env_ids)
            actual_tip_pos = u.fingertip_midpoint_pos[0] + o
            actual_tip_quat = u.fingertip_midpoint_quat[0]
            goal_pos_err = goal_tip_pos - actual_tip_pos
            goal_quat_err = _tu.quat_mul(goal_tip_quat, _tu.quat_conjugate(actual_tip_quat))
            goal_quat_err = goal_quat_err / goal_quat_err.norm().clamp(min=1e-6)
            goal_aa_err = 2.0 * torch.atan2(goal_quat_err[1:].norm(), goal_quat_err[0].abs().clamp(min=1e-6))
            if goal_pos_err.norm() <= 0.0005 and goal_aa_err <= 0.25 * _m.pi / 180.0:
                break
    finally:
        if hasattr(u, "_reset_ik_budget"):
            u._reset_ik_budget = saved_goal_ik_budget
    print(
        f"[goal-arm] measured fingertip residual: pos={goal_pos_err.norm().item() * 1e3:.2f} mm "
        f"rot={goal_aa_err.item() * 180.0 / _m.pi:.2f} deg",
        flush=True,
    )

    def place_goal():
        st = u._held_asset.data.root_state_w.clone()
        st[0, 0:3] = bore_center
        st[0, 3:7] = goal_quat
        st[0, 7:13] = 0.0
        u._held_asset.write_root_state_to_sim(st)
        u.scene.write_data_to_sim()

    def orbit_static(name, center, r=0.30, zc=0.12):
        n_frames = args_cli.orbit_frames
        if n_frames <= 0:
            return
        frames = []
        for k in range(n_frames):
            place_goal()
            a = 2.0 * _m.pi * k / n_frames
            eye = center + torch.tensor([r * _m.cos(a), r * _m.sin(a), zc], device=dev)
            u._scene_camera.set_world_poses_from_view(eye.unsqueeze(0), center.unsqueeze(0))
            env.step(zero)  # refresh the camera output at this pose; gravity is disabled for the held pipe.
            frames.append(u._scene_camera.data.output["rgb"][0, :, :, :3].clamp(0, 255).to(torch.uint8).cpu().numpy())
        import imageio.v2 as imageio
        vpath = os.path.join(args_cli.out_dir, f"pb_pipe_{name}.mp4")
        imageio.mimwrite(vpath, frames, fps=args_cli.fps, macro_block_size=None)
        print(f"[OK] wrote {vpath}  ({len(frames)} frames)", flush=True)

    orbit_static("goal_orbit", bore_center)
    # Stills of the seated pose (render-only, no physics step).
    for nm, eoff in [("goal_side", (0.0, 0.35, 0.08)), ("goal_iso", (0.38, -0.42, 0.30)),
                     ("goal_top", (0.08, 0.0, 0.4))]:
        place_goal()
        eye = bore_center + torch.tensor(eoff, device=dev)
        u._scene_camera.set_world_poses_from_view(eye.unsqueeze(0), bore_center.unsqueeze(0))
        env.step(zero)  # camera sensor data is updated only on a simulation step
        img = u._scene_camera.data.output["rgb"][0, :, :, :3].clamp(0, 255).to(torch.uint8).cpu().numpy()
        from PIL import Image
        Image.fromarray(img).save(os.path.join(args_cli.out_dir, f"pb_pipe_{nm}.png"))
        print(f"[OK] wrote diagnostics/renders/pb_pipe_{nm}.png", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
