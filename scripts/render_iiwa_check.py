"""Clean verification renders of the iiwa+gripper grasp and the wrist-camera mount.

Purpose-built to actually SEE the setup (unlike the DR-darkened training montage):
  * canonical pose: grasp misalignment AND pre-insert tilt = 0 -> the screw sits vertical in the jaws,
    so the closed grip is easy to verify (and it sidesteps the tilt-triggered reset collision);
  * appearance DR OFF + bright neutral lighting -> the gripper geometry is clearly visible;
  * a movable third-person camera rendered from several angles around ONE env, framed on the
    gripper<->socket region, including the +x side where the RealSense (D405) plate + wrist cam sit;
  * prints the wrist-camera world pose vs the gripper so the mount can be checked numerically.

Outputs /tmp/iiwa_<angle>.png (+ copies under renders/iiwa_bringup/).

    OMNI_KIT_ACCEPT_EULA=YES python scripts/render_iiwa_check.py --experience <physx1065.kit>
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Insertion-CoolingPeg-Iiwa-Vision-Direct-v0")
parser.add_argument("--env", type=int, default=0, help="which env to frame")
parser.add_argument("--realistic", action="store_true",
                    help="keep appearance-DR ON (per-env colored materials, lighting, D405 sensor noise) "
                         "so the wrist POV shows what a real RealSense would see, instead of the grey "
                         "geometry-check mode.")
parser.add_argument("--pov_focal", type=float, default=None,
                    help="override the wrist-camera focal_length (mm) for the POV render. Real D405 RGB "
                         "is ~70deg FOV -> ~15mm (vs the cfg's 28mm/~41deg). Lower = wider = sees more.")
parser.add_argument("--pov_aim", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="override wrist_cam_look_target_pos (aim, fingertip frame) for the POV.")
parser.add_argument("--cam_pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="override wrist_cam_offset_pos (D405 eye, fingertip frame) - for iterating the mount.")
parser.add_argument("--quick", action="store_true",
                    help="fast iteration: render ONLY cammount_side (skip grip close-ups, sweep, POV).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import torch
import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg, save_images_to_file

REPORT = []


def emit(s):
    REPORT.append(str(s))


def main():
    import isaaclab_tasks  # noqa: F401
    import insertion_policy.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=4)

    # Stable, repeatable pose (always): no grasp tilt / pre-insert tilt / reset noise / cam jitter, so the
    # frame is clean and the grip/mount are easy to read.
    env_cfg.task.grasp_misalign_max_deg = 0.0
    env_cfg.task.pre_insert_tilt_max_deg = 0.0
    env_cfg.task.fixed_asset_init_pos_noise = [0.0, 0.0, 0.0]
    env_cfg.task.hand_init_pos_noise = [0.0, 0.0, 0.0]
    env_cfg.task.hand_init_orn_noise = [0.0, 0.0, 0.0]
    env_cfg.cam_pos_jitter = 0.0
    env_cfg.cam_rot_jitter_deg = 0.0
    env_cfg.cam_log_interval = 0
    if args_cli.pov_focal is not None:
        env_cfg.tiled_camera.spawn.focal_length = args_cli.pov_focal
    if args_cli.pov_aim is not None:
        env_cfg.wrist_cam_look_target_pos = tuple(args_cli.pov_aim)
    if args_cli.cam_pos is not None:
        env_cfg.wrist_cam_offset_pos = tuple(args_cli.cam_pos)  # iterate the D405 mount without editing the cfg
    if args_cli.realistic:
        # Keep appearance DR ON -> the wrist POV shows what a real RealSense D405 would see (per-env
        # colored 3D-printed materials, directional lighting, RGB+depth sensor noise). The 4 envs render
        # with different DR draws, so the montage samples the sim2real appearance distribution.
        pass
    else:
        # Clean geometry mode: flat grey materials + bright flat light -> best for the grip/mount close-ups.
        env_cfg.randomize_part_materials = False
        env_cfg.photo_brightness = env_cfg.photo_contrast = env_cfg.photo_gain_rgb = env_cfg.photo_gamma = 0.0
        env_cfg.rgb_noise_std = env_cfg.depth_noise_std = env_cfg.depth_dropout_prob = 0.0
        env_cfg.light_intensity_range = (3000.0, 3000.0)
        env_cfg.key_light_intensity_range = (2000.0, 2000.0)

    # A movable third-person camera (wide FOV so we can frame the whole gripper close up).
    env_cfg.scene_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/scene_cam",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 8.0)
        ),
        width=600,
        height=600,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    env.reset()
    zero = torch.zeros((u.num_envs, u.cfg.action_space), device=u.device)
    for _ in range(4):
        env.step(zero)

    e = args_cli.env
    o = u.scene.env_origins[e]
    ft = (u.fingertip_midpoint_pos[e] + o)                 # fingertip (world)
    gripper_base = u._robot.data.body_pos_w[e, u._robot.body_names.index("gripper_base_link")]
    socket = u.fixed_pos_obs_frame[e]                      # socket opening (world)
    # Frame TIGHT on the grip: centre a bit below the fingertip so the fingers + gripped screw + the
    # socket just under them all sit in the middle (the fingertip is ~8cm above the table, so a low,
    # close eye clears the lab-table box that the workspace sits on).
    center = ft + torch.tensor([0.0, 0.0, -0.03], device=u.device)

    import isaacsim.core.utils.torch as tu

    emit("=== wrist-camera mount check (env %d) ===" % e)
    emit(f"gripper_base (world)   = {[round(x,3) for x in gripper_base.tolist()]}")
    emit(f"fingertip/tcp (world)  = {[round(x,3) for x in ft.tolist()]}")
    # Where the rigid mount PUTS the eye: fingertip + R(ft_quat)*offset (exactly what _update_camera_pose
    # does). Express it in the gripper_base frame -> should equal the vendor D405 prim (0.068, 0, 0.045).
    gb_quat = u._robot.data.body_quat_w[e, u._robot.body_names.index("gripper_base_link")]
    ft_quat0 = u.fingertip_midpoint_quat[e]
    off = torch.tensor(u.cfg.wrist_cam_offset_pos, device=u.device)
    cam_eye = ft + tu.quat_apply(ft_quat0.unsqueeze(0), off.unsqueeze(0))[0]
    cam_in_gb = tu.quat_apply(tu.quat_conjugate(gb_quat).unsqueeze(0), (cam_eye - gripper_base).unsqueeze(0))[0]
    emit(f"wrist_cam eye (world)  = {[round(x,3) for x in cam_eye.tolist()]}")
    emit(f"wrist_cam in gripper_base frame = {[round(x,3) for x in cam_in_gb.tolist()]}  "
         f"(vendor D405 prim = [0.068, 0.0, 0.045])")
    # Isolation check: the iiwa camera FOV override must NOT leak into the Franka cfg (trains tonight).
    from insertion_policy.tasks.insertion.cooling_tasks_cfg import ForgeTaskCoolingInsertCameraCfg
    _fr = ForgeTaskCoolingInsertCameraCfg().tiled_camera.spawn.focal_length
    emit(f"ISOLATION: Franka cam focal={_fr} (expect 28.0), iiwa cam focal="
         f"{u.cfg.tiled_camera.spawn.focal_length} (expect 15.0)")

    # ---- GRASP diagnostics: are the fingers actually closed on the screw HEAD, and where? ----
    li = u._robot.body_names.index("left_finger_link")
    ri = u._robot.body_names.index("right_finger_link")
    gbi = u._robot.body_names.index("gripper_base_link")
    lfj = u._robot.joint_names.index("left_finger_joint")
    rfj = u._robot.joint_names.index("right_finger_joint")
    gb_pos = u._robot.data.body_pos_w[e, gbi]
    gb_q = u._robot.data.body_quat_w[e, gbi]
    gbqi = tu.quat_conjugate(gb_q).unsqueeze(0)
    lf = tu.quat_apply(gbqi, (u._robot.data.body_pos_w[e, li] - gb_pos).unsqueeze(0))[0]  # finger origin in gripper frame
    rf = tu.quat_apply(gbqi, (u._robot.data.body_pos_w[e, ri] - gb_pos).unsqueeze(0))[0]
    held = tu.quat_apply(gbqi, (u.held_pos[e] + o - gb_pos).unsqueeze(0))[0]              # screw origin in gripper frame
    shaft = u._held_base_pose()[0][e]
    shaft_g = tu.quat_apply(gbqi, (shaft + o - gb_pos).unsqueeze(0))[0]
    emit("--- grasp check (gripper_base frame; fingers span z~0.070..0.150, tip=0.150, tcp=0.146) ---")
    emit(f"finger joints: left={float(u._robot.data.joint_pos[e, lfj]):.4f} "
         f"right={float(u._robot.data.joint_pos[e, rfj]):.4f}  (0=closed, 0.04=open)")
    emit(f"finger origins y: left={lf[1]:+.4f} right={rf[1]:+.4f}  gap(|Δy|)={abs(lf[1]-rf[1])*1000:.1f}mm "
         f"(screw head Ø=20mm)")
    emit(f"screw head/origin in gripper frame = {[round(float(v),4) for v in held.tolist()]}  "
         f"(z tells finger-length position; y should be ~0 = between pads)")
    emit(f"screw shaft-tip in gripper frame   = {[round(float(v),4) for v in shaft_g.tolist()]}")

    os.makedirs("renders/iiwa_bringup", exist_ok=True)
    n = u.num_envs

    def scene_shot(label, mcenter, off):
        eye = (mcenter + off).unsqueeze(0).repeat(n, 1)
        ctr = mcenter.unsqueeze(0).repeat(n, 1)
        u._scene_camera.set_world_poses_from_view(eye, ctr)
        for _ in range(2):
            env.step(zero)
        rgb = u._scene_camera.data.output["rgb"][e, ..., :3].float() / 255.0
        save_images_to_file(rgb.unsqueeze(0), f"renders/iiwa_bringup/{label}.png")
        emit(f"wrote renders/iiwa_bringup/{label}.png")

    # --- D405 MOUNT marker FIRST (so we can iterate the mount fast): a bright red sphere at the wrist-cam
    # eye (the D405 lens point). cammount_side frames the gripper body + marker so we can check the lens
    # sits ON the camera mount. In --quick mode we render ONLY this and stop.
    mk = sim_utils.SphereCfg(
        radius=0.006,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.12, 0.12), emissive_color=(0.8, 0.0, 0.0)),
    )
    mk.func("/World/d405_marker", mk, translation=tuple(cam_eye.tolist()))  # RED = D405 lens (cfg eye)
    # occlusion check for the wrist POV at this eye (higher std = clear; ~0.006 = buried in mesh).
    env.step(zero)
    wstd = float(u._tiled_camera.data.output["rgb"][e, ..., :3].float().std() / 255.0)
    emit(f"RED lens eye(fingertip)={tuple(round(float(v),3) for v in u.cfg.wrist_cam_offset_pos)} "
         f"world={[round(x,3) for x in cam_eye.tolist()]}  |  wrist POV std={wstd:.3f} "
         f"({'CLEAR' if wstd>0.05 else 'OCCLUDED'})  focal={float(u.cfg.tiled_camera.spawn.focal_length):.1f}mm")
    mount_center = 0.5 * (gripper_base + cam_eye)
    scene_shot("cammount_side", mount_center, torch.tensor([0.10, -0.30, 0.12], device=u.device))  # -y side
    if args_cli.quick:
        env.close()
        return

    # Full set: grip close-ups + the +x/front mount shot.
    center = ft + torch.tensor([0.0, 0.0, -0.03], device=u.device)  # tight on the grip
    for name, off in {
        "grip_side":    torch.tensor([0.015, -0.17, 0.055], device=u.device),
        "grip_front3q": torch.tensor([0.14, -0.11, 0.06], device=u.device),
        "grip_plate":   torch.tensor([0.19, 0.05, 0.09], device=u.device),
        "grip_topdown": torch.tensor([0.04, -0.05, 0.22], device=u.device),
    }.items():
        scene_shot(name, center, off)
    scene_shot("cammount_front", mount_center, torch.tensor([0.32, 0.02, 0.16], device=u.device))  # +x/plate side

    # The actual TRAINING image: the wrist RGB the policy sees at the CONFIGURED mount (env camera is
    # already at the cfg eye + vendor D405 orientation; no override needed).
    env.step(zero)
    wrist_rgb = u._tiled_camera.data.output["rgb"][:, ..., :3].float() / 255.0
    save_images_to_file(wrist_rgb, "renders/iiwa_bringup/wrist_pov_clean.png")
    r0 = wrist_rgb[e]
    emit(f"wrist POV env{e}: rgb min/mean/max/std = {r0.min():.3f}/{r0.mean():.3f}/{r0.max():.3f}/{r0.std():.3f} "
         f"(low std = flat/uniform view); focal={float(u.cfg.tiled_camera.spawn.focal_length):.1f}mm")
    emit("wrote renders/iiwa_bringup/wrist_pov_clean.png (wrist RGB the policy sees, real D405 mount)")

    # 1:1 comparison with the REAL frame the user sent (gripper OPEN, NO part): force the fingers open and
    # shove the held screw far away, re-writing the open joint state each step so the controller can't
    # re-close it, then dump the wrist POV. Same camera/FOV as training -> directly comparable to hardware.
    open_q = u.joint_pos.clone()
    open_q[:, 7:9] = 0.04  # both fingers fully open (0=closed, 0.04=open)
    held = u._held_asset.data.default_root_state.clone()
    held[:, 0:3] = torch.tensor([3.0, 3.0, 3.0], device=u.device) + u.scene.env_origins
    held[:, 7:] = 0.0
    u._held_asset.write_root_pose_to_sim(held[:, 0:7])
    u._held_asset.write_root_velocity_to_sim(held[:, 7:])
    for _ in range(4):
        u._robot.write_joint_state_to_sim(open_q, torch.zeros_like(open_q))
        u._held_asset.write_root_pose_to_sim(held[:, 0:7])
        env.step(zero)
    open_rgb = u._tiled_camera.data.output["rgb"][:, ..., :3].float() / 255.0
    save_images_to_file(open_rgb, "renders/iiwa_bringup/wrist_pov_open.png")
    emit("wrote renders/iiwa_bringup/wrist_pov_open.png (gripper OPEN, no part -> matches the real frame)")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        emit("FAILED\n" + traceback.format_exc())
    finally:
        with open("/tmp/render_iiwa_check.txt", "w") as f:
            f.write("\n".join(REPORT) + "\n")
        simulation_app.close()
