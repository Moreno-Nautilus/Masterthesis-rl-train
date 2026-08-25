"""Smoke test: instantiate the forked cooling-insertion env, reset, step with random actions.

Validates registration + asset loading + Forge control loop end-to-end. Writes a report to
/tmp/smoke_env_report.txt (Isaac Sim swallows stdout).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Insertion-CoolingPeg-Direct-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import traceback

import torch
import gymnasium as gym

REPORT = []


def emit(line: str) -> None:
    REPORT.append(str(line))


def main() -> None:
    import isaaclab_tasks  # noqa: F401  (ensures isaac base envs importable)
    import insertion_policy.tasks  # noqa: F401  (registers our env)
    from isaaclab_tasks.utils import parse_env_cfg

    emit(f"task={args_cli.task} num_envs={args_cli.num_envs}")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    # Escape hatch for a broken/mismatched omni.physx.fabric extension (route poses via USD XformCache
    # instead of Fabric). Default OFF -> normal fabric path unchanged. Set SMOKE_NO_FABRIC=1 to bypass.
    import os as _os

    if _os.environ.get("SMOKE_NO_FABRIC"):
        env_cfg.sim.use_fabric = False
        emit("SMOKE_NO_FABRIC=1 -> sim.use_fabric=False")
    # Vision variant: dump a rendered RGB/depth frame to /tmp to sanity-check the wrist mount pose.
    if hasattr(env_cfg, "write_image_to_file"):
        env_cfg.write_image_to_file = True
    # SMOKE_FULL_DR=1: zero the DR curricula so the smoke exercises the injection at FULL magnitude (the
    # goal-anchor lateral/angular error otherwise ramps to ~0 at step 0 -> nothing to see). Diagnostic only.
    if _os.environ.get("SMOKE_FULL_DR"):
        for _obj, _attr in ((getattr(env_cfg, "task", None), "goal_anchor_curriculum_steps"),
                            (getattr(env_cfg, "task", None), "pre_insert_tilt_curriculum_steps"),
                            (env_cfg, "image_dropout_start_steps"), (env_cfg, "anchor_dropout_start_steps")):
            if _obj is not None and hasattr(_obj, _attr):
                setattr(_obj, _attr, 0)
        emit("SMOKE_FULL_DR=1 -> zeroed DR curricula (full-magnitude injection)")
    env = gym.make(args_cli.task, cfg=env_cfg)
    emit(f"created env OK | action_space={env.action_space} obs_space={env.observation_space}")

    obs, _ = env.reset()
    obs_t = obs["policy"] if isinstance(obs, dict) else obs
    emit(f"reset OK | obs['policy'] shape={tuple(obs_t.shape)}")
    if isinstance(obs, dict) and "image" in obs:
        img = obs["image"]
        emit(f"reset OK | obs['image'] shape={tuple(img.shape)} dtype={img.dtype}")
        rgb = img[..., :3]
        emit(f"  rgb   min/mean/max = {rgb.min():.3f}/{rgb.mean():.3f}/{rgb.max():.3f}")
        if img.shape[-1] >= 4:  # RGB-D path (residual/state vision tasks)
            depth = img[..., 3]
            emit(f"  depth min/mean/max = {depth.min():.3f}/{depth.mean():.3f}/{depth.max():.3f}")
            emit("  wrote /tmp/wrist_rgb.png and /tmp/wrist_depth.png")
        else:  # RGB-only path (E2E sim2real rebuild)
            emit("  RGB-only (3ch): wrote /tmp/wrist_rgb.png")

    # --- geometry diagnostic (env 0, relative to its env origin) ---
    try:
        u = env.unwrapped
        hb, _q = u._held_base_pose()
        sb = u.fixed_pos[0]           # socket bottom (target)
        so = u.fixed_pos_obs_frame[0] # socket opening (tip / entry)
        emit(f"GEOM env0: socket_bottom={[round(x,4) for x in sb.tolist()]}")
        emit(f"GEOM env0: socket_opening={[round(x,4) for x in so.tolist()]}")
        emit(f"GEOM env0: shaft_tip(held_base)={[round(x,4) for x in hb[0].tolist()]}")
        emit(f"GEOM env0: screw_origin(held_pos)={[round(x,4) for x in u.held_pos[0].tolist()]}")
        emit(f"GEOM env0: fingertip={[round(x,4) for x in u.fingertip_midpoint_pos[0].tolist()]}")
        emit(f"GEOM env0: shaft_tip ABOVE socket_bottom by z = {(hb[0,2]-sb[2]).item():+.4f} m")
        emit(
            f"GEOM env0: shaft_tip ABOVE socket_opening by z = {(hb[0,2]-so[2]).item():+.4f} m  "
            "(want +0.010..+0.050 pre-insert)"
        )
        xy = ((hb[0,0]-sb[0])**2 + (hb[0,1]-sb[1])**2) ** 0.5
        emit(f"GEOM env0: shaft_tip<->socket xy offset = {xy.item():.4f} m")
        if hasattr(u, "_keypoint_distances"):
            kp_dist, tip_dist = u._keypoint_distances()
            shaft_axis_error = torch.rad2deg(u._shaft_axis_error())
            emit(f"GEOM env0: keypoint_mean_dist = {kp_dist[0].item():.4f} m")
            emit(f"GEOM env0: tip_dist = {tip_dist[0].item():.4f} m")
            emit(f"GEOM env0: shaft_axis_error = {shaft_axis_error[0].item():.2f} deg")
            # Distribution across all envs: tilt should now span ~0..pre_insert_tilt_max_deg, while
            # the tip stays near the socket mouth (xy offset driven by lateral noise, not the tilt).
            xy_all = torch.linalg.vector_norm(u.fixed_pos[:, 0:2] - hb[:, 0:2], dim=1) * 1000.0
            emit(
                f"GEOM all-{args_cli.num_envs}-envs: shaft_axis_error deg min/mean/max = "
                f"{shaft_axis_error.min().item():.2f}/{shaft_axis_error.mean().item():.2f}/{shaft_axis_error.max().item():.2f}"
            )
            emit(
                f"GEOM all-{args_cli.num_envs}-envs: tip<->socket xy offset mm min/mean/max = "
                f"{xy_all.min().item():.1f}/{xy_all.mean().item():.1f}/{xy_all.max().item():.1f}"
            )
        # Camera-pose diagnostic: where is the wrist cam and is it aimed at the socket?
        if getattr(u, "_tiled_camera", None) is not None:
            fq = u.fingertip_midpoint_quat[0]
            emit(f"CAM env0: fingertip_quat (wxyz) = {[round(x,4) for x in fq.tolist()]}")
            cam_pos = (u._tiled_camera.data.pos_w[0] - u.scene.env_origins[0])
            cam_qw = u._tiled_camera.data.quat_w_world[0]
            emit(f"CAM env0: cam_pos = {[round(x,4) for x in cam_pos.tolist()]}")
            emit(f"CAM env0: cam_quat_w_world (wxyz) = {[round(x,4) for x in cam_qw.tolist()]}")
            # world convention forward = +X; check alignment with direction cam->socket_opening.
            import isaacsim.core.utils.torch as tu
            fwd = tu.quat_apply(cam_qw.unsqueeze(0), torch.tensor([[1.0, 0.0, 0.0]], device=cam_qw.device))[0]
            to_tgt = (so - cam_pos); to_tgt = to_tgt / (to_tgt.norm() + 1e-9)
            cos = float((fwd / (fwd.norm() + 1e-9)) @ to_tgt)
            emit(f"CAM env0: forward(+X)={[round(x,3) for x in fwd.tolist()]} | dir->socket={[round(x,3) for x in to_tgt.tolist()]} | cos={cos:.3f} (want ~1)")
    except Exception as e:
        import traceback
        emit(f"GEOM diag skipped: {e}\n{traceback.format_exc()}")

    n_act = env.action_space.shape[1] if len(env.action_space.shape) > 1 else env.unwrapped.cfg.action_space
    for i in range(args_cli.steps):
        act = (2.0 * torch.rand((args_cli.num_envs, n_act), device=args_cli.device) - 1.0)
        obs, rew, terminated, truncated, info = env.step(act)
        if i % 5 == 0 or i == args_cli.steps - 1:
            r = rew.mean().item() if hasattr(rew, "mean") else float(rew)
            emit(f"step {i:3d}: mean_rew={r:+.4f} term={int(terminated.sum())} trunc={int(truncated.sum())}")

    emit("RESULT: SMOKE OK (env created, reset, stepped without crashing)")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        emit("RESULT: FAILED")
        emit(traceback.format_exc())
    finally:
        with open("/tmp/smoke_env_report.txt", "w") as f:
            f.write("\n".join(REPORT) + "\n")
        simulation_app.close()
