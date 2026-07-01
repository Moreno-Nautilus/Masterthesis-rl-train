"""Smoke-test the temporal frame-stack: verify the image obs widens to 4*N channels, the per-episode
ring buffer clears at reset (no cross-episode bleed), and stepping produces real inter-frame MOTION.

``ForgeTaskCoolingInsertCameraCfg.frame_stack = N`` stacks the last N wrist frames on the channel axis
(env-side ring buffer). This boots the vision env with N=3 and checks:

  * no crash; image obs channels == 4*N (== 12 for N=3);
  * RIGHT AFTER RESET the N sub-frames are IDENTICAL (buffer filled with the new episode's first frame
    -> no motion bleed from the previous episode);
  * after several push steps the NEWEST sub-frame DIFFERS from the OLDEST (motion is captured);
  * a second reset re-clears the history (sub-frames identical again).

    OMNI_KIT_ACCEPT_EULA=YES TORCHDYNAMO_DISABLE=1 python scripts/check_frame_stack.py --num_envs 8 \
        --frame_stack 3 --experience apps/isaaclab.python.headless.rendering.physx1065.kit
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke-test the temporal frame-stack.")
parser.add_argument("--task", type=str, default="Isaac-Insertion-CoolingPeg-Vision-Direct-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--frame_stack", type=int, default=3)
parser.add_argument("--steps", type=int, default=14, help="Push steps between the reset checks.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

# Isaac block-buffers stdout and hard-exits on close (buffer lost when piped to a file); mirror every
# result line into a file flushed immediately -- the reliable channel.
_RESULT_PATH = "/tmp/frame_stack_smoke_result.txt"
_result_fh = open(_RESULT_PATH, "w")


def log(msg=""):
    import sys
    sys.stdout.write(str(msg) + "\n")
    sys.stdout.flush()
    _result_fh.write(str(msg) + "\n")
    _result_fh.flush()


def main():
    import isaaclab_tasks  # noqa: F401
    import insertion_policy.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    n = args_cli.frame_stack
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.frame_stack = n
    per_frame_c = int(env_cfg.image_channels)
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    log(f"[INFO] task={args_cli.task} num_envs={args_cli.num_envs} frame_stack={n} "
        f"per_frame_channels={per_frame_c}")

    fails = []

    def sub(img, i):  # i-th stacked sub-frame (channel block), 0=oldest .. n-1=newest
        return img[..., i * per_frame_c:(i + 1) * per_frame_c]

    def newest_minus_oldest(img):
        return (sub(img, n - 1) - sub(img, 0)).abs()

    obs, _ = env.reset()
    img = obs["image"]
    ch = int(img.shape[-1])
    log(f"[INFO] image obs shape = {tuple(img.shape)}  (expected channels {per_frame_c * n})")
    if ch != per_frame_c * n:
        fails.append(f"image channels {ch} != {per_frame_c * n}")

    # (1) Right after reset: all n sub-frames identical (history filled with the first frame).
    d_reset = newest_minus_oldest(img).max().item()
    log(f"[check] post-reset max|newest-oldest| = {d_reset:.6f}  (expect ~0: history filled w/ 1st frame)")
    if d_reset > 1e-5:
        fails.append(f"post-reset sub-frames not identical (max diff {d_reset:.6f})")

    # Push the tilted shaft down so the scene visibly changes -> real inter-frame motion.
    action = torch.zeros((u.num_envs, u.single_action_space.shape[0]), device=u.device)
    action[:, 2] = -1.0
    action[:, 0] = 0.3
    for step in range(args_cli.steps):
        obs, _, _, _, _ = env.step(action)
    img = obs["image"]
    if not torch.isfinite(img).all():
        fails.append("image has NaN/inf after stepping")

    # (2) After motion: newest sub-frame differs from oldest (frames are genuinely time-shifted).
    d_motion = newest_minus_oldest(img).mean().item()
    d_motion_max = newest_minus_oldest(img).max().item()
    log(f"[check] post-step mean|newest-oldest| = {d_motion:.6f}  max = {d_motion_max:.6f}  "
        f"(expect >0: motion captured)")
    if d_motion <= 1e-4:
        fails.append(f"no inter-frame motion after {args_cli.steps} steps (mean diff {d_motion:.6f})")

    # (3) A fresh reset must re-clear the history -> sub-frames identical again.
    obs, _ = env.reset()
    img = obs["image"]
    d_reset2 = newest_minus_oldest(img).max().item()
    log(f"[check] second post-reset max|newest-oldest| = {d_reset2:.6f}  (expect ~0: history re-cleared)")
    if d_reset2 > 1e-5:
        fails.append(f"reset did not re-clear history (max diff {d_reset2:.6f})")

    log("\n" + ("=" * 60))
    if fails:
        log("FRAME-STACK SMOKE: FAIL")
        for f in fails:
            log(f"  - {f}")
    else:
        log(f"FRAME-STACK SMOKE: PASS  ({per_frame_c*n}-ch image, reset-cleared, motion captured)")
    log("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
