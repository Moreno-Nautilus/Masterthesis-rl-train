"""Smoke-test the 6-axis F/T feature: verify the policy proprio widens to 27-d and the appended
TORQUE channels are populated (non-zero on contact).

Forge already feeds the 3-axis contact FORCE into the policy obs; ``ForgeTaskCoolingInsertCfg.
use_torque_obs`` appends the matching 3 TORQUE channels (``force_sensor_smooth[:,3:6]``, with obs
noise), turning the 24-d proprio into 27-d. This boots the vision env with the toggle ON, drives the
tilted shaft DOWN into the socket (so the shaft presses the hole wall -> a contact MOMENT = torque),
and checks:

  * no crash;
  * proprio width == 27 (24 + 3 torque);
  * the last 3 proprio dims match ``force_sensor_smooth[:,3:6]`` up to the obs noise (correct wiring);
  * the torque channels are NON-ZERO once there is contact force (they share the sensor, so force!=0
    should imply torque!=0);
  * prints realized FORCE-vs-TORQUE magnitudes so the torque obs-noise (``torque_obs_noise``) can be
    sanity-checked against the signal it is meant to perturb.

    OMNI_KIT_ACCEPT_EULA=YES TORCHDYNAMO_DISABLE=1 python scripts/check_torque_obs.py --num_envs 8 \
        --steps 90 --experience apps/isaaclab.python.headless.rendering.physx1065.kit
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke-test the 6-axis F/T torque obs.")
parser.add_argument("--task", type=str, default="Isaac-Insertion-CoolingPeg-Vision-Direct-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=90, help="Downward-push steps to drive contact.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

# Isaac block-buffers stdout and hard-exits on close (buffer lost when piped to a file, see DECISIONS
# 8b), so mirror every result line into a file that is flushed immediately -- the reliable channel.
_RESULT_PATH = "/tmp/torque_smoke_result.txt"
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

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_torque_obs = True  # the feature under test
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    noise = float(getattr(u.cfg, "torque_obs_noise", 0.0))
    log(f"[INFO] task={args_cli.task} num_envs={args_cli.num_envs} use_torque_obs=True "
          f"torque_obs_noise={noise}")

    fails = []

    obs, _ = env.reset()
    policy = obs["policy"]
    width = int(policy.shape[1])
    log(f"[INFO] policy proprio width = {width}  (expected 27 = 24 base + 3 torque)")
    if width != 27:
        fails.append(f"policy width {width} != 27")

    # Drive the (pre-insert-tilted) shaft straight DOWN toward the socket: action[:,2] = -1 (z delta),
    # a touch of +x to press a wall. EMA (0.2) ramps it over a few steps; keep pushing to build contact.
    action = torch.zeros((u.num_envs, u.single_action_space.shape[0]), device=u.device)
    action[:, 2] = -1.0
    action[:, 0] = 0.3

    max_force = torch.zeros(u.num_envs, device=u.device)
    max_torque = torch.zeros(u.num_envs, device=u.device)
    contact_seen = False
    wiring_max_err = 0.0

    for step in range(args_cli.steps):
        obs, _, term, trunc, _ = env.step(action)
        policy = obs["policy"]
        if not torch.isfinite(policy).all():
            fails.append(f"step {step}: policy obs has NaN/inf")
            break

        # Ground-truth force/torque from the shared sensor (privileged, clean).
        f_gt = u.force_sensor_smooth[:, 0:3]
        t_gt = u.force_sensor_smooth[:, 3:6]
        f_mag = torch.linalg.vector_norm(f_gt, dim=1)
        t_mag = torch.linalg.vector_norm(t_gt, dim=1)
        max_force = torch.maximum(max_force, f_mag)
        max_torque = torch.maximum(max_torque, t_mag)

        # The appended obs torque (last 3 dims) must equal t_gt up to the obs noise (std=noise, so a
        # generous 6-sigma+eps bound catches a real wiring bug without flagging normal noise draws).
        obs_torque = policy[:, 24:27]
        wire_err = torch.linalg.vector_norm(obs_torque - t_gt, dim=1).max().item()
        wiring_max_err = max(wiring_max_err, wire_err)

        if f_mag.max().item() > 0.5:  # contact
            contact_seen = True

        if step % 15 == 0 or step == args_cli.steps - 1:
            log(f"  step {step:3d} | |force| mean/max = {f_mag.mean():6.3f}/{f_mag.max():6.3f} N | "
                  f"|torque| mean/max = {t_mag.mean():7.4f}/{t_mag.max():7.4f} N*m")

        if term.any() or trunc.any():
            # keep pushing across resets; the env auto-resets, contact rebuilds
            pass

    # Wiring bound: |obs_torque - true_torque| should be pure obs noise. Bound at 6*std*sqrt(3)+2mm.
    wire_bound = 6.0 * noise * (3 ** 0.5) + 0.002
    if wiring_max_err > wire_bound:
        fails.append(f"obs torque != force_sensor_smooth[:,3:6]+noise (max err {wiring_max_err:.4f} "
                     f"> bound {wire_bound:.4f}) -> wiring bug")

    torque_nonzero = max_torque.max().item() > 1e-6
    if contact_seen and not torque_nonzero:
        fails.append("contact force observed but torque stayed ~0 -> torque channels not populated")

    log("\n--- summary over run ---")
    log(f"  max |force|  over envs: {max_force.max().item():.3f} N")
    log(f"  max |torque| over envs: {max_torque.max().item():.5f} N*m")
    log(f"  torque_obs_noise (std): {noise} N*m  -> noise/max|torque| ~= "
          f"{(noise / max(max_torque.max().item(), 1e-9)):.2f}x")
    log(f"  contact seen: {contact_seen} | obs-vs-GT torque wiring max err: {wiring_max_err:.5f} "
          f"(bound {wire_bound:.4f})")

    log("\n" + ("=" * 60))
    if fails:
        log("TORQUE-OBS SMOKE: FAIL")
        for f in fails:
            log(f"  - {f}")
    else:
        log("TORQUE-OBS SMOKE: PASS  (27-d proprio, torque wired + non-zero on contact)")
    log("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
