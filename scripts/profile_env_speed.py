"""Profile where the Forge/Factory env spends wall-time: RESET (the IK servo loop) vs the per-step
rollout physics. Answers "is it the reset or the physics, and which knob helps?" without a full train.

Times a cold reset, a warm reset, and N zero-action steps (arm holds), then extrapolates the per-epoch
cost (horizon rollout + ~1 reset/epoch) and env-fps. Optional overrides let us A/B the obvious levers in
separate runs: --solver_iters (the 192-iteration tight-contact solver), --num_envs, --render_interval.

    OMNI_KIT_ACCEPT_EULA=YES python scripts/profile_env_speed.py \
        --task Isaac-Insertion-CoolingPeg-Iiwa-E2E-Direct-v0 --num_envs 128 --steps 128 \
        --experience /home/moreno/Masterthesis-rl-train/apps/isaaclab.python.headless.rendering.physx1065.kit

Report -> /tmp/profile_env_speed.txt (Isaac swallows stdout).
"""

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Insertion-CoolingPeg-Iiwa-E2E-Direct-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=128, help="zero-action steps to time (≈ one rollout horizon)")
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--resets", type=int, default=3, help="warm resets to average")
parser.add_argument("--solver_iters", type=int, default=None, help="override the PhysX position-iteration count everywhere")
parser.add_argument("--render_interval", type=int, default=None, help="override sim.render_interval")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import traceback

import torch
import gymnasium as gym

REPORT = []


def emit(line=""):
    REPORT.append(str(line))


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _set_solver_iters(cfg, n):
    """Lower the PhysX position-iteration count in every place the cooling cfgs set 192."""
    hits = []
    try:
        cfg.sim.physx.max_position_iteration_count = n
        hits.append("sim.physx")
    except Exception:
        pass
    targets = [("robot", getattr(cfg, "robot", None))]
    task = getattr(cfg, "task", None)
    for name in ("fixed_asset", "held_asset"):
        targets.append((name, getattr(task, name, None) if task else None))
    for name, art in targets:
        if art is None:
            continue
        for propname in ("rigid_props", "articulation_props"):
            props = getattr(getattr(art, "spawn", None), propname, None)
            if props is not None and hasattr(props, "solver_position_iteration_count"):
                props.solver_position_iteration_count = n
                hits.append(f"{name}.{propname}")
    return hits


def main():
    import isaaclab_tasks  # noqa: F401
    import insertion_policy.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    emit(f"task={args_cli.task} num_envs={args_cli.num_envs} steps={args_cli.steps}")
    emit(f"decimation={env_cfg.decimation} sim.dt={env_cfg.sim.dt} default render_interval={env_cfg.sim.render_interval}")

    if args_cli.render_interval is not None:
        env_cfg.sim.render_interval = args_cli.render_interval
        emit(f"OVERRIDE render_interval -> {args_cli.render_interval}")
    if args_cli.solver_iters is not None:
        hits = _set_solver_iters(env_cfg, args_cli.solver_iters)
        emit(f"OVERRIDE solver_position_iteration_count -> {args_cli.solver_iters} at {hits}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    n_act = env.action_space.shape[1] if len(env.action_space.shape) > 1 else env.unwrapped.cfg.action_space
    zero = torch.zeros((args_cli.num_envs, n_act), device=args_cli.device)

    # --- cold reset (includes one-time warmup) ---
    _sync(); t = time.perf_counter()
    env.reset()
    _sync(); cold_reset = time.perf_counter() - t
    emit(f"\ncold reset (incl warmup): {cold_reset:.2f} s")

    # --- warm resets (the recurring per-episode cost) ---
    warm = []
    for _ in range(args_cli.resets):
        _sync(); t = time.perf_counter()
        env.reset()
        _sync(); warm.append(time.perf_counter() - t)
    reset_mean = sum(warm) / len(warm)
    emit(f"warm reset x{args_cli.resets}: mean {reset_mean:.3f} s  (each: {[round(w,3) for w in warm]})")

    # --- per-step rollout cost (zero action = arm holds) ---
    for _ in range(args_cli.warmup):
        env.step(zero)
    _sync(); t = time.perf_counter()
    for _ in range(args_cli.steps):
        env.step(zero)
    _sync(); step_total = time.perf_counter() - t
    per_step = step_total / args_cli.steps
    emit(f"\n{args_cli.steps} zero-action steps: {step_total:.2f} s  -> {per_step*1000:.1f} ms/step  "
         f"({args_cli.num_envs/per_step:,.0f} env-fps)")

    # --- extrapolate a training epoch: horizon rollout + ~1 reset/epoch ---
    horizon = 128
    rollout = horizon * per_step
    epoch = rollout + reset_mean
    emit("\n--- extrapolated per training epoch (horizon=128) ---")
    emit(f"rollout (128 steps): {rollout:.1f} s   ({rollout/epoch*100:.0f}%)")
    emit(f"reset (~1/epoch):    {reset_mean:.1f} s   ({reset_mean/epoch*100:.0f}%)")
    emit(f"=> ~{epoch:.1f} s/epoch, ~{args_cli.num_envs*horizon/epoch:,.0f} frame-fps")

    env.close()
    emit("\nRESULT: PROFILE OK")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        emit("RESULT: FAILED")
        emit(traceback.format_exc())
    finally:
        with open("/tmp/profile_env_speed.txt", "w") as f:
            f.write("\n".join(REPORT) + "\n")
        simulation_app.close()
