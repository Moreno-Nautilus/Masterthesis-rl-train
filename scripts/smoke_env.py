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
    env = gym.make(args_cli.task, cfg=env_cfg)
    emit(f"created env OK | action_space={env.action_space} obs_space={env.observation_space}")

    obs, _ = env.reset()
    obs_t = obs["policy"] if isinstance(obs, dict) else obs
    emit(f"reset OK | obs['policy'] shape={tuple(obs_t.shape)}")

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
        emit(f"GEOM env0: shaft_tip ABOVE socket_opening by z = {(hb[0,2]-so[2]).item():+.4f} m  (want ~+0.010 pre-insert)")
        xy = ((hb[0,0]-sb[0])**2 + (hb[0,1]-sb[1])**2) ** 0.5
        emit(f"GEOM env0: shaft_tip<->socket xy offset = {xy.item():.4f} m")
    except Exception as e:
        emit(f"GEOM diag skipped: {e}")

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
