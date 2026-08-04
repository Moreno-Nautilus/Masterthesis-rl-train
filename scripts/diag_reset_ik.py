"""Root-cause the slow reset: instrument the hover-IK servo loop and report, per attempt, how many
envs are still 'bad' and the residual position/orientation error distribution.

The base reset (FactoryEnv.randomize_initial_state) retries any env whose IK servo doesn't converge to
1 mm AND 0.057 deg of its (already ±8 mm / ±45 deg random) target -> stragglers force ~20+ passes. This
tells us whether those stragglers sit JUST ABOVE that threshold (=> loosening the tolerance clears them,
the cheap fix) or are WILDLY off (=> a reachability/seed problem instead), and what looser tol would clear
~95% in 1-2 passes. Run with a HIGH budget so we see the natural convergence, not the capped one.

    OMNI_KIT_ACCEPT_EULA=YES python scripts/diag_reset_ik.py \
        --task Isaac-Insertion-CoolingPeg-Iiwa-E2E-Direct-v0 --num_envs 128 --budget 60 \
        --experience /home/moreno/Masterthesis-rl-train/apps/isaaclab.python.headless.rendering.physx1065.kit

Report -> /tmp/diag_reset_ik.txt
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Insertion-CoolingPeg-Iiwa-E2E-Direct-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--budget", type=int, default=60, help="reset hover-IK retry cap (high => see natural convergence)")
parser.add_argument("--resets", type=int, default=2)
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


def main():
    import isaaclab_tasks  # noqa: F401
    import insertion_policy.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if hasattr(env_cfg, "debug_reset_ik_budget"):
        env_cfg.debug_reset_ik_budget = args_cli.budget  # uncap so we see the natural attempt count
    emit(f"task={args_cli.task} num_envs={args_cli.num_envs} reset_ik_budget={args_cli.budget}")
    emit("base reset convergence threshold = 1.0 mm AND 0.057 deg (norm of pos_error / axis-angle error)\n")

    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    orig = u.set_pos_inverse_kinematics
    calls = []  # (n_env, pos_mm tensor, ang_deg tensor)

    def wrapped(*a, **k):  # base calls this with keyword args; the tilt path with positional -> pass through
        pe, ae = orig(*a, **k)
        pos_mm = torch.linalg.norm(pe, dim=1) * 1000.0
        ang_deg = torch.linalg.norm(ae, dim=1) * 180.0 / math.pi
        calls.append((int(pos_mm.numel()), pos_mm.detach().float().cpu(), ang_deg.detach().float().cpu()))
        return pe, ae

    u.set_pos_inverse_kinematics = wrapped

    def pct(t, q):
        return torch.quantile(t, q).item() if t.numel() else float("nan")

    for r in range(args_cli.resets):
        calls.clear()
        env.reset()
        emit(f"===== RESET {r} : {len(calls)} IK servo calls (hover retries + the 1 pre-insert-tilt call) =====")
        emit(f"{'call':>4} {'n_env':>6} | {'pos p50':>7} {'p90':>6} {'max':>6} mm | {'ang p50':>7} {'p90':>6} {'max':>6} deg")
        for i, (n, pmm, adeg) in enumerate(calls):
            emit(f"{i:>4} {n:>6} | {pct(pmm,.5):>7.2f} {pct(pmm,.9):>6.2f} {pmm.max().item():>6.1f} mm | "
                 f"{pct(adeg,.5):>7.2f} {pct(adeg,.9):>6.2f} {adeg.max().item():>6.1f} deg")
        # On the FIRST hover attempt, what fraction of envs would pass at various tolerances?
        if calls:
            n, pmm, adeg = calls[0]
            emit(f"\n  attempt-0 pass-rate (pos AND ang) at tolerances (n={n}):")
            for pt, at in [(1.0, 0.057), (2.0, 0.5), (3.0, 1.0), (5.0, 2.0), (8.0, 3.0)]:
                ok = ((pmm < pt) & (adeg < at)).float().mean().item() * 100
                tag = "  <- current" if pt == 1.0 else ""
                emit(f"    pos<{pt:>4.1f}mm AND ang<{at:>4.2f}deg : {ok:5.1f}% pass{tag}")
        emit("")

    env.close()
    emit("RESULT: DIAG OK")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        emit("RESULT: FAILED")
        emit(traceback.format_exc())
    finally:
        with open("/tmp/diag_reset_ik.txt", "w") as f:
            f.write("\n".join(REPORT) + "\n")
        simulation_app.close()
