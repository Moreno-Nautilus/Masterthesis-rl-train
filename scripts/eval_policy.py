# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--num_episodes", type=int, default=512, help="Number of episodes to evaluate over.")
parser.add_argument("--report", type=str, default="/tmp/eval_policy_report.txt", help="Where to write the report.")
parser.add_argument("--stochastic", action="store_true", default=False, help="Sample actions (non-deterministic) instead of the mean -> tests the det-vs-stochastic gap.")
parser.add_argument("--viz_estimator", type=str, default=None, help="Path (png) to save an EXPLICIT-ESTIMATOR visualization: the aux head's predicted hole gap vs the true gap, collected each step. Only meaningful for an estimator checkpoint (aux_head.feed_to_policy). Off by default -> the chain auto-eval is unchanged.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import math
import os
import random
import time

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401

import insertion_policy.tasks  # noqa: F401  (registers Isaac-Insertion-CoolingPeg-Direct-v0)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with RL-Games agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    u = env.unwrapped  # InsertionEnv: exposes _shaft_axis_error / _get_curr_successes / fixed_pos

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    # --- per-episode eval: pair each env's RESET error condition with whether it ever seats. ----
    n = u.num_envs
    thr = u.cfg_task.success_threshold

    def _reset_conditions():
        tilt_deg = torch.rad2deg(u._shaft_axis_error())  # angular pre-insert error per env
        hb, _ = u._held_base_pose()
        xy_mm = torch.linalg.vector_norm(u.fixed_pos[:, 0:2] - hb[:, 0:2], dim=1) * 1000.0
        return tilt_deg, xy_mm

    reset_tilt, reset_xy = _reset_conditions()
    succeeded = torch.zeros(n, dtype=torch.bool, device=u.device)  # ever-seated this episode
    best_zgap = torch.full((n,), 999.0, device=u.device)  # deepest tip->socket-bottom gap (mm) this episode
    rec_tilt, rec_xy, rec_succ, rec_zgap = [], [], [], []  # completed-episode records
    n_done = 0  # total episodes finished (each done-batch holds up to num_envs of them)

    # EXPLICIT-ESTIMATOR viz (opt-in): collect the aux head's predicted hole gap vs the true gap each step.
    _viz = args_cli.viz_estimator
    _vz_net = getattr(getattr(agent, "model", None), "a2c_network", None) if _viz else None
    _vz_pred, _vz_true = [], []  # predicted / true tip->socket gap (fingertip frame); pred is normalized-space

    while n_done < args_cli.num_episodes and simulation_app.is_running():
        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=(agent.is_deterministic and not args_cli.stochastic))
            if _viz and _vz_net is not None and getattr(_vz_net, "_estimator_pred", None) is not None:
                # captured AFTER the forward (get_action populated _estimator_pred) but BEFORE env.step,
                # so the true label matches the same pre-step image the prediction was computed from.
                _vz_pred.append(_vz_net._estimator_pred.detach().float().cpu())      # (n,3) normalized
                _vz_true.append(u._get_aux_label()[:, 1:4].detach().float().cpu())   # (n,3) raw meters
            obs, _, dones, _ = env.step(actions)
            succeeded |= u._get_curr_successes(thr)  # seated+centered+aligned at this step
            _hb, _ = u._held_base_pose()
            best_zgap = torch.minimum(best_zgap, (_hb[:, 2] - u.fixed_pos[:, 2]) * 1000.0)  # min tip->bottom gap (mm)
            dones = dones.nonzero(as_tuple=False).squeeze(-1)
            if len(dones) > 0:
                n_done += len(dones)
                rec_tilt.append(reset_tilt[dones].clone())
                rec_xy.append(reset_xy[dones].clone())
                rec_succ.append(succeeded[dones].clone())
                rec_zgap.append(best_zgap[dones].clone())
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
                # the done envs have auto-reset -> capture their NEW episode's reset condition
                new_tilt, new_xy = _reset_conditions()
                reset_tilt[dones], reset_xy[dones] = new_tilt[dones], new_xy[dones]
                succeeded[dones] = False
                best_zgap[dones] = 999.0

    tilt = torch.cat(rec_tilt)[: args_cli.num_episodes]
    xy = torch.cat(rec_xy)[: args_cli.num_episodes]
    succ = torch.cat(rec_succ)[: args_cli.num_episodes].float()
    zgap = torch.cat(rec_zgap)[: args_cli.num_episodes]

    def _bins(values, edges, label, unit):
        lines = [f"  success vs {label}:"]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (values >= lo) & (values < hi)
            k = int(m.sum())
            rate = float(succ[m].mean()) if k else float("nan")
            lines.append(f"    {lo:5.1f}-{hi:5.1f}{unit}: {rate:5.1%}  (n={k})")
        return "\n".join(lines)

    report = [
        f"checkpoint: {resume_path}",
        f"episodes: {len(succ)}   overall success: {succ.mean().item():.1%}",
        f"reset tilt deg  min/mean/max: {tilt.min():.1f}/{tilt.mean():.1f}/{tilt.max():.1f}",
        f"reset xy mm     min/mean/max: {xy.min():.1f}/{xy.mean():.1f}/{xy.max():.1f}",
        f"SEAT DEPTH (deepest tip->bottom gap, mm) min/mean/max: {zgap.min():.2f}/{zgap.mean():.2f}/{zgap.max():.2f}",
        f"  frac reaching  <1mm: {(zgap<1).float().mean():.1%}   <2mm: {(zgap<2).float().mean():.1%}   "
        f"<3mm: {(zgap<3).float().mean():.1%}   <4.4mm: {(zgap<4.4).float().mean():.1%}",
        _bins(tilt, [0, 5, 10, 15, 20, 25, 90], "angular error", "deg"),
        _bins(xy, [0, 3, 5, 7, 10, 50], "lateral error", "mm"),
    ]
    text = "\n".join(report)
    print("\n===== EVAL =====\n" + text)
    with open(args_cli.report, "w") as f:
        f.write(text + "\n")

    # Also persist a permanent copy in the checkpoint's run dir (the --report path defaults to /tmp,
    # which is overwritten every eval and lost on reboot). Tied to the checkpoint + timestamped.
    try:
        from datetime import datetime as _dt

        ckpt_abs = os.path.abspath(resume_path)
        run_dir = os.path.dirname(os.path.dirname(ckpt_abs))  # .../<run>/nn/<ckpt>.pth -> .../<run>
        stem = os.path.splitext(os.path.basename(ckpt_abs))[0]
        perm = os.path.join(run_dir, f"eval_{stem}_{_dt.now():%Y%m%d_%H%M%S}.txt")
        with open(perm, "w") as f:
            f.write(text + "\n")
        print(f"[INFO]: eval report also saved to: {perm}")
    except Exception as exc:  # never let report-saving break an eval
        print(f"[WARN]: could not save per-run eval report: {exc}")

    # ---- EXPLICIT-ESTIMATOR visualization (opt-in via --viz_estimator) --------------------------------
    if _viz and _vz_pred:
        pred = torch.cat(_vz_pred)  # (T*n, 3) normalized-space prediction
        true = torch.cat(_vz_true)  # (T*n, 3) raw meters
        # The aux head regressed the NORMALIZED label (rl_games normalize_input), so denormalize the
        # prediction with the model's aux_label RunningMeanStd to compare in physical meters.
        mean3, std3 = torch.zeros(3), torch.ones(3)
        denorm_ok = False
        try:
            store = getattr(agent.model.running_mean_std, "running_mean_std", None)
            aux = store["aux_label"] if (store is not None and "aux_label" in store) else None
            if aux is not None:
                mean3 = aux.running_mean.detach().cpu().float()[1:4]
                std3 = torch.sqrt(aux.running_var.detach().cpu().float()[1:4] + 1e-5)
                denorm_ok = True
        except Exception as exc:  # fall back to normalized-space plot
            print(f"[viz] aux_label RMS not found ({exc}); plotting in normalized space")
        pred_m = pred * std3 + mean3 if denorm_ok else pred
        err_mm = torch.linalg.vector_norm(pred_m - true, dim=1) * 1000.0
        dist_mm = torch.linalg.vector_norm(true, dim=1) * 1000.0  # true tip->hole distance
        near = dist_mm < 10.0
        vtext = [
            "===== ESTIMATOR VIZ =====",
            f"samples: {len(err_mm)}   denorm: {denorm_ok}",
            f"pred-vs-true hole-gap error (mm)  mean/median/p90: "
            f"{err_mm.mean():.2f}/{err_mm.median():.2f}/{err_mm.quantile(0.9):.2f}",
            f"  near hole (<10mm): mean {err_mm[near].mean():.2f}mm  (n={int(near.sum())})" if near.any() else "  (no near-hole samples)",
            f"true tip->hole dist (mm) min/mean/max: {dist_mm.min():.1f}/{dist_mm.mean():.1f}/{dist_mm.max():.1f}",
        ]
        vtxt = "\n".join(vtext)
        print("\n" + vtxt)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            sub = torch.randperm(len(err_mm))[:4000]  # subsample for a legible scatter
            fig, ax = plt.subplots(1, 3, figsize=(16, 5))
            ax[0].scatter(dist_mm[sub], err_mm[sub], s=3, alpha=0.3)
            ax[0].set_xlabel("true tip->hole dist (mm)"); ax[0].set_ylabel("estimator error (mm)")
            ax[0].set_title("estimator error vs approach"); ax[0].grid(alpha=0.3)
            u_mm = 1000.0
            # lateral (x,y) accuracy: true vs predicted socket-opening position in the fingertip frame
            ax[1].scatter(true[sub, 0] * u_mm, true[sub, 1] * u_mm, s=6, alpha=0.4, label="true", c="tab:green")
            ax[1].scatter(pred_m[sub, 0] * u_mm, pred_m[sub, 1] * u_mm, s=6, alpha=0.4, label="pred", c="tab:red")
            ax[1].set_xlabel("gap x (mm)"); ax[1].set_ylabel("gap y (mm)")
            ax[1].set_title("hole position: pred vs true (fingertip frame)"); ax[1].legend(); ax[1].grid(alpha=0.3)
            ax[1].set_aspect("equal", "box")
            ax[2].hist(err_mm.numpy(), bins=60)
            ax[2].set_xlabel("estimator error (mm)"); ax[2].set_ylabel("count")
            ax[2].set_title("error distribution"); ax[2].grid(alpha=0.3)
            fig.suptitle(f"Explicit estimator — {os.path.basename(resume_path)}")
            fig.tight_layout()
            out = args_cli.viz_estimator
            if not out.endswith(".png"):
                out = os.path.join(out, "estimator_viz.png")
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            fig.savefig(out, dpi=120)
            with open(os.path.splitext(out)[0] + ".txt", "w") as f:
                f.write(vtxt + "\n")
            print(f"[viz] saved estimator visualization -> {out}")
        except Exception as exc:
            print(f"[viz] plot failed ({exc}); stats above still valid")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
