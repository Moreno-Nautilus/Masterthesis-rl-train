#!/usr/bin/env python3

import os
import pickle as pkl
from pathlib import Path
import time
from typing import Any, Dict, Optional
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "serl_launcher"))
sys.path.insert(0, str(ROOT / "iiwa_serl"))

from absl import app, flags
try:
    import gym
except ImportError:
    import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from flax.training import checkpoints

from agentlace.data.data_store import QueuedDataStore
from agentlace.trainer import TrainerClient, TrainerServer
from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.common.evaluation import evaluate
from serl_launcher.utils.launcher import (
    make_drq_agent,
    make_replay_buffer,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.wrappers.chunking import ChunkingWrapper

import iiwa_serl  # noqa: F401

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "IiwaInsertReal-Vision-v0", "Environment name.")
flags.DEFINE_string("exp_name", None, "Experiment name.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("critic_actor_ratio", 4, "Critic to actor update ratio.")
flags.DEFINE_integer("max_steps", 1000000, "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 200000, "Replay buffer capacity.")
flags.DEFINE_integer("random_steps", 300, "Random exploration steps.")
flags.DEFINE_integer("training_starts", 300, "Minimum replay size before training.")
flags.DEFINE_integer("steps_per_update", 30, "Actor steps between parameter syncs.")
flags.DEFINE_integer("log_period", 10, "Logging period.")
flags.DEFINE_integer("eval_period", 2000, "Evaluation period.")
flags.DEFINE_integer("eval_n_trajs", 5, "Evaluation episodes.")
flags.DEFINE_integer(
    "eval_checkpoint_step",
    0,
    "If >0, run the actor in DEPLOY/EVAL-only mode: restore this checkpoint step from "
    "--checkpoint_path and run --eval_n_trajs episodes on the real robot, then exit. No learner needed.",
)
flags.DEFINE_boolean("learner", False, "Run learner.")
flags.DEFINE_boolean("actor", False, "Run actor.")
flags.DEFINE_string("ip", "localhost", "Learner host.")
flags.DEFINE_string("encoder_type", "resnet-pretrained", "Encoder type.")
flags.DEFINE_string("demo_path", None, "Optional pickle file of demo transitions.")
flags.DEFINE_integer("checkpoint_period", 0, "Checkpoint frequency.")
flags.DEFINE_string("checkpoint_path", None, "Checkpoint path.")
flags.DEFINE_boolean("debug", False, "Disable external logging.")
flags.DEFINE_string("log_rlds_path", None, "Optional RLDS log directory.")
flags.DEFINE_boolean(
    "manual_reset",
    False,
    "Real-robot MANUAL reset: before every episode the operator jogs the peg to a pre-insert "
    "pose (hold R1 + sticks) and presses X, exactly like record_demo. Env does NOT auto-move on "
    "reset (no FRI-drop lurch). Use for base-anywhere training. Disables periodic eval.",
)

devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(text):
    return print(f"\033[92m {text}\033[00m")


def make_env(fake_env: bool = False):
    kwargs = {}
    if "Real" in FLAGS.env:
        kwargs["fake_env"] = fake_env
    env = gym.make(FLAGS.env, **kwargs)
    if FLAGS.manual_reset and "Real" in FLAGS.env:
        # Same manual pre-insert workflow as record_demo: reset() captures the CURRENT pose as
        # the episode origin (NO auto-move → no FRI-drop lurch), workspace box is relative to it,
        # and env.step drives base-frame to match the jog feel. The operator jogs to pre-insert
        # BEFORE each reset (see jog_to_preinsert in the actor).
        cfg = env.unwrapped.config
        cfg.manual_reset = True
        cfg.relative_pose_limit = True
        cfg.base_frame_actions = True
        cfg.reward_mode = "manual"   # success = operator presses X (visual ground truth). The
                                     # force_depth auto-check was unusable (uncompensated force);
                                     # even with the external_torque fix, use manual X first.
        # D405 intermittently freezes (multi-second USB stalls). Reuse the last good frame
        # through a gap instead of crashing the actor mid-run (same as record_demo). Only helps
        # AFTER the first frame has arrived — the camera must be live at startup.
        cfg.reuse_last_camera_frame = True
        # Gentler motion so early random/unconverged actions don't ram the socket and drop FRI.
        # Halve XY, quarter Z (Z into a hard surface builds force fastest).
        cfg.action_scale_xyz_m = 0.005                       # 10mm -> 5mm base step
        cfg.action_scale_xyz_mult = np.array([1.0, 1.0, 0.5])  # Z half again -> 2.5mm/step
        cfg.action_scale_rot_rad = 0.05                      # gentler rotation too
        # Force-triggered retract: back off if contact force exceeds threshold, before FRI drops.
        cfg.force_retract_enable = True
        cfg.force_retract_thresh_n = 25.0                    # TUNE at rig, below the FRI drop point
        cfg.force_retract_step_m = 0.003
    # The iiwa env already emits a flat {"state", <image_keys>} obs dict (no nested
    # "images" group), so SERLObsWrapper is not needed here. But the memory-efficient
    # replay buffer requires image observations to carry a temporal axis
    # (B, T, H, W, C); ChunkingWrapper(obs_horizon=1) adds that T=1 dimension.
    # Without it the learner crashes in sample() at obs_pixels.transpose((0,4,1,2,3)).
    if any(k != "state" for k in env.observation_space.spaces.keys()):
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    return env


def actor(agent: DrQAgent, data_store, env, sampling_rng):
    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_store,
        wait_for_server=True,
    )

    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    # --- Manual pre-insert reset (base-anywhere) -------------------------------------------
    # Copies the WORKING record_demo jog: hold R1 + sticks to drive the peg to a pre-insert
    # pose, press X to arm. Then env.reset() (manual_reset) captures that pose as the episode
    # origin without auto-moving. This is what avoids the FRI-drop lurch of auto-reset.
    teleop = None
    jog_target = None
    pygame = None
    if FLAGS.manual_reset:
        import pygame

        from iiwa_serl.teleop import PS4TeleopProvider
        from iiwa_serl.teleop.pose_target import TeleopPoseTarget
        teleop = PS4TeleopProvider(joystick_index=0, server_url=None)
        _rc = env.unwrapped.config
        jog_target = TeleopPoseTarget(
            np.array([0.004, 0.004, _rc.action_scale_xyz_m * _rc.action_scale_xyz_mult[2]]),
            _rc.action_scale_rot_rad,
            base_frame_actions=True,
        )

    def poll_manual_buttons():
        """Poll motion state and consume manual-result buttons exactly once.

        The policy loop can block in inference/env.step long enough for a quick press and
        release to occur between get_button() snapshots. JOYBUTTONDOWN remains queued, so
        use it as the reliable edge source while retaining the provider's polled flags as
        a fallback. Always consume both flags so jog events cannot leak into an episode.
        """
        teleop_action = teleop.get_action()
        queued = pygame.event.get([pygame.JOYBUTTONDOWN, pygame.JOYBUTTONUP])

        js = teleop._js
        js_instance = js.get_instance_id() if hasattr(js, "get_instance_id") else None

        def is_this_joystick(event):
            event_instance = getattr(event, "instance_id", None)
            if js_instance is not None and event_instance is not None:
                return event_instance == js_instance
            return getattr(event, "joy", 0) == 0

        button_down = {
            int(event.button)
            for event in queued
            if event.type == pygame.JOYBUTTONDOWN and is_this_joystick(event)
        }
        cross = int(teleop._cfg["btn_cross"])
        triangle = int(teleop._cfg["btn_triangle"])
        success = teleop.is_success()
        failure = teleop.is_failure()
        success = success or cross in button_down
        failure = failure or triangle in button_down

        # TEMP live diagnostic: confirms the controller's actual SDL button indices.
        if button_down:
            raw_pressed = [
                i for i in range(js.get_numbuttons()) if js.get_button(i)
            ]
            print(
                f"\n[PS4 BUTTONS] down={sorted(button_down)} raw_pressed={raw_pressed} "
                f"configured(X={cross}, Triangle={triangle})",
                flush=True,
            )
        return teleop_action, success, failure

    def jog_to_preinsert():
        """Operator jogs peg to pre-insert (hold R1 + sticks), presses X to arm. Then returns
        env.reset() which captures the current pose as the episode origin (no auto-move)."""
        rob = env.unwrapped.client
        print("\n[TRAIN] JOG peg to PRE-INSERT pose (hold R1 + sticks). Press X when ready...")
        rob.hold_position()
        time.sleep(1.0 / float(env.unwrapped.config.hz))
        jog_target.reset(rob.get_state().pose.copy())
        while True:
            loop_start = time.monotonic()
            action, arm_pressed, _ = poll_manual_buttons()
            if arm_pressed:                         # X = armed
                rob.hold_position()
                break
            measured = rob.get_state().pose.copy()
            command = jog_target.update(action, measured)
            if command.kind == "move":
                rob.move_pose(command.target_pose)
            elif command.kind == "hold":
                rob.hold_position()
            time.sleep(max(0.0, 1.0 / float(env.unwrapped.config.hz) - (time.monotonic() - loop_start)))
        print("[TRAIN] Armed — episode running.")
        return env.reset()

    eval_env = None
    if not FLAGS.manual_reset:
        eval_env = gym.wrappers.RecordEpisodeStatistics(make_env(fake_env=False))

    if FLAGS.manual_reset:
        obs, _ = jog_to_preinsert()
    else:
        obs, _ = env.reset()
    timer = Timer()

    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True):
        timer.tick("total")
        with timer.context("sample_actions"):
            if step < FLAGS.random_steps:
                action = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                action = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    deterministic=False,
                )
                action = np.asarray(jax.device_get(action))

        # Manual reward: operator presses X (success) / triangle (abort) DURING the episode.
        # X -> reward=1, end episode as success. Triangle -> end with no reward.
        manual_success = False
        manual_abort = False
        if FLAGS.manual_reset and teleop is not None:
            _, manual_success, manual_abort = poll_manual_buttons()
            # Abort wins if both buttons are pressed together. Do not execute another
            # policy move after an operator termination request.
            if manual_abort:
                manual_success = False
            if manual_success or manual_abort:
                action = np.zeros_like(action)
            if manual_success:
                env.unwrapped.set_manual_success(True)  # env._success() returns True this step

        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(action)
            if manual_success:
                reward = 1.0
                done = True
            elif manual_abort:
                reward = 0.0
                done = True
            transition = dict(
                observations=obs,
                actions=action,
                next_observations=next_obs,
                rewards=np.asarray(reward, dtype=np.float32),
                masks=1.0 - float(done),
                dones=bool(done or truncated),
            )
            data_store.insert(transition)
            obs = next_obs
            if done or truncated:
                if manual_success:
                    print_green(f"  [SUCCESS] operator marked seat at step {step}")
                elif manual_abort:
                    print(f"\n[ABORT] operator ended episode at step {step}", flush=True)
                if FLAGS.manual_reset:
                    obs, _ = jog_to_preinsert()
                else:
                    obs, _ = env.reset()

        if step % FLAGS.steps_per_update == 0:
            client.update()

        # Periodic eval runs un-gated env.reset()s (no jog) — incompatible with manual reset,
        # so skip it there. You are watching the arm live anyway.
        if not FLAGS.manual_reset and step % FLAGS.eval_period == 0:
            with timer.context("eval"):
                stats = evaluate(
                    policy_fn=lambda observations, **_: agent.sample_actions(
                        observations=observations, argmax=True
                    ),
                    env=eval_env,
                    num_episodes=FLAGS.eval_n_trajs,
                )
            client.request("send-stats", {"eval": stats})

        timer.tock("total")
        if step % FLAGS.log_period == 0:
            client.request("send-stats", {"timer": timer.get_average_times()})


def learner(
    rng,
    agent: DrQAgent,
    replay_buffer,
    demo_buffer=None,
):
    wandb_logger = make_wandb_logger(
        project="serl_dev",
        description=FLAGS.exp_name or FLAGS.env,
        debug=FLAGS.debug,
    )
    update_steps = 0

    def stats_callback(kind: str, payload: dict) -> dict:
        assert kind == "send-stats"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=update_steps)
        return {}

    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)

    pbar = tqdm.tqdm(
        total=FLAGS.training_starts,
        initial=len(replay_buffer),
        desc="Filling replay buffer",
    )
    while len(replay_buffer) < FLAGS.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)
    pbar.close()

    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    single_buffer_batch_size = FLAGS.batch_size if demo_buffer is None else FLAGS.batch_size // 2
    replay_iterator = replay_buffer.get_iterator(
        sample_args={"batch_size": single_buffer_batch_size, "pack_obs_and_next_obs": True},
        device=sharding.replicate(),
    )
    demo_iterator = None
    if demo_buffer is not None:
        demo_iterator = demo_buffer.get_iterator(
            sample_args={"batch_size": FLAGS.batch_size // 2, "pack_obs_and_next_obs": True},
            device=sharding.replicate(),
        )

    timer = Timer()
    pbar = tqdm.tqdm(
        total=FLAGS.replay_buffer_capacity,
        initial=len(replay_buffer),
        desc="replay buffer",
    )
    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True, desc="learner"):
        for _critic_step in range(FLAGS.critic_actor_ratio - 1):
            with timer.context("sample_replay"):
                batch = next(replay_iterator)
                if demo_iterator is not None:
                    batch = concat_batches(batch, next(demo_iterator), axis=0)
            with timer.context("train_critic"):
                agent, critics_info = agent.update_critics(batch)

        with timer.context("train"):
            batch = next(replay_iterator)
            if demo_iterator is not None:
                batch = concat_batches(batch, next(demo_iterator), axis=0)
            agent, update_info = agent.update_high_utd(batch, utd_ratio=1)

        if update_steps % FLAGS.log_period == 0 and wandb_logger is not None:
            wandb_logger.log(update_info, step=update_steps)
            wandb_logger.log({"timer": timer.get_average_times()}, step=update_steps)

        if step > 0 and step % FLAGS.steps_per_update == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if FLAGS.checkpoint_period and update_steps % FLAGS.checkpoint_period == 0:
            assert FLAGS.checkpoint_path is not None
            checkpoints.save_checkpoint(
                FLAGS.checkpoint_path,
                agent.state,
                step=update_steps,
                keep=20,
            )

        pbar.update(len(replay_buffer) - pbar.n)
        update_steps += 1


def main(_):
    assert FLAGS.batch_size % num_devices == 0
    env = make_env(fake_env=FLAGS.learner)
    image_keys = [key for key in env.observation_space.keys() if key != "state"]

    agent = make_drq_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=image_keys,
        encoder_type=FLAGS.encoder_type,
    )
    agent = jax.device_put(jax.tree_map(jnp.array, agent), sharding.replicate())
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    # DEPLOY / EVAL-ONLY: restore a trained checkpoint and run the policy on the real robot.
    # This is how a SERL-trained policy is "deployed" — same stack as training, no extra backend,
    # no learner. Runs argmax=False (SERL evals with a small amount of exploration by default).
    if FLAGS.eval_checkpoint_step:
        assert FLAGS.checkpoint_path is not None, "--checkpoint_path required for eval."
        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)
        print_green(f"restored checkpoint {FLAGS.eval_checkpoint_step} from {FLAGS.checkpoint_path}")

        success_counter = 0
        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            done = False
            while not done:
                sampling_rng, key = jax.random.split(sampling_rng)
                action = agent.sample_actions(
                    observations=jax.device_put(obs), argmax=False, seed=key
                )
                action = np.asarray(jax.device_get(action))
                obs, reward, done, truncated, info = env.step(action)
                done = done or truncated
            success_counter += int(bool(info.get("success", reward > 0.5)))
            print(f"episode {episode + 1}/{FLAGS.eval_n_trajs}: "
                  f"success={info.get('success')} | running {success_counter}/{episode + 1}")
        print_green(f"SERL eval success rate: {success_counter}/{FLAGS.eval_n_trajs} "
                    f"= {success_counter / max(FLAGS.eval_n_trajs, 1):.1%}")
        return

    if FLAGS.learner:
        replay_buffer = make_replay_buffer(
            env,
            capacity=FLAGS.replay_buffer_capacity,
            rlds_logger_path=FLAGS.log_rlds_path,
            type="memory_efficient_replay_buffer",
            image_keys=image_keys,
        )
        demo_buffer = None
        if FLAGS.demo_path:
            demo_buffer = make_replay_buffer(
                env,
                capacity=FLAGS.replay_buffer_capacity,
                type="memory_efficient_replay_buffer",
                image_keys=image_keys,
            )
            if not os.path.exists(FLAGS.demo_path):
                raise FileNotFoundError(FLAGS.demo_path)
            with open(FLAGS.demo_path, "rb") as handle:
                trajs = pkl.load(handle)
            for traj in trajs:
                demo_buffer.insert(traj)
        learner(sampling_rng, agent, replay_buffer, demo_buffer=demo_buffer)
    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        actor(agent, QueuedDataStore(2000), env, sampling_rng)
    else:
        raise ValueError("Must pass --learner or --actor.")


if __name__ == "__main__":
    app.run(main)
