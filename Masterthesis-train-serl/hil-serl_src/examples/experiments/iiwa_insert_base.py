"""Shared per-insert HIL-SERL TrainConfig base for BOTH KUKA assemblies.

plumbers_block (4 inserts) and cooling_manifold (6 inserts) subclass this; each concrete
config sets `assembly` + `insertion_index`. Builds OUR real-KUKA env in RESIDUAL mode (or
E2E when RESIDUAL_ENABLE=0), wired with the tested PS4 intervention + manual-reward wrappers.

Frame handling: reset drives to the per-insert JOINT config (frame-independent); the goal
pose + insertion axis are derived at reset from the robot's OWN FK on the goal joints (see
base_real_env._resolve_goal_from_fk) — no handoff world->base transform, no flange/TCP bug.
"""

from __future__ import annotations

import os

import numpy as np

from experiments.config import DefaultTrainingConfig

from iiwa_serl.config import IiwaInsertionConfig, CameraConfig
from iiwa_serl.envs.iiwa_insert_env import KukaIiwaInsertionEnv
from iiwa_serl.envs.insertion_handoff import load_insert_spec
from iiwa_serl.envs.hil_wrappers import PS4Intervention, HumanReward
from serl_launcher.wrappers.chunking import ChunkingWrapper

_IMAGE_KEY = "wrist"


def _residual_enabled() -> bool:
    """RESIDUAL_ENABLE=0 flips to E2E fallback; default (unset) = residual ON."""
    return os.environ.get("RESIDUAL_ENABLE", "1") not in ("0", "false", "False", "")


class IiwaInsertBase(DefaultTrainingConfig):
    """Shared config; concrete subclasses set `assembly` and `insertion_index`."""

    assembly: str = "plumbers_block"
    insertion_index: int = 0

    # --- upstream train_rlpd knobs ---
    agent = "drq"
    max_traj_length = 400          # generous; the env sets the true per-insert horizon (#7)
    batch_size = 256
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"      # welded/pregrasped gripper, no learned gripper DoF
    image_keys = [_IMAGE_KEY]
    proprio_keys = ["state"]
    classifier_keys = []                          # manual reward — no classifier
    random_steps = 0
    training_starts = 100
    steps_per_update = 50
    checkpoint_period = 2000
    buffer_period = 2000
    eval_n_trajs = 0                              # manual-reward: no un-gated eval resets
    eval_period = 0

    def _make_iiwa_config(self) -> IiwaInsertionConfig:
        spec = load_insert_spec(self.insertion_index, assembly=self.assembly)
        residual = _residual_enabled()
        cfg = IiwaInsertionConfig(
            reward_mode="manual",                 # operator X=success / Square=abort
            base_frame_actions=True,
            relative_pose_limit=True,
            reuse_last_camera_frame=True,
            # GENTLER correction sensitivity (supervisor call: demos were too fast at full stick).
            action_scale_xyz_m=0.0025,
            action_scale_xyz_mult=np.array([1.0, 1.0, 0.5]),
            action_scale_rot_rad=0.03,
            # per-insert JOINT configs (frame-independent); goal/axis FK'd at reset.
            reset_joints=spec.reset_joints,
            reset_joints_target=spec.reset_joints,
            goal_joints=spec.goal_joints,
            joint_reset_on_reset=True,
            manual_reset=False,                   # AUTOMATED reset to the annotated pre-insert
            random_reset=False,
            force_retract_enable=True,
            force_retract_thresh_n=25.0,          # TUNE at rig
            force_retract_step_m=0.003,
            residual_enable=residual,
            nominal_speed_mm_s=float(os.environ.get("NOMINAL_SPEED_MM_S", "4.0")),
            nominal_pause_force_n=8.0,            # TUNE at rig
            residual_rotation_enable=(
                os.environ.get("RESIDUAL_ROTATION", "0") in ("1", "true", "True")
            ),
        )
        # Fixed ZED scene camera — RECORDING-ONLY (record_only), OPT-IN via RECORD_ZED=1 so a
        # missing/dead ZED can never break training/recording (#21). NOT a policy/training input.
        cams = [CameraConfig(image_key="wrist", ros_topic="/realsense_1/camera/color/image_raw")]
        if os.environ.get("RECORD_ZED", "0") in ("1", "true", "True"):
            cams.append(CameraConfig(
                image_key="scene", record_only=True,
                ros_topic=os.environ.get("ZED_TOPIC", "/zed/zed_node/left/image_rect_color"),
            ))
        cfg.cameras = cams
        return cfg

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = KukaIiwaInsertionEnv(
            config=self._make_iiwa_config(), include_image=True, fake_env=fake_env,
        )
        # #10: the LEARNER builds the env with fake_env=True purely for obs/action SPACES — it does
        # NOT drive the robot and has NO PS4 controller, so the hardware wrappers (PS4Intervention
        # opens the joystick; HumanReward reads its buttons) must be SKIPPED there or the learner
        # crashes with "No joystick". Only the ACTOR (fake_env=False) gets the teleop wrappers.
        # ChunkingWrapper is applied in BOTH so the observation_space (T=1 image axis) matches.
        if not fake_env:
            env = PS4Intervention(env)     # emits intervene_action, sets L1 stop-forward
            env = HumanReward(env)         # X=success / Square,Triangle=abort
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        return env

    def process_demos(self, demo):
        return demo
