"""Integration test: prove the STOCK record_demos.py logic works with our franka_plumbers
wrappers — WITHOUT real hardware. Mocks the robot client (contract dict) and the cameras,
then drives the exact record loop and asserts:

  1. every transition's obs has the SAME keys/shapes (incl. the X-terminated FINAL step) —
     i.e. PS4RewardWrapper never returns a differently-shaped/unwrapped obs (demo integrity).
  2. an X press ends the episode with reward 1 and info["succeed"]=True (the recorder saves it).
  3. intervene_action from a simulated R1+stick correction is captured into info.

Run:  source franka_env.sh
      python3 hil-serl_src/examples/experiments/franka_plumbers/test_demo_recording.py
"""

import copy
import numpy as np

from franka_env.envs.robot_client import RobotClient

CONTRACT_STATE = {
    "pose": [0.5, -0.03, 0.28, 0.0, 0.0, 0.0, 1.0], "vel": [0.0] * 6,
    "force": [1.0, 2.0, 3.0], "torque": [0.1, 0.2, 0.3], "jacobian": [0.0] * 42,
    "q": [0.0] * 7, "dq": [0.0] * 7, "gripper_pos": [1.0],
}


class MockClient(RobotClient):
    def get_state(self): return {k: list(v) for k, v in CONTRACT_STATE.items()}
    def send_pos(self, arr): pass
    def recover(self): pass
    def update_param(self, p): pass
    def joint_reset(self): pass
    def open_gripper(self): pass
    def close_gripper(self): pass
    def close_gripper_slow(self): pass
    def set_load(self, p): pass


class MockCap:
    """Mimics VideoCapture(RSCapture).read() -> HxWx3 uint8 frame."""
    def read(self): return np.zeros((720, 1280, 3), dtype=np.uint8)
    def close(self): pass


class ScriptedExpert:
    """Stands in for PS4Expert: emit a correction for a few steps, then press X."""
    def __init__(self, correct_steps=3, x_at=6):
        self.t = 0; self.correct_steps = correct_steps; self.x_at = x_at
    def get_action(self):
        # R1+stick correction (nonzero) for the first few steps, then release (zeros).
        if self.t < self.correct_steps:
            return np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0]), [0, 0]
        return np.zeros(6), [0, 0]
    def get_events(self):
        ev = {"success": self.t == self.x_at, "abort": False, "regrasp": False}
        return ev
    def deadman_held(self): return self.t < self.correct_steps


def build_env():
    """Assemble the ACTUAL actor wrapper stack (with reward+intervention) but inject a
    scripted expert + mock robot/cameras instead of real hardware — so we exercise the same
    code the actor runs, including PS4RewardWrapper and SpacemouseIntervention."""
    from collections import OrderedDict
    from experiments.mappings import CONFIG_MAPPING
    from franka_env.envs.wrappers import (
        GripperCloseEnv, SpacemouseIntervention, PS4RewardWrapper, Quat2EulerWrapper,
    )
    from franka_env.envs.relative_env import RelativeFrame
    from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
    from serl_launcher.wrappers.chunking import ChunkingWrapper
    from experiments.franka_plumbers.wrapper import FrankaPlumbersEnv

    cfg = CONFIG_MAPPING["franka_plumbers_insert0"]()
    scripted = ScriptedExpert()

    # Build the base env in fake_env mode (no hardware), then flip it to the real step path
    # with mocks so get_im + _get_obs + robot all run.
    inner = FrankaPlumbersEnv(fake_env=True, save_video=False, config=cfg.env_config_cls())
    inner.fake_env = False
    inner.robot = MockClient()
    inner.display_image = False
    inner.cap = OrderedDict(wrist=MockCap())
    inner.config.IMAGE_CROP = {}
    inner.post_reset_wait_s = 0.0
    inner.go_to_reset = lambda joint_reset=False: None  # skip real motion interpolation

    # Wrap exactly like config.get_environment's actor path, but inject the scripted expert.
    env = GripperCloseEnv(inner)
    si = SpacemouseIntervention.__new__(SpacemouseIntervention)
    import gymnasium as gym
    gym.ActionWrapper.__init__(si, env)
    si.gripper_enabled = si.action_space.shape == (7,)
    si.expert = scripted
    si.left = si.right = False
    si.action_indices = None
    si.deadman_gates_policy = False
    env = si
    env = PS4RewardWrapper(env, require_expert=True)
    env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env, proprio_keys=cfg.proprio_keys)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    return env, scripted, inner


def run_record_loop(env, scripted, inner):
    """Exactly the stock record_demos.py inner loop, one successful trajectory."""
    obs, info = env.reset()
    trajectory, shapes = [], []
    for _ in range(50):
        actions = np.zeros(env.action_space.sample().shape)
        next_obs, rew, done, truncated, info = env.step(actions)
        if "intervene_action" in info:
            actions = info["intervene_action"]
        trajectory.append(copy.deepcopy(dict(
            observations=obs, actions=actions, next_observations=next_obs,
            rewards=rew, masks=1.0 - done, dones=done, infos=info)))
        shapes.append({k: np.asarray(v).shape for k, v in next_obs.items()})
        obs = next_obs
        scripted.t += 1
        if done:
            return trajectory, shapes, info
    raise AssertionError("episode never terminated (X not honored?)")


def main():
    env, scripted, inner = build_env()
    traj, shapes, last_info = run_record_loop(env, scripted, inner)

    # 1. obs shape consistency across ALL transitions incl. the final X-terminated one
    ref = shapes[0]
    for i, s in enumerate(shapes):
        assert s == ref, f"obs shape drift at transition {i}: {s} != {ref}"
    print(f"  OK  obs shape consistent across {len(shapes)} transitions incl. final")

    # 2. X ended it with reward 1 + succeed
    assert traj[-1]["dones"] is True or traj[-1]["dones"] == True
    assert traj[-1]["rewards"] == 1, f"final reward {traj[-1]['rewards']} != 1"
    assert last_info.get("succeed") is True
    print("  OK  X press -> done + reward 1 + succeed (recorder would save this demo)")

    # 3. intervene_action captured on the correction steps
    intervened = [t for t in traj if "intervene_action" in t["infos"]]
    assert len(intervened) >= 1, "no intervene_action captured from R1+stick correction"
    print(f"  OK  intervene_action captured on {len(intervened)} step(s)")

    # 4. RECORD vs TRAIN space match — the recorded transition obs/action must fit the
    #    learner's demo_buffer (built from the SAME config with fake_env=True). If these ever
    #    diverge (e.g. someone changes obs_horizon/proprio in one path), demos won't load.
    from experiments.mappings import CONFIG_MAPPING
    learner_env = CONFIG_MAPPING["franka_plumbers_insert0"]().get_environment(fake_env=True)
    learner_obs_shapes = {k: tuple(v.shape) for k, v in
                          learner_env.observation_space.spaces.items()}
    rec_obs_shapes = {k: np.asarray(v).shape for k, v in traj[-1]["next_observations"].items()}
    assert rec_obs_shapes == learner_obs_shapes, (
        f"RECORD obs {rec_obs_shapes} != LEARNER demo_buffer space {learner_obs_shapes} "
        "— demos would not fit the buffer")
    assert traj[-1]["actions"].shape == tuple(learner_env.action_space.shape), (
        f"action shape {traj[-1]['actions'].shape} != learner {learner_env.action_space.shape}")
    print(f"  OK  record obs/action shapes == learner demo_buffer space {learner_obs_shapes}")

    print("DEMO-RECORDING INTEGRATION TEST PASS")


if __name__ == "__main__":
    main()
