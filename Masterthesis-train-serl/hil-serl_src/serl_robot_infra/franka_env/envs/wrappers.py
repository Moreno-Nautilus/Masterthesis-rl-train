import time
from gymnasium import Env, spaces
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
import copy
import os
import requests
from scipy.spatial.transform import Rotation as R
from franka_env.envs.franka_env import FrankaEnv
from typing import List

sigmoid = lambda x: 1 / (1 + np.exp(-x))


def make_teleop_expert():
    """Return the teleop expert selected by ``TELEOP_DEVICE`` env var.

    "spacemouse" (default) -> stock SpaceMouseExpert (behavior unchanged).
    "ps4"                  -> PS4Expert drop-in (same get_action() contract).
    """
    device = os.environ.get("TELEOP_DEVICE", "spacemouse").lower()
    if device == "ps4":
        from franka_env.spacemouse.ps4_expert import PS4Expert
        return PS4Expert()
    # Deferred import: SpaceMouseExpert pulls in easyhid (SpaceMouse HID lib), absent on
    # a PS4-only / offline box. Only import it when actually selected.
    from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert
    return SpaceMouseExpert()

class HumanClassifierWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        if done:
            while True:
                try:
                    rew = int(input("Success? (1/0)"))
                    assert rew == 0 or rew == 1
                    break
                except:
                    continue
        info['succeed'] = rew
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info


class PS4RewardWrapper(gym.Wrapper):
    """Human success/abort via the PS4 pad — ends the episode immediately on a press.

    Franka pivot (2026-09-08): reward is human-marked with the controller (no classifier).
    Each step it polls the PS4Expert's latched events:
      - X (Cross)  -> reward 1, done=True  (seated: freeze here, don't run to timeout)
      - Triangle   -> reward 0, done=True  (abort)
      - neither, step-limit reached -> reward 0, done=True (failed attempt; stock timeout)

    It reuses the SAME PS4Expert instance the SpacemouseIntervention wrapper already
    created (found by walking the wrapper chain for a `.expert` exposing get_events()),
    so no second controller reader is spawned.

    REWARD IS HUMAN-ONLY: reward is FORCED to 0 unless X is pressed — the underlying
    pose-threshold reward is deliberately ignored so X is the sole success signal (an
    unfilled/placeholder TARGET_POSE therefore cannot auto-succeed). X detected on a step
    ends the episode on THAT step (reward 1); Triangle ends it with reward 0; otherwise the
    stock step-limit timeout ends it (reward 0). We always take a real env.step so the
    returned obs is the SAME fully-wrapped shape as every other transition — never a
    hand-built unwrapped obs (that would corrupt the demo/replay buffer).

    Requires an event-capable expert on a real env (constructed with require_expert=True by
    the config on the actor path) — it will NOT silently fall back to pose reward. On the
    learner (fake_env) it is an inert passthrough.
    """

    def __init__(self, env, require_expert=False):
        super().__init__(env)
        self._expert = self._find_expert(env)
        if require_expert and self._expert is None:
            raise RuntimeError(
                "PS4RewardWrapper: no event-capable expert found in the wrapper chain. "
                "Human reward would silently fall back to pose-threshold reward. "
                "Ensure TELEOP_DEVICE=ps4 and SpacemouseIntervention is inside this wrapper."
            )
        self._regrasp_pending = False  # Circle pressed mid-episode -> regrasp at next reset

    @staticmethod
    def _find_expert(env):
        node = env
        while node is not None:
            expert = getattr(node, "expert", None)
            if expert is not None and hasattr(expert, "get_events"):
                return expert
            node = getattr(node, "env", None)
        return None

    def _poll(self):
        """Consume the expert events. Returns 'success'|'abort'|None for episode-end, and
        remembers a regrasp request (Circle) into self._regrasp_pending so it survives to
        the next reset. Single get_events() call per step (atomic consume)."""
        if self._expert is None:
            return None
        ev = self._expert.get_events()
        if ev.get("regrasp"):
            self._regrasp_pending = True
        if ev["success"]:
            return "success"
        if ev["abort"]:
            return "abort"
        return None

    def step(self, action):
        # Always take a real env.step so the returned obs is the SAME fully-wrapped shape as
        # every other transition (critical for demo/replay buffer integrity — do NOT hand-build
        # an obs from unwrapped._get_obs(), which bypasses RelativeFrame/Quat2Euler/SERLObs).
        obs, rew, done, truncated, info = self.env.step(action)

        # Human reward ONLY: ignore the stock pose-threshold reward; X is the sole success.
        # X/Triangle detected on THIS step ends the episode now (reward 1 / 0). At ~10Hz the
        # end is effectively immediate; the arm is compliant so the one delta step it took is
        # harmless. This keeps obs consistent AND ends promptly.
        #
        # CRITICAL: the stock `done` also fires on the POSE THRESHOLD (FrankaEnv.step does
        # `done = timeout or reward or terminate`). Suppressing only the reward would still
        # let a placeholder/measured TARGET_POSE silently end the episode with reward 0.
        # So we REBUILD done from the only legitimate sources: the stock TIMEOUT, the ESC
        # terminate flag, and the human buttons.
        inner = self.env.unwrapped
        timeout = bool(getattr(inner, "curr_path_length", 0) >= getattr(
            inner, "max_episode_length", float("inf")))
        esc = bool(getattr(inner, "terminate", False))

        rew = 0
        done = timeout or esc
        event = self._poll()
        if event == "success":
            rew, done = 1, True
        elif event == "abort":
            rew, done = 0, True
        info["succeed"] = bool(rew)
        return obs, rew, done, truncated, info

    def _drain(self):
        """Consume pending events, RETRYING until the latches are actually cleared.

        A single get_events() can return "no events" WITHOUT clearing them if it fails to
        take the lock in time — which previously let a stale X award a false success on the
        next episode's first step (poisoning the demo/replay data). We therefore retry
        briefly and report whether the drain is trustworthy. Regrasp is preserved.
        """
        if self._expert is None:
            return True, False
        regrasp = False
        # Count only confirmed reads. Legacy/scripted experts without read_ok retain
        # their synchronous get_events contract; PS4Expert always supplies read_ok.
        clean = 0
        for _ in range(20):  # at most ~1.2s with the default 50ms lock timeout
            ev = self._expert.get_events()
            regrasp = regrasp or bool(ev.get("regrasp"))
            if ev.get("read_ok", True) and not ev.get("success") and not ev.get("abort"):
                clean += 1
                if clean >= 2:
                    return True, regrasp
            else:
                clean = 0
            time.sleep(0.01)
        return False, regrasp

    def reset(self, **kwargs):
        # Pick up a Circle pressed AFTER the final step's poll but BEFORE reset starts —
        # otherwise that regrasp request would wait an extra full episode.
        drained, regrasp = self._drain()
        self._regrasp_pending |= regrasp
        if not drained:
            raise RuntimeError("PS4 event drain failed before reset; refusing reset motion. "
                               "Check the controller reader and retry reset.")

        # If Circle was pressed during the last episode, request a regrasp on the inner env
        # so this reset does the release -> hand-part-in -> close sequence.
        if self._regrasp_pending:
            unwrapped = self.env.unwrapped
            if hasattr(unwrapped, "should_regrasp"):
                unwrapped.should_regrasp = True
            self._regrasp_pending = False

        obs, info = self.env.reset(**kwargs)

        # Consume events that arrived DURING reset, but do NOT blindly discard everything:
        #  - success/abort from the just-ended episode (or a stray press) are dropped, since
        #    they must not terminate the first step of the new episode;
        #  - a REGRASP press is KEPT (re-latched) — the natural "X to end, then Circle for the
        #    next reset" sequence, and Circle pressed during the settle window, must survive.
        drained, regrasp = self._drain()
        self._regrasp_pending = regrasp
        if not drained:
            raise RuntimeError("PS4 event drain failed after reset; refusing to start an "
                               "episode with unconfirmed button latches. Check the "
                               "controller reader and retry reset.")
        return obs, info


class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """
    This wrapper uses the camera images to compute the reward,
    which is not part of the observation space
    """

    def __init__(self, env: Env, reward_classifier_func, target_hz = None):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.target_hz = target_hz

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            return self.reward_classifier_func(obs)
        return 0

    def step(self, action):
        start_time = time.time()
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = done or rew
        info['succeed'] = bool(rew)
        if self.target_hz is not None:
            time.sleep(max(0, 1/self.target_hz - (time.time() - start_time)))
            
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['succeed'] = False
        return obs, info
    
    
class MultiStageBinaryRewardClassifierWrapper(gym.Wrapper):
    def __init__(self, env: Env, reward_classifier_func: List[callable]):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.received = [False] * len(reward_classifier_func)
    
    def compute_reward(self, obs):
        rewards = [0] * len(self.reward_classifier_func)
        for i, classifier_func in enumerate(self.reward_classifier_func):
            if self.received[i]:
                continue

            logit = classifier_func(obs).item()
            if sigmoid(logit) >= 0.75:
                self.received[i] = True
                rewards[i] = 1

        reward = sum(rewards)
        return reward

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = (done or all(self.received)) # either environment done or all rewards satisfied
        info['succeed'] = all(self.received)
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.received = [False] * len(self.reward_classifier_func)
        info['succeed'] = False
        return obs, info


class Quat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation


class Quat2R2Wrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to rotation matrix
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(9,)
        )

    def observation(self, observation):
        tcp_pose = observation["state"]["tcp_pose"]
        r = R.from_quat(tcp_pose[3:]).as_matrix()
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], r[..., :2].flatten())
        )
        return observation


class DualQuat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["left/tcp_pose"].shape == (7,)
        assert env.observation_space["state"]["right/tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["left/tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )
        self.observation_space["state"]["right/tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["left/tcp_pose"]
        observation["state"]["left/tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        tcp_pose = observation["state"]["right/tcp_pose"]
        observation["state"]["right/tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

class GripperCloseEnv(gym.ActionWrapper):
    """
    Use this wrapper to task that requires the gripper to be closed
    """

    def __init__(self, env):
        super().__init__(env)
        ub = self.env.action_space
        assert ub.shape == (7,)
        self.action_space = Box(ub.low[:6], ub.high[:6])

    def action(self, action: np.ndarray) -> np.ndarray:
        new_action = np.zeros((7,), dtype=np.float32)
        new_action[:6] = action.copy()
        return new_action

    def step(self, action):
        new_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"][:6]
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    
class SpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None):
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        self.expert = make_teleop_expert()
        self.left, self.right = False, False
        self.action_indices = action_indices
        # OPT-IN hold-to-move gate (bring-up only; default OFF = stock HIL-SERL). When ON and
        # R1 is NOT held, NO motion is commanded (policy included) — the arm holds. Turn OFF
        # for real HIL training: the policy MUST be able to act autonomously to learn.
        self.deadman_gates_policy = (
            os.environ.get("PS4_DEADMAN_GATES_POLICY", "0") not in ("0", "false", "False")
            and hasattr(self.expert, "deadman_held")
        )
        # See the lock in action(). Default ON for teleop/demo recording: it is what makes
        # single-axis inserts drivable by hand. Set PS4_SINGLE_AXIS_LOCK=0 to disable.
        self._single_axis_lock = (
            os.environ.get("PS4_SINGLE_AXIS_LOCK", "1") not in ("0", "false", "False")
        )
        self._axis_active = float(os.environ.get("PS4_AXIS_ACTIVE", 0.05))
        # Mirror the env's orientation lock so the operator's rotation input is dropped
        # here too (see action()). Read off the unwrapped env's config.
        try:
            self._lock_orientation = bool(
                getattr(self.env.unwrapped.config, "LOCK_ORIENTATION", False))
        except Exception:
            self._lock_orientation = False

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        expert_a, buttons = self.expert.get_action()
        self.left, self.right = tuple(buttons)
        intervened = False

        # SINGLE-AXIS LOCK (opt-in, PS4_SINGLE_AXIS_LOCK=1). Keep only the dominant axis
        # of the human's input and zero the rest, so "push -x" is pure -x with no tilt,
        # and a rotation correction is pure rotation. Without this the operator could not
        # drive a screw in cleanly: translating dragged ~6 deg of pitch with it (the
        # controller's rotational loop lags translation), and a mixed stick input made
        # every correction ambiguous. Applies ONLY to the human action, never to the
        # policy's, so training stays stock.
        # When the env hard-locks orientation (config.LOCK_ORIENTATION), drop the operator's
        # rotation input too: it would otherwise be recorded as an intervention the env then
        # ignores, i.e. demo actions that do not match what the arm did.
        if getattr(self, "_lock_orientation", False):
            expert_a = np.asarray(expert_a, dtype=float).copy()
            expert_a[3:6] = 0.0

        # getattr guards subclasses/tests that build this wrapper without running __init__.
        if getattr(self, "_single_axis_lock", False) and np.linalg.norm(expert_a) > 1e-6:
            axis_active = getattr(self, "_axis_active", 0.05)
            lin, rot = expert_a[:3].copy(), expert_a[3:6].copy()
            lin_max, rot_max = float(np.abs(lin).max()), float(np.abs(rot).max())
            locked = np.zeros_like(expert_a)
            if lin_max >= axis_active and lin_max >= rot_max:
                k = int(np.argmax(np.abs(lin)))
                locked[k] = lin[k]
            elif rot_max >= axis_active:
                k = int(np.argmax(np.abs(rot)))
                locked[3 + k] = rot[k]
            expert_a = locked

        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        if self.gripper_enabled:
            if self.left:  # close gripper
                gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.right:  # open gripper
                gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                gripper_action = np.zeros((1,))
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if intervened:
            return expert_a, True

        return action, False

    def step(self, action):

        new_action, replaced = self.action(action)

        # Bring-up hold-to-move gate: if enabled and R1 is not held, freeze (zero action),
        # overriding both policy and expert. Default OFF (stock). Not an intervention label.
        if self.deadman_gates_policy and not self.expert.deadman_held():
            new_action = np.zeros_like(np.asarray(new_action, dtype=np.float32))
            replaced = False

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info

    def close(self):
        # Forward closure to the teleop reader, else its child process + Manager outlive the
        # env (leaks a controller reader per env construction).
        expert = getattr(self, "expert", None)
        if expert is not None and hasattr(expert, "close"):
            try:
                expert.close()
            except Exception:
                pass
        return self.env.close()

class DualSpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None, gripper_enabled=True):
        super().__init__(env)

        self.gripper_enabled = gripper_enabled

        self.expert = make_teleop_expert()
        self.left1, self.left2, self.right1, self.right2 = False, False, False, False
        self.action_indices = action_indices

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        intervened = False
        expert_a, buttons = self.expert.get_action()
        self.left1, self.left2, self.right1, self.right2 = tuple(buttons)


        if self.gripper_enabled:
            if self.left1:  # close gripper
                left_gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.left2:  # open gripper
                left_gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                left_gripper_action = np.zeros((1,))

            if self.right1:  # close gripper
                right_gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.right2:  # open gripper
                right_gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                right_gripper_action = np.zeros((1,))
            expert_a = np.concatenate(
                (expert_a[:6], left_gripper_action, expert_a[6:], right_gripper_action),
                axis=0,
            )

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        if intervened:
            return expert_a, True
        return action, False

    def step(self, action):

        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left1"] = self.left1
        info["left2"] = self.left2
        info["right1"] = self.right1
        info["right2"] = self.right2
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class GripperPenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty=0.1):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    def reward(self, reward: float, action) -> float:
        if (action[6] < -0.5 and self.last_gripper_pos > 0.95) or (
            action[6] > 0.5 and self.last_gripper_pos < 0.95
        ):
            return reward - self.penalty
        else:
            return reward

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]
        reward = self.reward(reward, action)
        self.last_gripper_pos = observation["state"][0, 0]
        return observation, reward, terminated, truncated, info

class DualGripperPenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty=0.1):
        super().__init__(env)
        assert env.action_space.shape == (14,)
        self.penalty = penalty
        self.last_gripper_pos_left = 0 #TODO: this assume gripper starts opened
        self.last_gripper_pos_right = 0 #TODO: this assume gripper starts opened
    
    def reward(self, reward: float, action) -> float:
        if (action[6] < -0.5 and self.last_gripper_pos_left==0):
            reward -= self.penalty
            self.last_gripper_pos_left = 1
        elif (action[6] > 0.5 and self.last_gripper_pos_left==1):
            reward -= self.penalty
            self.last_gripper_pos_left = 0
        if (action[13] < -0.5 and self.last_gripper_pos_right==0):
            reward -= self.penalty
            self.last_gripper_pos_right = 1
        elif (action[13] > 0.5 and self.last_gripper_pos_right==1):
            reward -= self.penalty
            self.last_gripper_pos_right = 0
        return reward
    
    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]
        reward = self.reward(reward, action)
        return observation, reward, terminated, truncated, info
