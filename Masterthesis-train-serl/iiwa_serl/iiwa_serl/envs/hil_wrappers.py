"""HIL-SERL wrappers for the real-KUKA iiwa env.

These bridge our WORKING PS4 teleop + manual-success stack into the upstream
HIL-SERL training loop (`hil-serl_src/examples/train_rlpd.py`), which expects:

  * an env whose `step()` may emit ``info["intervene_action"]`` when a human takes
    over — the loop then overrides the policy action with it and routes that
    transition into BOTH the demo and RL replay buffers (policy-only transitions
    go to the RL buffer alone). See `train_rlpd.py` (`if "intervene_action" in info`).
  * a scalar reward per step (here: the operator's manual X=success / Square=abort,
    reusing `env.set_manual_success()` and `PS4TeleopProvider.is_success/is_failure`).

Design mirrors upstream `franka_env/envs/wrappers.py::SpacemouseIntervention` and
`HumanClassifierWrapper`, but every controller/robot detail is copied from our
proven code (`async_drq_iiwa_insert.py::poll_manual_buttons`, the PS4 provider,
and `base_real_env`) rather than reinvented.

A SINGLE `PS4TeleopProvider` is shared between the two wrappers: it pumps pygame
events and updates button edges exactly ONCE per step (inside `PS4Intervention`),
then stashes the consumed edge results in `info` so `HumanReward` (the outer
wrapper) reads them without a second, race-prone event pump — matching the
single-poll design of the actor's `poll_manual_buttons`.
"""

from __future__ import annotations

from typing import Optional

# #5: the vendored HIL serl_launcher stack is gymnasium-based; wrappers must be gymnasium.
import gymnasium as gym
import numpy as np

from iiwa_serl.teleop import PS4TeleopProvider


# ---------------------------------------------------------------------------
# Intervention (inner wrapper) — the core HIL-SERL upgrade
# ---------------------------------------------------------------------------

class PS4Intervention(gym.ActionWrapper):
    """Replace the policy action with the operator's PS4 delta while they drive.

    Copies `PS4TeleopProvider.get_action()` semantics exactly: the returned 6D
    delta in [-1, 1] is all-zeros unless the R1 deadman is held, so a non-zero
    norm is an unambiguous "human is intervening" signal (same threshold idea as
    upstream's SpacemouseIntervention: ``norm > 1e-3``).

    No gripper action: this env is 6-DoF (welded/custom gripper), so
    ``gripper_enabled`` is always False — unlike upstream we never append a
    gripper channel.

    On takeover, emits ``info["intervene_action"]`` (the executed delta) for the
    training loop's dual-buffer routing. Always emits the consumed manual-result
    button edges (``info["manual_success"]`` / ``info["manual_abort"]``) so the
    outer `HumanReward` wrapper can act on them from the SAME event pump.
    """

    #: below this L2-norm the teleop delta counts as "not intervening"
    INTERVENE_EPS = 1e-3

    def __init__(self, env: gym.Env, teleop: Optional[PS4TeleopProvider] = None,
                 joystick_index: int = 0):
        super().__init__(env)
        if self.action_space.shape != (6,):
            raise ValueError(
                f"PS4Intervention expects a 6-DoF action space, got {self.action_space.shape}. "
                "This env has no gripper channel by design."
            )
        # Reuse a shared provider if one was passed (so HumanReward reads the same
        # button edges); otherwise open our own. server_url=None → no gripper HTTP
        # (matches the actor's manual-reset provider construction).
        self.teleop = teleop if teleop is not None else PS4TeleopProvider(
            joystick_index=joystick_index, server_url=None
        )
        self._intervened = False
        self._last_success = False
        self._last_abort = False

    def action(self, action: np.ndarray) -> np.ndarray:
        """Return motion only while R1 is held.

        This single call is the ONLY place per step that pumps pygame events /
        updates button edges (via `get_action`), so the manual success/abort
        flags consumed here are frame-aligned with this action (same guarantee
        the actor's `poll_manual_buttons` relies on)."""
        expert_a = self.teleop.get_action()          # pumps events, updates edges
        snapshot = (
            self.teleop.debug_snapshot()
            if hasattr(self.teleop, "debug_snapshot")
            else {"r1": 1}  # compatibility for test/custom providers predating snapshots
        )
        self._deadman_held = bool(snapshot.get("r1", 0))
        if hasattr(self.env.unwrapped, "set_deadman"):
            self.env.unwrapped.set_deadman(self._deadman_held)
        # Consume the one-shot manual-result edges set during that same pump.
        self._last_success = self.teleop.is_success()
        self._last_abort = self.teleop.is_failure()
        # Terminal markers are recorded transitions, so accept either result button
        # only during an R1-enabled interval. This prevents an unrecorded terminal
        # cycle from leaving the preceding replay transition without an episode end.
        if not self._deadman_held:
            self._last_success = False
            self._last_abort = False
        # Level read of L1 stop-forward (residual mode): freeze the nominal clock while held.
        # Push it into the env so the nominal clock (which lives in env.step) can pause forward
        # regardless of whether the executed action is the policy's or an intervention.
        if hasattr(self.env.unwrapped, "set_stop_forward"):
            stop_fwd = getattr(self.teleop, "is_stop_forward", lambda: False)()
            self.env.unwrapped.set_stop_forward(bool(stop_fwd))

        # #11: on a TERMINAL button (success/abort) this step, do NOT issue another motion —
        # zero the action AND freeze forward so the env commands a hold on the terminal step
        # (no last-instant push/jog leaking into the robot or the recorded transition).
        if self._last_success or self._last_abort:
            if hasattr(self.env.unwrapped, "set_stop_forward"):
                self.env.unwrapped.set_stop_forward(True)
            self._intervened = False
            return np.zeros_like(action)

        # R1 gates the autonomous policy as well as the human residual. Without
        # this, get_action() returned zero but the wrapper passed policy motion on.
        if not self._deadman_held:
            self._intervened = False
            return np.zeros_like(action)

        self._intervened = bool(np.linalg.norm(expert_a) > self.INTERVENE_EPS)
        if self._intervened:
            return np.asarray(expert_a, dtype=action.dtype)
        return action

    def step(self, action):
        new_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if self._intervened:
            info["intervene_action"] = new_action
        # Always surface the button edges so the reward wrapper can end the episode.
        info["manual_success"] = self._last_success
        info["manual_abort"] = self._last_abort
        info["deadman_held"] = bool(self._deadman_held)
        return obs, rew, done, truncated, info

    def close(self):
        try:
            self.teleop.close()
        finally:
            return self.env.close()


# ---------------------------------------------------------------------------
# Manual reward (outer wrapper) — keep the working X/Square scheme
# ---------------------------------------------------------------------------

class HumanReward(gym.Wrapper):
    """Operator marks the episode outcome by button (no classifier).

    Mirrors upstream `HumanClassifierWrapper` but the signal is our proven manual
    scheme: X (Cross) = success → reward 1.0 and terminate; Square/Triangle
    (abort) = end with reward 0.0. Reads the button edges from
    ``info["manual_success"]`` / ``info["manual_abort"]`` that `PS4Intervention`
    already consumed this step (single event pump — no double poll).

    On success it also flips the env's one-shot ``set_manual_success`` so
    ``env._success()`` reports True for this step (keeps env-side success stats /
    info["succeed"] consistent with the reward we hand the buffer).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        manual_success = bool(info.get("manual_success", False))
        manual_abort = bool(info.get("manual_abort", False))
        # Abort wins if both fire together (same tie-break as the actor).
        if manual_abort:
            manual_success = False

        if manual_success or manual_abort:
            import os as _os
            if _os.environ.get("HIL_DEBUG_BTN", "0") in ("1", "true", "True"):
                print(f"[HumanReward] manual_success={manual_success} manual_abort={manual_abort} "
                      f"deadman={info.get('deadman_held')} -> rew set", flush=True)
        if manual_success:
            # Keep env-side success bookkeeping in sync (one-shot flag).
            if hasattr(self.env.unwrapped, "set_manual_success"):
                self.env.unwrapped.set_manual_success(True)
            rew = 1.0
            done = True
            info["succeed"] = True
        elif manual_abort:
            rew = 0.0
            done = True
            info["succeed"] = False

        return obs, rew, done, truncated, info
