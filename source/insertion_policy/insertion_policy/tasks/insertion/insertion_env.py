"""InsertionEnv: socket-aware residual insertion env, subclassing Forge.

Two jobs on top of Forge:

1. Off-center sockets. Factory/Forge assume the socket sits at the fixed-asset XY origin, but the
   cooling_base sockets are at x=+/-30mm. We redefine ``self.fixed_pos`` as the ACTIVE socket frame
   (base origin (+) per-env socket offset) inside ``_compute_intermediate_values``. Because every
   downstream computation (tip frame, hand placement, observations, success/reward target) is built
   from ``self.fixed_pos``, they all become socket-centric automatically. Multi-socket: each env
   samples one of ``cfg_task.socket_offsets_local`` at reset.

2. Pure-residual reward (per supervisor). Reward = negative L2 from the screw shaft tip to the
   socket target (Fabrica), replacing Factory's keypoint reward. The screw seats when its shaft tip
   reaches the socket bottom (equivalently, the head bottoms flush on the platform).

The fixed_pos socket frame is the socket BOTTOM; with CoolingBase.height = socket depth and
base_height = 0, Factory's "tip" lands at the socket opening (entry) and the target stays at the
bottom — matching peg-insert semantics with the screw's shaft tip as the held reference.
"""

import torch

import isaacsim.core.utils.torch as torch_utils

from isaaclab_tasks.direct.forge.forge_env import ForgeEnv

from .cooling_tasks_cfg import ForgeTaskCoolingInsertCfg


class InsertionEnv(ForgeEnv):
    cfg: ForgeTaskCoolingInsertCfg

    def __init__(self, cfg: ForgeTaskCoolingInsertCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Socket offsets (socket BOTTOM, base-local frame); each env holds the active one.
        self._socket_table = torch.tensor(
            self.cfg_task.socket_offsets_local, device=self.device, dtype=torch.float32
        )  # (n_sockets, 3)
        self.socket_offset_local = self._socket_table[0].repeat(self.num_envs, 1).clone()

        # Held reference = screw shaft tip (shaft_length below the part origin).
        self.held_base_offset = torch.tensor(
            [0.0, 0.0, -self.cfg_task.screw_shaft_length], device=self.device
        ).repeat(self.num_envs, 1)

        self._identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

    # --- socket selection -------------------------------------------------------------------
    def _sample_sockets(self, env_ids):
        n = self._socket_table.shape[0]
        idx = torch.randint(0, n, (len(env_ids),), device=self.device)
        self.socket_offset_local[env_ids] = self._socket_table[idx]

    def randomize_initial_state(self, env_ids):
        # Pick which socket each resetting env targets, then let Forge place everything relative to
        # the (now socket-centric) fixed frame. (Guard: socket tensors are allocated after the
        # parent __init__, which may itself trigger a reset.)
        if hasattr(self, "_socket_table"):
            self._sample_sockets(env_ids)
        super().randomize_initial_state(env_ids)

    # --- socket-centric frame ----------------------------------------------------------------
    def _compute_intermediate_values(self, dt):
        super()._compute_intermediate_values(dt)
        if not hasattr(self, "socket_offset_local"):
            return  # parent __init__ may call this before our tensors exist
        # Shift the fixed frame from base origin to the active socket bottom (same orientation).
        _, socket_pos = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self._identity_quat, self.socket_offset_local
        )
        self.fixed_pos = socket_pos

    def _held_base_pose(self):
        """World pose of the screw shaft tip (the point that seats at the socket bottom)."""
        held_base_quat, held_base_pos = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self._identity_quat, self.held_base_offset
        )
        return held_base_pos, held_base_quat

    # --- success + reward (pure residual, neg-L2) -------------------------------------------
    def _get_curr_successes(self, success_threshold, check_rot=False):
        held_base_pos, _ = self._held_base_pose()
        target = self.fixed_pos  # socket bottom
        xy_dist = torch.linalg.vector_norm(target[:, 0:2] - held_base_pos[:, 0:2], dim=1)
        z_disp = held_base_pos[:, 2] - target[:, 2]
        is_centered = xy_dist < 0.0025
        height_threshold = self.cfg_task.fixed_asset_cfg.height * success_threshold
        is_seated = z_disp < height_threshold
        return torch.logical_and(is_centered, is_seated)

    def _get_rewards(self):
        held_base_pos, _ = self._held_base_pose()
        dist = torch.linalg.vector_norm(held_base_pos - self.fixed_pos, dim=1)
        curr_successes = self._get_curr_successes(self.cfg_task.success_threshold)

        rew_buf = -dist + self.cfg_task.success_bonus * curr_successes.float()

        self.prev_actions = self.actions.clone()
        self._log_factory_metrics({"neg_l2_dist": -dist, "success": curr_successes.float()}, curr_successes)
        return rew_buf
