"""InsertionEnvIiwa: the iiwa7 + custom-Y-gripper variant of the cooling insertion env.

Subclasses ``InsertionEnv`` (all task logic — sockets, residual reward, pre-insert tilt, grasp
misalignment, wrist RGB-D + appearance DR — is inherited unchanged) and overrides ONLY the two places
the Forge/Factory base env hardcodes Franka-specific robot geometry:

  1. ``_init_tensors`` — the base looks up ``panda_leftfinger`` / ``panda_rightfinger`` /
     ``panda_fingertip_centered`` by name (they don't exist on this robot). We replicate the base setup
     with the iiwa gripper's body names (``left_finger_link`` / ``right_finger_link`` / ``gripper_tcp``).
     ``force_sensor`` is NOT remapped here — we injected a real ``force_sensor`` link into the robot USD
     (build_iiwa_gripper_usd.py), so ForgeEnv's lookup succeeds untouched.

  2. ``_compute_intermediate_values`` — the base builds the fingertip jacobian as the average of the two
     finger-body jacobians. On the Franka the finger bodies sit at the fingertip; on this gripper they
     sit at the gripper BASE (~0.145 m behind the ``gripper_tcp`` control frame), so that average carries
     a large lever-arm error into the OSC/IK. We overwrite it with the EXACT ``gripper_tcp`` body
     jacobian (gripper_tcp is rigidly fixed to link7 -> its arm jacobian IS the fingertip jacobian).

Everything else (pressing axis = fingertip Y, grasp offset via ``robot_cfg.franka_fingerpad_length``,
camera mount) is driven by cfg values, so it needs no code override here. The Franka ``InsertionEnv``
and its registered tasks are left completely untouched.
"""

import torch

from .cooling_iiwa_tasks_cfg import ForgeTaskCoolingInsertIiwaCfg
from .insertion_env import InsertionEnv


class InsertionEnvIiwa(InsertionEnv):
    cfg: ForgeTaskCoolingInsertIiwaCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Use the REAL D405 mount orientation (vendor USD) as the rigid wrist-cam offset quat, instead
        # of the base env's synthetic look-at, so the sim wrist view matches the hardware camera. The
        # eye position (wrist_cam_offset_pos) is already the real D405 point; this fixes the pointing.
        q = getattr(cfg, "wrist_cam_offset_quat", None)
        if getattr(self, "_has_camera", False) and q is not None:
            self._cam_offset_quat = torch.tensor(q, device=self.device, dtype=torch.float32).repeat(
                self.num_envs, 1
            )

    def _init_tensors(self):
        """Replicate FactoryEnv._init_tensors, but resolve the iiwa+gripper body names.

        (Provenance: isaaclab_tasks/direct/factory/factory_env.py::_init_tensors. Kept in sync with the
        pinned Isaac Lab version; only the three body-name lookups differ from upstream.)
        """
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.ema_factor = self.cfg.ctrl.ema_factor
        self.dead_zone_thresholds = None

        self.fixed_pos_obs_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.init_fixed_pos_obs_noise = torch.zeros((self.num_envs, 3), device=self.device)

        # iiwa+gripper body names (vs the Franka's panda_leftfinger/rightfinger/fingertip_centered).
        self.left_finger_body_idx = self._robot.body_names.index("left_finger_link")
        self.right_finger_body_idx = self._robot.body_names.index("right_finger_link")
        self.fingertip_body_idx = self._robot.body_names.index("gripper_tcp")

        self.last_update_timestamp = 0.0
        self.prev_fingertip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_fingertip_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        self.prev_joint_pos = torch.zeros((self.num_envs, 7), device=self.device)

        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

    def _reset_idx(self, env_ids):
        # Budget the reset hover-IK retries so a single unreachable random target can NEVER hang the
        # whole run (the base env's servo loop is `while True` with no cap). After the budget is spent
        # we report convergence (below) so the loop exits with the best-effort pose for any straggler.
        self._reset_ik_budget = 25
        super()._reset_idx(env_ids)

    def set_pos_inverse_kinematics(self, ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, env_ids):
        pos_error, axis_angle_error = super().set_pos_inverse_kinematics(
            ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, env_ids
        )
        budget = getattr(self, "_reset_ik_budget", None)
        if budget is not None:
            self._reset_ik_budget = budget - 1
            if self._reset_ik_budget <= 0:
                # Give up servoing stragglers -> report converged so the reset loop terminates.
                return torch.zeros_like(pos_error), torch.zeros_like(axis_angle_error)
        return pos_error, axis_angle_error

    def _set_franka_to_default_pose(self, joints, env_ids):
        """Always seed the arm from the iiwa reset posture, ignoring the passed joints.

        The base env's reset servo loop (FactoryEnv.randomize_initial_state) re-seeds any env whose
        hover IK fails to a HARDCODED FRANKA posture and retries. For the iiwa that Franka seed is a poor
        (sometimes non-converging) IK start, so failed envs never recover and the ``while True`` loop can
        spin forever (intermittent reset hang). Forcing every seed to the folded, tool-down iiwa reset
        posture (cfg.ctrl.reset_joints, which places the TCP right over the socket) makes the DLS IK
        converge reliably. _reset_idx already passes this value; here we also override the fallback path.
        """
        super()._set_franka_to_default_pose(joints=self.cfg.ctrl.reset_joints, env_ids=env_ids)

    def _compute_intermediate_values(self, dt):
        super()._compute_intermediate_values(dt)
        # Use the exact gripper_tcp jacobian at the fingertip control frame, not the finger-midpoint
        # average (which for this gripper is ~0.145 m behind the TCP -> lever-arm error in the OSC).
        if getattr(self, "fingertip_body_idx", None) is not None:
            jacobians = self._robot.root_physx_view.get_jacobians()
            self.fingertip_midpoint_jacobian = jacobians[:, self.fingertip_body_idx - 1, 0:6, 0:7]
