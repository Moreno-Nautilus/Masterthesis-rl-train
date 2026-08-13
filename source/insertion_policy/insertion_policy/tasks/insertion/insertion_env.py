"""InsertionEnv: socket-aware residual insertion env, subclassing Forge.

Two jobs on top of Forge:

1. Off-center sockets. Factory/Forge assume the socket sits at the fixed-asset XY origin, but the
   cooling_base sockets are at x=+/-30mm. We redefine ``self.fixed_pos`` as the ACTIVE socket frame
   (base origin (+) per-env socket offset) inside ``_compute_intermediate_values``. Because every
   downstream computation (tip frame, hand placement, observations, success/reward target) is built
   from ``self.fixed_pos``, they all become socket-centric automatically. Multi-socket: each env
   samples one of ``cfg_task.socket_offsets_local`` at reset.

2. Pure-residual reward (per supervisor). Reward = multi-keypoint squashing kernel (Factory/
   IndustReal: dense, bounded, smooth near the goal) on the distance from the screw to the target
   seated pose, replacing Factory's centered-socket keypoint geometry. The first keypoint is the
   shaft tip and the rest run up the screw axis, so tilt is penalized even when the tip is close to
   the socket bottom. A binary seat bonus is added on top.

The fixed_pos socket frame is the socket BOTTOM; with CoolingBase.height = socket depth and
base_height = 0, Factory's "tip" lands at the socket opening (entry) and the target stays at the
bottom — matching peg-insert semantics with the screw's shaft tip as the held reference.
"""

import math

import carb
import gymnasium as gym
import torch

import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sensors import TiledCamera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.direct.factory import factory_utils
from isaaclab_tasks.direct.forge.forge_env import ForgeEnv

from .cooling_tasks_cfg import ForgeTaskCoolingInsertCfg

# --- Fabric/usdrt workaround (keep Fabric ON for physics; route only the camera to USD poses) -----
# This Isaac Sim build's usdrt lacks the `hierarchy` submodule, so Isaac Lab's Fabric world-pose path
# (XformPrimView._get_world_poses_fabric -> _initialize_fabric -> usdrt.hierarchy) crashes whenever a
# camera reads/writes its pose. Disabling Fabric globally "fixes" that but makes GPU PhysX unstable
# (CUDA-700 crashes / hangs in vision runs). Instead we keep Fabric ENABLED (stable physics) and force
# ONLY XformPrimView (used by cameras / xform-sensors, NOT the physics articulation views) onto the
# USD XformCache pose path, which needs no usdrt.hierarchy. Physics is untouched.
from isaaclab.sim.views import XformPrimView as _XformPrimView  # noqa: E402

if not getattr(_XformPrimView, "_insertion_usd_pose_patch", False):
    _orig_xpv_init = _XformPrimView.__init__

    def _xpv_init_usd_poses(self, *args, **kwargs):
        _orig_xpv_init(self, *args, **kwargs)
        self._use_fabric = False  # USD pose path (no usdrt.hierarchy); physics keeps Fabric

    _XformPrimView.__init__ = _xpv_init_usd_poses
    _XformPrimView._insertion_usd_pose_patch = True

# --- rl_games dict-obs input-normalization fix --------------------------------------------------
# rl_games normalizes dict observations with torch.jit.script(RunningMeanStdObs(obs_shape)), but this
# rl_games/torch version can't infer the Dict[str,Tensor] input type when scripting -> compile error
# ("'Tensor' object has no attribute 'items'"), so normalize_input=True crashes our hybrid
# (proprio+image) dict obs at model build. The EAGER module works fine; only the JIT compile fails.
# We intercept torch.jit.script for that one class and return it un-scripted, so we can normalize the
# proprio + privileged critic state (which Forge needs; force obs are large) while the image stays
# pre-normalized in-env. Everything else still scripts normally.
import torch as _torch  # noqa: E402
from rl_games.algos_torch.running_mean_std import RunningMeanStdObs as _RMSObs  # noqa: E402

if not getattr(_torch.jit, "_insertion_rmsobs_patch", False):
    _orig_jit_script = _torch.jit.script

    def _jit_script_skip_rmsobs(obj, *args, **kwargs):
        if isinstance(obj, _RMSObs):
            return obj  # eager; TorchScript can't infer the Dict input type in this rl_games version
        return _orig_jit_script(obj, *args, **kwargs)

    _torch.jit.script = _jit_script_skip_rmsobs
    _torch.jit._insertion_rmsobs_patch = True


class InsertionEnv(ForgeEnv):
    cfg: ForgeTaskCoolingInsertCfg

    def __init__(self, cfg: ForgeTaskCoolingInsertCfg, render_mode: str | None = None, **kwargs):
        # Appearance-DR bookkeeping must exist BEFORE super().__init__(): the base ctor builds the
        # scene (-> _setup_part_materials, which records shader paths here) and may trigger an early
        # reset, both before our camera branch below runs.
        self._appearance_shader_paths = {}  # {"screw": shader_path, "base": shader_path}
        self._material_dr_failed = False
        self._light_dr_failed = False
        super().__init__(cfg, render_mode, **kwargs)

        # 6-axis F/T: widen the policy proprio by the 3 torque channels (see _get_observations). Done
        # here (post-super, so the base 24-d policy space already exists) by rebuilding the "policy"
        # Box; the batched observation_space is regenerated to match. The hybrid/state nets infer the
        # new proprio width from this space, so no net change is needed. Applies to both state + vision.
        if getattr(self.cfg, "use_torque_obs", False):
            base_dim = int(self.single_observation_space["policy"].shape[0])
            self.single_observation_space["policy"] = gym.spaces.Box(
                low=-float("inf"), high=float("inf"), shape=(base_dim + 3,)
            )
            self.observation_space = gym.vector.utils.batch_space(
                self.single_observation_space["policy"], self.num_envs
            )

        # Vision variant: a wrist RGB-D camera adds an "image" observation group alongside Forge's
        # "policy" (proprio) and "critic" (state) groups. The rl_games wrapper consumes it via
        # obs_groups={"obs": ["policy", "image"], "states": ["critic"]} + concate_obs_groups=False.
        self._has_camera = getattr(self.cfg, "tiled_camera", None) is not None
        if self._has_camera:
            h, w, c = self.cfg.image_height, self.cfg.image_width, self.cfg.image_channels
            # Temporal frame-stack: the OBSERVED image has c*frame_stack channels (last N frames stacked
            # on the channel axis; see _get_camera_image / _stack_frames). N=1 => single frame (default).
            self._frame_stack = max(1, int(getattr(self.cfg, "frame_stack", 1) or 1))
            self._frame_hist = None  # lazy (num_envs, N, H, W, c) ring buffer, filled on first render
            self._frame_reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            obs_c = c * self._frame_stack
            self.single_observation_space["image"] = gym.spaces.Box(low=0.0, high=1.0, shape=(h, w, obs_c))
            # Privileged labels for the auxiliary vision heads (actor-UNOBSERVABLE; never fed to the
            # policy): signed grasp tilt (1) + true shaft-tip->socket-opening gap in the fingertip frame
            # (3). Routed to the policy net as its own obs group and sliced out there for the aux loss.
            self._aux_label_dim = 4
            self.single_observation_space["aux_label"] = gym.spaces.Box(
                low=-float("inf"), high=float("inf"), shape=(self._aux_label_dim,)
            )
            # Rigid wrist mount: a FIXED eye offset AND a FIXED aim point, both in the fingertip frame.
            self._cam_offset_pos = torch.tensor(
                self.cfg.wrist_cam_offset_pos, device=self.device, dtype=torch.float32
            ).repeat(self.num_envs, 1)
            self._cam_look_local = torch.tensor(
                self.cfg.wrist_cam_look_target_pos, device=self.device, dtype=torch.float32
            ).repeat(self.num_envs, 1)
            # FIXED camera orientation in the fingertip frame -> a truly rigid mount (roll locked to the
            # gripper, NOT world-up). Replicate Isaac's look-at basis (OpenGL: -z forward, +y up; R
            # columns = camera x,y,z) from the fixed eye->aim direction, with a gripper-fixed "up" =
            # fingertip -z (which maps to world-up at the nominal downward grasp, so renders stay upright
            # while the roll now rides the gripper). The world pose each step = quat_mul(ft_quat, this).
            from isaaclab.utils.math import quat_from_matrix
            eye0, aim0 = self._cam_offset_pos[0], self._cam_look_local[0]
            up0 = torch.tensor([0.0, 0.0, -1.0], device=self.device)
            zc = -torch.nn.functional.normalize(aim0 - eye0, dim=0, eps=1e-5)
            xc = torch.nn.functional.normalize(torch.cross(up0, zc, dim=0), dim=0, eps=1e-5)
            yc = torch.nn.functional.normalize(torch.cross(zc, xc, dim=0), dim=0, eps=1e-5)
            cam_R = torch.stack([xc, yc, zc], dim=1)  # columns = camera x,y,z axes (fingertip frame)
            self._cam_offset_quat = quat_from_matrix(cam_R.unsqueeze(0)).repeat(self.num_envs, 1)
            self._cam_roll_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
            # Per-episode camera extrinsics jitter (DR; resampled at reset; zero when the cfg knob is 0).
            self._cam_jitter = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
            self._cam_roll_jitter = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
            # Per-env photometric image-augmentation params (appearance DR; resampled per reset). Init
            # to the IDENTITY so each term is a no-op until its cfg knob enables it. Broadcast over H,W:
            # gain/brightness/contrast are per-channel (...,1,1,3); gamma is per-image (...,1,1,1).
            self._photo_gain = torch.ones((self.num_envs, 1, 1, 3), device=self.device)
            self._photo_brightness = torch.zeros((self.num_envs, 1, 1, 3), device=self.device)
            self._photo_contrast = torch.ones((self.num_envs, 1, 1, 3), device=self.device)
            self._photo_gamma = torch.ones((self.num_envs, 1, 1, 1), device=self.device)
            # Realistic-depth DR (sim2real): per-EPISODE modality + fill state, resampled at reset.
            #  _depth_modality in {0=RGB-only (depth zeroed), 1=heavily-corrupted depth, 2=normal-corrupted
            #  depth}; default 2 (all "normal") so with depth_modality_dropout OFF nothing changes.
            #  _depth_fill flags episodes whose synthetic holes get interpolation-filled (mimics the D405's
            #  hole-filling post-proc); default False. Both are no-ops unless the cfg knobs enable them.
            self._depth_modality = torch.full((self.num_envs,), 2, dtype=torch.long, device=self.device)
            self._depth_fill = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._cam_log_counter = 0  # drives the periodic training image gallery (cam_log_interval)
            # NOTE: _appearance_shader_paths / _material_dr_failed / _light_dr_failed are initialized at
            # the TOP of __init__ (before super), and _setup_part_materials populates the shader paths
            # during super().__init__() -- do NOT re-init them here or that binding would be wiped.

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
        self._local_z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)

        self._reward_keypoint_offsets = torch.zeros(
            (self.cfg_task.num_keypoints, 3), device=self.device, dtype=torch.float32
        )
        self._reward_keypoint_offsets[:, 2] = torch.linspace(
            0.0, self.cfg_task.held_asset_cfg.height, self.cfg_task.num_keypoints, device=self.device
        )

    def _weld_gripper_dof_pos(self) -> float:
        """Finger joint target for welded grasps.

        Both fingers stop at the calibrated screw-head contact aperture. The custom Y-gripper collision
        pads still have about a 4 mm face-to-face gap at q=0, so the contact joint value is not simply
        the screw-head radius.
        """
        closed_half_gap = float(getattr(self.cfg, "weld_gripper_closed_half_gap", 0.0))
        return max(0.0, 0.5 * float(self.cfg_task.held_asset_cfg.diameter) - closed_half_gap)

    def _weld_gripper_dof_pos_for_envs(self, env_ids) -> torch.Tensor:
        target = float(self._weld_gripper_dof_pos())
        return torch.full((len(env_ids), 2), target, device=self.device)

    def _set_welded_gripper_state(self, env_ids=None):
        """Force the reset-time finger state to the welded contact aperture.

        Factory's reset close-loop and IK fallback write raw joint states. For welded grasps, a stale
        q=0 finger state moves the one-pad weld parent inward and drags the screw into the opposite pad.
        """
        if not getattr(self, "_weld_held", False):
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        target = self._weld_gripper_dof_pos_for_envs(env_ids)
        joint_ids = [7, 8]
        zero_vel = torch.zeros_like(target)
        self.joint_pos[env_ids[:, None], joint_ids] = target
        self.joint_vel[env_ids[:, None], joint_ids] = zero_vel
        self.ctrl_target_joint_pos[env_ids[:, None], joint_ids] = target
        self._robot.write_joint_state_to_sim(target, zero_vel, joint_ids=joint_ids, env_ids=env_ids)
        self._robot.set_joint_position_target(target, joint_ids=joint_ids, env_ids=env_ids)

    def _weld_parent_to_tcp_pose(self, parent_body: str):
        """Gripper kinematics for a one-pad weld frame.

        The exported gripper USD defines q=0 as closed and q=0.04 as open. At the welded-pad target,
        the chosen finger link sits laterally at +/- (0.04 - q), while gripper_tcp is 145.5 mm past the
        finger-link origins along the local z axis.
        """
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        pos = torch.zeros((self.num_envs, 3), device=self.device)
        if parent_body == "gripper_tcp":
            return quat, pos

        pad_target = self._weld_gripper_dof_pos()
        closed_y = float(getattr(self.cfg, "weld_finger_closed_origin_y", 0.04))
        tcp_z = float(getattr(self.cfg, "weld_tcp_from_finger_z", 0.1455))
        lateral = closed_y - pad_target
        if parent_body == "left_finger_link":
            pos[:, 1] = -lateral
        elif parent_body == "right_finger_link":
            pos[:, 1] = lateral
        else:
            raise ValueError(f"Unsupported weld_held_parent_body={parent_body!r}")
        pos[:, 2] = tcp_z
        return quat, pos

    def generate_ctrl_signals(
        self, ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, ctrl_target_gripper_dof_pos
    ):
        if getattr(self, "_weld_held", False):
            ctrl_target_gripper_dof_pos = self._weld_gripper_dof_pos()
        super().generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=ctrl_target_gripper_dof_pos,
        )

    # --- scene (adds the wrist camera for the vision variant) -------------------------------
    def _weld_held_asset(self):
        """Rigidly WELD the held screw to a gripper body with a PER-ENV FixedJoint (created AFTER clone, so each
        env bakes its OWN grasp-misalign tilt). The screw is a RigidObject (no articulation root), so the
        joint to the robot articulation is legal -- welding two ARTICULATIONS crashes PhysX. It stays a
        DYNAMIC body: still physically seats in the socket + transmits contact to the wrist F/T, it just
        can't slip, fall out, or be pinched through by the welded pad.

        The joint frame is derived to EXACTLY match Factory's validated clamp grasp transform. The clamp
        (randomize_initial_state) places the screw at ``fingertip * flip_z * inverse(rel)`` -- note the
        flip_z (180deg about Y) AND the inverse of the fingertip->asset relative pose. For a TCP weld,
        body0 IS the gripper_tcp (= fingertip) body. For a one-pad weld, body0 is a finger link and frame0
        first maps that link to gripper_tcp, then applies the same TCP->screw transform. Hand-building it as
        a plain +z offset put the screw in UPSIDE DOWN (shaft in the jaws), so we compute it from the SAME
        tf ops. rel already includes the per-env grasp-misalign, so the DR is preserved (fixed per-env tilt,
        not per-reset -- a static joint frame can't be re-randomized cheaply). NOTE: Factory's per-reset
        +-3mm held_asset_pos_noise is dropped (minor)."""
        import re
        import omni.usd
        from pxr import Gf, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        # Sample the per-env grasp-misalign ONCE; stash it so the reset grasp placement + aux label reuse the
        # SAME angle (so the written reset pose matches the weld and there is no first-step snap/fight).
        max_deg = float(getattr(self.cfg_task, "grasp_misalign_max_deg", 0.0))
        if max_deg > 0.0:
            angles = math.radians(max_deg) * (2.0 * torch.rand(self.num_envs, device=self.device) - 1.0)
        else:
            angles = torch.zeros(self.num_envs, device=self.device)
        self._weld_misalign_signed = angles
        # Optional SECONDARY out-of-plane grasp tilt (about fingertip X) -- flat pads mostly block this, but
        # pad compliance / an off-centre head allow a few deg. Baked per-env like the primary Y tilt so the
        # weld frame and reset pose agree. 0 => off (the pure single-axis pressing-axis tilt).
        sec_deg = float(getattr(self.cfg_task, "grasp_misalign_secondary_deg", 0.0))
        if sec_deg > 0.0:
            self._weld_misalign_secondary = math.radians(sec_deg) * (
                2.0 * torch.rand(self.num_envs, device=self.device) - 1.0
            )
        else:
            self._weld_misalign_secondary = torch.zeros(self.num_envs, device=self.device)
        # Per-env grasp POSITION DR (2026-08-05): axial (z, shaft depth) + lateral (x, in-pad plane) jitter
        # of where the head sits between the pads. Sampled ONCE per env (like the misalign above) so the
        # weld frame (built from get_handheld_asset_relative_pose just below) and any later reset-written
        # pose agree -- no first-step snap. y (finger pressing axis) is clamp-fixed -> not jittered.
        _axial_m = float(getattr(self.cfg_task, "grasp_pos_jitter_axial_mm", 0.0)) * 1e-3
        _lat_m = float(getattr(self.cfg_task, "grasp_pos_jitter_lateral_mm", 0.0)) * 1e-3
        _jit = torch.zeros((self.num_envs, 3), device=self.device)
        _jit[:, 2] = _axial_m * (2.0 * torch.rand(self.num_envs, device=self.device) - 1.0)
        _jit[:, 0] = _lat_m * (2.0 * torch.rand(self.num_envs, device=self.device) - 1.0)
        self._weld_pos_jitter = _jit

        # TCP->screw = flip_z * inverse(held_asset_relative_pose), per env -- the clamp's exact hand->screw tf.
        rel_pos, rel_quat = self.get_handheld_asset_relative_pose()  # per-env, includes the baked misalign
        inv_quat, inv_pos = torch_utils.tf_inverse(rel_quat, rel_pos)
        flip_z = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device).repeat(self.num_envs, 1)  # 180deg about Y
        tcp_screw_quat, tcp_screw_pos = torch_utils.tf_combine(
            flip_z, torch.zeros((self.num_envs, 3), device=self.device), inv_quat, inv_pos
        )

        parent_body = str(getattr(self.cfg, "weld_held_parent_body", "gripper_tcp"))
        parent_tcp_quat, parent_tcp_pos = self._weld_parent_to_tcp_pose(parent_body)
        f0_quat, f0_pos = torch_utils.tf_combine(
            parent_tcp_quat, parent_tcp_pos, tcp_screw_quat, tcp_screw_pos
        )
        self._weld_parent_body = parent_body
        self._weld_frame0_quat = f0_quat.clone()
        self._weld_frame0_pos = f0_pos.clone()
        f0_quat, f0_pos = f0_quat.cpu(), f0_pos.cpu()

        parent_by_env, screw_by_env = {}, {}
        for prim in stage.Traverse():
            p = prim.GetPath().pathString
            m = re.search(r"/env_(\d+)/", p)
            if m is None:
                continue
            e = int(m.group(1))
            if p.endswith(f"/{parent_body}"):
                parent_by_env[e] = p
            elif "/HeldAsset" in p and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                screw_by_env[e] = p

        n_ok = 0
        for e in range(self.num_envs):
            parent, screw = parent_by_env.get(e), screw_by_env.get(e)
            if parent is None or screw is None:
                continue
            qw, qx, qy, qz = (float(x) for x in f0_quat[e])
            px, py, pz = (float(x) for x in f0_pos[e])
            j = UsdPhysics.FixedJoint.Define(stage, screw + f"/weld_to_{parent_body}")
            j.CreateBody0Rel().SetTargets([parent])
            j.CreateBody1Rel().SetTargets([screw])
            j.CreateLocalPos0Attr().Set(Gf.Vec3f(px, py, pz))
            j.CreateLocalRot0Attr().Set(Gf.Quatf(qw, qx, qy, qz))
            j.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            # CRITICAL: a joint from an articulation link to an external body pulls that body
            # INTO the robot articulation and breaks its creation ("Failed to create articulation at Robot").
            # Mark it excludeFromArticulation so PhysX treats it as a MAXIMAL (out-of-articulation) fixed
            # constraint -- a rigid weld that leaves the Robot articulation topology untouched.
            j.CreateExcludeFromArticulationAttr().Set(True)
            n_ok += 1
        print(f"[weld] created {n_ok}/{self.num_envs} per-env {parent_body}<-screw welds (clamp-matched frame)  "
              f"misalign<=+/-{max_deg:.1f}deg")

    def _sync_welded_held_pose_from_parent(self, env_ids=None):
        """Write the held screw pose implied by the fixed joint and its current parent body.

        Factory resets freely teleport the held asset into the gripper. Once the screw is welded, direct
        held-body teleports can fight the fixed joint and create visible snap/penetration artifacts. This
        helper keeps the RigidObject state exactly consistent with the authored parent->screw weld frame.
        """
        if not getattr(self, "_weld_held", False) or not hasattr(self, "_weld_frame0_pos"):
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        parent_body = str(getattr(self, "_weld_parent_body", getattr(self.cfg, "weld_held_parent_body", "gripper_tcp")))
        parent_idx = self._robot.body_names.index(parent_body)
        parent_pos = self._robot.data.body_pos_w[env_ids, parent_idx]
        parent_quat = self._robot.data.body_quat_w[env_ids, parent_idx]
        held_quat, held_pos = torch_utils.tf_combine(
            parent_quat, parent_pos, self._weld_frame0_quat[env_ids], self._weld_frame0_pos[env_ids]
        )
        held_pose = torch.cat((held_pos, held_quat), dim=1)
        self._held_asset.write_root_pose_to_sim(held_pose, env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids)
        self._held_asset.reset(env_ids)

    def _setup_scene(self):
        """Factory's scene plus a wrist-mounted TiledCamera (vision variant only).

        Mirrors ``FactoryEnv._setup_scene`` but instantiates the camera BEFORE
        ``clone_environments`` so the per-env camera prims are created under each cloned robot.
        When the cfg has no ``tiled_camera`` (the state-obs task), this is just Factory's setup.
        (We override rather than call super because the camera must exist before the clone, which
        the parent performs internally; the cooling task never uses the gear-mesh assets.)
        """
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))
        table_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        )
        table_cfg.func(
            "/World/envs/env_.*/Table", table_cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        self._robot = Articulation(self.cfg.robot)
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
        # Held screw: a RigidObject when we weld it to the gripper (no articulation root -> a FixedJoint to
        # the robot articulation is legal; welding two ARTICULATIONS crashes PhysX). Else Factory's 1-body
        # Articulation (physical clamp grasp). The weld itself is created AFTER clone (per-env, below).
        self._held_is_rigid = isinstance(self.cfg_task.held_asset, RigidObjectCfg)
        self._weld_held = self._held_is_rigid and bool(getattr(self.cfg, "weld_held_to_gripper", False))
        if self._held_is_rigid:
            self._held_asset = RigidObject(self.cfg_task.held_asset)
        else:
            self._held_asset = Articulation(self.cfg_task.held_asset)

        self._tiled_camera = None
        if getattr(self.cfg, "tiled_camera", None) is not None:
            # NOTE: this Isaac Sim build's usdrt lacks the ``hierarchy`` submodule Isaac Lab's Fabric
            # world-pose path needs. We keep Fabric ON (GPU-stable physics) and monkeypatch only the
            # camera's XformPrimView to the USD pose path (see top of file), so the camera works.
            self._tiled_camera = TiledCamera(self.cfg.tiled_camera)

        # Optional third-person debug camera (only set by scripts/viz_camera.py; None in training).
        self._scene_camera = None
        if getattr(self.cfg, "scene_camera", None) is not None:
            self._scene_camera = TiledCamera(self.cfg.scene_camera)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        if self._held_is_rigid:
            self.scene.rigid_objects["held_asset"] = self._held_asset
        else:
            self.scene.articulations["held_asset"] = self._held_asset
        # Per-env FixedJoint welds, created AFTER clone so each env can bake its OWN grasp-misalign tilt.
        if self._weld_held:
            self._weld_held_asset()
        if self._tiled_camera is not None:
            self.scene.sensors["tiled_camera"] = self._tiled_camera
        if self._scene_camera is not None:
            self.scene.sensors["scene_camera"] = self._scene_camera

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Appearance DR (vision variant): PER-ENV matte materials on the parts + a directional key
        # light, both randomized per reset (_randomize_part_materials / _randomize_scene_light). The
        # dome above is the ambient fill; the key light adds the directional shadows/specular a dome
        # can't. Fail-safe: any error in setup disables that layer; image-space DR still runs.
        if getattr(self.cfg, "tiled_camera", None) is not None:
            if getattr(self.cfg, "randomize_part_materials", False):
                self._setup_part_materials()
            self._setup_key_light()

    def _setup_part_materials(self):
        """Bind a PER-ENV matte PreviewSurface to each env's screw and base (best-effort).

        ``bind_visual_material`` takes a CONCRETE prim path (no regex), and cloned env prims are
        instanceable proxies that can't carry a per-prim binding, so we mirror Isaac Lab's
        ``randomize_visual_color`` recipe: expand the env-regex with ``find_matching_prim_paths``, flip
        each matched prim un-instanceable, then bind its OWN material (``@apply_nested`` -> reaches the
        visual mesh). Per-env (not one shared material) so colours vary WITHIN a batch -> the policy
        sees many filament colours at once, not just one per episode. Shader paths are stored per env
        (sorted by env index) so _randomize_part_materials can give screw_i and base_i a correlated
        colour. Wrapped so any failure cleanly disables material DR.
        """
        try:
            import re

            import isaacsim.core.utils.prims as prim_utils

            base_color = tuple(getattr(self.cfg, "material_base_color", (0.45, 0.45, 0.5)))
            rough_lo = float(getattr(self.cfg, "material_roughness_range", (0.4, 0.9))[0])
            metallic = float(getattr(self.cfg, "material_metallic", 0.0))

            def _env_idx(path):
                m = re.search(r"/env_(\d+)/", path)
                return int(m.group(1)) if m else 0

            shaders = {"screw": [], "base": []}
            for key, env_prim in (("screw", "HeldAsset"), ("base", "FixedAsset")):
                paths = sorted(
                    sim_utils.find_matching_prim_paths(f"/World/envs/env_.*/{env_prim}"), key=_env_idx
                )
                if not paths:
                    raise RuntimeError(f"no prims matched /World/envs/env_.*/{env_prim}")
                for p in paths:
                    mat_path = f"/World/Looks/{key}_mat_env{_env_idx(p)}"
                    mat_cfg = sim_utils.PreviewSurfaceCfg(
                        diffuse_color=base_color, roughness=rough_lo, metallic=metallic
                    )
                    mat_cfg.func(mat_path, mat_cfg)
                    prim = prim_utils.get_prim_at_path(p)
                    if prim and prim.IsValid() and prim.IsInstanceable():
                        prim.SetInstanceable(False)  # per-prim material binding needs a non-instanced prim
                    sim_utils.bind_visual_material(p, mat_path)
                    shaders[key].append(f"{mat_path}/Shader")
            self._appearance_shader_paths = shaders  # {"screw": [per-env...], "base": [per-env...]}
        except Exception as exc:  # noqa: BLE001
            carb.log_warn(f"[InsertionEnv] part-material DR setup failed, disabling it: {exc}")
            self._material_dr_failed = True
            self._appearance_shader_paths = {}

    def _setup_key_light(self):
        """Spawn a directional key light (DistantLight) whose direction is randomized per reset.

        A DistantLight is infinite/parallel (like the sun): no position -> it lights every env
        identically (no per-env brightness confound a positioned point light would add), but its
        DIRECTION casts the moving shadows/specular highlights a dome cannot. Spawned with an initial
        orientation so the xform op exists; _randomize_scene_light re-points it each reset. Best-effort.
        """
        try:
            angle = float(getattr(self.cfg, "key_light_angle", 1.0))
            inten = float(getattr(self.cfg, "key_light_intensity_range", (1000.0, 3000.0))[0])
            key_cfg = sim_utils.DistantLightCfg(intensity=inten, color=(1.0, 1.0, 1.0), angle=angle)
            key_cfg.func("/World/KeyLight", key_cfg, orientation=(0.966, 0.259, 0.0, 0.0))  # ~30deg tilt
        except Exception as exc:  # noqa: BLE001
            carb.log_warn(f"[InsertionEnv] key-light setup failed, disabling it: {exc}")
            self._light_dr_failed = True

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
        old_held_noise = None
        if getattr(self, "_weld_held", False):
            old_held_noise = self.cfg_task.held_asset_pos_noise
            self.cfg_task.held_asset_pos_noise = [0.0, 0.0, 0.0]
        # WRIST-YAW CURRICULUM: temporarily shrink the reset yaw noise toward hand_init_yaw_start_deg early
        # on, ramping to the full configured yaw over hand_init_yaw_curriculum_steps. Restored in finally so
        # cfg_task.hand_init_orn_noise[2] always reads the true max.
        old_orn_noise = None
        _yaw_steps = int(getattr(self.cfg, "hand_init_yaw_curriculum_steps", 0) or 0)
        if _yaw_steps > 0:
            _frac = min(1.0, max(0.0, float(self.common_step_counter) / float(_yaw_steps)))
            _yaw_start = math.radians(float(getattr(self.cfg, "hand_init_yaw_start_deg", 45.0)))
            _yaw_max = float(self.cfg_task.hand_init_orn_noise[2])
            _cur = min(_yaw_start + (_yaw_max - _yaw_start) * _frac, _yaw_max)
            old_orn_noise = list(self.cfg_task.hand_init_orn_noise)
            self.cfg_task.hand_init_orn_noise = [old_orn_noise[0], old_orn_noise[1], _cur]
        try:
            super().randomize_initial_state(env_ids)
        finally:
            if old_held_noise is not None:
                self.cfg_task.held_asset_pos_noise = old_held_noise
            if old_orn_noise is not None:
                self.cfg_task.hand_init_orn_noise = old_orn_noise
        if getattr(self, "_weld_held", False):
            self._set_welded_gripper_state(env_ids)
            self._sync_welded_held_pose_from_parent(env_ids)
            self.step_sim_no_action()
        # Replace Factory's inherited Gaussian fixed-asset observation noise with bounded upstream
        # socket/goal pose error. This is actor-side only: the critic still receives clean fixed_pos.
        # The physical reset offset is separate (cfg_task.hand_init_pos_noise).
        obs_noise_bound = getattr(self.cfg, "fixed_asset_pos_obs_noise_bound", None)
        if obs_noise_bound is not None:
            bound = torch.tensor(obs_noise_bound, device=self.device, dtype=torch.float32)
            # Socket-anchor obs-noise CURRICULUM: ramp the bound from _start -> _bound over
            # fixed_asset_pos_obs_noise_curriculum_steps control steps (0 => constant at _bound). Lets the
            # policy learn hole-finding at low anchor error first, then extend to the required 2.5cm --
            # from-scratch at 2.5cm doesn't crack it (s192). Same common_step_counter ramp as the tilt curric.
            _nramp = int(getattr(self.cfg, "fixed_asset_pos_obs_noise_curriculum_steps", 0) or 0)
            _nstart = getattr(self.cfg, "fixed_asset_pos_obs_noise_start", None)
            if _nramp > 0 and _nstart is not None:
                _start = torch.tensor(_nstart, device=self.device, dtype=torch.float32)
                _frac = min(1.0, max(0.0, float(self.common_step_counter) / float(_nramp)))
                bound = _start + (bound - _start) * _frac
                self._cur_obs_noise_bound = bound.tolist()  # for logging/inspection
            sample = 2.0 * torch.rand((len(env_ids), 3), device=self.device) - 1.0
            self.init_fixed_pos_obs_noise[env_ids] = sample * bound
        # Forge's reset leaves the screw vertical (yaw noise only), so the upstream angular error is
        # absent. Add it now by tilting the grasped screw about its shaft tip.
        if hasattr(self, "_socket_table"):
            self._apply_pre_insert_tilt(env_ids)
        # GRAVITY COMP: flag these envs to capture their pre-contact force baseline (= the constant
        # world-frame gravity wrench) on the next obs; the E2E obs subtracts it so ft_force is pure
        # CONTACT -- matching the real robot's gravity-compensated F/T (else the sim force has an
        # orientation-independent gravity offset the real one won't). Only if use_gravity_comp.
        if getattr(self.cfg, "use_gravity_comp", False):
            if not hasattr(self, "_grav_pending"):
                self._grav_pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                self._grav_base = torch.zeros((self.num_envs, 3), device=self.device)
            self._grav_pending[env_ids] = True
        # Resample per-episode camera extrinsics jitter (camera-pose DR; no-ops when the knobs are 0).
        if getattr(self, "_has_camera", False):
            if self.cfg.cam_pos_jitter > 0.0:
                self._cam_jitter[env_ids] = self.cfg.cam_pos_jitter * torch.randn(
                    (len(env_ids), 3), device=self.device
                )
            if self.cfg.cam_rot_jitter_deg > 0.0:
                self._cam_roll_jitter[env_ids] = math.radians(self.cfg.cam_rot_jitter_deg) * torch.randn(
                    (len(env_ids),), device=self.device
                )
            # Appearance DR: per-env photometric params + global scene material/light randomization.
            self._randomize_appearance(env_ids)
            # Realistic-depth MODALITY dropout (per-episode): sample which depth regime each env gets this
            # episode -- RGB-only (depth zeroed) / heavily-corrupted / normal-corrupted -- so the policy +
            # estimator can't over-trust the (now noisy) depth channel and must read the hole from RGB too.
            if getattr(self.cfg, "depth_modality_dropout", False):
                probs = torch.tensor(
                    getattr(self.cfg, "depth_modality_probs", (0.15, 0.35, 0.50)), device=self.device
                ).clamp(min=0.0)
                sample = torch.multinomial(probs, len(env_ids), replacement=True)
                # Curriculum: ramp dropout PREVALENCE 0->full -- keep an episode's sampled rgb-only/heavy
                # modality only with prob `curric`, else force "normal" (index 2). So early training is
                # mostly normal depth and rgb-only/heavy episodes phase in over the ramp.
                _cf = self._depth_curric_frac()
                if _cf < 1.0:
                    keep = torch.rand(len(env_ids), device=self.device) < _cf
                    sample = torch.where(keep, sample, torch.full_like(sample, 2))
                self._depth_modality[env_ids] = sample
            # Interpolation-fill variant: flag some episodes to have their synthetic depth holes filled.
            _fill_p = float(getattr(self.cfg, "depth_fill_prob", 0.0))
            if _fill_p > 0.0:
                self._depth_fill[env_ids] = torch.rand(len(env_ids), device=self.device) < _fill_p
            # Temporal frame-stack: flag these envs so their frame history is reset to the new episode's
            # first frame (no cross-episode motion bleed); consumed in the next _get_camera_image.
            if getattr(self, "_frame_stack", 1) > 1:
                self._frame_reset_mask[env_ids] = True

    def get_handheld_asset_relative_pose(self):
        """Grip the screw HEAD between the fingerpads, then add the rotational grasp misalignment.

        Position fix. Factory's peg formula sets ``relative_pos.z = held_height - fingerpad_length``,
        which assumes the asset ORIGIN is at its BASE. Our cooling_screw origin is at the head/shaft
        SHOULDER, with the shaft (``screw_shaft_length``) hanging below it -- so the formula places
        the screw ~one shaft-length too low and the gripper closes on empty air below the head (the
        screw then only dangles, wobbles, and can slip out). We instead grip the HEAD treated as the
        graspable cylinder (its base = the shoulder = our origin), so ``relative_pos.z =
        head_height - fingerpad_length`` with ``head_height = full_height - screw_shaft_length``.

        Misalignment. Factory grasps the peg perfectly aligned. Real grasps are not: the screw sits
        slightly tilted in the jaws. We tilt the shaft OFF the finger-pointing direction by a uniform
        angle in [0, grasp_misalign_max_deg] about a random horizontal axis (the shaft axis no longer
        points straight along the gripper), so the TRUE screw pose deviates from the fingertip-based
        estimate by an amount the actor cannot observe (held pose is privileged/critic-only) -- it
        must be recovered from vision. Resampled every reset. Independent of the pre-insert tilt.
        """
        held_asset_relative_pos, held_asset_relative_quat = super().get_handheld_asset_relative_pose()
        # Grip the head, not below it (correct for the shoulder-origin, two-diameter screw).
        head_height = self.cfg_task.held_asset_cfg.height - self.cfg_task.screw_shaft_length
        held_asset_relative_pos[:, 2] = head_height - self.cfg_task.robot_cfg.franka_fingerpad_length
        # Grasp POSITION DR: add the per-env axial(z)+lateral(x) jitter. Welded path reuses the offset
        # stashed in _weld_held_asset so the weld and the reset pose agree (see there). +z retracts the
        # screw INTO the pads, so keep the axial bound small (never pull the shaft shoulder in).
        if getattr(self, "_weld_held", False):
            _jit = getattr(self, "_weld_pos_jitter", None)
            if _jit is not None:
                held_asset_relative_pos = held_asset_relative_pos + _jit
        max_deg = getattr(self.cfg_task, "grasp_misalign_max_deg", 0.0)
        n = self.num_envs
        # Physical grasp tilt: a cylindrical head between flat parallel pads can only tip ABOUT THE
        # FINGER PRESSING AXIS (the pads' closing direction), flopping the shaft in the plane
        # perpendicular to it. Tipping toward/away from a pad FACE is blocked by the flat pad. The
        # Franka closes along the fingertip-frame Y axis (measured pressing axis ~= [0,-1,0] in the
        # fingertip frame), so we tilt about Y by a signed angle in [-max, max]. RE-MEASURE the pressing
        # axis for the custom gripper (viz_camera prints it).
        welded = getattr(self, "_weld_held", False)
        if welded:
            # Welded grasp: the tilt is BAKED into the per-env FixedJoint (_weld_held_asset). Reuse that same
            # per-env angle so the written reset pose matches the weld (no first-step snap) + the aux label
            # stays correct. (Per-env fixed, not per-reset -- a static joint frame can't be cheaply re-tilted.)
            mag = getattr(self, "_weld_misalign_signed", torch.zeros(n, device=self.device))
        elif max_deg > 0.0:
            mag = math.radians(max_deg) * (2.0 * torch.rand(n, device=self.device) - 1.0)
        else:
            mag = torch.zeros(n, device=self.device)
        if welded or max_deg > 0.0:
            axis = torch.zeros((n, 3), device=self.device)
            axis[:, 1] = 1.0  # fingertip-frame Y = finger pressing/closing axis
            misalign_quat = torch_utils.quat_from_angle_axis(mag, axis)
            held_asset_relative_quat = torch_utils.quat_mul(misalign_quat, held_asset_relative_quat)
        # Secondary out-of-plane tilt about fingertip X (pad compliance / off-centre head). Welded path
        # reuses the per-env angle baked in _weld_held_asset; non-weld samples per grasp. 0 => no-op.
        sec_deg = float(getattr(self.cfg_task, "grasp_misalign_secondary_deg", 0.0))
        if welded:
            sec = getattr(self, "_weld_misalign_secondary", torch.zeros(n, device=self.device))
        elif sec_deg > 0.0:
            sec = math.radians(sec_deg) * (2.0 * torch.rand(n, device=self.device) - 1.0)
        else:
            sec = torch.zeros(n, device=self.device)
        if welded or sec_deg > 0.0:
            axis2 = torch.zeros((n, 3), device=self.device)
            axis2[:, 0] = 1.0  # fingertip-frame X = out-of-plane (perpendicular to the pressing axis)
            sec_quat = torch_utils.quat_from_angle_axis(sec, axis2)
            held_asset_relative_quat = torch_utils.quat_mul(sec_quat, held_asset_relative_quat)
        # Stash the per-episode signed grasp tilt (rad) as the label for the auxiliary grasp head. Set
        # for ALL envs every reset (resets are synchronized), so it matches each env's current episode.
        # This is the COMMANDED tilt; the realized angle drifts a little under gravity sag, which is
        # acceptable noise for a representation-shaping aux target.
        self._grasp_misalign_signed = mag
        return held_asset_relative_pos, held_asset_relative_quat

    def _apply_pre_insert_tilt(self, env_ids):
        """Inject the angular pre-insert error (the upstream orientation uncertainty).

        Factory's reset leaves the screw vertical (only yaw noise), so the ~20-25 deg angular error
        never appears and the orientation-aware reward/success are never exercised. Here we rigidly
        rotate the already-grasped screw AND the wrist together about the screw's shaft tip, by a
        uniform angle in [0, pre_insert_tilt_max_deg] about a random horizontal axis. Pivoting about
        the tip keeps it over the socket mouth (preserving the lateral pre-insert offset) while the
        body angles -- a realistic angled pre-insert, not a tip swung out of the socket. Uses the
        live grasp poses, so it stays correct after the iiwa/gripper swap (no hard-coded geometry).
        """
        final_deg = float(self.cfg_task.pre_insert_tilt_max_deg)
        if final_deg <= 0.0:
            return
        # Tilt CURRICULUM: linearly ramp the effective max tilt from start->final over the first
        # pre_insert_tilt_curriculum_steps control steps (0 => constant at final). common_step_counter
        # counts control steps (~= iters * horizon_length). Stashed for logging/inspection.
        start_deg = float(getattr(self.cfg_task, "pre_insert_tilt_start_deg", 0.0))
        ramp = int(getattr(self.cfg_task, "pre_insert_tilt_curriculum_steps", 0) or 0)
        if ramp > 0:
            frac = min(1.0, max(0.0, float(self.common_step_counter) / float(ramp)))
            cur_deg = start_deg + (final_deg - start_deg) * frac
        else:
            cur_deg = final_deg
        self._cur_pre_insert_tilt_deg = cur_deg
        if cur_deg <= 0.0:
            return

        # No gravity while we re-pose (mirror Factory's reset; the screw also has gravity disabled).
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        # Sample an isotropic tilt of the vertical screw axis: magnitude in [0, max], random azimuth.
        n = self.num_envs
        max_rad = math.radians(cur_deg)
        mag = max_rad * torch.rand(n, device=self.device)
        az = 2.0 * math.pi * torch.rand(n, device=self.device)
        axis = torch.stack([torch.cos(az), torch.sin(az), torch.zeros_like(az)], dim=1)
        tilt_quat = torch_utils.quat_from_angle_axis(mag, axis)

        # Capture the vertical grasp BEFORE moving (resets are all-env at once): pivot (shaft tip),
        # the screw pose, and the arm joints, so envs whose IK can't reach the tilted pose fall back
        # to this known-good vertical grasp instead of an invalid one.
        tip, _ = self._held_base_pose()
        ft_pos, ft_quat = self.fingertip_midpoint_pos.clone(), self.fingertip_midpoint_quat.clone()
        held_pos0, held_quat0 = self.held_pos.clone(), self.held_quat.clone()
        saved_joint_pos = self.joint_pos.clone()

        # Tilt the wrist target about the tip and IK the arm there (teleports joints; screw floats).
        new_ft_pos = tip + torch_utils.quat_apply(tilt_quat, ft_pos - tip)
        new_ft_quat = torch_utils.quat_mul(tilt_quat, ft_quat)
        pos_err, aa_err = self.set_pos_inverse_kinematics(new_ft_pos, new_ft_quat, env_ids)
        ik_failed = (torch.linalg.norm(pos_err, dim=1) > 1e-3) | (torch.linalg.norm(aa_err, dim=1) > 1e-3)
        failed_env_ids = env_ids[ik_failed.nonzero(as_tuple=False).squeeze(-1)]

        # Carry the screw rigidly with the same tilt about the tip (tip stays put, body angles by the
        # sampled angle). Envs where IK failed keep the original vertical grasp -> arm + screw stay
        # consistent (no contact shove, no out-of-range tilt).
        new_held_pos = tip + torch_utils.quat_apply(tilt_quat, held_pos0 - tip)
        new_held_quat = torch_utils.quat_mul(tilt_quat, held_quat0)
        new_held_pos[failed_env_ids] = held_pos0[failed_env_ids]
        new_held_quat[failed_env_ids] = held_quat0[failed_env_ids]
        if len(failed_env_ids) > 0:
            self.joint_pos[failed_env_ids] = saved_joint_pos[failed_env_ids]
            self.joint_vel[:] = 0.0
            self._robot.write_joint_state_to_sim(self.joint_pos, self.joint_vel)
            self.ctrl_target_joint_pos[:, 0:7] = self.joint_pos[:, 0:7]
            self._robot.set_joint_position_target(self.ctrl_target_joint_pos)

        if getattr(self, "_weld_held", False):
            self._set_welded_gripper_state(env_ids)
            self._sync_welded_held_pose_from_parent(env_ids)
            self.step_sim_no_action()
            physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))
            self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
            self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()
            self.prev_joint_pos = self.joint_pos[:, 0:7].clone()
            self.ee_linvel_fd[:, :] = 0.0
            self.ee_angvel_fd[:, :] = 0.0
            return

        held_state = self._held_asset.data.default_root_state.clone()[env_ids]
        held_state[:, 0:3] = new_held_pos[env_ids] + self.scene.env_origins[env_ids]
        held_state[:, 3:7] = new_held_quat[env_ids]
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7], env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
        self._held_asset.reset(env_ids)
        self.step_sim_no_action()

        # Restore gravity; clear the reset-induced finite-difference velocities so step 0 is clean.
        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()
        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()
        self.ee_linvel_fd[:, :] = 0.0
        self.ee_angvel_fd[:, :] = 0.0

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
        self.fixed_pos_obs_frame[:] = self._socket_opening_pos()
        self._update_camera_pose()

    def _update_camera_pose(self):
        """Drive the wrist camera each step as a RIGID wrist mount (a "GoPro bolted to the wrist").

        BOTH the eye and the aim point are fixed offsets in the fingertip frame, so the camera is
        bolted to the gripper: it translates and rotates with the wrist and NEVER re-aims. Its world
        pose depends only on the wrist pose -- it uses NO privileged knowledge of the true socket
        location. (The earlier version aimed at the real socket, which kept the socket artificially
        centered and could not transfer to hardware.) Under lateral/tilt/grasp error the wrist -- and
        the camera with it -- move while the socket stays put in the world, so the socket appears
        displaced/tilted in the image; that drift IS the cue the policy reads.

        Orientation is a FIXED offset quaternion composed with the live gripper quaternion, so ALL
        three rotation axes (including ROLL) ride the gripper -- unlike a look-at, which would pin the
        roll to world-up and let the gripper spin in the image as the wrist yaws/tilts.

        Camera-pose DR (both sampled at reset): _cam_jitter shifts the eye (hand-eye/mount position
        error); _cam_roll_jitter rolls the camera a few degrees about its optical axis (mount roll-
        calibration error). Guarded so the state-obs task and the parent __init__'s early calls are
        unaffected.
        """
        if getattr(self, "_tiled_camera", None) is None:
            return
        ft_pos, ft_quat = self.fingertip_midpoint_pos, self.fingertip_midpoint_quat
        # Eye: fixed offset in the fingertip frame (+ DR position jitter) -> rides the wrist rigidly.
        eye = ft_pos + torch_utils.quat_apply(ft_quat, self._cam_offset_pos + self._cam_jitter)
        # Orientation: gripper quat * fixed mount quat * (DR roll about the optical axis).
        roll_quat = torch_utils.quat_from_angle_axis(self._cam_roll_jitter, self._cam_roll_axis)
        cam_quat = torch_utils.quat_mul(torch_utils.quat_mul(ft_quat, self._cam_offset_quat), roll_quat)
        # cam_quat is built in the OpenGL camera convention (-z forward, +y up; see _cam_offset_quat),
        # so declare it -- the default "ros" would re-convert and corrupt the orientation.
        self._tiled_camera.set_world_poses(eye + self.scene.env_origins, cam_quat, convention="opengl")

    def _socket_opening_pos(self):
        """World position of the active socket opening/entry frame."""
        fixed_tip_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
        fixed_tip_pos_local[:, 2] = self.cfg_task.fixed_asset_cfg.height + self.cfg_task.fixed_asset_cfg.base_height
        _, socket_opening_pos = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self._identity_quat, fixed_tip_pos_local
        )
        return socket_opening_pos

    def _held_base_pose(self):
        """World pose of the screw shaft tip (the point that seats at the socket bottom)."""
        held_base_quat, held_base_pos = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self._identity_quat, self.held_base_offset
        )
        return held_base_pos, held_base_quat

    def _keypoint_distances(self):
        """Mean keypoint distance between current screw pose and target seated pose."""
        held_base_pos, held_base_quat = self._held_base_pose()
        target_base_pos = self.fixed_pos
        target_base_quat = self.fixed_quat

        keypoints_held = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
        keypoints_target = torch.zeros_like(keypoints_held)
        for idx, keypoint_offset in enumerate(self._reward_keypoint_offsets):
            offset = keypoint_offset.repeat(self.num_envs, 1)
            keypoints_held[:, idx] = torch_utils.tf_combine(
                held_base_quat, held_base_pos, self._identity_quat, offset
            )[1]
            keypoints_target[:, idx] = torch_utils.tf_combine(
                target_base_quat, target_base_pos, self._identity_quat, offset
            )[1]

        keypoint_dist = torch.linalg.vector_norm(keypoints_held - keypoints_target, dim=-1).mean(dim=-1)
        tip_dist = torch.linalg.vector_norm(held_base_pos - target_base_pos, dim=1)
        return keypoint_dist, tip_dist

    def _shaft_axis_error(self):
        """Angle between the screw shaft axis and the active socket axis."""
        _, held_base_quat = self._held_base_pose()
        held_axis = torch_utils.quat_apply(held_base_quat, self._local_z_axis)
        socket_axis = torch_utils.quat_apply(self.fixed_quat, self._local_z_axis)
        axis_cos = torch.clamp(torch.sum(held_axis * socket_axis, dim=1), -1.0, 1.0)
        return torch.acos(axis_cos)

    # --- success + reward (pure residual, squashing kernel) ---------------------------------
    def _get_curr_successes(self, success_threshold, check_rot=False):
        held_base_pos, _ = self._held_base_pose()
        target = self.fixed_pos  # socket bottom
        xy_dist = torch.linalg.vector_norm(target[:, 0:2] - held_base_pos[:, 0:2], dim=1)
        z_disp = held_base_pos[:, 2] - target[:, 2]
        is_centered = xy_dist < 0.0025
        height_threshold = self.cfg_task.fixed_asset_cfg.height * success_threshold
        is_seated = z_disp < height_threshold
        is_aligned = self._shaft_axis_error() < self.cfg_task.success_orientation_threshold
        return torch.logical_and(torch.logical_and(is_centered, is_seated), is_aligned)

    def _get_rewards(self):
        keypoint_dist, tip_dist = self._keypoint_distances()
        shaft_axis_error = self._shaft_axis_error()
        curr_successes = self._get_curr_successes(self.cfg_task.success_threshold)

        # Multi-scale squashing kernel 1/(exp(a*d)+b+exp(-a*d)) on the mean keypoint distance:
        # each scale peaks (bounded) at d=0 and decays smoothly. baseline (small a) shapes the whole
        # approach; coarse/fine (large a) sharpen the gradient near the seated pose. This replaces the
        # unbounded neg-L2 + binary-dominated signal with the dense bounded reward Factory/IndustReal
        # use, while keeping our orientation-aware shaft-tip->top keypoints (so tilt still costs).
        a0, b0 = self.cfg_task.keypoint_coef_baseline
        a1, b1 = self.cfg_task.keypoint_coef_coarse
        a2, b2 = self.cfg_task.keypoint_coef_fine
        kp_baseline = factory_utils.squashing_fn(keypoint_dist, a0, b0)
        kp_coarse = factory_utils.squashing_fn(keypoint_dist, a1, b1)
        kp_fine = factory_utils.squashing_fn(keypoint_dist, a2, b2)

        rew_buf = kp_baseline + kp_coarse + kp_fine + self.cfg_task.success_bonus * curr_successes.float()

        # Restore Forge's success-prediction term that our override otherwise drops: train action[:,6]
        # to predict whether the insertion is currently successful. The penalty scale ramps from 0 to
        # 1 only once a delay_until_ratio fraction of envs have succeeded (so it doesn't fight early
        # learning), matching ForgeEnv._get_rewards. Needed before sim-to-real (the policy's done-signal).
        policy_success_pred = (self.actions[:, 6] + 1) / 2  # [-1,1] -> [0,1]
        success_pred_error = (curr_successes.float() - policy_success_pred).abs()
        if curr_successes.float().mean() >= self.cfg_task.delay_until_ratio:
            self.success_pred_scale = 1.0
        rew_buf = rew_buf - self.success_pred_scale * success_pred_error

        log_dict = {
            "kp_baseline": kp_baseline,
            "kp_coarse": kp_coarse,
            "kp_fine": kp_fine,
            "neg_keypoint_l2": -keypoint_dist,
            "neg_tip_l2": -tip_dist,
            "shaft_axis_error": shaft_axis_error,
            "success_pred_error": success_pred_error,
            "success": curr_successes.float(),
        }

        # Optional Forge contact-force penalty (off by default). Penalizes estimated EE contact
        # force above the per-env threshold -> discourages jamming/over-force, important for
        # sim-to-real and not damaging the part. Enable with cfg_task.contact_penalty_scale > 0;
        # the scale likely needs tuning against the neg-L2 reward magnitude (~0.05-0.3).
        if self.cfg_task.contact_penalty_scale > 0.0:
            contact_force = torch.linalg.vector_norm(self.force_sensor_smooth[:, 0:3], dim=-1)
            contact_penalty = torch.nn.functional.relu(contact_force - self.contact_penalty_thresholds)
            rew_buf = rew_buf - self.cfg_task.contact_penalty_scale * contact_penalty
            log_dict["contact_force"] = contact_force
            log_dict["contact_penalty"] = contact_penalty

        self.prev_actions = self.actions.clone()
        self._log_factory_metrics(log_dict, curr_successes)
        return rew_buf

    # --- appearance domain randomization (vision variant) -----------------------------------
    def _randomize_appearance(self, env_ids):
        """Per-reset appearance DR: per-env photometric params + per-env materials + scene lighting.

        Photometric params (gain/brightness/contrast/gamma) are per-env GPU buffers consumed in
        _get_camera_image. Part materials are PER-ENV (colour varies within the batch). The dome +
        directional key light are scene-wide (one set per reset). All scene-USD writes are best-effort
        and self-disable on error (the GPU photometric aug still runs).
        """
        cfg = self.cfg
        m = len(env_ids)

        def _half(width, shape):  # uniform in [-width, +width]
            return (2.0 * torch.rand(shape, device=self.device) - 1.0) * width

        if getattr(cfg, "photo_gain_rgb", 0.0) > 0.0:
            self._photo_gain[env_ids] = 1.0 + _half(cfg.photo_gain_rgb, (m, 1, 1, 3))
        if getattr(cfg, "photo_brightness", 0.0) > 0.0:
            self._photo_brightness[env_ids] = _half(cfg.photo_brightness, (m, 1, 1, 3))
        if getattr(cfg, "photo_contrast", 0.0) > 0.0:
            self._photo_contrast[env_ids] = 1.0 + _half(cfg.photo_contrast, (m, 1, 1, 3))
        if getattr(cfg, "photo_gamma", 0.0) > 0.0:
            self._photo_gamma[env_ids] = 1.0 + _half(cfg.photo_gamma, (m, 1, 1, 1))

        self._randomize_part_materials()
        self._randomize_scene_light()

    def _randomize_part_materials(self):
        """Per-env, CORRELATED screw/base albedo + roughness (best-effort).

        For each env we draw ONE scene colour (base +/- material_color_jitter), then screw and base
        get that colour +/- a SMALL material_part_color_jitter. So within an env the two parts are
        usually similar (matching reality: same filament => same colour => no colour contrast to lean
        on) while still occasionally diverging (robustness). The wide between-env spread is the actual
        DR; the small within-env spread stops the policy keying on a screw-vs-base colour contrast that
        won't exist on hardware. Colours are drawn as tensors then ``.tolist()`` once (no per-channel
        GPU sync); the cost is the per-env USD ``Set`` calls, paid only at reset.
        """
        if self._material_dr_failed or not self._appearance_shader_paths:
            return
        cfg = self.cfg
        base = torch.tensor(cfg.material_base_color)  # CPU
        jit = float(getattr(cfg, "material_color_jitter", 0.0))
        pjit = float(getattr(cfg, "material_part_color_jitter", 0.0))
        r_lo, r_hi = (float(x) for x in getattr(cfg, "material_roughness_range", (0.5, 0.5)))
        try:
            import isaacsim.core.utils.prims as prim_utils
            from pxr import Gf

            screw_paths = self._appearance_shader_paths.get("screw", [])
            base_paths = self._appearance_shader_paths.get("base", [])
            n = len(screw_paths)
            if n == 0:
                return
            if bool(getattr(cfg, "material_hue_randomize", False)):
                # FULL-HUE randomization (HSV): the real base is vivid BLUE, which the narrow RGB-jitter-
                # about-grey band does NOT span. Draw a full-range hue per env (so blue, and every other
                # colour, appears) with configurable saturation/value, then convert to RGB. The screw stays
                # CORRELATED to the base (same hue +/- a small jitter) so there's no screw-vs-base colour
                # cue to exploit (same filament in reality), while the between-env hue spread is the DR.
                s_lo, s_hi = (float(x) for x in getattr(cfg, "material_saturation_range", (0.4, 1.0)))
                v_lo, v_hi = (float(x) for x in getattr(cfg, "material_value_range", (0.3, 0.9)))
                hue = torch.rand(n)
                sat = s_lo + (s_hi - s_lo) * torch.rand(n)
                val = v_lo + (v_hi - v_lo) * torch.rand(n)
                scene = self._hsv_to_rgb(hue, sat, val)
                hjit = float(getattr(cfg, "material_hue_part_jitter", 0.02))
                screw_c = self._hsv_to_rgb((hue + (2.0 * torch.rand(n) - 1.0) * hjit) % 1.0, sat, val).clamp(0.0, 1.0).tolist()
                base_c = scene.clamp(0.0, 1.0).tolist()
            else:
                scene = base + (2.0 * torch.rand(n, 3) - 1.0) * jit  # per-env shared colour
                screw_c = (scene + (2.0 * torch.rand(n, 3) - 1.0) * pjit).clamp(0.0, 1.0).tolist()
                base_c = (scene + (2.0 * torch.rand(n, 3) - 1.0) * pjit).clamp(0.0, 1.0).tolist()
            screw_r = (r_lo + (r_hi - r_lo) * torch.rand(n)).tolist()
            base_r = (r_lo + (r_hi - r_lo) * torch.rand(n)).tolist()

            for paths, colors, roughs in ((screw_paths, screw_c, screw_r), (base_paths, base_c, base_r)):
                for sp, col, rgh in zip(paths, colors, roughs):
                    prim = prim_utils.get_prim_at_path(sp)
                    d_attr = prim.GetAttribute("inputs:diffuseColor") if (prim and prim.IsValid()) else None
                    if not (d_attr and d_attr.IsValid()):
                        self._material_dr_failed = True  # not the material we expect -> fall back to aug
                        return
                    d_attr.Set(Gf.Vec3f(*col))
                    r_attr = prim.GetAttribute("inputs:roughness")
                    if r_attr and r_attr.IsValid():
                        r_attr.Set(float(rgh))
        except Exception as exc:  # noqa: BLE001
            carb.log_warn(f"[InsertionEnv] part-material DR disabled after error: {exc}")
            self._material_dr_failed = True

    @staticmethod
    def _hsv_to_rgb(h, s, v):
        """Vectorized HSV->RGB on CPU tensors (h,s,v each (n,) in [0,1]); returns (n,3) RGB in [0,1]."""
        i = (h * 6.0).floor()
        f = h * 6.0 - i
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        i = (i % 6).long()
        r = torch.stack([v, q, p, p, t, v], dim=1).gather(1, i.unsqueeze(1)).squeeze(1)
        g = torch.stack([t, v, v, q, p, p], dim=1).gather(1, i.unsqueeze(1)).squeeze(1)
        b = torch.stack([p, p, t, v, v, q], dim=1).gather(1, i.unsqueeze(1)).squeeze(1)
        return torch.stack([r, g, b], dim=1)

    @staticmethod
    def _color_temp_to_rgb(kelvin):
        """Approximate a blackbody colour-temperature (K) as a normalized RGB tint (peak channel = 1).

        Tanner Helland's piecewise fit, clamped to [0,1]; used to tint the scene lights so the wrist
        image spans warm(~3000K)->cool(~8000K) white balances the D405 will see under different cell
        lighting. Returns a 3-tuple in [0,1].
        """
        t = max(1000.0, min(40000.0, float(kelvin))) / 100.0
        if t <= 66.0:
            r = 255.0
            g = 99.4708025861 * math.log(t) - 161.1195681661
        else:
            r = 329.698727446 * ((t - 60.0) ** -0.1332047592)
            g = 288.1221695283 * ((t - 60.0) ** -0.0755148492)
        if t >= 66.0:
            b = 255.0
        elif t <= 19.0:
            b = 0.0
        else:
            b = 138.5177312231 * math.log(t - 10.0) - 305.0447927307
        return tuple(max(0.0, min(255.0, x)) / 255.0 for x in (r, g, b))

    def _randomize_scene_light(self):
        """Per reset: dome (ambient) intensity+colour, and the key light's DIRECTION+intensity.

        Dome = soft fill; key DistantLight = the directional source casting moving shadows/specular.
        The key direction is sampled in the upper hemisphere (elevation off straight-down, random
        azimuth) so light arrives from a realistic-but-varied overhead angle each episode. Best-effort.
        """
        if self._light_dr_failed:
            return
        cfg = self.cfg
        try:
            import isaacsim.core.utils.prims as prim_utils
            from pxr import Gf

            def _attr(p, name):  # tolerate both UsdLux input schemas
                a = p.GetAttribute(f"inputs:{name}")
                return a if (a and a.IsValid()) else p.GetAttribute(name)

            # Optional COLOUR-TEMPERATURE tint (warm 3000K -> cool 8000K) shared by dome + key, so the
            # wrist image spans the white balances the D405 sees under different cell lighting. When the
            # range is unset, falls back to the legacy grey +/- light_color_jitter dome tint.
            temp_rng = getattr(cfg, "light_color_temp_range_k", None)
            temp_rgb = None
            if temp_rng and float(temp_rng[1]) > 0.0:
                kelvin = float(temp_rng[0]) + (float(temp_rng[1]) - float(temp_rng[0])) * torch.rand(1).item()
                temp_rgb = self._color_temp_to_rgb(kelvin)

            # --- dome ambient: intensity + colour ---
            rng = getattr(cfg, "light_intensity_range", None)
            jit = float(getattr(cfg, "light_color_jitter", 0.0))
            do_intensity = bool(rng) and float(rng[1]) > 0.0
            if do_intensity or jit > 0.0 or temp_rgb is not None:
                dome = prim_utils.get_prim_at_path("/World/Light")
                if not (dome and dome.IsValid()):
                    self._light_dr_failed = True
                    return
                if do_intensity:
                    lo, hi = float(rng[0]), float(rng[1])
                    a = _attr(dome, "intensity")
                    if a and a.IsValid():
                        a.Set(lo + (hi - lo) * torch.rand(1).item())
                if temp_rgb is not None:
                    a = _attr(dome, "color")
                    if a and a.IsValid():
                        a.Set(Gf.Vec3f(*temp_rgb))
                elif jit > 0.0:
                    c = [min(1.0, max(0.0, 0.75 + (2.0 * torch.rand(1).item() - 1.0) * jit)) for _ in range(3)]
                    a = _attr(dome, "color")
                    if a and a.IsValid():
                        a.Set(Gf.Vec3f(*c))

            # --- directional key light: re-point + intensity ---
            krng = getattr(cfg, "key_light_intensity_range", None)
            elev_lo, elev_hi = (float(x) for x in getattr(cfg, "key_light_elev_range_deg", (15.0, 60.0)))
            key = prim_utils.get_prim_at_path("/World/KeyLight")
            if key and key.IsValid():
                if krng and float(krng[1]) > 0.0:
                    a = _attr(key, "intensity")
                    if a and a.IsValid():
                        a.Set(float(krng[0]) + (float(krng[1]) - float(krng[0])) * torch.rand(1).item())
                if temp_rgb is not None:
                    a = _attr(key, "color")
                    if a and a.IsValid():
                        a.Set(Gf.Vec3f(*temp_rgb))
                # tilt straight-down by a random elevation about a random azimuth -> overhead but varied.
                elev = math.radians(elev_lo + (elev_hi - elev_lo) * torch.rand(1).item())
                az = 2.0 * math.pi * torch.rand(1).item()
                axis = torch.tensor([math.cos(az), math.sin(az), 0.0])
                q = torch_utils.quat_from_angle_axis(torch.tensor([elev]), axis.unsqueeze(0))[0].tolist()
                o_attr = key.GetAttribute("xformOp:orient")
                if o_attr and o_attr.IsValid():
                    cur = o_attr.Get()
                    QuatT = type(cur) if cur is not None else Gf.Quatd
                    o_attr.Set(QuatT(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
        except Exception as exc:  # noqa: BLE001
            carb.log_warn(f"[InsertionEnv] scene-light DR disabled after error: {exc}")
            self._light_dr_failed = True

    # --- observations (adds the RGB-D image group for the vision variant) -------------------
    def _get_observations(self):
        obs = super()._get_observations()  # {"policy": proprio, "critic": state}
        if getattr(self.cfg, "use_torque_obs", False):
            # 6-axis F/T: the base env already put the 3-axis contact FORCE in the policy obs
            # (`ft_force`); append the matching 3 TORQUE channels (`force_sensor_smooth[:,3:6]`, the
            # contact moment of a tilted shaft), with obs noise mirroring Forge's force-channel noise.
            # Appended at the END of the proprio vector -> policy 24-d becomes 27-d. The critic keeps
            # the clean privileged state; the aux-label group is a separate obs group, untouched.
            torque = self.force_sensor_smooth[:, 3:6]
            noisy_torque = torque + self.cfg.torque_obs_noise * torch.randn_like(torque)
            obs["policy"] = torch.cat([obs["policy"], noisy_torque], dim=-1)
        if getattr(self, "_has_camera", False):
            obs["image"] = self._get_camera_image()
            obs["aux_label"] = self._get_aux_label()
        return obs

    def _get_aux_label(self):
        """Privileged supervised targets for the auxiliary vision heads (NOT fed to the policy).

        [0]   signed grasp-misalignment angle (rad) about the finger pressing axis -- the per-episode,
              actor-unobservable shaft tilt the policy must recover (set in
              ``get_handheld_asset_relative_pose``).
        [1:4] true shaft-tip -> socket-opening gap, expressed in the fingertip frame (m) -- the clean
              relative pose the wrist camera sees, vs the +/-8mm-noisy socket estimate the actor's
              proprio carries. Uses the realized tip (so it already encodes the grasp-induced tip shift).

        Both come from privileged (critic-only) state, so the labels are available server-side at train
        time and simply unused at deploy (the heads are a training-time representation prior, not a
        runtime estimator).
        """
        grasp = getattr(self, "_grasp_misalign_signed", None)
        if grasp is None:
            grasp = torch.zeros(self.num_envs, device=self.device)
        tip_pos, _ = self._held_base_pose()
        gap_world = self._socket_opening_pos() - tip_pos
        gap_ft = torch_utils.quat_rotate_inverse(self.fingertip_midpoint_quat, gap_world)
        return torch.cat([grasp.unsqueeze(-1), gap_ft], dim=-1)

    def _get_camera_image(self):
        """Wrist RGB-D as a (num_envs, H, W, 4) tensor: normalized RGB (3) + normalized depth (1).

        RGB is scaled to [0,1], then (appearance DR) per-env photometric augmentation + sensor noise,
        then mean-subtracted per image (as in Isaac Lab's camera examples). Depth has inf (no-hit)
        zeroed, (appearance DR) range noise on real returns + dropout holes, then far-clip scaling into
        ~[0,1]. The CNN in the hybrid network consumes this directly (it permutes HWC->CHW).
        """
        if getattr(self.cfg, "blank_image", False):
            # Ablation: zeroed image (camera still renders; the policy just gets no visual signal).
            # Returned BEFORE any DR so the blank control stays exactly blank. Channels include the
            # frame-stack (stacked zeros are still zeros) so the shape matches the widened obs space.
            h, w, c = self.cfg.image_height, self.cfg.image_width, self.cfg.image_channels
            return torch.zeros((self.num_envs, h, w, c * getattr(self, "_frame_stack", 1)), device=self.device)

        out = self._tiled_camera.data.output
        cfg = self.cfg

        # RGB: per-env photometric augmentation (exposure/white-balance, gamma tone curve, contrast),
        # then sensor noise. Brightness/gain go BEFORE gamma so the additive shift survives the final
        # per-image mean re-centring (a pure DC shift would otherwise cancel).
        rgb01 = out["rgb"][..., :3].float() / 255.0
        rgb01 = torch.clamp(rgb01 * self._photo_gain + self._photo_brightness, 0.0, 1.0)
        if (self._photo_gamma != 1.0).any():
            rgb01 = torch.pow(rgb01, self._photo_gamma)
        ch_mean = rgb01.mean(dim=(1, 2), keepdim=True)
        rgb01 = torch.clamp((rgb01 - ch_mean) * self._photo_contrast + ch_mean, 0.0, 1.0)
        rgb_ns = float(getattr(cfg, "rgb_noise_std", 0.0))
        if rgb_ns > 0.0:
            rgb01 = torch.clamp(rgb01 + rgb_ns * torch.randn_like(rgb01), 0.0, 1.0)
        rgb = rgb01 - torch.mean(rgb01, dim=(1, 2), keepdim=True)

        depth = out["depth"].clone()
        if depth.dim() == 4:
            depth = depth.squeeze(-1)  # (N,H,W); a trailing 1-channel axis is re-added at the end
        # ``finite`` = the sensor's ORIGINAL real returns. The socket interior (a no-hit void) is already
        # non-finite here, so it stays a real void through every branch below (never filled) -- matching
        # the real D405, whose depth is missing over the hole interior.
        finite = torch.isfinite(depth)
        depth[~finite] = 0.0
        far = float(self.cfg.tiled_camera.spawn.clipping_range[1])
        d_ns = float(getattr(cfg, "depth_noise_std", 0.0))
        if d_ns > 0.0:  # range noise on REAL returns only (no-hit stays 0)
            depth = torch.where(finite, depth + d_ns * torch.randn_like(depth), depth)
        depth = torch.clamp(depth, 0.0, far) / far
        mode = str(getattr(cfg, "depth_corruption_mode", "uniform"))
        if mode == "structured":
            # Aggressive, D405-like STRUCTURED corruption (clustered blobs + edge/gradient holes + dark-
            # region dropout + per-episode heavy/normal/RGB-only modality + optional hole-fill). Replaces
            # the legacy uniform per-pixel dropout for the sim2real vision run. See _corrupt_depth_structured.
            depth = self._corrupt_depth_structured(depth, finite, rgb01)
        else:
            d_drop = float(getattr(cfg, "depth_dropout_prob", 0.0))
            if d_drop > 0.0:  # legacy per-pixel no-return holes, like a real depth sensor
                depth[torch.rand_like(depth) < d_drop] = 0.0
        depth = depth.unsqueeze(-1)  # (N,H,W,1)

        if getattr(self.cfg, "write_image_to_file", False):
            from isaaclab.sensors import save_images_to_file

            save_images_to_file(rgb01, "/tmp/wrist_rgb.png")  # the augmented image the policy sees
            save_images_to_file(depth, "/tmp/wrist_depth.png")

        # Periodic training gallery: every cam_log_interval calls, dump a tiled montage (all envs in one
        # grid, via save_images_to_file) of exactly what the policy SEES -> watch the env/DR/framing
        # during an 8h run without a second GPU job. Off when cam_log_interval<=0. renders/ is gitignored.
        log_int = int(getattr(self.cfg, "cam_log_interval", 0))
        if log_int > 0:
            if self._cam_log_counter % log_int == 0:
                import os

                from isaaclab.sensors import save_images_to_file

                d = str(getattr(self.cfg, "cam_log_dir", "renders/train_cam"))
                os.makedirs(d, exist_ok=True)
                tag = f"{self._cam_log_counter:08d}"
                rgb_log, depth_log = rgb01, depth
                # Optional: stamp the TRUE socket opening on the policy-seen frame (cam_log_overlay) so the
                # gallery shows whether the hole is in-frame and whether depth is void/corrupted there.
                if bool(getattr(self.cfg, "cam_log_overlay", False)):
                    rgb_log, depth_log = self._draw_true_hole_overlay(rgb01, depth)
                save_images_to_file(rgb_log, f"{d}/rgb_{tag}.png")
                save_images_to_file(depth_log.expand(-1, -1, -1, 3) if depth_log.shape[-1] == 1 else depth_log,
                                    f"{d}/depth_{tag}.png")
            self._cam_log_counter += 1

        frame = torch.cat([rgb, depth], dim=-1)  # (num_envs, H, W, c) -- this timestep's processed frame
        return self._stack_frames(frame)

    def _stack_frames(self, frame):
        """Temporal frame-stack: return the last N processed frames stacked on the channel axis.

        Keeps a per-env ring buffer of the FULLY-PROCESSED frames (post DR/normalisation), so the
        stacked channels carry real motion (approach speed, contact onset) the CNN can read. N=1 is a
        pass-through. On the first call (or a size change) the buffer is filled with the current frame
        so there is no zero-pad startup transient; envs flagged at reset get all N slots overwritten
        with their current frame (no cross-episode bleed). Channel order is oldest -> newest.
        """
        n = getattr(self, "_frame_stack", 1)
        if n <= 1:
            return frame
        if self._frame_hist is None or self._frame_hist.shape[0] != frame.shape[0]:
            self._frame_hist = frame.unsqueeze(1).repeat(1, n, 1, 1, 1)  # (N_env, N, H, W, c)
        else:
            self._frame_hist = torch.roll(self._frame_hist, shifts=-1, dims=1)
            self._frame_hist[:, -1] = frame
        m = self._frame_reset_mask
        if bool(m.any()):
            self._frame_hist[m] = frame[m].unsqueeze(1)  # clear history for just-reset envs
            self._frame_reset_mask = torch.zeros_like(m)
        b, h, w, c = frame.shape
        return self._frame_hist.permute(0, 2, 3, 1, 4).reshape(b, h, w, n * c)

    # --- realistic (D405-like) structured depth corruption ----------------------------------------
    def _depth_curric_frac(self):
        """Depth-corruption curriculum fraction in [0,1] (1.0 when the knob is 0 = no ramp)."""
        steps = int(getattr(self.cfg, "depth_corruption_curriculum_steps", 0) or 0)
        if steps <= 0:
            return 1.0
        return min(1.0, max(0.0, float(self.common_step_counter) / float(steps)))

    def _corrupt_depth_structured(self, depth, finite, rgb01):
        """Turn a clean sim depth map into a real-D405-like one (measured ~45% invalid, structured).

        ``depth`` is (N,H,W) in [0,1] (no-hit already 0); ``finite`` (N,H,W bool) marks the sensor's
        ORIGINAL real returns (so the socket-interior void is preserved and never filled); ``rgb01`` is
        (N,H,W,3) in [0,1] used for the dark-slot + gradient cues. Returns the corrupted (N,H,W) depth.

        Four structured invalidation sources, each cfg-weighted, combined into a synthetic-hole mask that
        is applied ONLY to originally-valid pixels (``& finite``):
          * clustered BLOBS   -- low-frequency-noise thresholding (contiguous drop regions, size set by
                                 ``depth_blob_scale_px``, coverage by ``depth_blob_frac``),
          * EDGE / gradient   -- drop along strong depth+luma gradients (occlusion boundaries, fin edges),
          * DARK-region       -- high drop prob where RGB is dark (the deep fin slots that swallow the IR),
          * per-episode heavy/normal/RGB-only MODALITY (``_depth_modality``) scales the whole thing and,
            for RGB-only episodes, zeros depth entirely.
        Optionally (``_depth_fill``) the synthetic holes are interpolation-filled (mimicking the sensor's
        hole-fill post-proc) -- the socket void is still left empty.
        """
        import torch.nn.functional as F

        cfg = self.cfg
        n, h, w = depth.shape
        # DEPTH CURRICULUM: scale the structural drop fractions by the ramp fraction (0->1) so early
        # training sees near-clean depth and full ~45%-invalid corruption only later.
        curric = self._depth_curric_frac()
        # Per-env corruption strength (heavy modality gets a multiplier; normal/RGB-only = 1.0).
        strength = torch.ones((n, 1, 1), device=self.device)
        heavy_mult = float(getattr(cfg, "depth_heavy_strength", 1.6))
        strength[self._depth_modality == 1] = heavy_mult
        strength = strength * curric
        luma = rgb01.mean(dim=-1)  # (N,H,W)

        drop = torch.zeros((n, h, w), dtype=torch.bool, device=self.device)

        # (1) Clustered blobs via low-frequency noise: coarse random field upsampled -> contiguous regions.
        blob_frac = float(getattr(cfg, "depth_blob_frac", 0.20))
        if blob_frac > 0.0:
            scale = max(2, int(getattr(cfg, "depth_blob_scale_px", 20)))
            ch, cw = max(1, h // scale), max(1, w // scale)
            coarse = torch.rand((n, 1, ch, cw), device=self.device)
            field = F.interpolate(coarse, size=(h, w), mode="bilinear", align_corners=False).squeeze(1)
            drop |= field < (blob_frac * strength).clamp(max=1.0)

        # (2) Edge / gradient holes: strong depth OR luma gradient -> unreliable return.
        edge_p = float(getattr(cfg, "depth_edge_drop_prob", 0.5))
        if edge_p > 0.0:
            thr = float(getattr(cfg, "depth_edge_grad_thresh", 0.015))
            dgx = (depth - torch.roll(depth, 1, dims=2)).abs()
            dgy = (depth - torch.roll(depth, 1, dims=1)).abs()
            lgx = (luma - torch.roll(luma, 1, dims=2)).abs()
            lgy = (luma - torch.roll(luma, 1, dims=1)).abs()
            edge = (dgx + dgy > thr) | (lgx + lgy > thr)
            drop |= edge & (torch.rand((n, h, w), device=self.device) < (edge_p * strength).clamp(max=1.0))

        # (3) Dark-region dropout: the fin slots read low RGB and swallow the projected IR pattern.
        dark_p = float(getattr(cfg, "depth_dark_drop_prob", 0.6))
        if dark_p > 0.0:
            dark_thr = float(getattr(cfg, "depth_dark_value_thresh", 0.30))
            dark = luma < dark_thr
            drop |= dark & (torch.rand((n, h, w), device=self.device) < (dark_p * strength).clamp(max=1.0))

        synth_holes = drop & finite  # only invalidate originally-valid pixels (keep the socket void as-is)
        depth = torch.where(synth_holes, torch.zeros_like(depth), depth)

        # (4) Interpolation-fill variant (per-episode): fill ONLY the synthetic holes from valid neighbours,
        # leaving the socket void empty. A few 3x3 propagation passes (cheap; runs at the render cadence).
        if bool(self._depth_fill.any()):
            known = (finite & ~synth_holes).float()
            d = depth.clone()
            v = known
            for _ in range(int(getattr(cfg, "depth_fill_iters", 4))):
                d_sum = F.avg_pool2d((d * v).unsqueeze(1), 3, 1, 1).squeeze(1) * 9.0
                v_sum = F.avg_pool2d(v.unsqueeze(1), 3, 1, 1).squeeze(1) * 9.0
                fill_here = synth_holes & (v_sum > 0) & (v < 0.5)
                d = torch.where(fill_here, d_sum / v_sum.clamp(min=1e-6), d)
                v = (v.bool() | fill_here).float()
            fill_env = self._depth_fill.view(n, 1, 1)
            depth = torch.where(fill_env & synth_holes, d, depth)

        # Per-episode RGB-only modality: zero depth entirely for those envs (policy sees RGB alone).
        depth = depth * (self._depth_modality != 0).float().view(n, 1, 1)
        return depth

    # --- diagnostic overlay: mark the TRUE socket opening on the policy-seen wrist image -----------
    def _draw_true_hole_overlay(self, rgb01, depth):
        """Stamp a marker at the projected TRUE socket-opening on COPIES of the wrist RGB+depth.

        Reuses the wrist camera's own intrinsics/extrinsics to project ``_socket_opening_pos()`` into the
        image the policy sees, so the periodic gallery shows WHERE the hole actually is (and whether depth
        is void there). Returns (rgb_marked, depth_marked); the live obs tensors are untouched. The
        estimator's PREDICTED hole (pred-vs-GT) is overlaid offline by scripts/render_estimator_overlay.py,
        which has the trained aux head. Best-effort: any failure returns the inputs unchanged.
        """
        try:
            from isaaclab.utils.math import quat_rotate_inverse

            cam = self._tiled_camera
            K = cam.data.intrinsic_matrices  # (N,3,3)
            pos = cam.data.pos_w             # (N,3) world
            quat = cam.data.quat_w_ros       # (N,4) ROS optical frame
            hole_w = self._socket_opening_pos() + self.scene.env_origins
            p_cam = quat_rotate_inverse(quat, hole_w - pos)  # (N,3) optical frame
            z = p_cam[:, 2].clamp(min=1e-4)
            u = (K[:, 0, 0] * p_cam[:, 0] / z + K[:, 0, 2]).round().long()
            v = (K[:, 1, 1] * p_cam[:, 1] / z + K[:, 1, 2]).round().long()
            rgb_m, depth_m = rgb01.clone(), depth.clone()
            _, hh, ww, _ = rgb_m.shape
            r = 3
            for e in range(rgb_m.shape[0]):
                if not (0 <= int(u[e]) < ww and 0 <= int(v[e]) < hh) or z[e] <= 1e-4:
                    continue
                y0, y1 = max(0, int(v[e]) - r), min(hh, int(v[e]) + r + 1)
                x0, x1 = max(0, int(u[e]) - r), min(ww, int(u[e]) + r + 1)
                rgb_m[e, y0:y1, x0:x1, 0] = 1.0  # red marker on RGB
                rgb_m[e, y0:y1, x0:x1, 1:] = 0.0
                depth_m[e, y0:y1, x0:x1, :] = 1.0  # white marker on depth
            return rgb_m, depth_m
        except Exception as exc:  # noqa: BLE001
            carb.log_warn(f"[InsertionEnv] true-hole overlay failed: {exc}")
            return rgb01, depth
