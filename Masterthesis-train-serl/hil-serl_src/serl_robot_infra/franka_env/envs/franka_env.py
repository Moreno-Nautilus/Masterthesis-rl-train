"""Gym Interface for Franka"""
import os
import numpy as np
import gymnasium as gym
import cv2
import copy
from scipy.spatial.transform import Rotation
import time
import requests
import queue
import threading
from datetime import datetime
from collections import OrderedDict
from typing import Dict

from franka_env.camera.video_capture import VideoCapture
from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.zed_capture import ZEDCapture
from franka_env.envs.robot_client import make_robot_client
from franka_env.utils.rotations import euler_2_quat, quat_2_euler


class ImageDisplayer(threading.Thread):
    def __init__(self, queue, name):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True  # make this a daemon thread
        self.name = name

    def run(self):
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break

            frame = np.concatenate(
                [cv2.resize(v, (128, 128)) for k, v in img_array.items() if "full" not in k], axis=1
            )

            cv2.imshow(self.name, frame)
            cv2.waitKey(1)


##############################################################################


class DefaultEnvConfig:
    """Default configuration for FrankaEnv. Fill in the values below."""

    SERVER_URL: str = "http://127.0.0.1:5000/"
    # Robot-communication backend: "http" (stock Flask server, default/unchanged) or
    # "ros2" (Franka-pivot: cartesian_impedance_control over franka_ros2). See robot_client.py.
    ROBOT_CLIENT: str = "http"
    # Camera backend: "realsense" (stock RSCapture, default) or "zed" (ZED Mini wrist).
    # The env keys cameras off REALSENSE_CAMERAS either way; ZEDCapture is RSCapture-compatible.
    CAMERA_TYPE: str = "realsense"
    REALSENSE_CAMERAS: Dict = {
        "wrist_1": "130322274175",
        "wrist_2": "127122270572",
    }
    IMAGE_CROP: dict[str, callable] = {}
    TARGET_POSE: np.ndarray = np.zeros((6,))
    GRASP_POSE: np.ndarray = np.zeros((6,))
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    ACTION_SCALE = np.zeros((3,))
    RESET_POSE = np.zeros((6,))
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_RZ_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    COMPLIANCE_PARAM: Dict[str, float] = {}
    RESET_PARAM: Dict[str, float] = {}
    PRECISION_PARAM: Dict[str, float] = {}
    LOAD_PARAM: Dict[str, float] = {
        "mass": 0.0,
        "F_x_center_load": [0.0, 0.0, 0.0],
        "load_inertia": [0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    DISPLAY_IMAGE: bool = True
    GRIPPER_SLEEP: float = 0.6
    MAX_EPISODE_LENGTH: int = 100
    JOINT_RESET_PERIOD: int = 0


##############################################################################


# FR3 per-joint torque limits (Nm), datasheet. Joints 5-7 (the wrist) take only 12 Nm and
# are what actually trips the "joint torque limit" reflex during an insert.
JOINT_TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
# Back off once ANY joint exceeds this fraction of its limit.
# MEASURED at the rig 2026-09-09 at RESET_POSE (no contact): worst joint 3.2%, |F| 2.5 N.
# With the screw pressed into the base it was 75% on the wrist (|F| 20 N) — a ~23x jump.
# 0.40 therefore sits an order of magnitude above free space while still leaving 60% of
# the budget before the robot's own reflex fires, so the guard reacts long before a fault.
JOINT_TORQUE_FRACTION = 0.40


class FrankaEnv(gym.Env):
    JOINT_TORQUE_LIMITS = JOINT_TORQUE_LIMITS
    JOINT_TORQUE_FRACTION = JOINT_TORQUE_FRACTION

    def __init__(
        self,
        hz=10,
        fake_env=False,
        save_video=False,
        config: DefaultEnvConfig = None,
        set_load=False,
    ):
        self.fake_env = fake_env
        self.action_scale = config.ACTION_SCALE
        self._TARGET_POSE = config.TARGET_POSE
        self._RESET_POSE = config.RESET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.url = config.SERVER_URL
        self.config = config
        # Robot-communication seam: all robot I/O goes through self.robot. The "http"
        # backend reproduces the stock requests.post calls exactly; "ros2" targets the
        # Franka cartesian_impedance_control controller. Selected by config.ROBOT_CLIENT.
        # Skip under fake_env: the learner has no robot (and must not rclpy.init()).
        self.robot = None if fake_env else make_robot_client(config, self.url)
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.display_image = config.DISPLAY_IMAGE
        self.gripper_sleep = config.GRIPPER_SLEEP

        # convert last 3 elements from euler to quat, from size (6,) to (7,)
        self.resetpos = np.concatenate(
            [config.RESET_POSE[:3], euler_2_quat(config.RESET_POSE[3:])]
        )
        if fake_env:
            # Learner path: no robot to query. Seed the state buffers with safe zeros so
            # any incidental _get_obs works; real values are irrelevant (obs are sampled).
            self.currpos = np.zeros((7,)); self.currpos[6] = 1.0  # unit quat
            self.currvel = np.zeros((6,))
            self.currforce = np.zeros((3,))
            self.currtorque = np.zeros((3,))
            self.currjacobian = np.zeros((6, 7))
            self.q = np.zeros((7,))
            self.dq = np.zeros((7,))
            self.curr_gripper_pos = np.zeros((1,))
        else:
            self._update_currpos()
        self.last_gripper_act = time.time()
        self.lastsent = time.time()
        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rz_range = config.RANDOM_RZ_RANGE
        self.hz = hz
        self.joint_reset_cycle = config.JOINT_RESET_PERIOD  # reset the robot joint every 200 cycles

        self.save_video = save_video
        if self.save_video:
            print("Saving videos!")
            self.recording_frames = []

        # boundary box
        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )
        # Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),  # xyz + quat
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {key: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8) 
                                for key in config.REALSENSE_CAMERAS}
                ),
            }
        )
        self.cycle_count = 0

        if fake_env:
            return

        self.cap = None
        self.init_cameras(config.REALSENSE_CAMERAS)
        if self.display_image:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue, self.url)
            self.displayer.start()

        if set_load:
            input("Put arm into programing mode and press enter.")
            self.robot.set_load(self.config.LOAD_PARAM)
            input("Put arm into execution mode and press enter.")
            for _ in range(2):
                self._recover()
                time.sleep(1)

        self.terminate = False
        self.listener = None
        if not fake_env:
            # Stock ESC-to-terminate listener. pynput needs an X display, which a headless
            # / SSH rig shell does not have — it raised ImportError and killed env
            # construction on the robot PC (2026-09-09). It is a CONVENIENCE only: episodes
            # end via the PS4 buttons (X = success, Triangle = abort) or the step limit, so
            # degrade to "no ESC key" rather than refusing to build the env.
            try:
                from pynput import keyboard

                def on_press(key):
                    if key == keyboard.Key.esc:
                        self.terminate = True

                self.listener = keyboard.Listener(on_press=on_press)
                self.listener.start()
            except Exception as e:
                print(f"[franka_env] ESC-terminate listener unavailable ({type(e).__name__}); "
                      "use the PS4 buttons (X = success, Triangle = abort) to end episodes.")

        print("Initialized Franka")

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        euler = Rotation.from_quat(pose[3:]).as_euler("xyz")

        # Clip first euler angle separately due to discontinuity from pi to -pi
        sign = np.sign(euler[0])
        euler[0] = sign * (
            np.clip(
                np.abs(euler[0]),
                self.rpy_bounding_box.low[0],
                self.rpy_bounding_box.high[0],
            )
        )

        euler[1:] = np.clip(
            euler[1:], self.rpy_bounding_box.low[1:], self.rpy_bounding_box.high[1:]
        )
        pose[3:] = Rotation.from_euler("xyz", euler).as_quat()

        return pose

    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        xyz_delta = action[:3].copy()

        # Z FORCE LIMIT: refuse further DOWNWARD z once the measured contact force exceeds
        # Z_FORCE_LIMIT_N. Protects the screw/socket from being driven in harder once it
        # bottoms out — that overload is what tripped the JOINT TORQUE LIMITS during demo
        # recording (rig 2026-09-09). Upward motion is always allowed so the arm (or the
        # operator) can always back out. Set Z_FORCE_LIMIT_N = 0 to disable.
        z_limit = float(getattr(self.config, "Z_FORCE_LIMIT_N", 0.0))
        if z_limit > 0.0:
            # Use a FRESH force reading: self.currforce is from the previous step, which is
            # a whole control period stale — long enough for the force to spike past the
            # limit before the guard ever sees it.
            try:
                self.currforce = np.array(self.robot.get_state()["force"])
            except Exception:
                pass
            fz = float(self.currforce[2])

            # (a) stop pushing DOWN once the z contact force is over the limit
            if xyz_delta[2] < 0.0 and abs(fz) > z_limit:
                xyz_delta[2] = 0.0

            # (b) JOINT-TORQUE guard — this is what actually faults the robot.
            # The "joint torque limit" reflex is per-joint tau_J against the FR3's limits
            # (87 Nm for joints 1-4, only 12 Nm for the WRIST joints 5-7), and the EE wrench
            # does not capture it: measured on the rig with the arm merely holding position,
            # |F| was already 20 N and the wrist torque 9.0 of its 12 Nm budget. A force-only
            # guard therefore either never fires or fires constantly, and neither prevents the
            # fault. Guard the joint torques directly, with a margin, and RETREAT along +z
            # (out of the socket) so the arm unloads instead of sitting pinned until it trips.
            tau = np.abs(np.asarray(self.currtorque_j, dtype=float)) \
                if getattr(self, "currtorque_j", None) is not None else None
            if tau is not None and tau.size == 7:
                util = float(np.max(tau / self.JOINT_TORQUE_LIMITS))
                if util > self.JOINT_TORQUE_FRACTION:
                    xyz_delta = np.zeros(3)
                    xyz_delta[2] = 0.5      # back off, scaled by ACTION_SCALE[0]
                    j = int(np.argmax(tau / self.JOINT_TORQUE_LIMITS))
                    if not getattr(self, "_torque_warned", False):
                        self._torque_warned = True
                        print(f"[franka_env] TORQUE GUARD: joint {j+1} at {100*util:.0f}% of "
                              f"its limit ({tau[j]:.1f}/{self.JOINT_TORQUE_LIMITS[j]:.0f} Nm) "
                              "— translation blocked, backing off. (Further hits are silent.)")

        # HARD AXIS SEPARATION (LOCK_TRANSLATION_DURING_ROTATION).
        # The operator must be able to tilt WITHOUT the tool wandering in x/y, and to
        # translate WITHOUT it tilting — otherwise aligning a screw to a hole is
        # impossible. Measured coupling on this arm is NOT a fixed geometric lever
        # (rotation-only segments gave apparent levers of 3.2 / 3.8 / 4.1 / 6.1 / 9.2 /
        # 29.8 cm), so it is the controller's rotational loop dragging translation and no
        # lever compensation can cancel it. What we CAN do is refuse to command both at
        # once and pin the un-commanded half to the pose we started the move from, so the
        # controller is actively told to hold it rather than left to drift.
        rotating = bool(np.linalg.norm(action[3:6]) > 1e-6)
        translating = bool(np.linalg.norm(action[:3]) > 1e-6)
        if getattr(self.config, "LOCK_TRANSLATION_DURING_ROTATION", False):
            if rotating and not translating:
                # pure rotation: hold the position we had when this rotation began
                if getattr(self, "_axis_lock_pos", None) is None:
                    self._axis_lock_pos = self.currpos[:3].copy()
            else:
                self._axis_lock_pos = None
            if translating and not rotating:
                # pure translation: hold the orientation we had when it began
                if getattr(self, "_axis_lock_rot", None) is None:
                    self._axis_lock_rot = self.currpos[3:].copy()
            else:
                self._axis_lock_rot = None

        self.nextpos = self.currpos.copy()
        step_xyz = xyz_delta * self.action_scale[0]

        # CAP THE DOWNWARD Z STEP. The demos had z at full stick 41% of the time, so the
        # policy drives down at maximum speed and rams the base hard enough to trip the
        # joint-torque reflex — a transient too fast for any 10 Hz guard to catch (measured:
        # filtered tau_j read 9.6% while the robot was already faulted). Capping only the
        # DESCENT keeps x/y and lifting at full speed, so the arm stays responsive while
        # approaching contact gently. Set MAX_Z_DOWN_STEP = 0 to disable.
        max_down = float(getattr(self.config, "MAX_Z_DOWN_STEP", 0.0))
        if max_down > 0.0 and step_xyz[2] < -max_down:
            step_xyz[2] = -max_down

        self.nextpos[:3] = self.nextpos[:3] + step_xyz

        # GET ORIENTATION FROM ACTION
        # FULLY REVERTED TO STOCK 2026-09-09. Two changes were tried on the rig and BOTH
        # made things worse, so this is back to exactly what upstream HIL-SERL does:
        #   1. tool-frame rotation (right-multiply instead of left) -> rotation commands
        #      felt much larger to the operator;
        #   2. a persistent orientation target held across steps -> the arm floated
        #      uncommanded.
        # Do not reintroduce either without testing on the robot first. The residual tilt
        # during fast translation is a CONTROLLER property (its rotational loop lags
        # translation), not something this line can fix.
        rot_now = Rotation.from_quat(self.currpos[3:])
        rot_next = (
            Rotation.from_rotvec(action[3:6] * self.action_scale[1]) * rot_now
        )
        self.nextpos[3:] = rot_next.as_quat()

        # ROTATE ABOUT THE TOOL TIP, not the flange.
        # The controller rotates about the flange, but the screw tip sits ROTATION_LEVER_M
        # below it, so a tilt swings the tip sideways: at a 10 cm lever one 2.3 deg step
        # drags the tip ~4 mm in x (measured). That makes "tilt back to align" useless —
        # every correction shoves the part out of the hole.
        # Compensating the position by the tip's displacement keeps the TIP fixed while the
        # tool pivots around it, which is what the operator actually wants when aligning.
        # Set ROTATION_LEVER_M = 0 (default) for stock flange-centred behaviour.
        lever = float(getattr(self.config, "ROTATION_LEVER_M", 0.0))
        if lever != 0.0:
            offset = np.array([0.0, 0.0, lever])
            self.nextpos[:3] += rot_now.apply(offset) - rot_next.apply(offset)

        # Apply the axis locks captured above: command the HELD value explicitly, so the
        # controller is driven back to it instead of being allowed to drift.
        if getattr(self, "_axis_lock_pos", None) is not None:
            self.nextpos[:3] = self._axis_lock_pos      # tilting: x/y/z pinned
        if getattr(self, "_axis_lock_rot", None) is not None:
            self.nextpos[3:] = self._axis_lock_rot      # translating: attitude pinned

        # HARD ORIENTATION LOCK. Every commanded pose carries RESET_POSE's orientation
        # (tool straight down) and rotation actions are ignored entirely.
        # WHY: this arm's controller couples translation and rotation in a way that is NOT
        # a fixed geometric lever (rotation-only jogs implied levers of 3-30 cm), so it
        # cannot be cancelled by compensating the command — every attempt (tool-frame
        # rotation, persistent orientation target, tip-centred rotation, per-axis locking)
        # failed on the real robot. The insert is a pure vertical descent, so orientation
        # is not a DoF the task needs: locking it REMOVES the coupling instead of fighting
        # it, and the arm cannot tilt away from vertical during a demo.
        if getattr(self.config, "LOCK_ORIENTATION", False):
            self.nextpos[3:] = self.resetpos[3:]

        gripper_action = action[6] * self.action_scale[2]

        self._send_gripper_command(gripper_action)
        self._send_pos_command(self.clip_safety_box(self.nextpos))

        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        self._update_currpos()
        ob = self._get_obs()
        reward = self.compute_reward(ob)
        done = self.curr_path_length >= self.max_episode_length or reward or self.terminate
        return ob, int(reward), done, False, {"succeed": reward}

    def compute_reward(self, obs) -> bool:
        current_pose = obs["state"]["tcp_pose"]
        # convert from quat to euler first
        current_rot = Rotation.from_quat(current_pose[3:]).as_matrix()
        target_rot = Rotation.from_euler("xyz", self._TARGET_POSE[3:]).as_matrix()
        diff_rot = current_rot.T  @ target_rot
        diff_euler = Rotation.from_matrix(diff_rot).as_euler("xyz")
        delta = np.abs(np.hstack([current_pose[:3] - self._TARGET_POSE[:3], diff_euler]))
        # print(f"Delta: {delta}")
        if np.all(delta < self._REWARD_THRESHOLD):
            return True
        else:
            # print(f'Goal not reached, the difference is {delta}, the desired threshold is {self._REWARD_THRESHOLD}')
            return False

    def get_im(self) -> Dict[str, np.ndarray]:
        """Get images from the realsense cameras."""
        images = {}
        display_images = {}
        full_res_images = {}  # New dictionary to store full resolution cropped images
        for key, cap in self.cap.items():
            try:
                rgb = cap.read()
                cropped_rgb = self.config.IMAGE_CROP[key](rgb) if key in self.config.IMAGE_CROP else rgb
                resized = cv2.resize(
                    cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
                )
                images[key] = resized[..., ::-1]
                display_images[key] = resized
                display_images[key + "_full"] = cropped_rgb
                full_res_images[key] = copy.deepcopy(cropped_rgb)  # Store the full resolution cropped image
            except queue.Empty:
                input(
                    f"{key} camera frozen. Check connect, then press enter to relaunch..."
                )
                cap.close()
                self.init_cameras(self.config.REALSENSE_CAMERAS)
                return self.get_im()

        # Store full resolution cropped images separately
        if self.save_video:
            self.recording_frames.append(full_res_images)

        if self.display_image:
            self.img_queue.put(display_images)
        return images

    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """Move the robot to the goal position with linear interpolation."""
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
        steps = int(timeout * self.hz)
        self._update_currpos()
        path = np.linspace(self.currpos, goal, steps)
        for p in path:
            self._send_pos_command(p)
            time.sleep(1 / self.hz)
        self.nextpos = p
        self._update_currpos()

    def go_to_reset(self, joint_reset=False):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """
        # Change to precision mode for reset        # Use compliance mode for coupled reset
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.1)
        self.robot.update_param(self.config.PRECISION_PARAM)
        # (0.3 + 0.5 s here were settle time for the stiffness switch. On the ROS2 backend
        # update_param is a no-op — the controller's gains are compile-time — so there is
        # nothing to settle. Trimmed to keep resets short during demo recording.)

        # Perform joint reset if needed
        if joint_reset:
            print("JOINT RESET")
            self.robot.joint_reset()
            time.sleep(0.5)

        # Perform Carteasian reset
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)
            self.interpolate_move(reset_pose, timeout=1)
        else:
            reset_pose = self.resetpos.copy()
            self.interpolate_move(reset_pose, timeout=1)

        # Change to compliance mode
        self.robot.update_param(self.config.COMPLIANCE_PARAM)

    def reset(self, joint_reset=False, **kwargs):
        self.last_gripper_act = time.time()
        self.robot.update_param(self.config.COMPLIANCE_PARAM)
        if self.save_video:
            self.save_video_recording()

        self.cycle_count += 1
        if self.joint_reset_cycle!=0 and self.cycle_count % self.joint_reset_cycle == 0:
            self.cycle_count = 0
            joint_reset = True

        self._recover()
        self.go_to_reset(joint_reset=joint_reset)
        self._recover()
        self.curr_path_length = 0
        self._axis_lock_pos = None
        self._axis_lock_rot = None

        self._update_currpos()
        obs = self._get_obs()
        self.terminate = False
        return obs, {"succeed": False}

    def save_video_recording(self):
        try:
            if len(self.recording_frames):
                if not os.path.exists('./videos'):
                    os.makedirs('./videos')
                
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                
                for camera_key in self.recording_frames[0].keys():
                    if self.url == "http://127.0.0.1:5000/":
                        video_path = f'./videos/left_{camera_key}_{timestamp}.mp4'
                    else:
                        video_path = f'./videos/right_{camera_key}_{timestamp}.mp4'
                    
                    # Get the shape of the first frame for this camera
                    first_frame = self.recording_frames[0][camera_key]
                    height, width = first_frame.shape[:2]
                    
                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        10,
                        (width, height),
                    )
                    
                    for frame_dict in self.recording_frames:
                        video_writer.write(frame_dict[camera_key])
                    
                    video_writer.release()
                    print(f"Saved video for camera {camera_key} at {video_path}")
                
            self.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")

    def init_cameras(self, name_serial_dict=None):
        """Init both wrist cameras."""
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        camera_type = getattr(self.config, "CAMERA_TYPE", "realsense")
        capture_cls = ZEDCapture if camera_type == "zed" else RSCapture

        self.cap = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
            cap = VideoCapture(
                capture_cls(name=cam_name, **kwargs)
            )
            self.cap[cam_name] = cap

    def close_cameras(self):
        """Close both wrist cameras."""
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")

    def _recover(self):
        """Internal function to recover the robot from error state."""
        self.robot.recover()

    def _send_pos_command(self, pos: np.ndarray):
        """Internal function to send position command to the robot."""
        self._recover()
        self.robot.send_pos(pos)

    def _send_gripper_command(self, pos: float, mode="binary"):
        """Internal function to send gripper command to the robot."""
        if mode == "binary":
            if (pos <= -0.5) and (self.curr_gripper_pos > 0.85) and (time.time() - self.last_gripper_act > self.gripper_sleep):  # close gripper
                self.robot.close_gripper()
                self.last_gripper_act = time.time()
                time.sleep(self.gripper_sleep)
            elif (pos >= 0.5) and (self.curr_gripper_pos < 0.85) and (time.time() - self.last_gripper_act > self.gripper_sleep):  # open gripper
                self.robot.open_gripper()
                self.last_gripper_act = time.time()
                time.sleep(self.gripper_sleep)
            else:
                return
        elif mode == "continuous":
            raise NotImplementedError("Continuous gripper control is optional")

    def _update_currpos(self):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        ps = self.robot.get_state()
        self.currpos = np.array(ps["pose"])
        self.currvel = np.array(ps["vel"])

        self.currforce = np.array(ps["force"])
        self.currtorque = np.array(ps["torque"])
        self.currjacobian = np.reshape(np.array(ps["jacobian"]), (6, 7))

        self.q = np.array(ps["q"])
        self.dq = np.array(ps["dq"])
        # per-joint external torque, for the torque guard in step(). Absent on backends that
        # do not provide it (e.g. the stock HTTP server) -> None, and the guard is skipped.
        self.currtorque_j = np.array(ps["tau_j"]) if "tau_j" in ps else None

        self.curr_gripper_pos = np.array(ps["gripper_pos"])

    def update_currpos(self):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        ps = self.robot.get_state()
        self.currpos = np.array(ps["pose"])
        self.currvel = np.array(ps["vel"])

        self.currforce = np.array(ps["force"])
        self.currtorque = np.array(ps["torque"])
        self.currjacobian = np.reshape(np.array(ps["jacobian"]), (6, 7))

        self.q = np.array(ps["q"])
        self.dq = np.array(ps["dq"])

        self.curr_gripper_pos = np.array(ps["gripper_pos"])

    def _get_obs(self) -> dict:
        images = self.get_im()
        state_observation = {
            "tcp_pose": self.currpos,
            "tcp_vel": self.currvel,
            "gripper_pose": self.curr_gripper_pos,
            "tcp_force": self.currforce,
            "tcp_torque": self.currtorque,
        }
        return copy.deepcopy(dict(images=images, state=state_observation))

    def close(self):
        # listener is None when pynput was unavailable (headless rig shell) — see __init__.
        if getattr(self, 'listener', None) is not None:
            self.listener.stop()
        if getattr(self, 'robot', None) is not None:
            self.robot.close()
        # Under fake_env (learner) cameras/display were never initialized — nothing to close.
        if getattr(self, 'fake_env', False):
            return
        self.close_cameras()
        if self.display_image:
            self.img_queue.put(None)
            cv2.destroyAllWindows()
            self.displayer.join()
