from __future__ import annotations

from collections import OrderedDict
import time

import cv2
import numpy as np

from iiwa_serl.config import CameraConfig


class DummyCameraProvider:
    def __init__(self, configs: list[CameraConfig]):
        self.configs = configs

    def read(self) -> OrderedDict[str, np.ndarray]:
        frames: OrderedDict[str, np.ndarray] = OrderedDict()
        for cfg in self.configs:
            frames[cfg.image_key] = np.zeros((cfg.height, cfg.width, 3), dtype=np.uint8)
        return frames

    def close(self) -> None:
        return None


class RealSenseCameraProvider:
    def __init__(self, configs: list[CameraConfig]):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("pyrealsense2 is required for RealSense capture.") from exc

        self.rs = rs
        self.configs = configs
        self.pipelines = OrderedDict()
        for cfg in configs:
            pipeline = rs.pipeline()
            rs_cfg = rs.config()
            if cfg.serial:
                rs_cfg.enable_device(cfg.serial)
            rs_cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipeline.start(rs_cfg)
            self.pipelines[cfg.image_key] = pipeline

    def _crop_and_resize(self, image: np.ndarray, cfg: CameraConfig) -> np.ndarray:
        if cfg.crop_y is not None:
            image = image[cfg.crop_y[0] : cfg.crop_y[1], :, :]
        if cfg.crop_x is not None:
            image = image[:, cfg.crop_x[0] : cfg.crop_x[1], :]
        return cv2.resize(image, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA)

    def read(self) -> OrderedDict[str, np.ndarray]:
        frames: OrderedDict[str, np.ndarray] = OrderedDict()
        for cfg in self.configs:
            pipeline = self.pipelines[cfg.image_key]
            frame_set = pipeline.wait_for_frames()
            color = frame_set.get_color_frame()
            if not color:
                raise RuntimeError(f"No color frame for camera {cfg.image_key}.")
            image = np.asanyarray(color.get_data())
            frames[cfg.image_key] = self._crop_and_resize(image, cfg)
        return frames

    def close(self) -> None:
        for pipeline in self.pipelines.values():
            pipeline.stop()


class ROSCameraProvider:
    """Reads camera frames from ROS2 sensor_msgs/Image topics (e.g. the mv_launch RealSense
    driver) instead of a direct pyrealsense2 pipeline. One subscription per configured camera;
    the latest frame is cached and returned on read()."""

    def __init__(self, configs: list[CameraConfig]):
        import threading

        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image

        self.configs = configs
        self._Image = Image
        if not rclpy.ok():
            rclpy.init()
        self._node = Node("serl_ros_camera")
        self._lock = threading.Lock()
        self._latest: dict[str, np.ndarray] = {}
        self._latest_sequence: dict[str, int] = {}
        self._last_read_sequence: dict[str, int] = {}
        self.last_read_stale = False

        def _make_cb(key: str, cfg: CameraConfig):
            def _cb(msg):
                img = self._decode(msg)
                with self._lock:
                    self._latest[key] = self._crop_and_resize(img, cfg)
                    self._latest_sequence[key] = self._latest_sequence.get(key, 0) + 1
            return _cb

        for cfg in configs:
            if not cfg.ros_topic:
                raise RuntimeError(f"ROSCameraProvider needs ros_topic set for '{cfg.image_key}'.")
            self._node.create_subscription(
                Image, cfg.ros_topic, _make_cb(cfg.image_key, cfg), qos_profile_sensor_data
            )

        self._spin_thread = threading.Thread(target=rclpy.spin, args=(self._node,), daemon=True)
        self._spin_thread.start()

    def _decode(self, msg) -> np.ndarray:
        # sensor_msgs/Image -> HxWx3 BGR uint8 (match RealSenseCameraProvider output)
        h, w = msg.height, msg.width
        buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, -1)
        enc = msg.encoding.lower()
        if enc == "rgb8":
            return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
        if enc == "bgr8":
            return buf
        if enc in ("mono8",):
            return cv2.cvtColor(buf, cv2.COLOR_GRAY2BGR)
        # fall back: assume 3-channel, no conversion
        return buf[:, :, :3]

    def _crop_and_resize(self, image: np.ndarray, cfg: CameraConfig) -> np.ndarray:
        if cfg.crop_y is not None:
            image = image[cfg.crop_y[0] : cfg.crop_y[1], :, :]
        if cfg.crop_x is not None:
            image = image[:, cfg.crop_x[0] : cfg.crop_x[1], :]
        return cv2.resize(image, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA)

    def read(self) -> OrderedDict[str, np.ndarray]:
        frames: OrderedDict[str, np.ndarray] = OrderedDict()
        any_stale = False
        for cfg in self.configs:
            # Wait briefly for the first frame on startup.
            deadline = time.time() + 5.0
            while time.time() < deadline:
                with self._lock:
                    img = self._latest.get(cfg.image_key)
                    sequence = self._latest_sequence.get(cfg.image_key, 0)
                if img is not None:
                    break
                time.sleep(0.02)
            if img is None:
                raise RuntimeError(
                    f"No frame on ROS topic '{cfg.ros_topic}' for camera '{cfg.image_key}'. "
                    "Is the RealSense launch running?"
                )
            previous_sequence = self._last_read_sequence.get(cfg.image_key)
            if previous_sequence is not None and sequence == previous_sequence:
                any_stale = True
            self._last_read_sequence[cfg.image_key] = sequence
            frames[cfg.image_key] = img
        self.last_read_stale = any_stale
        return frames

    def close(self) -> None:
        try:
            self._node.destroy_node()
        except Exception:
            pass


def make_camera_provider(configs: list[CameraConfig], real: bool):
    if not real:
        return DummyCameraProvider(configs)
    # Route to ROS if any camera specifies a ros_topic (mv_launch RealSense driver).
    if any(cfg.ros_topic for cfg in configs):
        return ROSCameraProvider(configs)
    return RealSenseCameraProvider(configs)
