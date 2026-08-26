from __future__ import annotations

from collections import OrderedDict

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


def make_camera_provider(configs: list[CameraConfig], real: bool):
    if not real:
        return DummyCameraProvider(configs)
    return RealSenseCameraProvider(configs)
