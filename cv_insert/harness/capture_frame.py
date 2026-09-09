#!/usr/bin/env python3
"""Grab ONE synced wrist (RealSense_2) RGB-D + flange pose and save an .npz the detector eats.

Saves exactly the keys the offline harness/eval_detector + detector expect:
  color_<tag>.png              (rgb8 -> BGR png, like the fixture captures)
  frame_<tag>.npz: depth_m (float32, m), depth_raw (uint16), K_depth (3x3),
                   T_base_flange (4x4), encoding, stamp_*, feature tag.
Optionally also grabs the ZED (zed2i_3) RGB + registered depth.

Run (rig, ROS + ws sourced):
  python3 capture_frame.py --feature pb_top --out ~/pbtop_caps
  python3 capture_frame.py --feature pb_top --out ~/pbtop_caps --zed     # also ZED
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped

try:
    import cv2
except Exception:
    cv2 = None

WRIST_COLOR = "/realsense_2/camera/color/image_raw"
WRIST_DEPTH = "/realsense_2/camera/aligned_depth_to_color/image_raw"
WRIST_INFO = "/realsense_2/camera/aligned_depth_to_color/camera_info"
EE_POSE = "/right/ee_pose"

ZED_COLOR = "/zed2i_3/zed_node/rgb/color/rect/image"
ZED_DEPTH = "/zed2i_3/zed_node/depth/depth_registered"
ZED_INFO = "/zed2i_3/zed_node/depth/camera_info"


def img_to_np(msg: Image) -> np.ndarray:
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    enc = msg.encoding.lower()
    if enc in ("rgb8", "bgr8"):
        a = buf.reshape(msg.height, msg.width, 3)
        return a[:, :, ::-1] if enc == "rgb8" else a  # -> BGR for cv2.imwrite
    if enc in ("16uc1", "mono16"):
        return buf.view(np.uint16).reshape(msg.height, msg.width)
    if enc == "32fc1":
        return buf.view(np.float32).reshape(msg.height, msg.width)
    raise ValueError(f"unhandled encoding {msg.encoding}")


def pose_to_T(msg: PoseStamped) -> tuple[np.ndarray, str]:
    p, q = msg.pose.position, msg.pose.orientation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [p.x, p.y, p.z]
    return T, msg.header.frame_id


class Grabber(Node):
    def __init__(self, feature, out, do_zed):
        super().__init__("cv_insert_capture")
        self.feature, self.out, self.do_zed = feature, out, do_zed
        os.makedirs(out, exist_ok=True)
        self.pose = None
        self.create_subscription(PoseStamped, EE_POSE, self._pose, 10)
        subs = [Subscriber(self, Image, WRIST_COLOR, qos_profile=qos_profile_sensor_data),
                Subscriber(self, Image, WRIST_DEPTH, qos_profile=qos_profile_sensor_data),
                Subscriber(self, CameraInfo, WRIST_INFO, qos_profile=qos_profile_sensor_data)]
        self.sync = ApproximateTimeSynchronizer(subs, queue_size=10, slop=0.05)
        self.sync.registerCallback(self._wrist)
        self.zed = {}
        if do_zed:
            zsubs = [Subscriber(self, Image, ZED_COLOR, qos_profile=qos_profile_sensor_data),
                     Subscriber(self, Image, ZED_DEPTH, qos_profile=qos_profile_sensor_data),
                     Subscriber(self, CameraInfo, ZED_INFO, qos_profile=qos_profile_sensor_data)]
            self.zsync = ApproximateTimeSynchronizer(zsubs, queue_size=10, slop=0.08)
            self.zsync.registerCallback(self._zed)
        self.done = False
        self.got_wrist = False

    def _pose(self, msg):
        self.pose = msg

    def _wrist(self, c, d, info):
        if self.done:
            return
        rgb = img_to_np(c)
        depth_raw = img_to_np(d).astype(np.uint16)
        depth_m = depth_raw.astype(np.float32) / 1000.0  # 16UC1 mm -> m
        K = np.array(info.k, float).reshape(3, 3)
        extra = {}
        if self.pose is not None:
            T_bf, base_frame = pose_to_T(self.pose)
            extra = dict(T_base_flange=T_bf, base_frame=base_frame)
            pose_note = f" base_frame={base_frame}"
        else:
            pose_note = " (NO ee_pose — bringup off; image-only, no base geometry)"
        tag = f"{self.feature}_{time.strftime('%H%M%S')}"
        cv2.imwrite(os.path.join(self.out, f"color_{tag}.png"), rgb)
        np.savez(os.path.join(self.out, f"frame_{tag}.npz"),
                 depth_m=depth_m, depth_raw=depth_raw, K_depth=K,
                 encoding=c.encoding, feature=self.feature, stamp=time.time(), **extra)
        self.get_logger().info(f"SAVED wrist frame_{tag}.npz  depth med="
                               f"{np.median(depth_m[depth_m>0]):.3f}m{pose_note}")
        self.got_wrist = True
        # ZED is best-effort: if it hasn't synced within a short grace window, stop anyway.
        if not self.do_zed:
            self.done = True

    def _zed(self, c, d, info):
        if self.done or self.pose is None:
            return
        rgb = img_to_np(c)
        depth = img_to_np(d)
        depth_m = depth.astype(np.float32) if depth.dtype == np.float32 else depth.astype(np.float32) / 1000.0
        K = np.array(info.k, float).reshape(3, 3)
        tag = f"{self.feature}_zed_{time.strftime('%H%M%S')}"
        cv2.imwrite(os.path.join(self.out, f"color_{tag}.png"), rgb)
        np.savez(os.path.join(self.out, f"frame_{tag}.npz"),
                 depth_m=depth_m, K_depth=K, camera="zed2i_3",
                 feature=self.feature, stamp=time.time())
        self.get_logger().info(f"SAVED zed frame_{tag}.npz")
        self.done = True


def main():
    if cv2 is None:
        raise SystemExit("cv2 not available")
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zed", action="store_true")
    a = ap.parse_args()
    rclpy.init()
    node = Grabber(a.feature, os.path.expanduser(a.out), a.zed)
    t0 = time.time()
    while rclpy.ok() and not node.done and time.time() - t0 < 12:
        rclpy.spin_once(node, timeout_sec=0.1)
        # if we've got the wrist frame and ZED hasn't synced within 3s, stop (ZED best-effort)
        if node.got_wrist and (not node.do_zed or time.time() - t0 > 3):
            break
    if not node.got_wrist:
        node.get_logger().error("no wrist frame — check topics/bringup")
    elif a.zed and not any("zed" in f for f in os.listdir(os.path.expanduser(a.out))):
        node.get_logger().warn("wrist saved, but ZED never synced (topic/QoS) — wrist-only")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
