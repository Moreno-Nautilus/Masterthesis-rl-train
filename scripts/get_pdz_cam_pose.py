"""Print the D405 depth-optical frame pose expressed in the pdz_gripper_tcp frame (for wrist_cam_offset_*)."""
import os, argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser)
a = parser.parse_args(); a.headless = True
app = AppLauncher(a).app
from pxr import Usd, UsdGeom  # noqa: E402
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDZ = os.path.join(ROOT, "source/insertion_policy/insertion_policy/assets/robots/pdz_gripper.usd")
s = Usd.Stage.Open(PDZ); xc = UsdGeom.XformCache()
tcp = s.GetPrimAtPath("/pdz_gripper/pdz_gripper_tcp")
for camname in ("camera_depth_optical_frame", "camera_color_optical_frame", "camera_link"):
    cam = s.GetPrimAtPath(f"/pdz_gripper/{camname}")
    if not cam or not cam.IsValid():
        print(f"{camname}: MISSING"); continue
    rel = xc.GetLocalToWorldTransform(cam) * xc.GetLocalToWorldTransform(tcp).GetInverse()
    p = rel.ExtractTranslation(); q = rel.ExtractRotationQuat()
    im = q.GetImaginary()
    print(f"{camname}: pos_in_tcp=({p[0]:.5f}, {p[1]:.5f}, {p[2]:.5f})  "
          f"quat_wxyz=({q.GetReal():.5f}, {im[0]:.5f}, {im[1]:.5f}, {im[2]:.5f})", flush=True)
app.close()
