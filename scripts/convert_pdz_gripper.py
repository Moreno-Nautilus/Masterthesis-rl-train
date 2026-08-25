"""Convert the PDZ Slim gripper URDF -> USD (standalone) for inspection before the iiwa7 merge.

Keeps fixed-joint frames (pdz_gripper_tcp + camera_*), does NOT merge fixed joints, fixes the base for a
clean static render. Prints the resulting body/joint tree so we can verify links/joints/frames.

    OMNI_KIT_ACCEPT_EULA=YES python scripts/convert_pdz_gripper.py
"""
import os
from isaaclab.app import AppLauncher

import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = os.path.join(ROOT, "source/insertion_policy/insertion_policy/assets/robots/pdz_gripper_description/urdf/pdz_gripper_clean.urdf")
OUT_DIR = os.path.join(ROOT, "source/insertion_policy/insertion_policy/assets/robots")

cfg = UrdfConverterCfg(
    asset_path=URDF,
    usd_dir=OUT_DIR,
    usd_file_name="pdz_gripper.usd",
    fix_base=True,               # standalone: pin the base for a static render/inspection
    merge_fixed_joints=False,    # KEEP the tcp + camera fixed-joint frames
    force_usd_conversion=True,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        target_type="position",
        drive_type="force",
        # PhysX PD for the belt-driven finger prismatic (mirror the old Y-gripper's 2000/100 position PD).
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=2000.0, damping=100.0),
    ),
)
conv = UrdfConverter(cfg)
print(f"[convert] wrote USD: {conv.usd_path}", flush=True)

# Print the body/joint tree for verification.
from pxr import Usd, UsdPhysics  # noqa: E402
stage = Usd.Stage.Open(conv.usd_path)
print("=== PRIM TREE (links / joints) ===", flush=True)
for prim in stage.Traverse():
    t = prim.GetTypeName()
    if t in ("Xform",) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
        print(f"  LINK  {prim.GetPath()}")
    if prim.IsA(UsdPhysics.Joint) or "Joint" in str(t):
        print(f"  JOINT {prim.GetPath()}  ({t})")
simulation_app.close()
