"""Verify the merged iiwa7+pdz USD parses as one articulation + render the gripper end.

    OMNI_KIT_ACCEPT_EULA=YES python scripts/render_merged_robot.py
"""
import os, argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch, numpy as np  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USD = os.path.join(ROOT, "source/insertion_policy/insertion_policy/assets/robots/kuka_iiwa7_pdz_gripper.usd")
OUT = os.path.join(ROOT, "diagnostics/renders"); os.makedirs(OUT, exist_ok=True)

sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cpu"))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg(), translation=(0, 0, 0))
sim_utils.DomeLightCfg(intensity=2500.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2500.0))
sim_utils.DistantLightCfg(intensity=1500.0).func("/World/Key", sim_utils.DistantLightCfg(intensity=1500.0), orientation=(0.92, 0.38, 0, 0))

robot = Articulation(ArticulationCfg(
    prim_path="/World/Robot",
    spawn=sim_utils.UsdFileCfg(usd_path=USD),
    init_state=ArticulationCfg.InitialStateCfg(),
    actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)},
))

cam_cfg = CameraCfg(prim_path="/World/Cam", height=600, width=800, data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.01, 20.0)))
cam = Camera(cam_cfg)
sim.reset()

print("=== ARTICULATION PARSED ===", flush=True)
print("bodies:", robot.body_names, flush=True)
print("joints:", robot.joint_names, flush=True)
print("num_dof:", robot.num_joints, flush=True)

# Auto-aim at the ACTUAL grafted gripper base body position (arm default pose puts the flange off-axis).
gidx = robot.body_names.index("pdz_gripper_base_link")
gpos = robot.data.body_pos_w[0, gidx].cpu()
tidx = robot.body_names.index("pdz_gripper_tcp")
tpos = robot.data.body_pos_w[0, tidx].cpu()
print(f"gripper_base world pos = {gpos.tolist()}   tcp world pos = {tpos.tolist()}", flush=True)
gp = gpos.unsqueeze(0)
views = {
    "merged_gripper": (gp + torch.tensor([[0.18, -0.18, 0.05]]), gp),
    "merged_full":    (torch.tensor([[1.6, -1.6, 1.2]]),          torch.tensor([[0.0, 0.0, 0.7]])),
}
try:
    import imageio.v2 as imageio
    save = lambda p, a: imageio.imwrite(p, a)
except Exception:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    save = lambda p, a: plt.imsave(p, a)

for name, (eye, tgt) in views.items():
    cam.set_world_poses_from_view(eye, tgt)
    for _ in range(8):
        sim.step()
    cam.update(0.0)
    rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
    p = os.path.join(OUT, name + ".png"); save(p, rgb); print(f"[OK] wrote {p}", flush=True)

simulation_app.close()
