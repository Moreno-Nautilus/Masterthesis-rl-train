"""Render the converted standalone PDZ gripper USD to a PNG for visual verification.

    OMNI_KIT_ACCEPT_EULA=YES python scripts/render_pdz_gripper.py
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

import torch  # noqa: E402
import numpy as np  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USD = os.path.join(ROOT, "source/insertion_policy/insertion_policy/assets/robots/pdz_gripper.usd")
OUT = os.path.join(ROOT, "diagnostics/renders")
os.makedirs(OUT, exist_ok=True)

sim = SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cpu"))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg(), translation=(0, 0, -0.5))
sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)).func("/World/Light", sim_utils.DomeLightCfg(intensity=2500.0))
sim_utils.DistantLightCfg(intensity=2000.0).func("/World/Key", sim_utils.DistantLightCfg(intensity=2000.0), orientation=(0.92, 0.38, 0, 0))

# Spawn the gripper (flange at origin, +Z up = out through the gripper).
g = sim_utils.UsdFileCfg(usd_path=USD)
g.func("/World/Gripper", g)

cam_cfg = CameraCfg(
    prim_path="/World/Cam", height=600, width=800, data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.01, 10.0)),
)
cam = Camera(cam_cfg)
sim.reset()

# Two views: a 3/4 view and a front (finger-opening) view.
views = {
    "pdz_gripper_iso":   (torch.tensor([[0.22, -0.22, 0.18]]), torch.tensor([[0.0, 0.0, 0.09]])),
    "pdz_gripper_front": (torch.tensor([[0.0, -0.30, 0.10]]),  torch.tensor([[0.0, 0.0, 0.09]])),
}
try:
    import imageio.v2 as imageio
    save = lambda p, a: imageio.imwrite(p, a)
except Exception:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    def save(p, a):
        plt.imsave(p, a)

for name, (eye, tgt) in views.items():
    cam.set_world_poses_from_view(eye, tgt)
    for _ in range(6):
        sim.step()
    cam.update(0.0)
    rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
    out = os.path.join(OUT, name + ".png")
    save(out, rgb)
    print(f"[OK] wrote {out}  shape={rgb.shape}", flush=True)

simulation_app.close()
