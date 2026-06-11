"""Minimal physics sanity check for the converted part USDs.

Fixes cooling_base (kinematic) on the ground and drops cooling_screw shaft-down over one of
the two sockets at x=+30mm. Steps the sim and logs the screw's height so we can confirm SDF
collision behaves (screw is caught / seats rather than passing through). Writes a report to
/tmp/part_physics_report.txt (Isaac Sim swallows stdout).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=400)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg

ASSETS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "source", "insertion_policy", "insertion_policy", "assets")
)

# cooling_base centered-frame: 35mm thick -> place center at z=0.0175 so it rests on z=0.
BASE_Z = 0.0175
PLATFORM_TOP = 0.035  # top surface of base in world (centered + BASE_Z)
SOCKET_X = 0.030      # one of the two sockets at x = +/-30mm
# screw is 35mm tall, shaft (lower 17.5mm) points -z by default. Start just above the socket.
SCREW_START_Z = PLATFORM_TOP + 0.0175 + 0.004  # head-center height with shaft tip ~4mm above hole


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device))
    sim.set_camera_view([0.3, 0.3, 0.2], [0.0, 0.0, 0.02])

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2000.0))

    base = RigidObject(
        RigidObjectCfg(
            prim_path="/World/Base",
            spawn=sim_utils.UsdFileCfg(
                usd_path=os.path.join(ASSETS, "cooling_base.usd"),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),  # immovable but collidable
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, BASE_Z)),
        )
    )
    screw = RigidObject(
        RigidObjectCfg(
            prim_path="/World/Screw",
            spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(ASSETS, "cooling_screw.usd")),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(SOCKET_X, 0.0, SCREW_START_Z)),
        )
    )

    sim.reset()
    dt = sim.get_physics_dt()
    report = [f"start screw z={SCREW_START_Z:.4f}  socket_x={SOCKET_X}  platform_top={PLATFORM_TOP}"]
    for i in range(args_cli.steps):
        sim.step()
        screw.update(dt)
        if i % 40 == 0 or i == args_cli.steps - 1:
            p = screw.data.root_pos_w[0]
            report.append(f"step {i:4d}: screw pos=({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})")

    final_z = float(screw.data.root_pos_w[0, 2])
    # If SDF collision works, the shaft enters but the head (20mm > 14mm socket) is caught near the
    # platform top; the screw should NOT fall to the ground (head-center would end ~>= platform top).
    caught = final_z > 0.015
    report.append(f"RESULT: final screw z={final_z:.4f}  -> {'CAUGHT (collision OK)' if caught else 'FELL THROUGH (!)'}")
    with open("/tmp/part_physics_report.txt", "w") as f:
        f.write("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
