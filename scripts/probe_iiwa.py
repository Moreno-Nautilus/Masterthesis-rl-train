"""Probe the injected iiwa7+gripper articulation: body/DOF order, and FK of candidate reset postures.

Loads the robot alone (no task) to confirm the assumptions Forge's env hardcodes:
  * body_names[0] is the articulation root (so jacobian index = body_idx-1 holds),
  * the 7 arm DOFs come first (indices 0:6) and the 2 finger DOFs last (7:8),
  * gripper_tcp / left_finger_link / right_finger_link / force_sensor bodies exist,
and prints the gripper_tcp world pose for several candidate arm configs so we can pick a
``reset_joints`` seed whose TCP lands above the socket (~x=0.6, z~0.13) pointing down.

    OMNI_KIT_ACCEPT_EULA=YES python scripts/probe_iiwa.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext

USD = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "source",
        "insertion_policy",
        "insertion_policy",
        "assets",
        "robots",
        "kuka_iiwa7_y_gripper.usd",
    )
)

REPORT = []


def emit(s):
    REPORT.append(str(s))


def main():
    # use_fabric=False: this Isaac build's physx.fabric plugin fails to acquire in a bare
    # SimulationContext (version mismatch). No camera here, so the USD pose path is fine.
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device=args_cli.device, use_fabric=False))
    sim.set_camera_view([2.0, 2.0, 2.0], [0.0, 0.0, 0.5])

    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=USD),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-7]"], stiffness=2000.0, damping=200.0
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["left_finger_joint", "right_finger_joint"],
                stiffness=5000.0,
                damping=200.0,
            ),
        },
    )
    robot = Articulation(robot_cfg)
    sim.reset()

    emit(f"num_bodies={robot.num_bodies}  num_joints={robot.num_joints}")
    emit(f"body_names (index order): {list(enumerate(robot.body_names))}")
    emit(f"joint_names (DOF order):  {list(enumerate(robot.joint_names))}")
    for name in ("force_sensor", "gripper_tcp", "left_finger_link", "right_finger_link"):
        try:
            emit(f"  body '{name}' -> index {robot.body_names.index(name)}")
        except ValueError:
            emit(f"  body '{name}' -> MISSING")
    lo = robot.data.joint_pos_limits[0, :, 0].tolist()
    hi = robot.data.joint_pos_limits[0, :, 1].tolist()
    emit("joint limits (rad):")
    for i, n in enumerate(robot.joint_names):
        emit(f"  {i} {n:20s} [{lo[i]:+.3f}, {hi[i]:+.3f}]")

    d2r = math.pi / 180.0
    tcp_idx = robot.body_names.index("gripper_tcp")
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=sim.device)
    down_vec = torch.tensor([0.0, 0.0, -1.0], device=sim.device)
    target = torch.tensor([0.60, 0.0, 0.15], device=sim.device)  # seed goal: above the socket
    import isaacsim.core.utils.torch as tu

    # ---- IK reachability across socket distances: for target (x,0,z) pointing DOWN, DLS-IK from a
    # folded seed and report convergence + how far each joint sits from its limit. Picks the socket x
    # + reset posture where the iiwa points down COMFORTABLY (max joint-limit margin) so the env-side
    # reset IK converges reliably (no near-limit stalls) -> no reset hang / base launch.
    from isaaclab.utils.math import axis_angle_from_quat, quat_from_euler_xyz, quat_mul, quat_conjugate

    lo_t = robot.data.joint_pos_limits[0, :7, 0]
    hi_t = robot.data.joint_pos_limits[0, :7, 1]
    down_quat = quat_from_euler_xyz(
        torch.tensor([math.pi], device=sim.device), torch.tensor([0.0], device=sim.device),
        torch.tensor([0.0], device=sim.device))[0]

    def ik(target_xyz, seed_deg, iters=120, tol=1.5e-3):
        q = torch.zeros((1, robot.num_joints), device=sim.device)
        q[0, :7] = torch.tensor([x * d2r for x in seed_deg], device=sim.device)
        tgt = torch.tensor(target_xyz, device=sim.device)
        perr = rerr = 9.9
        for _ in range(iters):
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            robot.set_joint_position_target(q)
            robot.write_data_to_sim(); sim.step(); robot.update(sim.get_physics_dt())
            p = robot.data.body_pos_w[0, tcp_idx]
            quat = robot.data.body_quat_w[0, tcp_idx]
            jac = robot.root_physx_view.get_jacobians()[0, tcp_idx - 1, 0:6, 0:7]
            pos_e = tgt - p
            q_e = quat_mul(down_quat, quat_conjugate(quat))
            aa_e = axis_angle_from_quat(q_e.unsqueeze(0))[0]
            perr = float(pos_e.norm()); rerr = float(aa_e.norm())
            if perr < tol and rerr < tol:
                break
            dpose = torch.cat([pos_e, aa_e]).unsqueeze(-1)
            JT = jac.T
            dq = (JT @ torch.inverse(jac @ JT + (0.1**2) * torch.eye(6, device=sim.device)) @ dpose).squeeze(-1)
            q[0, :7] = torch.clamp(q[0, :7] + dq, lo_t, hi_t)
        margin = torch.minimum(q[0, :7] - lo_t, hi_t - q[0, :7]).min().item()
        return q[0, :7].clone(), perr, rerr, margin

    emit("\nIK reachability vs socket distance (target z=0.14, pointing down), seed [0,100,0,-45,0,100,0]:")
    best = None
    for x in (0.62, 0.67, 0.72, 0.77, 0.82, 0.87):
        qj, pe, re, mg = ik([x, 0.0, 0.14], [0, 100, 0, -45, 0, 100, 0])
        ok = pe < 2e-3 and re < 2e-3
        deg = [round(float(v) / d2r, 1) for v in qj]
        emit(f"  x={x:.2f}  conv={ok}  pos_err={pe*1000:5.1f}mm rot_err={re:.3f}  "
             f"min_margin={mg/d2r:5.1f}deg  q_deg={deg}")
        if ok and (best is None or mg > best[1]):
            best = (x, mg, deg, qj)
    if best is not None:
        emit(f"\nBEST: socket x={best[0]:.2f}, min joint margin={best[1]/d2r:.1f}deg, reset q_deg={best[2]}")
        emit(f"      reset_joints_rad = {[round(float(v),4) for v in best[3].tolist()]}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        emit("FAILED\n" + traceback.format_exc())
    finally:
        with open("/tmp/probe_iiwa_report.txt", "w") as f:
            f.write("\n".join(REPORT) + "\n")
        simulation_app.close()
