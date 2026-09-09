"""Rig preflight self-check — run BEFORE enabling any autonomous motion.

Verifies the pieces the actor depends on and prints a clear PASS/FAIL per item:
  1. ROS2 topics/actions the Ros2FrankaClient needs are present.
  2. PS4 reader is healthy (heartbeat) and R1/buttons read.
  3. ZED camera opens and yields a frame.

Degrades gracefully off-rig: anything unavailable is reported as SKIP/FAIL, never crashes.
This is a CHECK, not a fix — it does not command the robot.

    source franka_env.sh
    python3 hil-serl_src/examples/experiments/franka_plumbers/preflight.py
"""

import os
import sys

from franka_env.envs.robot_client import Ros2FrankaClient

# Topics/actions the ROS backend needs (mirrors Ros2FrankaClient's names). Confirm/adjust
# with `ros2 topic list` / `ros2 action list` (see FRANKA_RIG_CHECKLIST.md).
REQUIRED_TOPICS = [
    Ros2FrankaClient.CMD_POSE_TOPIC,
    Ros2FrankaClient.ROBOT_STATE_TOPIC,
]
REQUIRED_ACTIONS = [
    Ros2FrankaClient.GRIPPER_MOVE_ACTION,
    Ros2FrankaClient.GRIPPER_GRASP_ACTION,
    Ros2FrankaClient.ERROR_RECOVERY_ACTION,
]

GREEN, RED, YEL, RST = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def _p(status, name, detail=""):
    """Print one result. Returns PASS->True, FAIL->False, SKIP->None (SKIP is NOT a pass and
    must not be conflated with one — callers propagate None separately)."""
    color = {"PASS": GREEN, "FAIL": RED, "SKIP": YEL}[status]
    print(f"  [{color}{status:4}{RST}] {name}" + (f" — {detail}" if detail else ""))
    return True if status == "PASS" else (None if status == "SKIP" else False)


def check_backend_wired():
    """The actor CANNOT run while Ros2FrankaClient methods are still NotImplementedError
    stubs. Preflight must catch that instead of reporting green."""
    print("ROS backend source screen (NOT live backend validation):")
    print("       PASS means no obvious stub was found. Delegated code, state frames/units,")
    print("       and actual controller behavior still require rig validation.")
    import ast
    import inspect
    import textwrap

    def _stub_state(fn):
        """-> (is_stub, detail). Fail CLOSED: if we cannot inspect it, we cannot certify it."""
        if fn is None:
            return True, "method missing entirely"
        try:
            src = textwrap.dedent(inspect.getsource(fn))
        except Exception as e:
            # Swallowing this and reporting PASS is exactly backwards — an uninspectable
            # method is UNVERIFIED, so treat it as not-wired.
            return True, f"could not inspect source ({e!r}) — cannot certify as wired"
        try:
            tree = ast.parse(src)
        except Exception as e:
            return True, f"could not parse source ({e!r})"
        # Any `raise NotImplementedError...` anywhere in the body means still a stub.
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise):
                exc = node.exc
                target = exc.func if isinstance(exc, ast.Call) else exc
                nm = getattr(target, "id", None) or getattr(target, "attr", None)
                if nm == "NotImplementedError":
                    return True, "still a TODO(rig) stub — wire it before enabling motion"
        # A body that is only a docstring/comments/pass is also not an implementation.
        fnodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if fnodes:
            body = [n for n in fnodes[0].body
                    if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
            if not body or all(isinstance(n, ast.Pass) for n in body):
                return True, "body is `pass` only — not implemented"
        return False, ""

    ok = True
    for name in ("get_state", "update_param", "joint_reset", "open_gripper",
                 "close_gripper", "close_gripper_slow", "set_load"):
        stubbed, detail = _stub_state(getattr(Ros2FrankaClient, name, None))
        ok &= bool(_p("FAIL" if stubbed else "PASS", f"Ros2FrankaClient.{name}()", detail))
    return ok


def check_ros():
    print("ROS2 interfaces:")
    try:
        import rclpy
        from rclpy.node import Node
    except Exception as e:
        _p("SKIP", "rclpy import", f"{e!r} (not on the robot PC?)")
        return None
    ok = True
    try:
        if not rclpy.ok():
            rclpy.init()
        node = Node("franka_preflight")
        # settle for discovery
        import time
        end = time.time() + 2.0
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
        topic_types = {n: ts for n, ts in node.get_topic_names_and_types()}
        for t in REQUIRED_TOPICS:
            present = t in topic_types
            detail = "" if present else "NOT FOUND (check ros2 topic list)"
            if present and t == Ros2FrankaClient.CMD_POSE_TOPIC:
                # verify the TYPE too — a name match with the wrong msg type still breaks us
                types = topic_types[t]
                # cartesian_impedance_control v0.1.8 subscribes to a geometry_msgs/Pose
                # on reference_pose (VERIFIED at the rig 2026-09-09) — NOT the
                # Float64MultiArray the pre-rig docs assumed.
                if not any("geometry_msgs/msg/Pose" in x for x in types):
                    present, detail = False, f"wrong type {types} (expected geometry_msgs/msg/Pose)"
            ok &= bool(_p("PASS" if present else "FAIL", f"topic {t}", detail))
        try:
            from rclpy.action import get_action_names_and_types
            actions = {n for n, _ in get_action_names_and_types(node)}
        except Exception:
            actions = set()
        for a in REQUIRED_ACTIONS:
            present = a in actions   # EXACT match (substring matching gave false PASSes)
            ok &= bool(_p("PASS" if present else "FAIL", f"action {a}",
                          "" if present else "NOT FOUND (check ros2 action list)"))
        node.destroy_node()
    except Exception as e:
        _p("FAIL", "ROS discovery", repr(e))
        ok = False
    return ok


def check_ps4():
    print("PS4 controller:")
    if os.environ.get("TELEOP_DEVICE", "").lower() != "ps4":
        os.environ["TELEOP_DEVICE"] = "ps4"
    try:
        from franka_env.spacemouse.ps4_expert import PS4Expert
        expert = PS4Expert()  # raises loudly if the reader can't start / no controller
    except Exception as e:
        return _p("FAIL", "PS4Expert start", repr(e))
    try:
        import time
        time.sleep(0.3)
        # Health comes from the READER's own liveness, not from "get_action() returned
        # something" — zeros are also what a dead/stale reader returns (fail-safe), so
        # treating any return as healthy would report green on a dead controller.
        h = expert.health()
        healthy = h["alive"] and not h["stale"] and not h["init_error"] and not h["runtime_error"]
        ok = _p("PASS" if healthy else "FAIL", "PS4 reader alive+fresh",
                f"alive={h['alive']} stale={h['stale']} age={h['age_s']:.2f}s "
                f"err={h['init_error'] or h['runtime_error'] or '-'}")
        a, btns = expert.get_action()
        _p("PASS", "PS4 action readable", f"action={list(a.round(2))} buttons={btns}")
        _p("PASS", "R1 deadman readable", f"held={expert.deadman_held()}")
        print("       (hold R1 + move sticks / press X·Triangle·Circle to verify LIVE — a")
        print("        zero action here only proves the reader runs, not that axes work)")
        return ok
    finally:
        expert.close()


def check_zed():
    print("ZED camera:")
    try:
        import pyzed.sl as sl  # noqa: F401
    except Exception as e:
        return _p("SKIP", "pyzed import", f"{e!r} (not on the robot PC?)")
    try:
        from franka_env.camera.zed_capture import ZEDCapture
        cam = ZEDCapture(name="wrist")
        ok, frame = cam.read()
        cam.close()
        if ok and frame is not None:
            return _p("PASS", "ZED frame", f"shape={frame.shape}")
        return _p("FAIL", "ZED frame", "no frame returned")
    except Exception as e:
        return _p("FAIL", "ZED open", repr(e))


def main():
    print("=== Franka plumbers preflight ===")
    results = {
        "BACKEND": check_backend_wired(),
        "ROS": check_ros(),
        "PS4": check_ps4(),
        "ZED": check_zed(),
    }
    print("=== summary ===")
    for k, v in results.items():
        label = "PASS" if v is True else ("SKIP" if v is None else "FAIL")
        print(f"  {k}: {label}")

    hard_fail = any(v is False for v in results.values())
    skipped = [k for k, v in results.items() if v is None]
    if hard_fail:
        print(f"{RED}PREFLIGHT FAILED — do NOT enable autonomous motion until resolved.{RST}")
        sys.exit(1)
    if skipped:
        # A SKIP is NOT a pass: off-rig it's expected, ON the rig it means we could not
        # verify something the actor needs. Exit non-zero so a rig run can't be waved through.
        print(f"{YEL}Preflight incomplete — SKIPPED: {', '.join(skipped)}.{RST}")
        print(f"{YEL}Off-rig this is expected. ON THE RIG, treat a SKIP as a FAIL.{RST}")
        sys.exit(2)
    print(f"{GREEN}Static/interface preflight passed — NOT clearance for autonomous motion.{RST}")
    print("Verify live state conversion, compliance, E-stop, teleop, and full reset clearance "
          "using FRANKA_RIG_CHECKLIST.md before demos/training.")


if __name__ == "__main__":
    main()
