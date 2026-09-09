"""Contract tests for the RobotClient seam — the get_state() dict shape + method surface
the FR3 Ros2FrankaClient MUST satisfy at the rig. Run offline (no ROS, no robot):

    source franka_env.sh
    python3 hil-serl_src/examples/experiments/franka_plumbers/test_robot_client_contract.py

These pin the exact contract so wiring the real FrankaRobotState -> dict at the rig has a
target to hit, and prove the env consumes that dict correctly (jacobian reshape, wrench,
gripper, pose) without any hardware.
"""

import numpy as np

from franka_env.envs.robot_client import RobotClient, HttpFrankaClient, make_robot_client


# The exact dict shape the HTTP /getstate returns and Ros2FrankaClient.get_state() must
# reproduce. (pose=7 xyz+quat, vel=6, force=3, torque=3, jacobian=42 flat, q/dq=7, gripper=1)
CONTRACT_STATE = {
    "pose": [0.5, -0.03, 0.28, 0.0, 0.0, 0.0, 1.0],
    "vel": [0.0] * 6,
    "force": [1.0, 2.0, 3.0],
    "torque": [0.1, 0.2, 0.3],
    "jacobian": [0.0] * 42,
    "q": [0.0] * 7,
    "dq": [0.0] * 7,
    "gripper_pos": [1.0],
}
REQUIRED_KEYS = set(CONTRACT_STATE)


class MockClient(RobotClient):
    """A fully-implemented fake backend returning the contract dict — stands in for a
    finished Ros2FrankaClient so we can exercise the env end-to-end offline."""

    def __init__(self):
        self.calls = []

    def get_state(self):
        return {k: list(v) for k, v in CONTRACT_STATE.items()}

    def send_pos(self, arr):
        self.calls.append(("send_pos", np.asarray(arr).shape))

    def recover(self):
        self.calls.append(("recover", None))

    def update_param(self, params):
        self.calls.append(("update_param", params))

    def joint_reset(self):
        self.calls.append(("joint_reset", None))

    def open_gripper(self):
        self.calls.append(("open_gripper", None))

    def close_gripper(self):
        self.calls.append(("close_gripper", None))

    def close_gripper_slow(self):
        self.calls.append(("close_gripper_slow", None))

    def set_load(self, params):
        self.calls.append(("set_load", params))


def test_contract_keys():
    st = MockClient().get_state()
    assert set(st) == REQUIRED_KEYS, f"missing/extra keys: {set(st) ^ REQUIRED_KEYS}"
    assert len(st["pose"]) == 7 and len(st["vel"]) == 6
    assert len(st["force"]) == 3 and len(st["torque"]) == 3
    assert len(st["jacobian"]) == 42
    assert len(st["q"]) == 7 and len(st["dq"]) == 7
    assert len(st["gripper_pos"]) == 1
    print("  OK  get_state() contract keys + shapes")


def test_env_consumes_contract():
    """Build the plumbers env (fake_env, so no real robot client), then swap in MockClient
    and prove _update_currpos + _get_obs parse the contract dict without error."""
    from experiments.mappings import CONFIG_MAPPING

    env = CONFIG_MAPPING["franka_plumbers_insert0"]().get_environment(fake_env=True)
    base = env
    while hasattr(base, "env"):
        base = base.env
    base = base.unwrapped

    base.robot = MockClient()
    base._update_currpos()  # must parse pose/vel/force/torque/jacobian/q/dq/gripper
    assert base.currpos.shape == (7,)
    assert base.currjacobian.shape == (6, 7), "jacobian must reshape to (6,7)"
    assert base.currforce.shape == (3,) and base.currtorque.shape == (3,)
    assert base.curr_gripper_pos.shape == (1,)
    # (images come from cameras, not the robot client — not part of this contract.)
    print("  OK  env consumes contract dict (jacobian reshape, wrench, pose, gripper)")


def test_http_default_and_method_surface():
    class Cfg:
        ROBOT_CLIENT = "http"
    c = make_robot_client(Cfg, "http://x/")
    assert isinstance(c, HttpFrankaClient)
    # every RobotClient method the env/wrapper calls must exist
    for m in ("get_state", "send_pos", "recover", "update_param", "joint_reset",
              "open_gripper", "close_gripper", "close_gripper_slow", "set_load"):
        assert callable(getattr(c, m)), f"HttpFrankaClient missing {m}"
    print("  OK  http default + full method surface")


def test_send_pos_accepts_interpolated_poses():
    """REGRESSION: Ros2FrankaClient.send_pos must accept the poses interpolate_move actually
    produces. That path LERPs raw quaternion components (np.linspace), so intermediates are
    sub-unit (|q|~0.9962 for a 20 deg change). A strict unit-norm check rejected them and
    would have aborted every orientation-changing reset/regrasp on the rig — and the demo
    test could not catch it because it MOCKS send_pos. Validate the real method here."""
    import types
    from franka_env.envs.robot_client import Ros2FrankaClient
    from franka_env.utils.rotations import euler_2_quat

    published = []

    class _Vec3:
        x = y = z = 0.0

    class _Quat:
        x = y = z = 0.0
        w = 1.0

    class _Msg:
        """Stand-in for geometry_msgs/Pose (what the controller subscribes to)."""
        def __init__(self):
            self.position = _Vec3()
            self.orientation = _Quat()

    def _capture(m):
        published.append([m.position.x, m.position.y, m.position.z,
                          m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w])

    c = object.__new__(Ros2FrankaClient)  # no ROS needed for the validation logic
    c._Pose = _Msg
    c._cmd_pub = types.SimpleNamespace(publish=_capture)

    start = np.concatenate([[0.5, 0, 0.3], euler_2_quat(np.array([np.pi, 0, 0]))])
    goal = np.concatenate([[0.5, 0, 0.3], euler_2_quat(np.array([np.pi, 0, np.deg2rad(20)]))])
    for p in np.linspace(start, goal, 10):   # exactly what interpolate_move builds
        c.send_pos(p)
    assert len(published) == 10, f"send_pos rejected interpolated poses ({len(published)}/10)"
    for d in published:
        n = float(np.linalg.norm(np.asarray(d[3:])))
        assert abs(n - 1.0) < 1e-9, f"published a non-unit quaternion (|q|={n})"

    # genuinely broken input must STILL be refused
    for bad in (np.zeros(7), np.array([0, 0, 0, np.nan, 0, 0, 1.0]), np.zeros(6)):
        try:
            c.send_pos(bad)
            raise AssertionError(f"send_pos accepted invalid input {bad}")
        except ValueError:
            pass
    print("  OK  send_pos accepts LERP-interpolated poses, normalizes, rejects bad input")


if __name__ == "__main__":
    test_contract_keys()
    test_http_default_and_method_surface()
    test_env_consumes_contract()
    test_send_pos_accepts_interpolated_poses()
    print("ALL CONTRACT TESTS PASS")
