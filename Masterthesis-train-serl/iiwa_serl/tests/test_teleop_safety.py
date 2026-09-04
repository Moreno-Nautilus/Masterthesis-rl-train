from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from iiwa_serl.config import IiwaInsertionConfig
from iiwa_serl.envs.base_real_env import BaseIiwaSERLEnv
from iiwa_serl.robot.api import RobotServerClient
from iiwa_serl.robot_servers.iiwa_server import create_app
from iiwa_serl.teleop.pose_target import TeleopPoseTarget
from iiwa_serl.teleop import ps4_teleop_provider as ps4
from iiwa_serl.utils.transformations import clip_pose7_relative


class _FakeJoystick:
    def __init__(self):
        self.axes = [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]
        self.buttons = [0] * 6

    def get_axis(self, index):
        return self.axes[index]

    def get_button(self, index):
        return self.buttons[index]

    def get_numbuttons(self):
        return len(self.buttons)

    def get_numhats(self):
        return 1

    def get_hat(self, _index):
        return (0, 0)


def _provider_without_sdl(monkeypatch):
    provider = ps4.PS4TeleopProvider.__new__(ps4.PS4TeleopProvider)
    provider._cfg = ps4._DS4_DEFAULTS.copy()
    provider._js = _FakeJoystick()
    provider._prev_buttons = [False] * provider._js.get_numbuttons()
    provider._success_flag = False
    provider._failure_flag = False
    provider._server_url = None
    provider._ff_enabled = False
    provider._last_ff_time = 0.0
    provider._trigger_ready = {"l2": False, "r2": False}
    provider._trigger_warning_shown = set()
    provider._last_snapshot = {}
    monkeypatch.setattr(ps4.pygame.event, "pump", lambda: None)
    return provider


def test_uninitialized_zero_trigger_is_ignored_until_released(monkeypatch):
    provider = _provider_without_sdl(monkeypatch)
    provider._js.buttons[provider._cfg["btn_r1"]] = 1

    # Linux/SDL reports R2=0 until its first physical event.  It must not be
    # interpreted as a half press when the configured released value is -1.
    provider._js.axes[provider._cfg["axis_r2"]] = 0.0
    assert provider.get_action()[2] == 0.0
    assert provider.debug_snapshot()["r2_ready"] == 0

    provider._js.axes[provider._cfg["axis_r2"]] = -1.0
    assert provider.get_action()[2] == 0.0
    assert provider.debug_snapshot()["r2_ready"] == 1

    provider._js.axes[provider._cfg["axis_r2"]] = 1.0
    assert provider.get_action()[2] == 1.0


def test_debug_snapshot_is_the_same_sample_as_action(monkeypatch):
    provider = _provider_without_sdl(monkeypatch)
    provider._js.buttons[provider._cfg["btn_r1"]] = 1
    provider._js.axes[provider._cfg["axis_left_x"]] = 0.5

    action = provider.get_action()
    assert action[1] > 0.0
    assert provider.debug_snapshot()["raw_lx"] == 0.5

    # Changing the fake device without another get_action must not silently make
    # diagnostics describe a different controller frame.
    provider._js.axes[provider._cfg["axis_left_x"]] = 0.9
    assert provider.debug_snapshot()["raw_lx"] == 0.5


def test_pose_target_rebases_released_axis_and_requests_one_hold():
    pose = np.array([0.6, -0.4, 0.1, 0.0, 0.0, 0.0, 1.0])
    target = TeleopPoseTarget(0.004, 0.03, base_frame_actions=True)
    target.reset(pose)

    first = target.update(np.array([0, 0, 1, 0, 0, 0]), pose)
    assert first.kind == "move"
    assert np.isclose(first.target_pose[2], 0.104)

    measured = pose.copy()
    measured[2] = 0.102
    second = target.update(np.array([0, 0, 1, 0, 0, 0]), measured)
    assert np.isclose(second.target_pose[2], 0.108)  # accumulated target, not measured + step

    measured[0] = 0.601
    measured[2] = 0.103
    z_released = target.update(np.array([1, 0, 0, 0, 0, 0]), measured)
    assert z_released.kind == "move"
    assert np.isclose(z_released.target_pose[0], 0.605)
    assert np.isclose(z_released.target_pose[2], 0.103)

    release_pose = measured.copy()
    release_pose[0] = 0.603
    hold = target.update(np.zeros(6), release_pose)
    assert hold.kind == "hold"
    np.testing.assert_allclose(hold.target_pose, release_pose)

    drifting_pose = release_pose.copy()
    drifting_pose[0] += 0.01
    idle = target.update(np.zeros(6), drifting_pose)
    assert idle.kind == "idle"
    np.testing.assert_allclose(idle.target_pose, release_pose)


def test_relative_rotation_clip_uses_the_preinsert_tool_frame():
    origin_rotation = Rotation.from_euler("xyz", [3.05, 0.2, -0.1])
    origin = np.array([0.6, -0.4, 0.1, *origin_rotation.as_quat()])
    requested_rotvec = np.array([0.45, -0.30, 0.60])
    requested_rotation = origin_rotation * Rotation.from_rotvec(requested_rotvec)
    requested = np.array([0.66, -0.46, -0.04, *requested_rotation.as_quat()])
    low = np.array([-0.04, -0.04, -0.12, -0.35, -0.35, -0.50])
    high = np.array([0.04, 0.04, 0.04, 0.35, 0.35, 0.50])

    clipped = clip_pose7_relative(requested, origin, low, high)
    np.testing.assert_allclose(clipped[:3] - origin[:3], [0.04, -0.04, -0.12])
    clipped_relative = (
        origin_rotation.inv() * Rotation.from_quat(clipped[3:7])
    ).as_rotvec()
    np.testing.assert_allclose(clipped_relative, [0.35, -0.30, 0.50], atol=1e-8)


def test_integrated_env_sends_one_direct_hold_then_no_idle_commands():
    config = IiwaInsertionConfig(
        hz=10_000,
        manual_reset=True,
        sticky_gripper_closed=False,
        integrate_action_target=True,
        base_frame_actions=True,
        random_reset=False,
    )
    env = BaseIiwaSERLEnv(config=config, include_image=False, fake_env=True)
    env.reset()

    moves = []
    holds = []
    real_move = env.client.move_pose
    real_hold = env.client.hold_position

    def move_spy(pose):
        moves.append(np.asarray(pose).copy())
        real_move(pose)

    def hold_spy():
        holds.append(True)
        real_hold()

    env.client.move_pose = move_spy
    env.client.hold_position = hold_spy

    _, _, _, _, moving = env.step(np.array([1, 0, 0, 0, 0, 0], dtype=np.float32))
    _, _, _, _, release = env.step(np.zeros(6, dtype=np.float32))
    _, _, _, _, idle = env.step(np.zeros(6, dtype=np.float32))

    assert len(moves) == 1
    assert len(holds) == 1
    assert moving["command_kind"] == "move"
    assert release["command_kind"] == "hold"
    assert idle["command_kind"] == "idle"
    assert release["command_gap_mm"] == 0.0


def test_camera_drop_reuses_last_good_frame_when_demo_mode_enables_it():
    env = BaseIiwaSERLEnv.__new__(BaseIiwaSERLEnv)
    env.include_image = True
    env.config = SimpleNamespace(reuse_last_camera_frame=True)
    good = {"wrist": np.full((2, 2, 3), 7, dtype=np.uint8)}
    env.latest_images = good
    env._camera_stale = False
    env._last_camera_warning = 0.0
    env.camera_provider = SimpleNamespace(
        read=lambda: (_ for _ in ()).throw(RuntimeError("transient frame drop"))
    )

    frames = env._get_images()
    assert frames is good
    assert env._camera_stale is True

    # A ROS provider can successfully return its cached image when no new topic
    # frame arrived.  That must still be visible in the diagnostic CSV.
    env.camera_provider = SimpleNamespace(read=lambda: good, last_read_stale=True)
    env._camera_stale = False
    assert env._get_images() is good
    assert env._camera_stale is True


def test_hold_http_route_uses_explicit_backend_hold():
    backend = SimpleNamespace(hold_calls=0)

    def hold_position():
        backend.hold_calls += 1

    backend.hold_position = hold_position
    response = create_app(backend).test_client().post("/hold")
    assert response.status_code == 200
    assert backend.hold_calls == 1

    routes = []
    client = RobotServerClient("http://unused")
    client._post = lambda route, payload=None: routes.append((route, payload))
    client.hold_position()
    assert routes == [("hold", None)]
