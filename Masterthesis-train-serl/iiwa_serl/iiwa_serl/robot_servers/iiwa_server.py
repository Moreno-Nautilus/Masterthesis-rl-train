from __future__ import annotations

import json
import time

from absl import app, flags
from flask import Flask, jsonify, request
import numpy as np

from iiwa_serl.robot import make_backend

FLAGS = flags.FLAGS
flags.DEFINE_string("backend", "mock", "Backend name: 'mock' or 'module.path:ClassName'.")
flags.DEFINE_string("backend_kwargs_json", "", "JSON object passed to the backend constructor.")
flags.DEFINE_string("host", "127.0.0.1", "HTTP host.")
flags.DEFINE_integer("port", 5000, "HTTP port.")


def _to_json_state(state):
    return {
        "pose": np.asarray(state.pose).tolist(),
        "vel": np.asarray(state.vel).tolist(),
        "force": np.asarray(state.force).tolist(),
        "torque": np.asarray(state.torque).tolist(),
        "q": np.asarray(state.q).tolist(),
        "dq": np.asarray(state.dq).tolist(),
        "jacobian": np.asarray(state.jacobian).tolist(),
        "gripper": float(state.gripper),
        "timestamp": float(state.timestamp),
    }


def create_app(backend):
    webapp = Flask(__name__)

    @webapp.route("/pose", methods=["POST"])
    def pose():
        payload = request.json or {}
        backend.move_pose(np.asarray(payload["pose"], dtype=np.float64))
        return "Moved"

    @webapp.route("/getpos", methods=["POST"])
    def get_pos():
        return jsonify({"pose": np.asarray(backend.get_state().pose).tolist()})

    @webapp.route("/getvel", methods=["POST"])
    def get_vel():
        return jsonify({"vel": np.asarray(backend.get_state().vel).tolist()})

    @webapp.route("/getforce", methods=["POST"])
    def get_force():
        return jsonify({"force": np.asarray(backend.get_state().force).tolist()})

    @webapp.route("/gettorque", methods=["POST"])
    def get_torque():
        return jsonify({"torque": np.asarray(backend.get_state().torque).tolist()})

    @webapp.route("/getq", methods=["POST"])
    def get_q():
        return jsonify({"q": np.asarray(backend.get_state().q).tolist()})

    @webapp.route("/getdq", methods=["POST"])
    def get_dq():
        return jsonify({"dq": np.asarray(backend.get_state().dq).tolist()})

    @webapp.route("/getjacobian", methods=["POST"])
    def get_jacobian():
        return jsonify({"jacobian": np.asarray(backend.get_state().jacobian).tolist()})

    @webapp.route("/get_gripper", methods=["POST"])
    def get_gripper():
        return jsonify({"gripper": float(backend.get_state().gripper)})

    @webapp.route("/getstate", methods=["POST"])
    def get_state():
        state = backend.get_state()
        return jsonify(_to_json_state(state))

    @webapp.route("/jointreset", methods=["POST"])
    def joint_reset():
        backend.reset_joints()
        return "Reset Joint"

    @webapp.route("/activate_gripper", methods=["POST"])
    def activate_gripper():
        backend.activate_gripper()
        return "Activated"

    @webapp.route("/reset_gripper", methods=["POST"])
    def reset_gripper():
        backend.reset_gripper()
        return "Reset"

    @webapp.route("/open_gripper", methods=["POST"])
    def open_gripper():
        backend.open_gripper()
        return "Opened"

    @webapp.route("/close_gripper", methods=["POST"])
    def close_gripper():
        backend.close_gripper()
        return "Closed"

    @webapp.route("/move_gripper", methods=["POST"])
    def move_gripper():
        payload = request.json or {}
        backend.move_gripper(float(payload.get("gripper", 0.0)))
        return "Moved Gripper"

    @webapp.route("/clearerr", methods=["POST"])
    def clearerr():
        backend.clear_errors()
        return "Cleared"

    @webapp.route("/update_param", methods=["POST"])
    def update_param():
        backend.update_params(request.json or {})
        return "Updated"

    @webapp.route("/healthz", methods=["GET"])
    def healthz():
        return jsonify({"ok": True, "timestamp": time.time()})

    return webapp


def main(_):
    backend_kwargs = json.loads(FLAGS.backend_kwargs_json) if FLAGS.backend_kwargs_json else {}
    backend = make_backend(FLAGS.backend, **backend_kwargs)
    if hasattr(backend, "start"):
        backend.start()
    webapp = create_app(backend)
    webapp.run(host=FLAGS.host, port=FLAGS.port)


if __name__ == "__main__":
    app.run(main)
