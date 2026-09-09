"""Compile and exercise the actual FR3 reference callback without ROS or hardware.

Run with python3. Requires c++ and Eigen headers. Override FRANKA_CONTROLLER_SOURCE
if the controller checkout is not at ~/fr3_ws/src/cartesian_impedance_control.
Only a temporary test executable is built; the controller library is not touched.
"""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class ControllerOrientationHoldTest(unittest.TestCase):
    def test_actual_reference_callback(self):
        default = (Path.home() / 'fr3_ws/src/cartesian_impedance_control/src/'
                   'cartesian_impedance_controller.cpp')
        source = Path(os.environ.get('FRANKA_CONTROLLER_SOURCE', str(default))).read_text()
        start = source.index('void CartesianImpedanceController::reference_pose_callback(')
        end = source.index('void CartesianImpedanceController::control_mode_callback(', start)
        callback = source[start:end]
        harness = r'''
#include <Eigen/Geometry>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
unsigned int rejected = 0;
#define RCLCPP_ERROR(...) (++rejected)
namespace geometry_msgs { namespace msg {
struct Pose {
  using SharedPtr = std::shared_ptr<Pose>;
  struct { double x=0, y=0, z=0; } position;
  struct { double x=0, y=0, z=0, w=1; } orientation;
};
}}
namespace cartesian_impedance_control {
class CartesianImpedanceController {
public:
  Eigen::Vector3d position_d_target_ = Eigen::Vector3d::Zero();
  Eigen::Quaterniond orientation_d_target_ = Eigen::Quaterniond::Identity();
  Eigen::Matrix<double, 6, 1> I_error = Eigen::Matrix<double, 6, 1>::Zero();
  void reference_pose_callback(const geometry_msgs::msg::Pose::SharedPtr msg);
};
''' + callback + r'''
}
int main() {
  using cartesian_impedance_control::CartesianImpedanceController;
  CartesianImpedanceController c;
  auto msg = std::make_shared<geometry_msgs::msg::Pose>();
  std::ostringstream quiet;
  auto old_output = std::cout.rdbuf(quiet.rdbuf());
  const Eigen::Vector3d correction(0.2, -0.3, 0.1);
  c.I_error.head(3).setConstant(0.5);
  c.I_error.tail(3) = correction;
  // Streaming pure-Z references must preserve all rotational integral components.
  for (int i=0; i<100; ++i) {
    msg->position.z = 0.3 + 0.001*i;
    c.reference_pose_callback(msg);
    assert(c.I_error.tail(3).isApprox(correction, 1e-12));
    assert(c.I_error.head(3).isZero());
    assert(c.orientation_d_target_.angularDistance(Eigen::Quaterniond::Identity()) < 1e-12);
  }
  // Quaternion sign or scaling does not request a different orientation.
  msg->orientation.w = -3.0;
  c.reference_pose_callback(msg);
  assert(c.I_error.tail(3).isApprox(correction, 1e-12));
  // An intentional orientation change still resets the rotational integral.
  const Eigen::Quaterniond rotated(Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitX()));
  msg->orientation.x = rotated.x();
  msg->orientation.w = rotated.w();
  c.reference_pose_callback(msg);
  assert(c.I_error.tail(3).isZero());
  c.I_error.tail(3) = correction;
  c.reference_pose_callback(msg); // identical hold re-publication
  assert(c.I_error.tail(3).isApprox(correction, 1e-12));
  // Invalid messages must leave the previous target and correction untouched.
  const auto saved_position = c.position_d_target_;
  const auto saved_orientation = c.orientation_d_target_;
  msg->position.z = 9;
  msg->orientation.x = msg->orientation.w = 0;
  c.reference_pose_callback(msg);
  assert(rejected == 1);
  assert(c.position_d_target_.isApprox(saved_position));
  assert(c.orientation_d_target_.angularDistance(saved_orientation) < 1e-12);
  assert(c.I_error.tail(3).isApprox(correction, 1e-12));
  msg->orientation.w = std::numeric_limits<double>::quiet_NaN();
  c.reference_pose_callback(msg);
  assert(rejected == 2);
  msg->orientation.w = 1;
  msg->position.z = std::numeric_limits<double>::infinity();
  c.reference_pose_callback(msg);
  assert(rejected == 3);
  // At 20Hz reference traffic, 1kHz integral updates must no longer be wiped out.
  // This checks accumulator behavior only, NOT simulated robot stability/tracking.
  msg->position.z = 0.3;
  msg->orientation.x = 0;
  c.reference_pose_callback(msg);
  c.I_error.setZero();
  for (int tick=1; tick<=1000; ++tick) {
    c.I_error(3) += 50.0 * 0.01 * 0.001;
    if (tick % 50 == 0) c.reference_pose_callback(msg);
  }
  assert(std::abs(c.I_error(3) - 0.5) < 1e-10);
  std::cout.rdbuf(old_output);
  std::cout << "PASS: translation/hold/sign invariance, deliberate rotation reset, "
               "invalid input rejection, and 20Hz streaming retention\n";
}
'''
        with tempfile.TemporaryDirectory(prefix='franka-orientation-test-') as temp:
            binary = str(Path(temp) / 'callback_test')
            compiled = subprocess.run(
                ['c++', '-std=c++17', '-O0', '-I/usr/include/eigen3',
                 '-x', 'c++', '-', '-o', binary], input=harness,
                text=True, capture_output=True, timeout=60)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            result = subprocess.run([binary], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            print(result.stdout.strip())


if __name__ == '__main__':
    unittest.main()
