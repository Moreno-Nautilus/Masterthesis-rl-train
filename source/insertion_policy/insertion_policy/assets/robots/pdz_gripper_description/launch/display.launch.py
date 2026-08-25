from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = Path(get_package_share_directory("pdz_gripper_description"))
    model_path = package_share / "urdf" / "pdz_gripper.urdf.xacro"
    rviz_path = package_share / "rviz" / "display.rviz"

    use_pads = LaunchConfiguration("use_pads")
    pad_thickness = LaunchConfiguration("pad_thickness")
    use_camera = LaunchConfiguration("use_camera")
    camera_use_nominal_extrinsics = LaunchConfiguration(
        "camera_use_nominal_extrinsics"
    )
    robot_description = ParameterValue(
        Command(
            [
                "xacro ",
                str(model_path),
                " use_pads:=",
                use_pads,
                " pad_thickness:=",
                pad_thickness,
                " use_camera:=",
                use_camera,
                " camera_use_nominal_extrinsics:=",
                camera_use_nominal_extrinsics,
            ]
        ),
        value_type=str,
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_pads",
                default_value="true",
                description="Include TPU finger-pad visual and collision geometry.",
            ),
            DeclareLaunchArgument(
                "pad_thickness",
                default_value="0.008",
                description="TPU pad thickness in metres; supported range 0.005-0.014.",
            ),
            DeclareLaunchArgument(
                "use_camera",
                default_value="true",
                description="Include the mounted RealSense D405 model.",
            ),
            DeclareLaunchArgument(
                "camera_use_nominal_extrinsics",
                default_value="true",
                description=(
                    "Publish nominal D405 stream frames from the URDF. Set false "
                    "when realsense2_camera publishes calibrated stream TFs."
                ),
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[{"robot_description": robot_description}],
            ),
            Node(
                package="joint_state_publisher_gui",
                executable="joint_state_publisher_gui",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", str(rviz_path)],
                output="screen",
            ),
        ]
    )
