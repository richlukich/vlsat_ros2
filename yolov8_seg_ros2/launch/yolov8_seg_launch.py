import launch
import launch.actions
import launch.substitutions
import launch_ros.actions


def generate_launch_description():
    return launch.LaunchDescription(
        [
            # Segmentator
            launch.actions.DeclareLaunchArgument("device", default_value="cuda:0"),
            launch.actions.DeclareLaunchArgument(
                "weights",
                default_value="/home/docker_yolov8_seg/colcon_ws/src/yolov8_seg_ros2/weights/roboseg_S_5_cats.pt",
            ),
            launch.actions.DeclareLaunchArgument("confidence", default_value="0.25"),
            launch.actions.DeclareLaunchArgument("treshold", default_value="0.5"),
            # Topics
            launch.actions.DeclareLaunchArgument("queue_size", default_value="10"),
            launch.actions.DeclareLaunchArgument(
                "camera_ns", default_value="/sensum/left/"
            ),
            launch.actions.DeclareLaunchArgument(
                "image_topic", default_value="image_raw"
            ),
            launch.actions.DeclareLaunchArgument(
                "segmentation_topic", default_value="segmentation"
            ),
            launch.actions.DeclareLaunchArgument(
                "segmentation_color_topic", default_value="segmentation_color"
            ),
            # Nodes
            launch_ros.actions.Node(
                package="yolov8_seg_ros2",
                namespace=launch.substitutions.LaunchConfiguration("camera_ns"),
                executable="yolov8_seg_node",
                name="yolov8_seg_node",
                remappings=[
                    ("image", launch.substitutions.LaunchConfiguration("image_topic")),
                    (
                        "segmentation",
                        launch.substitutions.LaunchConfiguration("segmentation_topic"),
                    ),
                ],
                parameters=[
                    {
                        "queue_size": launch.substitutions.LaunchConfiguration(
                            "queue_size"
                        ),
                        "weights": launch.substitutions.LaunchConfiguration("weights"),
                        "confidence": launch.substitutions.LaunchConfiguration(
                            "confidence"
                        ),
                        "treshold": launch.substitutions.LaunchConfiguration(
                            "treshold"
                        ),
                    }
                ],
                output="screen",
            ),
            launch_ros.actions.Node(
                package="yolov8_seg_ros2",
                namespace=launch.substitutions.LaunchConfiguration("camera_ns"),
                executable="visualizer_node",
                name="visualizer_node",
                remappings=[
                    ("image", launch.substitutions.LaunchConfiguration("image_topic")),
                    (
                        "segmentation",
                        launch.substitutions.LaunchConfiguration("segmentation_topic"),
                    ),
                    (
                        "segmentation_color",
                        launch.substitutions.LaunchConfiguration(
                            "segmentation_color_topic"
                        ),
                    ),
                ],
                output="screen",
            ),
        ]
    )
