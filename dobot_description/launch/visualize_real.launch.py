#!/usr/bin/env python3
import sys
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_path

def generate_launch_description():
    urdf_tutorial_path = get_package_share_path('dobot_description')
    default_model_path = urdf_tutorial_path / 'model/magician_standalone.urdf.xacro'
    default_rviz_config_path = urdf_tutorial_path / 'rviz/urdf_full.rviz'

    tool_arg = DeclareLaunchArgument('tool', default_value='gripper', choices=['gripper', 'extended_gripper', 'suction_cup', 'pen', 'none'])
    dof_arg = DeclareLaunchArgument('DOF', default_value='4')
    camera_arg = DeclareLaunchArgument('use_camera', default_value='false')
    rviz_arg = DeclareLaunchArgument('rviz', default_value='true')

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': Command([
                'xacro ', str(default_model_path),
                ' DOF:=', LaunchConfiguration('DOF'),
                ' use_camera:=', LaunchConfiguration('use_camera'),
                ' tool:=', LaunchConfiguration('tool')
            ])
        }]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', str(default_rviz_config_path)],
        condition=IfCondition(LaunchConfiguration('rviz'))
    )

    return LaunchDescription([
        tool_arg,
        dof_arg,
        camera_arg,
        rviz_arg,
        robot_state_publisher_node,
        rviz_node
    ])
