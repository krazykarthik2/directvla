#!/bin/bash
# ==============================================================================
# Launch ROS2 Robot State Publisher + RViz2 alongside the Dobot Runner
# ==============================================================================
ACTION="$1"
shift || true

# Setup ROS2 environment
source /opt/ros/jazzy/setup.bash
if [ -f /home/karthikkrazy/dobot/install/setup.bash ]; then
    source /home/karthikkrazy/dobot/install/setup.bash
fi

export PYTHONPATH="/home/karthikkrazy/dobot:$PYTHONPATH"

# Function to reliably cleanup background processes and child trees on exit
cleanup() {
    echo ""
    echo ">> Shutting down RViz2 and ROS2 background processes..."
    
    # 1. Kill recorded background PID tree if it exists
    if [ -n "$LAUNCH_PID" ]; then
        pkill -P "$LAUNCH_PID" 2>/dev/null || true
        kill -9 "$LAUNCH_PID" 2>/dev/null || true
    fi

    # 2. Kill all background sub-jobs of this shell
    kill $(jobs -p) 2>/dev/null || true

    # 3. Explicitly terminate rviz2 and robot_state_publisher
    pkill -9 -f "rviz2" 2>/dev/null || true
    pkill -9 -f "robot_state_publisher" 2>/dev/null || true
}
trap cleanup EXIT INT TERM HUP

echo "======================================================"
echo "          Dobot Magician Controller & Visualizer      "
echo "======================================================"
echo " Select an End Effector configuration for RViz:"
echo ""
echo "  [1] Standard Gripper               (4-DOF)"
echo "  [2] Extended Gripper               (4-DOF)"
echo "  [3] Suction Cup                    (4-DOF)"
echo "  [4] Writing / Drawing Pen          (3-DOF)"
echo "  [5] Bare Tool Flange (No Tool)     (3-DOF)"
echo "  [6] Gripper + RealSense D435i Cam  (4-DOF)"
echo "  [7] Suction Cup + RealSense Cam    (4-DOF)"
echo "======================================================"

if [ -t 0 ]; then
    read -p "Enter choice [1-7] (default: 1): " -n 1 -r CHOICE
    echo ""
else
    CHOICE="1"
fi

case "$CHOICE" in
    2)
        TOOL="extended_gripper"
        DOF="4"
        CAMERA="false"
        TOOL_NAME="Extended Gripper (4-DOF)"
        ;;
    3)
        TOOL="suction_cup"
        DOF="4"
        CAMERA="false"
        TOOL_NAME="Suction Cup (4-DOF)"
        ;;
    4)
        TOOL="pen"
        DOF="3"
        CAMERA="false"
        TOOL_NAME="Writing / Drawing Pen (3-DOF)"
        ;;
    5)
        TOOL="none"
        DOF="3"
        CAMERA="false"
        TOOL_NAME="Bare Tool Flange (3-DOF)"
        ;;
    6)
        TOOL="gripper"
        DOF="4"
        CAMERA="true"
        TOOL_NAME="Standard Gripper + RealSense D435i Camera (4-DOF)"
        ;;
    7)
        TOOL="suction_cup"
        DOF="4"
        CAMERA="true"
        TOOL_NAME="Suction Cup + RealSense D435i Camera (4-DOF)"
        ;;
    1|*)
        TOOL="gripper"
        DOF="4"
        CAMERA="false"
        TOOL_NAME="Standard Gripper (4-DOF)"
        ;;
esac

echo ">> Configured Attachment: $TOOL_NAME"
echo ">> Starting ROS 2 Robot State Publisher and RViz2..."

# 1. Launch RViz & Robot State Publisher in background
ros2 launch dobot_description visualize_real.launch.py tool:="$TOOL" DOF:="$DOF" use_camera:="$CAMERA" > /dev/null 2>&1 &
LAUNCH_PID=$!

sleep 1.5

# 2. Launch interactive task runner directly (manages both hardware & live RViz publishing)
python3 /home/karthikkrazy/dobot/dobot_driver/dobot_task_runner.py "$ACTION" --tool "$TOOL" "$@"
