#!/usr/bin/env bash
# ==============================================================================
# Launch Full 3D Interactive Dobot Magician Demonstration in ROS 2 & RViz
# Starts:
#   1. robot_state_publisher with official Dobot Magician URDF
#   2. RViz2 3D Visualizer with robot model and 3D trajectory markers
#   3. 3D Demo Playback Node publishing joint states and live interactive objects
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$DIR:$PYTHONPATH"

# Source ROS 2 Jazzy
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source /opt/ros/jazzy/setup.bash
fi

# Source Dobot workspace
if [ -f "/home/karthikkrazy/dobot/install/setup.bash" ]; then
    source /home/karthikkrazy/dobot/install/setup.bash
fi

RSP_PID=""
RVIZ_PID=""
PLAYER_PID=""

cleanup() {
    echo ""
    echo "[3D Demo] Stopping ROS 2 nodes and RViz..."
    if [ -n "$PLAYER_PID" ] && kill -0 "$PLAYER_PID" 2>/dev/null; then
        kill "$PLAYER_PID" 2>/dev/null || true
    fi
    if [ -n "$RVIZ_PID" ] && kill -0 "$RVIZ_PID" 2>/dev/null; then
        kill "$RVIZ_PID" 2>/dev/null || true
    fi
    if [ -n "$RSP_PID" ] && kill -0 "$RSP_PID" 2>/dev/null; then
        kill "$RSP_PID" 2>/dev/null || true
    fi
    echo "[3D Demo] Closed."
}
trap cleanup EXIT INT TERM

echo "========================================================================"
echo "   LAUNCHING DOBOT MAGICIAN 3D DEMONSTRATION IN ROS 2 & RVIZ"
echo "========================================================================"

# 1. Launch robot_state_publisher with URDF
echo "[3D Demo] 1/3 Starting robot_state_publisher..."
ros2 launch dobot_description visualize_real.launch.py rviz:=false > /dev/null 2>&1 &
RSP_PID=$!
sleep 1

# 2. Launch RViz2
RVIZ_CONFIG="$DIR/dobot_description/rviz/urdf_full.rviz"
if [ -n "$DISPLAY" ]; then
    echo "[3D Demo] 2/3 Launching RViz2 3D window..."
    rviz2 -d "$RVIZ_CONFIG" > /dev/null 2>&1 &
    RVIZ_PID=$!
    sleep 2
else
    echo "[3D Demo] Warning: DISPLAY not found. RViz GUI skipped (headless mode)."
fi

# 3. Launch 3D Demo Trajectory Player
echo "[3D Demo] 3/3 Starting 3D Trajectory Player..."
python3 "$DIR/visualize_3d_demo.py" "$@" &
PLAYER_PID=$!

wait "$PLAYER_PID"
