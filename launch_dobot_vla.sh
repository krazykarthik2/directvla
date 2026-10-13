#!/usr/bin/env bash
# ==============================================================================
# Launch Dobot Magician VLA Closed-Loop Controller
# Connects Real Desk Top-Angle Camera + Real ROS 2 Dobot Bridge + SmolVLA Policy
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$DIR:$PYTHONPATH"

echo "========================================================================"
echo "   LAUNCHING DOBOT MAGICIAN REAL VLA CONTROLLER"
echo "========================================================================"

# Source ROS 2 if installed
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source /opt/ros/jazzy/setup.bash
    echo "[ROS 2] Sourced ROS 2 Jazzy environment."
elif [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
    echo "[ROS 2] Sourced ROS 2 Humble environment."
fi

# Launch VLA Controller
python3 "$DIR/run_real_dobot_vla.py" "$@"
