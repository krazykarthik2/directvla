#!/usr/bin/env bash
# ==============================================================================
# Launch Dobot Magician VLA in Pure Virtual ROS 2 Environment
# Uses:
#   1. ROS 2 Driver in Mock Mode (--ros-args -p mock:=true)
#   2. Virtual / Synthetic Top-Angle Desk Camera with dynamic visual overlay
#   3. SmolVLA Policy for closed-loop manipulation
#
# Usage:
#   ./launch_virtual.sh --prompt "pick up the red cube and place it on the green platform"
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$DIR:$PYTHONPATH"

# Default prompt if none provided
PROMPT="pick up the red cube and place it on the green platform"
NUM_DISTRACTORS=2

# Parse CLI arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --prompt|-p) PROMPT="$2"; shift ;;
        --checkpoint|-c) CHECKPOINT="$2"; shift ;;
        --steps|-s) STEPS="$2"; shift ;;
        --distractors|--num_distractors|-d) NUM_DISTRACTORS="$2"; shift ;;
        --headless) HEADLESS="--headless" ;;
        *) EXTRA_ARGS+=("$1") ;;
    esac
    shift
done

CHECKPOINT="${CHECKPOINT:-$DIR/checkpoints/dobot_vla_best.pth}"
STEPS="${STEPS:-12}"

echo "========================================================================"
echo "   LAUNCHING DOBOT MAGICIAN VLA (VIRTUAL ROS 2 ENVIRONMENT)"
echo "   Prompt: \"$PROMPT\""
echo "   Checkpoint: $CHECKPOINT"
echo "========================================================================"

# Source ROS 2 if available
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source /opt/ros/jazzy/setup.bash
    echo "[ROS 2] Sourced ROS 2 Jazzy."
elif [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
    echo "[ROS 2] Sourced ROS 2 Humble."
fi

# Source dobot workspace if available
if [ -f "/home/karthikkrazy/dobot/install/setup.bash" ]; then
    source /home/karthikkrazy/dobot/install/setup.bash
    echo "[ROS 2] Sourced Dobot workspace."
fi

DRIVER_PID=""
RSP_PID=""
RVIZ_PID=""

cleanup() {
    echo ""
    echo "[Virtual Launch] Cleaning up background processes..."
    if [ -n "$DRIVER_PID" ] && kill -0 "$DRIVER_PID" 2>/dev/null; then
        kill "$DRIVER_PID" 2>/dev/null || true
    fi
    if [ -n "$RSP_PID" ] && kill -0 "$RSP_PID" 2>/dev/null; then
        kill "$RSP_PID" 2>/dev/null || true
    fi
    if [ -n "$RVIZ_PID" ] && kill -0 "$RVIZ_PID" 2>/dev/null; then
        kill "$RVIZ_PID" 2>/dev/null || true
    fi
    echo "[Virtual Launch] Done."
}
trap cleanup EXIT INT TERM

# Start ROS 2 Dobot Mock Driver node in background if rclpy is available
if command -v ros2 &> /dev/null || python3 -c "import rclpy" 2>/dev/null; then
    echo "[Virtual Launch] Starting virtual ROS 2 Dobot driver (mock mode with joint publisher)..."
    python3 "$DIR/dobot_driver/dobot_ros2_driver.py" --ros-args -p mock:=true > /dev/null 2>&1 &
    DRIVER_PID=$!
    sleep 1
    echo "[Virtual Launch] Virtual ROS 2 driver active (PID $DRIVER_PID)."

    # Start robot_state_publisher and RViz if dobot_description is installed
    if ros2 pkg prefix dobot_description &> /dev/null; then
        echo "[Virtual Launch] Launching Dobot robot_state_publisher with URDF..."
        ros2 launch dobot_description visualize_real.launch.py rviz:=false > /dev/null 2>&1 &
        RSP_PID=$!

        # Launch RViz visualizer if not headless and DISPLAY is set
        if [ -z "$HEADLESS" ] && [ -n "$DISPLAY" ]; then
            echo "[Virtual Launch] Launching RViz2 for Dobot arm visualization..."
            RVIZ_CONFIG="$(ros2 pkg prefix dobot_description)/share/dobot_description/rviz/urdf_full.rviz"
            rviz2 -d "$RVIZ_CONFIG" > /dev/null 2>&1 &
            RVIZ_PID=$!
        fi
    fi
else
    echo "[Virtual Launch] ROS 2 not detected in PATH; falling back to direct kinematic simulation bridge."
fi

# Run VLA Controller in virtual camera mode
echo "[Virtual Launch] Starting closed-loop VLA Controller..."
python3 "$DIR/run_real_dobot_vla.py" \
    --prompt "$PROMPT" \
    --checkpoint "$CHECKPOINT" \
    --camera_index -1 \
    --steps "$STEPS" \
    --num_distractors "$NUM_DISTRACTORS" \
    $HEADLESS \
    "${EXTRA_ARGS[@]}"
