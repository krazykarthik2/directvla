#!/usr/bin/env bash
# ==============================================================================
# Full High-Quality Wallclock-Sharded Demonstration Generation & Training Pipeline
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$DIR:$PYTHONPATH"

NUM_DEMOS=${1:-100}
EPOCHS=${2:-50}

echo "========================================================================"
echo "   DOBOT VLA WALLCLOCK-SHARDED DATASET & TRAINING PIPELINE"
echo "   Target Episodes: $NUM_DEMOS | Epochs: $EPOCHS"
echo "========================================================================"

echo "[1/3] Generating $NUM_DEMOS verified synthetic multi-task demos..."
python3 "$DIR/generate_demos.py" --num_demos "$NUM_DEMOS" --precompute_clip

echo "[2/3] Training SmolVLA Multimodal Flow-Matching Policy ($EPOCHS epochs)..."
python3 "$DIR/train_vla.py" --epochs "$EPOCHS" --batch_size 16

echo "========================================================================"
echo "   PIPELINE COMPLETE! BEST MODEL SAVED IN checkpoints/dobot_vla_best.pth"
echo "========================================================================"
