#!/usr/bin/env bash
# ==============================================================================
# Delete Trained Model Checkpoints
# Removes all saved checkpoints in checkpoints/ and models/
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINTS_DIR="$DIR/checkpoints"
MODELS_DIR="$DIR/models"

echo "========================================================================"
echo "   DELETING TRAINED MODEL CHECKPOINTS"
echo "========================================================================"

COUNT=0

if [ -d "$CHECKPOINTS_DIR" ]; then
    NUM_FILES=$(find "$CHECKPOINTS_DIR" -type f \( -name "*.pth" -o -name "*.pt" -o -name "*.tmp" \) | wc -l)
    if [ "$NUM_FILES" -gt 0 ]; then
        find "$CHECKPOINTS_DIR" -type f \( -name "*.pth" -o -name "*.pt" -o -name "*.tmp" \) -delete
        echo ">> Removed $NUM_FILES model checkpoint file(s) from $CHECKPOINTS_DIR"
        COUNT=$((COUNT + NUM_FILES))
    else
        echo ">> No checkpoint files found in $CHECKPOINTS_DIR"
    fi
fi

if [ -d "$MODELS_DIR" ]; then
    NUM_FILES=$(find "$MODELS_DIR" -type f \( -name "*.pth" -o -name "*.pt" -o -name "*.tmp" \) | wc -l)
    if [ "$NUM_FILES" -gt 0 ]; then
        find "$MODELS_DIR" -type f \( -name "*.pth" -o -name "*.pt" -o -name "*.tmp" \) -delete
        echo ">> Removed $NUM_FILES model checkpoint file(s) from $MODELS_DIR"
        COUNT=$((COUNT + NUM_FILES))
    fi
fi

echo ">> Done. Total checkpoints deleted: $COUNT"
echo "========================================================================"
