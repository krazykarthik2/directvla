#!/usr/bin/env bash
# ==============================================================================
# Delete All Generated Data and Models
# Removes:
#   1. All demonstration shards & CLIP cache files (.npz, .pt, manifest)
#   2. All trained model checkpoints (.pth)
#   3. Python __pycache__ bytecode and temporary test directories
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================================"
echo "   CLEANING ENTIRE PROJECT (MODELS + DEMOS + SHARDS + CACHES)"
echo "========================================================================"

# 1. Delete models
if [ -f "$DIR/deletemodel.sh" ]; then
    bash "$DIR/deletemodel.sh"
fi

# 2. Delete demos and shards
if [ -f "$DIR/deletedemosandshards.sh" ]; then
    bash "$DIR/deletedemosandshards.sh"
fi

# 3. Clean python bytecode & caches
echo ">> Cleaning Python bytecode and cache files..."
find "$DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$DIR" -type f -name "*.pyc" -delete 2>/dev/null || true
find "$DIR" -type f -name "*.tmp" -delete 2>/dev/null || true

# Recreate essential clean folders
mkdir -p "$DIR/checkpoints"
mkdir -p "$DIR/data/shards"

echo ">> Complete cleanup finished. Clean workspace state restored."
echo "========================================================================"
