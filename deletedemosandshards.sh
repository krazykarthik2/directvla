#!/usr/bin/env bash
# ==============================================================================
# Delete Demonstrations and Shards
# Removes all dataset files in data/shards and demo recordings in data/
# Recreates empty directory structure ready for fresh data collection.
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$DIR/data"
SHARDS_DIR="$DATA_DIR/shards"
TEST_SHARDS_DIR="$DATA_DIR/test_shards"

echo "========================================================================"
echo "   DELETING DEMONSTRATIONS AND SHARDS"
echo "========================================================================"

COUNT=0

if [ -d "$SHARDS_DIR" ]; then
    NUM_FILES=$(find "$SHARDS_DIR" -type f | wc -l)
    if [ "$NUM_FILES" -gt 0 ]; then
        find "$SHARDS_DIR" -type f -delete
        echo ">> Removed $NUM_FILES shard files/caches from $SHARDS_DIR"
        COUNT=$((COUNT + NUM_FILES))
    else
        echo ">> No shard files found in $SHARDS_DIR"
    fi
fi

if [ -d "$TEST_SHARDS_DIR" ]; then
    NUM_FILES=$(find "$TEST_SHARDS_DIR" -type f | wc -l)
    if [ "$NUM_FILES" -gt 0 ]; then
        rm -rf "$TEST_SHARDS_DIR"
        echo ">> Removed $NUM_FILES test shard files from $TEST_SHARDS_DIR"
        COUNT=$((COUNT + NUM_FILES))
    fi
fi

# Ensure clean directory structure exists
mkdir -p "$SHARDS_DIR"

echo ">> Done. Total demo/shard files deleted: $COUNT"
echo ">> Fresh data directory initialized: $SHARDS_DIR"
echo "========================================================================"
