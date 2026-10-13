#!/usr/bin/env bash
# ==============================================================================
# Auto-Scaling Pipeline Runner for Dobot Magician SmolVLA
#
# Runs:
#   Phase 1 (Parallel): generate_and_shard.py -> Generates verified demos with
#                       intermediate camera renders and joint outputs, sharding
#                       by wallclock.
#   Phase 2 (Parallel): extract_clip_embeddings.py --watch -> Concurrently watches
#                       for new shards and extracts frozen OpenAI CLIP ViT-B/32
#                       patch & token embeddings with configurable caching batch size.
#                       Optionally deletes raw images (--deleteraw) to save disk space.
#   Phase 3 (Training): train_vla.py -> Trains SmolVLA Optimal Transport
#                       Flow-Matching Policy over the cached multimodal shards.
#
# Usage:
#   ./autoscale_train.sh --num_demos 100 --epochs 100 --batchsize 16 --cachingbatchsize 16 --deleteraw
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$DIR:$PYTHONPATH"

# Defaults
NUM_DEMOS=100
EPOCHS=100
BATCH_SIZE=16
CACHING_BATCH_SIZE=16
DELETE_RAW_FLAG=""
EXTRA_ARGS=()

# Parse CLI arguments flexibly
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --num_demos|-n)
            NUM_DEMOS="$2"; shift 2 ;;
        --epochs|-e)
            EPOCHS="$2"; shift 2 ;;
        --batchsize|--batch_size|-b)
            BATCH_SIZE="$2"; shift 2 ;;
        --cachingbatchsize|--caching_batch_size|-c)
            CACHING_BATCH_SIZE="$2"; shift 2 ;;
        --deleteraw|--delete_raw)
            DELETE_RAW_FLAG="--delete_raw"; shift 1 ;;
        --import_real)
            EXTRA_ARGS+=("--import_real"); shift 1 ;;
        --device)
            EXTRA_ARGS+=("--device" "$2"); shift 2 ;;
        --lr)
            EXTRA_ARGS+=("--lr" "$2"); shift 2 ;;
        *)
            EXTRA_ARGS+=("$1"); shift 1 ;;
    esac
done

echo "========================================================================"
echo "   AUTOSCALE TRAINING PIPELINE: DOBOT MAGICIAN SMOLVLA"
echo "   Total Target Demos : $NUM_DEMOS"
echo "   Training Epochs    : $EPOCHS"
echo "   Training Batch Size: $BATCH_SIZE"
echo "   Caching Batch Size : $CACHING_BATCH_SIZE"
echo "   Delete Raw Images  : $([ -n "$DELETE_RAW_FLAG" ] && echo "YES" || echo "NO")"
echo "========================================================================"

# Run the master orchestrator which executes Phase 1 & 2 in parallel, followed by Phase 3
python3 "$DIR/pipeline.py" \
    --num_demos "$NUM_DEMOS" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --caching_batch_size "$CACHING_BATCH_SIZE" \
    $DELETE_RAW_FLAG \
    "${EXTRA_ARGS[@]}"
