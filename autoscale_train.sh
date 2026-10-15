#!/usr/bin/env bash
# ==============================================================================
# Auto-Scaling Pipeline Runner for Dobot Magician SmolVLA
#
# Automatically probes GPU hardware and VRAM to scale:
#   - Training Batch Size (up to 256-512 on 48GB L40S)
#   - Parallel CLIP Embedding Caching Batch Size (up to 128 on L40S)
#   - Flow-matching model capacity (d_model=256, num_layers=3)
#   - Multi-distractor 10k demonstration generation & wallclock sharding
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$DIR:$PYTHONPATH"

# Defaults
NUM_DEMOS=10000
EPOCHS=200
BATCH_SIZE=0            # 0 = auto-scale based on VRAM
CACHING_BATCH_SIZE=0    # 0 = auto-scale based on VRAM
D_MODEL=256
NUM_LAYERS=3
SHARD_SIZE=50
DELETE_RAW_FLAG="--delete_raw"
DEVICE="cuda"
LR="0.0015"
EXTRA_ARGS=()

# 1. Detect Hardware & VRAM
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    GPU_MEM_GB=$(( GPU_MEM_MB / 1024 ))
    echo "========================================================================"
    echo "   HARDWARE ACCELERATION DETECTED"
    echo "   GPU(s) Found : $GPU_COUNT x $GPU_NAME"
    echo "   VRAM Per GPU : ~${GPU_MEM_GB} GB (${GPU_MEM_MB} MiB)"
    echo "========================================================================"
else
    GPU_COUNT=0
    GPU_MEM_GB=0
    DEVICE="cpu"
    echo "[WARN] No NVIDIA GPU detected. Falling back to CPU."
fi

# 2. Parse CLI arguments
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
        --d_model)
            D_MODEL="$2"; shift 2 ;;
        --num_layers)
            NUM_LAYERS="$2"; shift 2 ;;
        --shard_size)
            SHARD_SIZE="$2"; shift 2 ;;
        --keep_raw)
            DELETE_RAW_FLAG=""; shift 1 ;;
        --device)
            DEVICE="$2"; shift 2 ;;
        --lr)
            LR="$2"; shift 2 ;;
        *)
            EXTRA_ARGS+=("$1"); shift 1 ;;
    esac
done

# 3. Dynamic VRAM Auto-scaling
if [ "$BATCH_SIZE" -eq 0 ]; then
    if [ "$GPU_MEM_GB" -ge 40 ]; then
        # 40GB - 80GB (A100, H100, L40S)
        BATCH_SIZE=256
        CACHING_BATCH_SIZE=128
    elif [ "$GPU_MEM_GB" -ge 20 ]; then
        # 24GB (RTX 3090, 4090, A10)
        BATCH_SIZE=128
        CACHING_BATCH_SIZE=64
    elif [ "$GPU_MEM_GB" -ge 10 ]; then
        # 12GB - 16GB (RTX 4070, T4, V100)
        BATCH_SIZE=64
        CACHING_BATCH_SIZE=32
    elif [ "$GPU_MEM_GB" -ge 4 ]; then
        BATCH_SIZE=32
        CACHING_BATCH_SIZE=16
    else
        # CPU or low memory
        BATCH_SIZE=16
        CACHING_BATCH_SIZE=16
    fi
fi

if [ "$CACHING_BATCH_SIZE" -eq 0 ]; then
    CACHING_BATCH_SIZE=64
fi

echo "========================================================================"
echo "   AUTOSCALE PIPELINE CONFIGURATION"
echo "   Target Demonstrations : $NUM_DEMOS (7 colors equal mix, up to 20 distractors)"
echo "   Policy Epochs         : $EPOCHS"
echo "   Auto Scaled Batch Size: $BATCH_SIZE"
echo "   CLIP Caching Batch    : $CACHING_BATCH_SIZE"
echo "   Model Architecture    : d_model=$D_MODEL, layers=$NUM_LAYERS (127 context tokens)"
echo "   Compute Device        : $DEVICE"
echo "   Shards Storage Policy : $([ -n "$DELETE_RAW_FLAG" ] && echo "Auto-purge raw .npz to save disk" || echo "Retain raw images")"
echo "========================================================================"

# Run full pipeline with automatic scaling
python3 "$DIR/pipeline.py" \
    --num_demos "$NUM_DEMOS" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --caching_batch_size "$CACHING_BATCH_SIZE" \
    --d_model "$D_MODEL" \
    --num_layers "$NUM_LAYERS" \
    --shard_size "$SHARD_SIZE" \
    --lr "$LR" \
    --device "$DEVICE" \
    $DELETE_RAW_FLAG \
    "${EXTRA_ARGS[@]}"
