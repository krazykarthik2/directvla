#!/usr/bin/env python3
r"""
Master Pipeline: runs all 3 phases, with Phase 1 & 2 in parallel.

  Phase 1  (generate_and_shard.py)     \   run in parallel
  Phase 2  (extract_clip_embeddings.py) /
  Phase 3  (train_vla.py)               <- after both finish

Usage:
  python3 pipeline.py                           # 100 demos, 60 epochs
  python3 pipeline.py --num_demos 10000 --epochs 200
  python3 pipeline.py --import_real --num_demos 500
"""

import os
import sys
import time
import signal
import argparse
import subprocess

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
SHARDS_DIR = os.path.join(_CUR_DIR, "data", "shards")
DONE_SIGNAL = os.path.join(SHARDS_DIR, ".phase1_done")


def cleanup_signal():
    """Remove stale done-signal from a previous run."""
    if os.path.exists(DONE_SIGNAL):
        os.remove(DONE_SIGNAL)


def write_done_signal():
    with open(DONE_SIGNAL, "w") as f:
        f.write("done\n")


def run_pipeline(num_demos, epochs, batch_size, lr, device,
                 d_model=256, num_layers=3, shard_size=50,
                 caching_batch_size=64, delete_raw=False):
    os.makedirs(SHARDS_DIR, exist_ok=True)
    cleanup_signal()

    py = sys.executable  # same python interpreter

    # ─────────────────────────────────────────────────────────────
    #  PHASE 1 + PHASE 2  (parallel)
    # ─────────────────────────────────────────────────────────────
    print("=" * 72)
    print("  PHASE 1 + PHASE 2  (parallel)")
    print("    Phase 1: Generate demos + camera renders + joints → shards")
    print(f"    Phase 2: CLIP ViT-B/32 embedding extraction (caching_batch={caching_batch_size}, delete_raw={delete_raw})")
    print("=" * 72, flush=True)

    # Phase 2: start watcher FIRST so it's ready when shards land
    phase2_cmd = [
        py, os.path.join(_CUR_DIR, "extract_clip_embeddings.py"),
        "--watch",
        "--shards_dir", SHARDS_DIR,
        "--device", device,
        "--done_signal", DONE_SIGNAL,
        "--poll_interval", "2",
        "--caching_batch_size", str(caching_batch_size),
    ]
    if delete_raw:
        phase2_cmd.append("--delete_raw")

    p2 = subprocess.Popen(phase2_cmd, stdout=sys.stdout, stderr=sys.stderr)
    print(f"[Pipeline] Phase 2 watcher started  (pid {p2.pid})", flush=True)

    # Phase 1: generate + shard
    phase1_cmd = [
        py, os.path.join(_CUR_DIR, "generate_and_shard.py"),
        "--num_demos", str(num_demos),
        "--shard_size", str(shard_size),
    ]

    print(f"[Pipeline] Phase 1 generator started (pid will follow)", flush=True)
    p1_ret = subprocess.call(phase1_cmd, stdout=sys.stdout, stderr=sys.stderr)

    if p1_ret != 0:
        print(f"[Pipeline] ERROR: Phase 1 exited with code {p1_ret}", flush=True)
        p2.terminate()
        sys.exit(p1_ret)

    # Signal Phase 2 that no more shards are coming
    write_done_signal()
    print("[Pipeline] Phase 1 complete. Signalled Phase 2 to finish remaining shards...",
          flush=True)

    # Wait for Phase 2 to drain
    try:
        p2.wait(timeout=1800)  # up to 30 min for large shard volumes
    except subprocess.TimeoutExpired:
        print("[Pipeline] Phase 2 watch mode timed out. Terminating watcher process...", flush=True)
        p2.terminate()
        p2.wait(timeout=10)

    # Reconcile any remaining uncached shards synchronously
    from extract_clip_embeddings import find_uncached_shards
    leftovers = find_uncached_shards(SHARDS_DIR)
    if leftovers:
        print(f"[Pipeline] Finalizing {len(leftovers)} remaining uncached shards before Phase 3...", flush=True)
        finalize_cmd = [
            py, os.path.join(_CUR_DIR, "extract_clip_embeddings.py"),
            "--shards_dir", SHARDS_DIR,
            "--device", device,
            "--caching_batch_size", str(caching_batch_size),
        ]
        if delete_raw:
            finalize_cmd.append("--delete_raw")
        subprocess.call(finalize_cmd, stdout=sys.stdout, stderr=sys.stderr)

    from vla_dataset import WallclockDatasetManager
    mgr = WallclockDatasetManager(SHARDS_DIR)
    manifest = mgr.update_manifest()
    print(f"[Pipeline] Phase 1 + Phase 2 complete. {manifest['total_shards']} shards ({manifest['total_episodes']} demos) fully indexed.\n", flush=True)

    # ─────────────────────────────────────────────────────────────
    #  PHASE 3  (training)
    # ─────────────────────────────────────────────────────────────
    print("=" * 72)
    print("  PHASE 3: Train SmolVLA Policy on cached CLIP embeddings")
    print("=" * 72, flush=True)

    phase3_cmd = [
        py, os.path.join(_CUR_DIR, "train_vla.py"),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--d_model", str(d_model),
        "--num_layers", str(num_layers),
        "--device", device,
        "--shards_dir", SHARDS_DIR,
    ]
    p3_ret = subprocess.call(phase3_cmd, stdout=sys.stdout, stderr=sys.stderr)

    if p3_ret != 0:
        print(f"[Pipeline] ERROR: Phase 3 exited with code {p3_ret}", flush=True)
        sys.exit(p3_ret)

    # cleanup
    cleanup_signal()

    print("\n" + "=" * 72)
    print("  ALL 3 PHASES COMPLETE")
    print("    Checkpoint → checkpoints/dobot_vla_best.pth")
    print("    Run with   → ./launch_virtual.sh --prompt \"your instruction\"")
    print("=" * 72, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full 3-Phase VLA Pipeline")
    parser.add_argument("--num_demos", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--shard_size", type=int, default=50)
    parser.add_argument("--device", type=str,
                        default="cuda" if "torch" in sys.modules and __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--caching_batch_size", type=int, default=64,
                        help="Batch size for parallel CLIP embedding extraction")
    parser.add_argument("--delete_raw", action="store_true",
                        help="Delete raw images .npz after extracting CLIP embeddings to save space")
    args = parser.parse_args()

    run_pipeline(
        num_demos=args.num_demos,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        d_model=args.d_model,
        num_layers=args.num_layers,
        shard_size=args.shard_size,
        caching_batch_size=args.caching_batch_size,
        delete_raw=args.delete_raw,
    )
