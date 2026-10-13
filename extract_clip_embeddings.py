#!/usr/bin/env python3
"""
PHASE 2: Extract CLIP Embeddings from Wallclock Shards (runs PARALLEL to Phase 1).

Scans data/shards/ for .npz shard files that do NOT yet have a matching
_clip_cache.pt file and extracts frozen OpenAI CLIP ViT-B/32 embeddings:
  - 49 visual patch tokens  [N, 49, 512]  from the initial observation image
  - 77 language BPE tokens   [N, 77, 512]  from the instruction prompt

Modes:
  --once     Process whatever is there right now, then exit.
  --watch    Poll every few seconds for new shards (for parallel pipeline).

Designed to be launched alongside generate_and_shard.py so shards are
CLIP-cached as soon as they land on disk.
"""

import os
import sys
import glob
import time
import argparse
import numpy as np
import torch
import clip

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if _CUR_DIR not in sys.path:
    sys.path.insert(0, _CUR_DIR)

from vla_model import PretrainedVLMEncoder
from vla_camera import preprocess_image_for_clip
from vla_dataset import WallclockDatasetManager


def find_uncached_shards(shards_dir):
    """Return list of shard .npz paths that have no matching _clip_cache.pt."""
    all_shards = sorted(glob.glob(os.path.join(shards_dir, "shard_*_*.npz")))
    uncached = []
    for s in all_shards:
        cache_path = s.replace(".npz", "_clip_cache.pt")
        if not os.path.exists(cache_path):
            uncached.append(s)
    return uncached


def extract_one_shard(shard_path, vlm, device, caching_batch_size=16, delete_raw=False):
    """
    Extract CLIP visual patches + text embeddings for every episode in a shard.
    Optionally batches image preprocessing and deletes raw image npz to free disk.
    """
    cache_path = shard_path.replace(".npz", "_clip_cache.pt")

    # Guard: another process may have cached it already
    if os.path.exists(cache_path):
        return False

    d = np.load(shard_path, allow_pickle=True)
    images_np = d["images"]      # [N, 224, 224, 3] uint8
    prompts_arr = d["prompts"]   # [N] object array of strings
    proprios_arr = d.get("proprioceptions", None)
    actions_arr = d.get("actions", None)
    n_episodes = len(images_np)

    patches_list = []
    text_list = []

    for i in range(0, n_episodes, caching_batch_size):
        end_idx = min(i + caching_batch_size, n_episodes)
        batch_imgs = images_np[i:end_idx]
        batch_prompts = [str(p) for p in prompts_arr[i:end_idx]]

        img_t = preprocess_image_for_clip(batch_imgs, device=device)
        tok_t = clip.tokenize(batch_prompts).to(device)

        with torch.no_grad():
            patches, _ = vlm.encode_vision(img_t)
            text_feats = vlm.encode_text(tok_t)

        patches_list.append(patches.float().cpu())
        text_list.append(text_feats.float().cpu())

    cached_patches = torch.cat(patches_list, dim=0)
    cached_text = torch.cat(text_list, dim=0)

    save_payload = {
        "shard_file": os.path.basename(shard_path),
        "num_episodes": n_episodes,
        "cached_patches": cached_patches,   # [N, 49, 512]
        "cached_text": cached_text,         # [N, 77, 512]
    }

    # If delete_raw requested, store proprioceptions & actions in the cache file
    # so raw uncompressed images can be safely deleted without losing trajectory data
    if delete_raw and proprios_arr is not None and actions_arr is not None:
        save_payload["proprioceptions"] = proprios_arr
        save_payload["actions"] = actions_arr
        save_payload["prompts"] = prompts_arr

    # Atomic save via tmp rename
    tmp_path = cache_path + ".tmp"
    torch.save(save_payload, tmp_path)
    os.replace(tmp_path, cache_path)

    # Delete raw images shard if requested
    if delete_raw:
        try:
            os.remove(shard_path)
            print(f"[Phase2]   deleted raw images -> {os.path.basename(shard_path)}", flush=True)
        except OSError as e:
            print(f"[Phase2]   warning: could not remove {shard_path}: {e}", flush=True)

    return True


def run_extraction(shards_dir, device, watch_mode=False, poll_interval=3.0,
                   done_signal_file=None, caching_batch_size=16, delete_raw=False):
    """
    Main extraction loop.
      watch_mode=False : process all uncached shards once and exit.
      watch_mode=True  : keep polling until <done_signal_file> appears AND
                         all shards are cached.
    """
    device = torch.device(device)
    print(f"[Phase2] Loading frozen CLIP ViT-B/32 on {device} (batch_size={caching_batch_size}, delete_raw={delete_raw})...", flush=True)
    vlm = PretrainedVLMEncoder(device=str(device))
    total_cached = 0

    if not watch_mode:
        # --- single pass ---
        uncached = find_uncached_shards(shards_dir)
        if not uncached:
            print("[Phase2] All shards already have CLIP caches. Nothing to do.", flush=True)
            return
        print(f"[Phase2] Processing {len(uncached)} uncached shards...", flush=True)
        for s in uncached:
            if extract_one_shard(s, vlm, device, caching_batch_size=caching_batch_size, delete_raw=delete_raw):
                total_cached += 1
                print(f"[Phase2]   cached -> {os.path.basename(s).replace('.npz', '_clip_cache.pt')}  "
                      f"({total_cached})", flush=True)
    else:
        # --- watch loop (runs parallel with Phase 1) ---
        print(f"[Phase2] Watching {shards_dir} for new shards (poll every {poll_interval}s)...",
              flush=True)
        idle_cycles = 0
        while True:
            uncached = find_uncached_shards(shards_dir)
            if uncached:
                idle_cycles = 0
                for s in uncached:
                    if extract_one_shard(s, vlm, device, caching_batch_size=caching_batch_size, delete_raw=delete_raw):
                        total_cached += 1
                        print(f"[Phase2]   cached -> "
                              f"{os.path.basename(s).replace('.npz', '_clip_cache.pt')}  "
                              f"({total_cached})", flush=True)
            else:
                idle_cycles += 1

            # Check if Phase 1 signalled completion
            if done_signal_file and os.path.exists(done_signal_file):
                # One final sweep to catch anything that landed right at the end
                leftovers = find_uncached_shards(shards_dir)
                for s in leftovers:
                    if extract_one_shard(s, vlm, device, caching_batch_size=caching_batch_size, delete_raw=delete_raw):
                        total_cached += 1
                        print(f"[Phase2]   cached (final) -> "
                              f"{os.path.basename(s).replace('.npz', '_clip_cache.pt')}  "
                              f"({total_cached})", flush=True)
                print(f"[Phase2] Phase 1 done signal received. Exiting after caching {total_cached} shards.",
                      flush=True)
                break

            # Safety exit: if we've been idle for a long time with no signal file,
            # assume phase 1 ended without writing the signal (e.g. standalone run)
            if idle_cycles > 60:  # ~3 minutes idle
                print(f"[Phase2] No new shards for {idle_cycles * poll_interval:.0f}s. Exiting.",
                      flush=True)
                break

            time.sleep(poll_interval)

    # Update manifest to reflect cache status
    mgr = WallclockDatasetManager(shards_dir=shards_dir)
    mgr.update_manifest()
    print(f"[Phase2] Complete. {total_cached} new CLIP caches written.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Extract CLIP Embeddings from Shards")
    parser.add_argument("--shards_dir", type=str,
                        default=os.path.join(_CUR_DIR, "data", "shards"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--watch", action="store_true",
                        help="Poll for new shards continuously (for parallel pipeline)")
    parser.add_argument("--poll_interval", type=float, default=3.0)
    parser.add_argument("--caching_batch_size", type=int, default=16,
                        help="Batch size for CLIP image/text encoding")
    parser.add_argument("--delete_raw", action="store_true",
                        help="Delete raw images .npz after extracting CLIP embeddings to save space")
    parser.add_argument("--done_signal", type=str, default=None,
                        help="Path to a file that signals Phase 1 is done")
    args = parser.parse_args()

    run_extraction(
        shards_dir=args.shards_dir,
        device=args.device,
        watch_mode=args.watch,
        poll_interval=args.poll_interval,
        done_signal_file=args.done_signal,
        caching_batch_size=args.caching_batch_size,
        delete_raw=args.delete_raw,
    )
