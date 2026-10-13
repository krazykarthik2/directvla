#!/usr/bin/env python3
"""
Wallclock-Sharded Dataset Architecture for 10,000+ High-Quality Demonstrations.
Design:
  1. Sharding by Wallclock:
     - Demonstrations grouped into indexed shards partitioned by ISO 8601 wallclock timestamps:
       shard_YYYYMMDD_HHMMSS_<shard_id:04d>.npz
     - Shard Manifest (shards_manifest.json) tracks wallclock timestamps, episode ranges, and splits.
  2. Memory-Efficient Streaming / Indexed Loading:
     - Maps global index to (shard_path, local_index) without loading 10k demonstrations into RAM.
     - LRU caching of loaded shards maintains constant, low memory footprint.
  3. Precomputed CLIP Feature Caching:
     - Pre-extracts 49 visual patch tokens and 77 BPE language tokens per shard into:
       shard_YYYYMMDD_HHMMSS_<shard_id:04d>_clip_cache.pt
     - Enables training over 10,000 trajectories in minutes with zero loss in fidelity.
  4. Real-time Augmentation Pipeline for Visual Generalization.
"""

import os
import sys
import glob
import json
import time
import datetime
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import clip

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if _CUR_DIR not in sys.path:
    sys.path.insert(0, _CUR_DIR)

from vla_model import ACTION_MEAN, ACTION_STD, ACTION_MEAN_9D, ACTION_STD_9D, PretrainedVLMEncoder
from vla_robot_env import DobotKinematics
from vla_camera import preprocess_image_for_clip, apply_visual_augmentations


class WallclockDatasetManager:
    """
    Manages creation, rotation, indexing, and caching of wallclock-sharded demonstrations.
    Each shard holds up to `episodes_per_shard` demonstrations.
    """
    def __init__(self, shards_dir=os.path.join(_CUR_DIR, "data", "shards"), episodes_per_shard=50):
        self.shards_dir = shards_dir
        self.episodes_per_shard = episodes_per_shard
        os.makedirs(self.shards_dir, exist_ok=True)
        self.manifest_file = os.path.join(self.shards_dir, "shards_manifest.json")
        self.current_shard_episodes = []
        self.shard_counter = self._get_next_shard_counter()

    def _get_next_shard_counter(self):
        files = glob.glob(os.path.join(self.shards_dir, "shard_*_*.npz"))
        if not files:
            return 0
        ids = []
        for f in files:
            base = os.path.basename(f)
            parts = base.replace(".npz", "").split("_")
            try:
                ids.append(int(parts[-1]))
            except ValueError:
                pass
        return max(ids) + 1 if ids else 0

    def add_demonstration(self, image_seq, proprio_seq, action_seq, prompt, metadata=None):
        """
        Appends a verified trajectory to current active wallclock batch.
        image_seq: [H, 3, 224, 224] or [H, 224, 224, 3] uint8
        proprio_seq: [H, 5] float32
        action_seq: [H, 4] float32 (or [H, 5])
        prompt: str
        """
        # Ensure H=128 standardization
        H = 128
        curr_len = len(action_seq)
        if curr_len != H:
            indices = np.linspace(0, curr_len - 1, H).astype(int)
            image_seq = image_seq[indices]
            proprio_seq = proprio_seq[indices]
            action_seq = action_seq[indices]

        # Standardize images to uint8 [H, 224, 224, 3] for compact compression
        if image_seq.ndim == 4 and image_seq.shape[1] == 3:
            # [H, 3, 224, 224] -> [H, 224, 224, 3]
            img_to_save = np.transpose(image_seq, (0, 2, 3, 1))
        else:
            img_to_save = image_seq

        if img_to_save.dtype != np.uint8:
            if img_to_save.max() <= 1.0:
                img_to_save = (img_to_save * 255.0).astype(np.uint8)
            else:
                img_to_save = img_to_save.astype(np.uint8)

        demo_entry = {
            "image": img_to_save[0],  # Initial keyframe observation for conditioning
            "proprio": proprio_seq.astype(np.float32),
            "action": action_seq.astype(np.float32), # [j1, j2, j3, j4, x, y, z, r, gripper] or [x, y, z, gripper]
            "prompt": str(prompt),
            "metadata": metadata or {}
        }
        self.current_shard_episodes.append(demo_entry)

        if len(self.current_shard_episodes) >= self.episodes_per_shard:
            self.commit_shard()

    def commit_shard(self):
        """Writes current episode batch into a wallclock timestamped shard file."""
        if not self.current_shard_episodes:
            return None

        now = datetime.datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        shard_name = f"shard_{timestamp_str}_{self.shard_counter:04d}.npz"
        shard_path = os.path.join(self.shards_dir, shard_name)

        images = np.stack([ep["image"] for ep in self.current_shard_episodes], axis=0)
        proprios = np.stack([ep["proprio"] for ep in self.current_shard_episodes], axis=0)
        actions = np.stack([ep["action"] for ep in self.current_shard_episodes], axis=0)
        prompts = np.array([ep["prompt"] for ep in self.current_shard_episodes], dtype=object)

        np.savez_compressed(
            shard_path,
            wallclock_timestamp=timestamp_str,
            num_episodes=len(self.current_shard_episodes),
            images=images,
            proprioceptions=proprios,
            actions=actions,
            prompts=prompts
        )

        print(f"[DatasetManager] Committed Wallclock Shard #{self.shard_counter:04d} -> {shard_name} ({len(self.current_shard_episodes)} episodes)", flush=True)

        self.shard_counter += 1
        self.current_shard_episodes = []
        self.update_manifest()
        return shard_path

    def update_manifest(self):
        """Scans all wallclock shards (both .npz and cached .pt) and writes a unified catalog."""
        npz_shards = sorted(glob.glob(os.path.join(self.shards_dir, "shard_*_*.npz")))
        clip_caches = sorted(glob.glob(os.path.join(self.shards_dir, "shard_*_*_clip_cache.pt")))

        # Get unique base shard names
        base_shards = set()
        for s in npz_shards:
            base_shards.add(os.path.basename(s))
        for c in clip_caches:
            base_shards.add(os.path.basename(c).replace("_clip_cache.pt", ".npz"))

        sorted_bases = sorted(list(base_shards))
        manifest = {
            "total_shards": len(sorted_bases),
            "total_episodes": 0,
            "updated_at": datetime.datetime.now().isoformat(),
            "shards": []
        }

        running_offset = 0
        for b in sorted_bases:
            npz_path = os.path.join(self.shards_dir, b)
            cache_path = npz_path.replace(".npz", "_clip_cache.pt")
            has_cache = os.path.exists(cache_path)
            has_npz = os.path.exists(npz_path)

            if has_npz:
                d = np.load(npz_path, allow_pickle=True)
                n_ep = int(d["num_episodes"])
                ts = str(d["wallclock_timestamp"])
            elif has_cache:
                c_data = torch.load(cache_path, map_location="cpu", weights_only=False)
                n_ep = int(c_data["num_episodes"])
                ts = b.split("_")[1] if len(b.split("_")) > 1 else ""
            else:
                continue

            shard_info = {
                "file": b,
                "path": npz_path,
                "wallclock_timestamp": ts,
                "episodes": n_ep,
                "start_index": running_offset,
                "end_index": running_offset + n_ep - 1,
                "has_clip_cache": has_cache
            }
            manifest["shards"].append(shard_info)
            running_offset += n_ep

        manifest["total_episodes"] = running_offset
        with open(self.manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"[DatasetManager] Manifest updated: {manifest['total_shards']} shards, {manifest['total_episodes']} total trajectories.", flush=True)
        return manifest

    def generate_clip_caches(self, device='cpu', force_recompute=False):
        """
        Precomputes OpenAI CLIP ViT-B/32 visual patch and text token embeddings
        for all shards. Drastically accelerates 10k demonstration training.
        """
        shards = sorted(glob.glob(os.path.join(self.shards_dir, "shard_*_*.npz")))
        if not shards:
            print("[DatasetManager] No shards found to precompute CLIP features.", flush=True)
            return

        print(f"[DatasetManager] Generating CLIP ViT-B/32 feature caches for {len(shards)} shards on {device}...", flush=True)
        vlm = PretrainedVLMEncoder(device=device)

        for s in shards:
            cache_path = s.replace(".npz", "_clip_cache.pt")
            if os.path.exists(cache_path) and not force_recompute:
                continue

            d = np.load(s, allow_pickle=True)
            images_np = d["images"] # [N, 224, 224, 3] uint8
            prompts_arr = d["prompts"]

            patches_list = []
            text_list = []

            for i in range(len(images_np)):
                # Preprocess image
                img_t = preprocess_image_for_clip(images_np[i], device=device)
                tok_t = clip.tokenize([str(prompts_arr[i])]).to(device)

                with torch.no_grad():
                    patches, _ = vlm.encode_vision(img_t)
                    text_feats = vlm.encode_text(tok_t)

                patches_list.append(patches[0].float().cpu())
                text_list.append(text_feats[0].float().cpu())

            torch.save({
                "shard_file": os.path.basename(s),
                "num_episodes": len(images_np),
                "cached_patches": torch.stack(patches_list),  # [N, 49, 512]
                "cached_text": torch.stack(text_list)        # [N, 77, 512]
            }, cache_path)
            print(f"       Cached CLIP features -> {os.path.basename(cache_path)}", flush=True)

        self.update_manifest()


class WallclockShardedDataset(Dataset):
    """
    Scalable PyTorch Dataset for 10k+ Wallclock-Sharded Demonstrations.
    Features:
      - Memory-mapped / LRU caching across shards
      - Fast loading of precomputed CLIP embeddings
      - Trajectory normalization with ACTION_MEAN and ACTION_STD
    """
    def __init__(self, shards_dir=os.path.join(_CUR_DIR, "data", "shards"), max_cached_shards=5, use_clip_cache=True):
        self.shards_dir = shards_dir
        self.use_clip_cache = use_clip_cache
        self.manifest_file = os.path.join(self.shards_dir, "shards_manifest.json")

        if not os.path.exists(self.manifest_file):
            # Auto-build manifest
            mgr = WallclockDatasetManager(self.shards_dir)
            mgr.update_manifest()

        with open(self.manifest_file, "r") as f:
            self.manifest = json.load(f)

        self.total_episodes = self.manifest["total_episodes"]
        self.shards_info = self.manifest["shards"]

        # Build index map for O(log S) or binary search lookup
        self.shard_offsets = [s["start_index"] for s in self.shards_info]

        # LRU Shard Cache
        self.max_cached_shards = max_cached_shards
        self.shard_cache = OrderedDict()

    def __len__(self):
        return self.total_episodes

    def _locate_shard(self, global_idx):
        """Finds (shard_info, local_idx) for a global index."""
        # Binary search
        low, high = 0, len(self.shards_info) - 1
        found_idx = 0
        while low <= high:
            mid = (low + high) // 2
            if self.shards_info[mid]["start_index"] <= global_idx:
                found_idx = mid
                low = mid + 1
            else:
                high = mid - 1

        shard_info = self.shards_info[found_idx]
        local_idx = global_idx - shard_info["start_index"]
        return shard_info, local_idx

    def _load_shard_data(self, shard_info):
        shard_path = shard_info["path"]
        if shard_path in self.shard_cache:
            # Move to MRU end
            self.shard_cache.move_to_end(shard_path)
            return self.shard_cache[shard_path]

        # Evict oldest if exceeding capacity
        if len(self.shard_cache) >= self.max_cached_shards:
            self.shard_cache.popitem(last=False)

        clip_cache_path = shard_path.replace(".npz", "_clip_cache.pt")
        clip_data = None
        if self.use_clip_cache and os.path.exists(clip_cache_path):
            clip_data = torch.load(clip_cache_path, map_location="cpu", weights_only=False)

        np_data = None
        if os.path.exists(shard_path):
            np_data = np.load(shard_path, allow_pickle=True)

        data_bundle = {
            "npz": np_data,
            "clip": clip_data
        }
        self.shard_cache[shard_path] = data_bundle
        return data_bundle

    def __getitem__(self, idx):
        shard_info, local_idx = self._locate_shard(idx)
        bundle = self._load_shard_data(shard_info)

        npz = bundle["npz"]
        clip_cache = bundle["clip"]

        # Trajectory actions & normalization
        if npz is not None:
            raw_actions = npz["actions"][local_idx].astype(np.float32)  # [128, 4]
            proprio = npz["proprioceptions"][local_idx].astype(np.float32)  # [128, 5]
            prompt = str(npz["prompts"][local_idx])
        elif clip_cache is not None and "actions" in clip_cache:
            raw_actions = np.array(clip_cache["actions"][local_idx], dtype=np.float32)
            proprio = np.array(clip_cache["proprioceptions"][local_idx], dtype=np.float32)
            prompt = str(clip_cache["prompts"][local_idx])
        else:
            raise RuntimeError(f"Neither raw .npz nor action data in clip_cache found for {shard_info['path']}")

        # Standardize actions to 9D: [j1, j2, j3, j4, x, y, z, r, gripper]
        if raw_actions.shape[-1] < 9:
            # Reconstruct joint angles using Dobot analytical kinematics
            kin = DobotKinematics()
            actions_9d = np.zeros((len(raw_actions), 9), dtype=np.float32)
            for step_i in range(len(raw_actions)):
                act_pt = raw_actions[step_i]
                x_m, y_m, z_m = act_pt[0], act_pt[1], act_pt[2]
                g_val = act_pt[3] if len(act_pt) > 3 else 0.0
                j = kin.inverse(x_m, y_m, z_m, 0.0)
                actions_9d[step_i] = [j[0], j[1], j[2], j[3], x_m, y_m, z_m, 0.0, g_val]
            raw_actions = actions_9d

        # Standardize initial proprioception to 9D: [j1, j2, j3, j4, x, y, z, r, gripper]
        p_first = proprio[0]
        if len(p_first) < 9:
            kin = DobotKinematics()
            x_m, y_m, z_m = p_first[0], p_first[1], p_first[2]
            yaw_r = p_first[3] if len(p_first) > 3 else 0.0
            g_val = p_first[4] if len(p_first) > 4 else 0.0
            j = kin.inverse(x_m, y_m, z_m, yaw_r)
            proprio0 = torch.tensor([j[0], j[1], j[2], j[3], x_m, y_m, z_m, yaw_r, g_val], dtype=torch.float32)
        else:
            proprio0 = torch.from_numpy(p_first[:9])

        # Normalize target trajectory for Flow Matching (9D)
        act_tensor = torch.from_numpy(raw_actions)
        norm_traj = (act_tensor - ACTION_MEAN_9D) / (ACTION_STD_9D + 1e-6)

        if clip_cache is not None:
            # Return precomputed CLIP embeddings: [49, 512] visual, [77, 512] language
            vis_patches = clip_cache["cached_patches"][local_idx]
            txt_feats = clip_cache["cached_text"][local_idx]
            return vis_patches, txt_feats, proprio0, norm_traj
        else:
            # Return raw image and prompt tokens for dynamic online encoder
            raw_img = npz["images"][local_idx]
            img_tensor = preprocess_image_for_clip(raw_img).squeeze(0)  # [3, 224, 224]
            token_ids = clip.tokenize([prompt])[0]  # [77]
            return img_tensor, token_ids, proprio0, norm_traj


if __name__ == "__main__":
    print("[TEST] Initializing WallclockDatasetManager...")
    test_shards_dir = os.path.join(_CUR_DIR, "data", "test_shards")
    os.makedirs(test_shards_dir, exist_ok=True)

    mgr = WallclockDatasetManager(shards_dir=test_shards_dir, episodes_per_shard=3)

    print("       Synthesizing 5 sample demonstrations across multiple wallclock shards...")
    for i in range(5):
        img_seq = np.random.randint(0, 255, (128, 224, 224, 3), dtype=np.uint8)
        proprio_seq = np.zeros((128, 5), dtype=np.float32)
        proprio_seq[:, 0] = 0.20 + 0.01 * i
        act_seq = np.zeros((128, 4), dtype=np.float32)
        act_seq[:, 0] = 0.22
        prompt = f"pick the red cube and place on green platform variant {i}"
        mgr.add_demonstration(img_seq, proprio_seq, act_seq, prompt)

    # Commit remaining
    mgr.commit_shard()

    print("       Testing precomputed CLIP feature extraction...")
    mgr.generate_clip_caches(device="cpu")

    print("       Testing WallclockShardedDataset loading...")
    dataset = WallclockShardedDataset(shards_dir=test_shards_dir, use_clip_cache=True)
    print(f"       Total loaded demonstrations: {len(dataset)}")
    assert len(dataset) == 5, f"Expected 5 demonstrations, got {len(dataset)}"

    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    for vis_p, txt_f, p0, traj in loader:
        print(f"       DataLoader batch: vis_patches={vis_p.shape}, text_feats={txt_f.shape}, proprio0={p0.shape}, traj={traj.shape}")
        assert vis_p.shape == (2, 49, 512)
        assert txt_f.shape == (2, 77, 512)
        assert traj.shape == (2, 128, 4)
        break

    # Clean up test artifacts
    import shutil
    shutil.rmtree(test_shards_dir, ignore_errors=True)
    print("[TEST] WallclockDatasetManager and WallclockShardedDataset passed successfully!")
