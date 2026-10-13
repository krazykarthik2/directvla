import os
import sys
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Device Configuration (Auto CUDA / Multi-core CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    NUM_CORES = os.cpu_count() or 4
    torch.set_num_threads(NUM_CORES)
    torch.set_num_interop_threads(NUM_CORES)
    os.environ["OMP_NUM_THREADS"] = str(NUM_CORES)
    os.environ["MKL_NUM_THREADS"] = str(NUM_CORES)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "demos")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def safe_save_model(state_dict, path):
    tmp_path = path + ".tmp"
    for attempt in range(5):
        try:
            torch.save(state_dict, tmp_path)
            if os.path.exists(path):
                try:
                    os.replace(tmp_path, path)
                except OSError:
                    import time
                    time.sleep(0.05)
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                    os.rename(tmp_path, path)
            else:
                os.rename(tmp_path, path)
            return
        except Exception as e:
            if attempt == 4:
                print(f"[WARN] Failed to save checkpoint: {e}", flush=True)
            import time
            time.sleep(0.1)

# -----------------------------------------------------------------------------
# Parameterized Embeddings (Without heavy LLM overhead: intent, source, dest)
# -----------------------------------------------------------------------------
COLOR_MAP = {
    "red": 0, "blue": 1, "yellow": 2, "green": 3,
    "purple": 4, "orange": 5, "cyan": 6
}
ACTION_MAP = {"pick_place": 0, "push": 1}

def get_intent_embedding_vector(act_str="pick_place", src_str="red", dst_str="green", dim=128):
    """
    Constructs a 128-dim structured continuous parameter embedding:
    - Action type (Pick & Place vs Push)
    - Source object color/target
    - Destination platform color/target
    """
    act_idx = ACTION_MAP.get(act_str, 0)
    src_idx = COLOR_MAP.get(src_str, 0)
    dst_idx = COLOR_MAP.get(dst_str, 3)

    act_onehot = np.zeros(4, dtype=np.float32)
    act_onehot[act_idx] = 1.0

    src_onehot = np.zeros(8, dtype=np.float32)
    src_onehot[src_idx] = 1.0

    dst_onehot = np.zeros(8, dtype=np.float32)
    dst_onehot[dst_idx] = 1.0

    raw_vec = np.concatenate([act_onehot, src_onehot, dst_onehot]) # 20 dims
    return raw_vec

# Normalization constants for 4D actions (X, Y, Z, Gripper) matching empirical dataset
ACTION_MEAN = torch.tensor([0.2028, 0.0008, 0.0854, 0.5360], dtype=torch.float32)
ACTION_STD  = torch.tensor([0.0330, 0.0837, 0.0362, 0.4987], dtype=torch.float32)

# -----------------------------------------------------------------------------
# 1. Optimal Transport Dataset
# -----------------------------------------------------------------------------
class OTFlowMatchingDataset(Dataset):
    def __init__(self, data_dir):
        files = sorted(glob.glob(os.path.join(data_dir, "demo_*.npz")))
        if not files:
            raise ValueError(f"No demonstration files found in {data_dir}. Run auto_generate_demos.py first!")

        self.samples = []
        print(f">> Indexing {len(files)} demonstrations for Flow-Matching...", flush=True)

        for f in files:
            d = np.load(f, allow_pickle=True)
            img0 = d['images'][0].astype(np.float32) # [3, 64, 64]
            proprio = d['proprioception'].astype(np.float32) # [128, 5]
            acts = d['actions'].astype(np.float32) # [128, 6]
            
            act_type = str(d['action_type'][0]) if 'action_type' in d else "pick_place"
            tgt_color = str(d['target_color'][0]) if 'target_color' in d else "red"
            tgt_plat = str(d['target_plat_color'][0]) if 'target_plat_color' in d else "green"

            intent_vec = get_intent_embedding_vector(act_type, tgt_color, tgt_plat)

            raw_traj = np.concatenate([proprio[:, :3], acts[:, 4:5]], axis=-1) # [128, 4]
            norm_traj = (torch.tensor(raw_traj, dtype=torch.float32) - ACTION_MEAN) / (ACTION_STD + 1e-6)
            proprio0 = proprio[0] # [5] (x, y, z, yaw, gripper)
            self.samples.append((img0, intent_vec, proprio0, norm_traj))

        print(f">> Successfully indexed {len(self.samples)} trajectory demonstrations.", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, intent_vec, proprio0, norm_traj = self.samples[idx]
        return torch.tensor(img, dtype=torch.float32), torch.tensor(intent_vec, dtype=torch.float32), torch.tensor(proprio0, dtype=torch.float32), norm_traj

# -----------------------------------------------------------------------------
# 2. Vision Patch & Flow Matching Policy Architecture
# -----------------------------------------------------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
    def forward(self, x):
        return x + self.conv(x)

class Sinusoidal2DPositionalEmbedding(nn.Module):
    """2D Positional Embeddings for 16x16 visual patch grid."""
    def __init__(self, dim=128, grid_size=16):
        super().__init__()
        self.dim = dim
        self.grid_size = grid_size
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim // 4, dtype=torch.float32) / (dim // 4)))
        
        pos_x = torch.arange(grid_size, dtype=torch.float32)
        pos_y = torch.arange(grid_size, dtype=torch.float32)
        
        sin_x = torch.sin(pos_x.unsqueeze(1) * inv_freq.unsqueeze(0))
        cos_x = torch.cos(pos_x.unsqueeze(1) * inv_freq.unsqueeze(0))
        sin_y = torch.sin(pos_y.unsqueeze(1) * inv_freq.unsqueeze(0))
        cos_y = torch.cos(pos_y.unsqueeze(1) * inv_freq.unsqueeze(0))
        
        # [16, 16, dim]
        pe = torch.zeros(grid_size, grid_size, dim)
        pe[:, :, 0::4] = sin_x.unsqueeze(1).repeat(1, grid_size, 1)
        pe[:, :, 1::4] = cos_x.unsqueeze(1).repeat(1, grid_size, 1)
        pe[:, :, 2::4] = sin_y.unsqueeze(0).repeat(grid_size, 1, 1)
        pe[:, :, 3::4] = cos_y.unsqueeze(0).repeat(grid_size, 1, 1)
        
        self.register_buffer("pe", pe.view(grid_size * grid_size, dim).unsqueeze(0)) # [1, 256, dim]

    def forward(self, x):
        return x + self.pe

class CrossAttentionBlock(nn.Module):
    """Transformer Decoder Block with Self-Attention and Visual Cross-Attention."""
    def __init__(self, d_model=128, nhead=4, d_ff=256):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, tgt, memory):
        # 1. Self Attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt)
        tgt = self.norm1(tgt + tgt2)
        
        # 2. Cross Attention to visual & goal tokens
        tgt2, attn_weights = self.cross_attn(tgt, memory, memory)
        tgt = self.norm2(tgt + tgt2)
        
        # 3. Feed Forward
        tgt = self.norm3(tgt + self.ffn(tgt))
        return tgt, attn_weights

class ManualVLAPolicy(nn.Module):
    """
    True Vision-Language-Action Cross-Attention Transformer Policy (ACT-style):
    - 256 Visual Patch Tokens (16x16 grid from 4x4 patches on 64x64 top camera)
    - 2D Sinusoidal Positional Embeddings
    - Goal & Intent Tokenizer (Action type, Target Cube color, Destination platform)
    - Proprioception State Tokenizer (live x, y, z, yaw, gripper)
    - Cross-Attention Transformer Decoder with Flow Matching Action Head
    """
    def __init__(self, raw_intent_dim=20, horizon=128, action_dim=4, d_model=128, num_layers=4):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.d_model = d_model
        
        # 1. Visual Tokenizer (64x64 image -> 16x16 grid of 4x4 patches = 256 tokens)
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=4, stride=4) # [B, d_model, 16, 16]
        self.pos_embed = Sinusoidal2DPositionalEmbedding(d_model, grid_size=16)
        
        # 2. Intent and Proprioception Tokenizers
        self.intent_proj = nn.Sequential(
            nn.Linear(raw_intent_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.proprio_proj = nn.Sequential(
            nn.Linear(5, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # 3. Continuous Time Embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(64),
            nn.Linear(64, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # 4. Action Query Tokens & Cross-Attention Transformer
        self.action_in_proj = nn.Linear(action_dim, d_model)
        self.pos_queries = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)
        
        self.layers = nn.ModuleList([
            CrossAttentionBlock(d_model=d_model, nhead=4, d_ff=d_model * 2)
            for _ in range(num_layers)
        ])
        
        # 5. Output Flow Vector Field Head
        self.out_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, action_dim)
        )

    def extract_visual_tokens(self, img):
        B = img.size(0)
        patches = self.patch_embed(img) # [B, d_model, 16, 16]
        tokens = patches.flatten(2).permute(0, 2, 1) # [B, 256, d_model]
        tokens = self.pos_embed(tokens)
        return tokens

    def forward_flow(self, x_t, t, img, intent_vec, proprio=None):
        B = img.size(0)
        
        # 1. Visual tokens (256 tokens)
        vis_tokens = self.extract_visual_tokens(img) # [B, 256, d_model]
        
        # 2. Context tokens (Intent + Proprioception + Time)
        i_token = self.intent_proj(intent_vec).unsqueeze(1) # [B, 1, d_model]
        
        if proprio is None:
            proprio = torch.zeros(B, 5, device=img.device)
        elif proprio.dim() == 1:
            proprio = proprio.unsqueeze(0)
        p_token = self.proprio_proj(proprio).unsqueeze(1) # [B, 1, d_model]
        
        t_token = self.time_embed(t).unsqueeze(1) # [B, 1, d_model]
        
        # Full Memory Sequence for Cross-Attention: 256 + 3 = 259 tokens
        memory = torch.cat([i_token, p_token, t_token, vis_tokens], dim=1) # [B, 259, d_model]
        
        # 3. Action Sequence Queries
        act_tokens = self.action_in_proj(x_t) + self.pos_queries # [B, horizon, d_model]
        
        # 4. Cross-Attention Transformer Layers
        x = act_tokens
        for layer in self.layers:
            x, attn_weights = layer(x, memory)
            
        # 5. Predict vector field for each timestep in horizon
        v_pred = self.out_head(x) # [B, horizon, action_dim]
        return v_pred

    @torch.no_grad()
    def sample(self, img, intent_vec, proprio=None, num_steps=20):
        """Continuous Euler ODE integration from noise to predicted robot trajectory."""
        B = img.size(0)
        x = torch.randn(B, self.horizon, self.action_dim, device=img.device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((B,), (i + 0.5) * dt, device=img.device)
            v = self.forward_flow(x, t, img, intent_vec, proprio=proprio)
            x = x + v * dt

        raw_x = x * ACTION_STD.to(img.device) + ACTION_MEAN.to(img.device)
        
        # Temporal smoothing filter across horizon
        kernel = torch.ones(1, 1, 5, device=img.device) / 5.0
        raw_perm = raw_x.permute(0, 2, 1).reshape(B * self.action_dim, 1, self.horizon)
        smoothed = F.conv1d(raw_perm, kernel, padding=2)
        smoothed = smoothed.view(B, self.action_dim, self.horizon).permute(0, 2, 1)
        return smoothed

        return smoothed

# Aliases
TrueSmolVLAPolicy = ManualVLAPolicy

# -----------------------------------------------------------------------------
# 3. Flow Matching Training
# -----------------------------------------------------------------------------
def train(epochs=120, batch_size=16, lr=1.8e-3):
    print("=" * 68, flush=True)
    print("   Manual VLA Flow-Matching Policy Training (Cross-Attention Transformer)", flush=True)
    print("   - Action Conditioning: Pick & Place vs Push", flush=True)
    print("   - Patch Visual Extractor & Parameterized Embeddings", flush=True)
    print("   - Optimal Transport Vector Field Regression | 100% CPU", flush=True)
    print("=" * 68, flush=True)

    dataset = OTFlowMatchingDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ManualVLAPolicy()
    model_path = os.path.join(MODEL_DIR, "dobot_bc_policy.pth")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.SmoothL1Loss(reduction='none') # Huber loss
    # Weighting: X, Y, Z spatial accuracy + 2.5x boosting on gripper grasp/release transitions
    dim_weights = torch.tensor([1.0, 1.0, 1.2, 2.5], device=DEVICE).view(1, 1, 4)

    print(f"\n>> Training Flow-Matching across {len(dataset)} trajectories ({epochs} epochs)...", flush=True)
    best_loss = float('inf')

    try:
        epoch_pbar = tqdm(range(1, epochs + 1), desc="Training VLA", unit="epoch", dynamic_ncols=True)
        for epoch in epoch_pbar:
            model.train()
            total_loss = 0.0

            for img_b, intent_b, proprio_b, traj_x0 in dataloader:
                B = img_b.size(0)
                optimizer.zero_grad(set_to_none=True)

                x_1 = torch.randn_like(traj_x0)
                t = torch.rand(B, device=img_b.device)
                t_expand = t.view(B, 1, 1)

                # Optimal Transport path interpolation
                x_t = (1.0 - t_expand) * traj_x0 + t_expand * x_1
                target_v = x_1 - traj_x0

                pred_v = model.forward_flow(x_t, t, img_b, intent_vec=intent_b, proprio=proprio_b)
                # Huber loss with gripper boosting
                raw_loss = loss_fn(pred_v, target_v) # [B, horizon, 4]
                weighted_loss = (raw_loss * dim_weights).mean()
                
                weighted_loss.backward()
                optimizer.step()
                total_loss += weighted_loss.item() * B

            scheduler.step()
            avg_loss = total_loss / len(dataset)

            if avg_loss < best_loss or epoch % 20 == 0:
                best_loss = min(best_loss, avg_loss)
                safe_save_model(model.state_dict(), model_path)

            current_lr = scheduler.get_last_lr()[0]
            epoch_pbar.set_postfix({
                "OT_Loss": f"{avg_loss:.5f}",
                "Best": f"{best_loss:.5f}",
                "LR": f"{current_lr:.6f}"
            })

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted. Saving checkpoint...", flush=True)
        safe_save_model(model.state_dict(), model_path)
        return

    safe_save_model(model.state_dict(), model_path)
    print(f"\n[SUCCESS] Manual VLA checkpoint saved -> {model_path}", flush=True)

# -----------------------------------------------------------------------------
# 4. Reinforcement Learning Fine-Tuning (Environment in the Loop)
# -----------------------------------------------------------------------------
def compute_episode_reward(sim, trajectory, action_type):
    plat_pos = sim.target_platform_pos.copy()
    min_dist_to_cube = float('inf')
    min_dist_to_plat = float('inf')
    grasped_at_any_point = False

    # Execute full trajectory to completion without premature termination
    for pt in trajectory:
        diff = pt[:3] - sim.ee_pos[:3]
        grip_cmd = 1.0 if pt[3] > 0.45 else 0.0
        act = np.array([diff[0], diff[1], diff[2], 0.0, grip_cmd], dtype=np.float32)
        obs, _ = sim.step_delta(act, max_step=0.010)

        d_cube = np.linalg.norm(sim.ee_pos[:3] - sim.target_cube_pos)
        min_dist_to_cube = min(min_dist_to_cube, d_cube)

        if sim.grasped:
            grasped_at_any_point = True

        # Monitor distance between target cube and target platform
        d_plat = np.linalg.norm(sim.target_cube_pos[:2] - plat_pos[:2])
        min_dist_to_plat = min(min_dist_to_plat, d_plat)

    # Assess final physical placement: is the right cube on top of / at the right platform?
    # Platform radius is 0.035m.
    final_cube_dist_to_plat = np.linalg.norm(sim.target_cube_pos[:2] - plat_pos[:2])
    if action_type == "pick_place":
        # Must be on top of platform (z <= 0.025) and gripper released
        task_succeeded = bool(
            final_cube_dist_to_plat < 0.040 and
            sim.target_cube_pos[2] <= 0.025 and
            not sim.gripper_closed
        )
    else:
        # Push action: cube pushed to / onto platform
        task_succeeded = bool(
            final_cube_dist_to_plat < 0.045 and
            not sim.gripper_closed
        )

    # Dense reward shaping:
    # 1. Approach bonus: max +10 if within reach of cube
    r_approach = max(0.0, (0.20 - min_dist_to_cube) / 0.20) * 10.0
    # 2. Manipulation bonus: +25 if target cube grasped (pick) or contacted (push)
    if action_type == "pick_place":
        r_manip = 25.0 if grasped_at_any_point else 0.0
    else:
        r_manip = 25.0 if min_dist_to_cube < 0.025 else 0.0
    # 3. Transport/Push bonus: max +20 if brought to destination platform
    r_transport = max(0.0, (0.25 - final_cube_dist_to_plat) / 0.25) * 20.0
    # 4. Success bonus: +50 when right thing is on/at right platform
    r_success = 50.0 if task_succeeded else 0.0

    total_reward = r_approach + r_manip + r_transport + r_success
    return total_reward, task_succeeded, grasped_at_any_point, min_dist_to_cube

def rl_finetune(num_episodes=500, lr=2e-5, update_every=4, target_success_rate=90.0):
    sys.path.append(os.path.join(os.path.dirname(__file__), "env"))
    from dobot_env import DobotPickPlaceSim

    print("=" * 68, flush=True)
    print("   MANUAL VLA POLICY: LONG-HORIZON REINFORCEMENT LEARNING", flush=True)
    print("   - Initialized from Pretrained Cross-Attention Transformer Weights", flush=True)
    print("   - Optimization Goal: Train continuously until high success rate", flush=True)
    print("   - Target Window Success Rate: >= {:.1f}%".format(target_success_rate), flush=True)
    print("=" * 68, flush=True)

    sim = DobotPickPlaceSim()
    model = ManualVLAPolicy().to(DEVICE)
    model_path = os.path.join(MODEL_DIR, "dobot_bc_policy.pth")

    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f">> Loaded pre-trained weights from {model_path}", flush=True)
        except Exception as e:
            print(f">> Starting fresh ({e})", flush=True)

    # Demo anchor dataset to prevent catastrophic forgetting
    demo_dataset = OTFlowMatchingDataset(DATA_DIR)
    demo_loader = DataLoader(demo_dataset, batch_size=4, shuffle=True)
    demo_iter = iter(demo_loader)
    loss_fn = nn.SmoothL1Loss(reduction='mean')

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    running_baseline = 10.0
    best_success_rate = 0.0
    recent_successes = []

    ep = 0
    while True:
        ep += 1
        if num_episodes > 0 and ep > num_episodes:
            print(f"\n[INFO] Reached requested episode limit ({num_episodes}). Finishing RL.", flush=True)
            break

        action_type = "pick_place" if (ep % 2 == 0) else "push"
        obs = sim.reset(random_scene=True, num_distractors=2, action_type=action_type)

        img_t = torch.tensor(obs["image"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        intent_raw = get_intent_embedding_vector(action_type, sim.target_color, sim.target_plat_color)
        intent_t = torch.tensor(intent_raw, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        proprio_t = torch.tensor(obs["proprio"], dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            clean_traj = model.sample(img_t, intent_t, proprio=proprio_t, num_steps=20).squeeze(0)

        # Subtle exploration noise (decaying as policy matures)
        noise_std = max(0.004, 0.010 * (0.998 ** ep))
        noise = torch.randn_like(clean_traj) * noise_std
        exp_traj_t = clean_traj + noise
        exp_traj = exp_traj_t.cpu().numpy()

        reward, succ, grasped, min_d = compute_episode_reward(sim, exp_traj, action_type)
        recent_successes.append(1.0 if succ else 0.0)
        if len(recent_successes) > 30:
            recent_successes.pop(0)

        advantage = reward - running_baseline
        running_baseline = 0.95 * running_baseline + 0.05 * reward

        # 1. Flow-Matching Loss on current exploration trajectory (weighted by advantage)
        t_rand = torch.rand(1, device=DEVICE)
        x_target = exp_traj_t.unsqueeze(0)
        # Reconstruct vector field from random noise to target trajectory
        x_noise = torch.randn_like(x_target)
        x_interp = (1.0 - t_rand.view(1, 1, 1)) * x_target + t_rand.view(1, 1, 1) * x_noise
        v_target = x_noise - x_target
        v_pred = model.forward_flow(x_interp, t_rand, img_t, intent_t, proprio=proprio_t)

        flow_l2 = loss_fn(v_pred, v_target)
        # Reinforce positive advantage trajectories; penalize bad moves
        adv_clipped = float(np.clip(advantage, -20.0, 30.0))
        # Positive advantage -> guide towards explored trajectory; Negative advantage -> move away
        rl_loss = - (adv_clipped / 30.0) * flow_l2 * 0.5

        # 2. Demonstration Replay Anchor (Behavioral Regularization):
        # Keeps policy grounded in clean human/expert demos so RL doesn't collapse or drift
        try:
            d_img, d_intent, d_proprio, d_traj = next(demo_iter)
        except StopIteration:
            demo_iter = iter(demo_loader)
            d_img, d_intent, d_proprio, d_traj = next(demo_iter)

        d_B = d_img.size(0)
        d_noise = torch.randn_like(d_traj)
        d_t = torch.rand(d_B, device=DEVICE)
        d_t_exp = d_t.view(d_B, 1, 1)
        d_interp = (1.0 - d_t_exp) * d_traj + d_t_exp * d_noise
        d_v_target = d_noise - d_traj
        d_v_pred = model.forward_flow(d_interp, d_t, d_img, intent_vec=d_intent, proprio=d_proprio)
        demo_loss = loss_fn(d_v_pred, d_v_target)

        # Combined Loss: 70% Grounded in Demonstrations + 30% RL Performance Optimization
        total_loss = demo_loss * 0.70 + rl_loss * 0.30
        total_loss.backward()

        if ep % update_every == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        succ_pct = sum(recent_successes) / len(recent_successes) * 100.0
        max_str = f"{num_episodes}" if num_episodes > 0 else "INF"
        print(f"EP {ep:04d}/{max_str} | Act: {action_type[:4].upper()} | MinDist: {min_d*1000:.1f}mm | Grasped: {grasped} | Succ: {succ} | R: {reward:.1f} | Adv: {advantage:+.1f} | Win30: {succ_pct:.1f}%", flush=True)

        if succ_pct > best_success_rate or ep % 20 == 0:
            best_success_rate = max(best_success_rate, succ_pct)
            safe_save_model(model.state_dict(), model_path)

        # Convergence criteria: if reached target success rate over rolling window of 30
        if len(recent_successes) >= 30 and succ_pct >= target_success_rate:
            print(f"\n[GOAL REACHED] Model achieved {succ_pct:.1f}% success rate over last 30 trials! Task solved.", flush=True)
            safe_save_model(model.state_dict(), model_path)
            break

    safe_save_model(model.state_dict(), model_path)
    print(f"\n[DONE] Long-Horizon RL complete. Model checkpoint saved -> {model_path}", flush=True)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "rl":
        episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 0 # 0 means train until success
        target_pct = float(sys.argv[3]) if len(sys.argv) > 3 else 90.0
        rl_finetune(num_episodes=episodes, target_success_rate=target_pct)
    else:
        epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 250
        train(epochs=epochs)
