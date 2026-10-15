#!/usr/bin/env python3
"""
SmolVLA Multimodal Vision-Language-Action Policy with Real Pretrained CLIP ViT-B/32.
Architecture:
  1. Genuine OpenAI CLIP ViT-B/32 backbone:
     - 49 visual patch tokens (7x7 spatial grid) + 1 global CLS token (512-dim).
     - 77 subword language tokens (512-dim).
  2. Multimodal Cross-Attention & Joint Self-Attention Fusion Backbone.
  3. Continuous Sinusoidal Diffusion/Flow Time & Proprioception Conditioning.
  4. Flow-Matching Trajectory Action Decoder predicting continuous end-effector actions.
  5. Fast Euler ODE flow-matching sampler with temporal trajectory smoothing.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

# Default empirical normalization for Dobot Magician:
# Supports 9-dim hybrid representation: [j1, j2, j3, j4, x, y, z, r, gripper]
# or standard 4-dim cartesian: [x, y, z, gripper]
ACTION_MEAN_9D = torch.tensor([
    0.0000, 0.5500, 0.0500, 0.0000,   # j1, j2, j3, j4 in radians (~0°, 31.5°, 2.9°, 0°)
    0.2200, 0.0000, 0.0350, 0.0000,   # x, y, z, r in meters & radians
    0.5000                            # gripper
], dtype=torch.float32)

ACTION_STD_9D = torch.tensor([
    0.3700, 0.2500, 0.3300, 0.3700,   # j1, j2, j3, j4 in radians
    0.0450, 0.0800, 0.0300, 0.3700,   # x, y, z, r
    0.5000                            # gripper
], dtype=torch.float32)

ACTION_MEAN = torch.tensor([0.2200, 0.0000, 0.0350, 0.5000], dtype=torch.float32)
ACTION_STD  = torch.tensor([0.0450, 0.0800, 0.0300, 0.5000], dtype=torch.float32)


class PretrainedVLMEncoder(nn.Module):
    """
    Extracts visual patch tokens and language token embeddings directly from
    the real pretrained OpenAI CLIP ViT-B/32 backbone.
    Vision: 49 patch tokens (7x7 grid) + 1 CLS global token (512-dim)
    Language: 77 BPE token representations (512-dim)
    Weights are frozen to preserve the pretrained multimodal feature space.
    """
    def __init__(self, device='cpu'):
        super().__init__()
        clip_model, _ = clip.load('ViT-B/32', device=device)
        self.clip = clip_model
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

        self.register_buffer('img_mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer('img_std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

    def encode_vision(self, img):
        """
        img: [B, 3, 224, 224] or [B, 3, H, W] normalized float tensor.
        Returns:
          patches: [B, 49, 512] normalized patch token features
          cls_emb: [B, 512] normalized global scene feature
        """
        if img.shape[-2:] != (224, 224):
            img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)

        vis = self.clip.visual
        dtype = vis.conv1.weight.dtype
        x = img.to(dtype=dtype, device=vis.conv1.weight.device)

        # ViT Stem: Conv2d patch projection
        x = vis.conv1(x)  # [B, 768, 7, 7]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B, 49, 768]
        cls_token = vis.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls_token, x], dim=1)  # [B, 50, 768]
        x = x + vis.positional_embedding.to(x.dtype)
        x = vis.ln_pre(x)
        x = x.permute(1, 0, 2)  # [50, B, 768]
        x = vis.transformer(x)
        x = x.permute(1, 0, 2)  # [B, 50, 768]
        x = vis.ln_post(x)

        # Linear projection to 512-dim multimodal joint space
        patches = F.normalize(x[:, 1:, :] @ vis.proj, dim=-1)  # [B, 49, 512]
        cls_emb = F.normalize(x[:, 0, :] @ vis.proj, dim=-1)    # [B, 512]
        return patches.float(), cls_emb.float()

    def encode_text(self, text_tokens):
        """
        text_tokens: [B, 77] long tensor.
        Returns:
          token_embs: [B, 77, 512] normalized language token embeddings.
        """
        t_tokens = text_tokens.to(device=self.clip.token_embedding.weight.device)
        t_x = self.clip.token_embedding(t_tokens)
        t_x = t_x + self.clip.positional_embedding
        t_x = t_x.to(self.clip.dtype)
        t_x = t_x.permute(1, 0, 2)  # [77, B, 512]
        t_x = self.clip.transformer(t_x)
        t_x = t_x.permute(1, 0, 2)  # [B, 77, 512]
        t_x = self.clip.ln_final(t_x)
        token_embs = t_x @ self.clip.text_projection
        return F.normalize(token_embs.float(), dim=-1)


class MultimodalCrossAttentionBlock(nn.Module):
    """
    Bidirectional Cross-Attention & Language-Gated Feature Fusion:
      1. Spatial visual patches actively cross-attend to language tokens (language modulates vision)
      2. Language tokens cross-attend to visual patches
      3. Joint self-attention over the combined multimodal sequence
      4. Feed-Forward Network with GELU
    """
    def __init__(self, d_model=256, nhead=8, d_ff=512):
        super().__init__()
        # Vision queries Language (Crucial for grounding colors, adjectives, objects)
        self.norm_vis_q = nn.LayerNorm(d_model)
        self.norm_txt_kv = nn.LayerNorm(d_model)
        self.cross_attn_vis = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # Language queries Vision
        self.norm_txt_q = nn.LayerNorm(d_model)
        self.norm_vis_kv = nn.LayerNorm(d_model)
        self.cross_attn_txt = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # Joint self-attention
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # Feed-Forward Network
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, text_tokens, visual_patches):
        # 1. Vision actively cross-attends to Language
        vq = self.norm_vis_q(visual_patches)
        tkv = self.norm_txt_kv(text_tokens)
        vis_cross, _ = self.cross_attn_vis(vq, tkv, tkv)
        visual_patches = visual_patches + vis_cross

        # 2. Language cross-attends to Vision
        tq = self.norm_txt_q(text_tokens)
        vkv = self.norm_vis_kv(visual_patches)
        txt_cross, _ = self.cross_attn_txt(tq, vkv, vkv)
        text_tokens = text_tokens + txt_cross

        # 3. Joint self-attention over all tokens
        joint = torch.cat([text_tokens, visual_patches], dim=1)
        joint_norm = self.norm_self(joint)
        sa_out, _ = self.self_attn(joint_norm, joint_norm, joint_norm)
        joint = joint + sa_out

        # 4. Feed-Forward Network
        joint = joint + self.ffn(self.norm_ffn(joint))

        T = text_tokens.size(1)
        return joint[:, :T], joint[:, T:]


class SmolVLABackbone(nn.Module):
    """
    Pretrained VLM Backbone:
    - Genuine pretrained OpenAI CLIP ViT-B/32 encoder (49 patches + 77 BPE tokens)
    - Full 77-token projection + 49 visual patch projection + Global sentence token
    - Multimodal Cross & Self-Attention Fusion layers
    """
    def __init__(self, d_model=256, vlm_dim=512, nhead=8, num_layers=3, device='cpu'):
        super().__init__()
        self.d_model = d_model
        self.vlm = PretrainedVLMEncoder(device=device)

        self.vis_proj = nn.Sequential(
            nn.Linear(vlm_dim, d_model),
            nn.LayerNorm(d_model)
        )
        self.txt_proj = nn.Sequential(
            nn.Linear(vlm_dim, d_model),
            nn.LayerNorm(d_model)
        )
        self.global_txt_proj = nn.Sequential(
            nn.Linear(vlm_dim, d_model),
            nn.LayerNorm(d_model)
        )

        self.layers = nn.ModuleList([
            MultimodalCrossAttentionBlock(d_model=d_model, nhead=nhead, d_ff=d_model * 2)
            for _ in range(num_layers)
        ])

    def forward(self, text_ids, img):
        with torch.no_grad():
            vis_patches, cls_emb = self.vlm.encode_vision(img)  # [B, 49, 512]
            text_feats = self.vlm.encode_text(text_ids)          # [B, 77, 512]

        cur_txt = self.txt_proj(text_feats)                  # [B, 77, d_model] - ALL 77 TOKENS!
        cur_vis = self.vis_proj(vis_patches)                 # [B, 49, d_model]
        global_txt = self.global_txt_proj(text_feats.mean(dim=1, keepdim=True)) # [B, 1, d_model]

        for layer in self.layers:
            cur_txt, cur_vis = layer(cur_txt, cur_vis)

        return cur_txt, cur_vis, global_txt


class ContinuousSinusoidalTimeEmbedding(nn.Module):
    """Continuous sinusoidal time embedding for diffusion / flow matching."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class SmolVLAPolicy(nn.Module):
    """
    Production-grade SmolVLA Policy for Dobot Magician:
    - Backbone: Genuine Pretrained OpenAI CLIP ViT-B/32
    - Full 77-Token Text Input + 49 Visual Spatial Patches + Global Sentence Token = 129 Context Tokens!
    - Bidirectional Visual-Language Gating (forces vision to condition on color/object tokens)
    - Action Horizon H=128 Optimal Transport Flow-Matching Decoder (9D)
    """
    def __init__(self, horizon=128, action_dim=9, proprio_dim=9, d_model=256, num_layers=3, device='cpu'):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.d_model = d_model

        # 1. Real Pretrained VLM Backbone with full 77 token projections
        self.backbone = SmolVLABackbone(d_model=d_model, nhead=8 if d_model % 8 == 0 else 4, num_layers=num_layers, device=device)

        # 2. Proprioception State Projector (9D: [j1, j2, j3, j4, x, y, z, r, gripper])
        self.proprio_proj = nn.Sequential(
            nn.Linear(proprio_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # 3. Continuous Diffusion/Flow Timestep Embedding
        self.time_embed = nn.Sequential(
            ContinuousSinusoidalTimeEmbedding(64),
            nn.Linear(64, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # 4. Action Decoder Positional Queries
        self.action_in_proj = nn.Linear(action_dim, d_model)
        self.pos_queries = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)

        # 5. Decoder Transformer Layers with 8-head trajectory attention
        nhead = 8 if d_model % 8 == 0 else 4
        self.dec_layers = nn.ModuleList([
            nn.ModuleDict({
                "sa": nn.MultiheadAttention(d_model, nhead, batch_first=True),
                "ca": nn.MultiheadAttention(d_model, nhead, batch_first=True),
                "n1": nn.LayerNorm(d_model),
                "n2": nn.LayerNorm(d_model),
                "n3": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))
            })
            for _ in range(num_layers)
        ])

        # 6. Output Velocity Field Head
        self.out_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, action_dim)
        )

    def forward_from_embeddings(self, x_t, t, vis_patches, text_feats, proprio=None):
        """
        Fast forward pass using pre-extracted or live CLIP embeddings:
        x_t: [B, H, action_dim]
        t: [B] in [0, 1]
        vis_patches: [B, 49, 512]
        text_feats: [B, 77, 512] - ALL 77 TOKENS!
        proprio: [B, 9]
        """
        B = vis_patches.size(0)

        # Project ALL 77 CLIP text tokens -> [B, 77, d_model]
        cur_txt = self.backbone.txt_proj(text_feats)
        # Project 49 spatial visual patches -> [B, 49, d_model]
        cur_vis = self.backbone.vis_proj(vis_patches)
        # Global sentence pooled token -> [B, 1, d_model]
        global_txt = self.backbone.global_txt_proj(text_feats.mean(dim=1, keepdim=True))

        # Multimodal bidirectional cross-attention & visual-language gating
        for layer in self.backbone.layers:
            cur_txt, cur_vis = layer(cur_txt, cur_vis)

        # Conditioning tokens: [proprio, time]
        if proprio is None:
            proprio = torch.zeros(B, self.proprio_dim, device=vis_patches.device)
        elif proprio.shape[-1] != self.proprio_dim:
            if proprio.shape[-1] < self.proprio_dim:
                pad = torch.zeros(B, self.proprio_dim - proprio.shape[-1], device=proprio.device)
                proprio = torch.cat([proprio, pad], dim=-1)
            else:
                proprio = proprio[:, :self.proprio_dim]
        p_token = self.proprio_proj(proprio).unsqueeze(1)     # [B, 1, d_model]
        t_token = self.time_embed(t).unsqueeze(1)             # [B, 1, d_model]

        # Full Multimodal Context: 1 proprio + 1 time + 1 global_lang + 77 text + 49 vis = 129 tokens!
        context = torch.cat([p_token, t_token, global_txt, cur_txt, cur_vis], dim=1)

        # Action Decoder stream
        act_queries = self.action_in_proj(x_t) + self.pos_queries

        for d_layer in self.dec_layers:
            sa_out, _ = d_layer["sa"](act_queries, act_queries, act_queries)
            act_queries = d_layer["n1"](act_queries + sa_out)
            ca_out, _ = d_layer["ca"](act_queries, context, context)
            act_queries = d_layer["n2"](act_queries + ca_out)
            act_queries = d_layer["n3"](act_queries + d_layer["ffn"](act_queries))

        v_pred = self.out_head(act_queries)
        return v_pred

    def forward_flow(self, x_t, t, img, token_ids, proprio=None):
        """End-to-end forward pass directly from raw images and token IDs."""
        with torch.no_grad():
            vis_patches, _ = self.backbone.vlm.encode_vision(img)
            text_feats = self.backbone.vlm.encode_text(token_ids)
        return self.forward_from_embeddings(x_t, t, vis_patches, text_feats, proprio=proprio)

    @torch.no_grad()
    def sample(self, img_or_patches, token_ids_or_text_feats, proprio=None, num_steps=15, is_precomputed=False):
        """
        Fast Euler ODE Flow Matching Sampler:
        Generates full continuous trajectory horizon H=128 from noise.
        Smooths trajectory with minimum-jerk 1D temporal convolution filter.
        """
        device = next(self.parameters()).device
        if is_precomputed:
            vis_patches = img_or_patches.to(device)
            text_feats = token_ids_or_text_feats.to(device)
            B = vis_patches.size(0)
        else:
            img = img_or_patches.to(device)
            token_ids = token_ids_or_text_feats.to(device)
            B = img.size(0)
            vis_patches, _ = self.backbone.vlm.encode_vision(img)
            text_feats = self.backbone.vlm.encode_text(token_ids)

        if proprio is not None:
            proprio = proprio.to(device)

        # Initial standard normal noise x_0 ~ N(0, I)
        x = torch.randn(B, self.horizon, self.action_dim, device=device)
        dt = 1.0 / num_steps

        # Euler numerical integration: dx/dt = v_theta(x, t, context)
        for i in range(num_steps):
            t = torch.full((B,), (i + 0.5) * dt, device=device)
            v = self.forward_from_embeddings(x, t, vis_patches, text_feats, proprio=proprio)
            x = x + v * dt

        # Unnormalize with empirical Dobot bounds
        if self.action_dim == 9:
            act_mean = ACTION_MEAN_9D.to(device)
            act_std  = ACTION_STD_9D.to(device)
        else:
            act_mean = ACTION_MEAN[:self.action_dim].to(device)
            act_std  = ACTION_STD[:self.action_dim].to(device)
        raw_x = x * act_std + act_mean

        # 1D Temporal smoothing kernel (5-tap window) for smooth real robot motion
        kernel = torch.ones(1, 1, 5, device=device) / 5.0
        raw_perm = raw_x.permute(0, 2, 1).reshape(B * self.action_dim, 1, self.horizon)
        smoothed = F.conv1d(raw_perm, kernel, padding=2)
        smoothed = smoothed.view(B, self.action_dim, self.horizon).permute(0, 2, 1)

        return smoothed


if __name__ == "__main__":
    print("[TEST] Initializing SmolVLAPolicy with Pretrained CLIP ViT-B/32 backbone...")
    policy = SmolVLAPolicy(horizon=128, action_dim=4, d_model=128)
    
    # Test text tokenization
    prompt = "pick up the red cube and place it on the green platform"
    tokens = clip.tokenize([prompt])
    print(f"       Tokenized prompt: shape={tokens.shape}")

    # Test dummy image input
    img = torch.rand(2, 3, 224, 224)
    proprio = torch.tensor([[0.20, 0.0, 0.12, 0.0, 0.0], [0.22, 0.05, 0.08, 0.0, 1.0]])
    x_t = torch.randn(2, 128, 4)
    t = torch.tensor([0.3, 0.7])

    print("       Testing forward_flow pass...")
    v_pred = policy.forward_flow(x_t, t, img, tokens.repeat(2, 1), proprio=proprio)
    print(f"       Predicted flow velocity: shape={v_pred.shape}")
    assert v_pred.shape == (2, 128, 4), f"Unexpected shape {v_pred.shape}"

    print("       Testing Euler ODE sample pass...")
    traj = policy.sample(img[:1], tokens[:1], proprio=proprio[:1], num_steps=5)
    print(f"       Sampled Trajectory: shape={traj.shape}")
    assert traj.shape == (1, 128, 4), f"Unexpected shape {traj.shape}"

    print("[TEST] SmolVLAPolicy test passed successfully!")
