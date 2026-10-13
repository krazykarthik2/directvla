import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

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
        
        # 7x7 spatial patch coordinates buffer in [-1, 1] range
        y_g, x_g = torch.meshgrid(torch.linspace(-1, 1, 7), torch.linspace(-1, 1, 7), indexing='ij')
        self.register_buffer('grid_x', x_g.reshape(1, 1, 49))
        self.register_buffer('grid_y', y_g.reshape(1, 1, 49))

    def encode_vision(self, img):
        # img: [B, 3, 64, 64] float in [0, 1]
        img_224 = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
        # Ensure normalization tensors are on the same device as the input
        img_mean = self.img_mean.to(img.device)
        img_std = self.img_std.to(img.device)
        x = (img_224 - img_mean) / img_std
        
        vis = self.clip.visual
        x = x.to(vis.conv1.weight.dtype)  # CLIP loads as float16 on CUDA
        x = vis.conv1(x) # [B, 768, 7, 7]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1) # [B, 49, 768]
        cls_token = vis.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls_token, x], dim=1) # [B, 50, 768]
        x = x + vis.positional_embedding.to(x.dtype)
        x = vis.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = vis.transformer(x)
        x = x.permute(1, 0, 2)
        x = vis.ln_post(x)
        
        # Project to 512-dim
        patches = F.normalize(x[:, 1:, :] @ vis.proj, dim=-1) # [B, 49, 512]
        cls_emb = F.normalize(x[:, 0, :] @ vis.proj, dim=-1)   # [B, 512]
        return patches, cls_emb

    def encode_text(self, text_tokens):
        # text_tokens: [B, 77]
        t_x = self.clip.token_embedding(text_tokens)
        t_x = t_x + self.clip.positional_embedding
        t_x = t_x.to(self.clip.dtype)  # match CLIP weights (float16 on CUDA)
        t_x = t_x.permute(1, 0, 2)
        t_x = self.clip.transformer(t_x)
        t_x = t_x.permute(1, 0, 2)
        t_x = self.clip.ln_final(t_x)
        token_embs = t_x @ self.clip.text_projection
        return F.normalize(token_embs, dim=-1) # [B, 77, 512]

class MultimodalCrossAttentionBlock(nn.Module):
    def __init__(self, d_model=128, nhead=4, d_ff=256):
        super().__init__()
        self.norm_cross_txt = nn.LayerNorm(d_model)
        self.norm_cross_vis = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, text_tokens, visual_patches):
        # 1. Cross-attention: Language attends to vision
        q = self.norm_cross_txt(text_tokens)
        kv = self.norm_cross_vis(visual_patches)
        cross_out, weights = self.cross_attn(q, kv, kv, need_weights=True)
        text_tokens = text_tokens + cross_out
        
        # 2. Joint multimodal self-attention
        joint = torch.cat([text_tokens, visual_patches], dim=1)
        joint_norm = self.norm_self(joint)
        sa_out, _ = self.self_attn(joint_norm, joint_norm, joint_norm)
        joint = joint + sa_out
        
        # 3. FFN
        joint = joint + self.ffn(self.norm_ffn(joint))
        
        T = text_tokens.size(1)
        return joint[:, :T], joint[:, T:], weights

class SmolVLABackbone(nn.Module):
    """
    Pretrained VLM Backbone:
    - Genuine pretrained OpenAI CLIP ViT-B/32 encoder (49 patches + 77 BPE tokens)
    - Projections from 512 -> d_model (128)
    - Multimodal Cross & Self-Attention Fusion layers
    """
    def __init__(self, d_model=128, vlm_dim=512, nhead=4, num_layers=2, device='cpu'):
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
        
        self.layers = nn.ModuleList([
            MultimodalCrossAttentionBlock(d_model=d_model, nhead=nhead, d_ff=d_model * 2)
            for _ in range(num_layers)
        ])

    def forward(self, text_ids, img):
        with torch.no_grad():
            vis_patches, cls_emb = self.vlm.encode_vision(img) # [B, 49, 512]
            text_feats = self.vlm.encode_text(text_ids)         # [B, 77, 512]
            
            # Zero-shot pretrained similarity heatmap
            base_weights = torch.einsum('bld,bpd->blp', text_feats, vis_patches) # [B, 77, 49]
            base_probs = F.softmax(base_weights * 5.0, dim=-1)

        # Project 512 -> 128
        cur_txt = self.txt_proj(text_feats[:, :64]) # [B, 64, 128]
        cur_vis = self.vis_proj(vis_patches)        # [B, 49, 128]

        for layer in self.layers:
            cur_txt, cur_vis, _ = layer(cur_txt, cur_vis)

        return cur_txt, cur_vis, base_probs

