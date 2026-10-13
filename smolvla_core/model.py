import torch
import torch.nn as nn
import numpy as np

COLOR_MAP = {
    "red": [1, 0, 0, 0, 0, 0, 0],
    "blue": [0, 1, 0, 0, 0, 0, 0],
    "yellow": [0, 0, 1, 0, 0, 0, 0],
    "green": [0, 0, 0, 1, 0, 0, 0],
    "purple": [0, 0, 0, 0, 1, 0, 0],
    "orange": [0, 0, 0, 0, 0, 1, 0],
    "cyan": [0, 0, 0, 0, 0, 0, 1]
}

ACTION_TYPE_MAP = {
    "pick_place": [1, 0],
    "push": [0, 1]
}

def build_intent_vector(action_type="pick_place", target_obj_color="red", target_dest_color="green"):
    act_vec = ACTION_TYPE_MAP.get(action_type, [1, 0])
    src_vec = COLOR_MAP.get(target_obj_color, COLOR_MAP["red"])
    dst_vec = COLOR_MAP.get(target_dest_color, COLOR_MAP["green"])
    # Total dim: 2 + 7 + 7 = 16
    return np.array(act_vec + src_vec + dst_vec, dtype=np.float32)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, 3, 64, 64)
        x = self.proj(x)  # (B, embed_dim, 4, 4)
        x = x.flatten(2).transpose(1, 2)  # (B, 16, embed_dim)
        return x

class VLA_Transformer(nn.Module):
    def __init__(self, img_size=64, patch_size=16, embed_dim=128, intent_dim=16, proprio_dim=5, action_dim=5):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(3, patch_size, embed_dim)
        num_patches = (img_size // patch_size) ** 2  # 16 patches
        
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        
        # Hadamard MLP Intent conditioning
        self.intent_embed = nn.Sequential(
            nn.Linear(intent_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Proprioception state encoder
        self.proprio_embed = nn.Sequential(
            nn.Linear(proprio_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Lightweight Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=4, 
            dim_feedforward=embed_dim * 2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Action Head: predicts delta EE [dx, dy, dz, dyaw, gripper]
        self.action_head = nn.Sequential(
            nn.Linear((num_patches + 1) * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, action_dim)
        )
        
    def forward(self, img, intent, proprio):
        # img: (B, 3, 64, 64)
        # intent: (B, 16)
        # proprio: (B, 5)
        
        # 1. Vision tokens + Position embedding
        patches = self.patch_embed(img) + self.pos_embed # (B, 16, embed_dim)
        
        # 2. Intent token via Hadamard Product conditioning on visual patches
        i_feat = self.intent_embed(intent).unsqueeze(1) # (B, 1, embed_dim)
        conditioned_patches = patches * i_feat # Hadamard MLP modulation
        
        # 3. Proprioception token
        p_feat = self.proprio_embed(proprio).unsqueeze(1) # (B, 1, embed_dim)
        
        # Combine conditioned visual patches and proprioception
        tokens = torch.cat([conditioned_patches, p_feat], dim=1) # (B, 17, embed_dim)
        
        # 4. Transformer reasoning
        feat = self.transformer(tokens)
        
        # 5. Output action prediction
        actions = self.action_head(feat.flatten(1)) # (B, 5)
        return actions

if __name__ == "__main__":
    m = VLA_Transformer()
    dummy_img = torch.randn(2, 3, 64, 64)
    dummy_intent = torch.randn(2, 16)
    dummy_prop = torch.randn(2, 5)
    out = m(dummy_img, dummy_intent, dummy_prop)
    print("Action output shape:", out.shape)
