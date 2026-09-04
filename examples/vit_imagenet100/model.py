"""
Vision Transformer (ViT) for ImageNet-100 (224x224, 100 classes)

Supports APA (Adaptive Precision Architecture), native FP8/FP16/TF32/FP32 training,
and standard ViT model scales: ViT-Tiny, ViT-Small, and ViT-Base.
"""

import math
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from apa import APALinear, APAConfig

MODEL_PRESETS = {
    'tiny': {
        'dim': 192,
        'depth': 12,
        'heads': 3,
        'mlp_dim': 768,
        'desc': "ViT-Tiny (ViT-Ti/16, ~5.7M params)",
    },
    'small': {
        'dim': 384,
        'depth': 12,
        'heads': 6,
        'mlp_dim': 1536,
        'desc': "ViT-Small (ViT-S/16, ~22M params)",
    },
    'base': {
        'dim': 768,
        'depth': 12,
        'heads': 12,
        'mlp_dim': 3072,
        'desc': "ViT-Base (ViT-B/16, ~86M params)",
    },
}

def _create_linear(in_features, out_features, bias=True, config=None, use_apa=True):
    """Creates an APALinear or standard nn.Linear depending on mode."""
    if use_apa and config is not None:
        return APALinear(in_features, out_features, bias=bias, config=config)
    return nn.Linear(in_features, out_features, bias=bias)

class PatchEmbedding(nn.Module):
    """Splits image into non-overlapping patches and projects to hidden dimension."""
    def __init__(self, in_channels=3, patch_size=16, dim=384, config=None, use_apa=True):
        super().__init__()
        self.patch_size = patch_size
        self.proj = _create_linear(in_channels * patch_size * patch_size, dim, config=config, use_apa=use_apa)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        p = self.patch_size
        # [B, C, H/p, p, W/p, p] -> [B, (H/p)*(W/p), C*p*p]
        x = x.view(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, (H // p) * (W // p), C * p * p)
        return self.proj(x)

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention supporting FlashAttention / fast SDPA."""
    def __init__(self, dim, heads=6, config=None, use_apa=True, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout_p = dropout
        
        self.qkv_proj = _create_linear(dim, 3 * dim, config=config, use_apa=use_apa)
        self.out_proj = _create_linear(dim, dim, config=config, use_apa=use_apa)

    def forward(self, x):
        B, N, C = x.shape
        # qkv: [B, N, 3 * C]
        qkv = self.qkv_proj(x)
        # Reshape to [3, B, heads, N, head_dim]
        qkv = qkv.reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Fast PyTorch scaled_dot_product_attention (FlashAttention when FP16/BF16)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            scale=self.scale
        )
        # Reshape back to [B, N, C]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    """Pre-LN Transformer Block with Attention and Feed-Forward MLP."""
    def __init__(self, dim, heads, mlp_dim, config=None, use_apa=True, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads=heads, config=config, use_apa=use_apa, dropout=dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = _create_linear(dim, mlp_dim, config=config, use_apa=use_apa)
        self.act = nn.GELU()
        self.fc2 = _create_linear(mlp_dim, dim, config=config, use_apa=use_apa)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        mlp_out = self.fc2(self.act(self.fc1(self.norm2(x))))
        x = x + self.dropout(mlp_out)
        return x

class VisionTransformerImageNet(nn.Module):
    """
    Vision Transformer tailored for ImageNet-100.
    
    Defaults:
        image_size: 224
        patch_size: 16 (196 patches + 1 [CLS] token = 197 tokens)
        num_classes: 100
    """
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=100,
        dim=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
        config=None,
        use_apa=True,
        preserve_critical_layers=True,
        dropout=0.0
    ):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.dim = dim
        self.num_classes = num_classes
        
        # Boundary layer preservation (patch embedding and classification head
        # are preserved in high precision FP16/FP32 for numerical stability)
        use_apa_boundary = use_apa and not preserve_critical_layers
        
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            dim=dim,
            config=config,
            use_apa=use_apa_boundary
        )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Initialize positional embeddings and cls token with truncated normal
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                heads=heads,
                mlp_dim=mlp_dim,
                config=config,
                use_apa=use_apa,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.head = _create_linear(dim, num_classes, config=config, use_apa=use_apa_boundary)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        # x: [B, N_patches, dim]
        x = self.patch_embed(x)
        
        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embedding)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x[:, 0])  # [CLS] token representation
        x = self.head(x)        # [B, num_classes]
        return x

def build_vit_imagenet100(
    model_size='small',
    config=None,
    use_apa=True,
    preserve_critical_layers=True,
    image_size=224,
    patch_size=16,
    num_classes=100,
    dropout=0.0
):
    """
    Factory function to instantiate ViT-Tiny, ViT-Small, or ViT-Base for ImageNet-100.
    """
    if model_size not in MODEL_PRESETS:
        raise ValueError(f"Unknown model_size '{model_size}'. Choose from: {list(MODEL_PRESETS.keys())}")
        
    preset = MODEL_PRESETS[model_size]
    model = VisionTransformerImageNet(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=preset['dim'],
        depth=preset['depth'],
        heads=preset['heads'],
        mlp_dim=preset['mlp_dim'],
        config=config,
        use_apa=use_apa,
        preserve_critical_layers=preserve_critical_layers,
        dropout=dropout
    )
    return model
