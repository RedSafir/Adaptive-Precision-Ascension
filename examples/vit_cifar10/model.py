import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from apa import APALinear, APAConfig

def _create_linear(in_features, out_features, bias=True, config=None, use_apa=True):
    if use_apa:
        return APALinear(in_features, out_features, bias=bias, config=config)
    else:
        return nn.Linear(in_features, out_features, bias=bias)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, dim=256, config=None, use_apa=True):
        super().__init__()
        self.patch_size = patch_size
        self.proj = _create_linear(in_channels * patch_size**2, dim, config=config, use_apa=use_apa)

    def forward(self, x):
        # x is (B, C, H, W)
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        # x is (B, C, H//p, W//p, p, p)
        x = x.contiguous().view(B, C, -1, p, p)
        # x is (B, C, N, p, p) where N = (H//p)*(W//p)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B, -1, C * p * p)
        # x is (B, N, C*p*p)
        x = self.proj(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=4, config=None, use_apa=True):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv_proj = _create_linear(dim, 3 * dim, config=config, use_apa=use_apa)
        self.out_proj = _create_linear(dim, dim, config=config, use_apa=use_apa)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, N, self.heads, self.head_dim).transpose(1, 2), qkv)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_dim=512, config=None, use_apa=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.mha = MultiHeadAttention(dim, heads=heads, config=config, use_apa=use_apa)
        
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            _create_linear(dim, mlp_dim, config=config, use_apa=use_apa),
            nn.GELU(),
            _create_linear(mlp_dim, dim, config=config, use_apa=use_apa)
        )

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self, 
        image_size=32, 
        patch_size=4, 
        in_channels=3, 
        num_classes=10, 
        dim=256, 
        depth=6, 
        heads=4, 
        mlp_dim=512, 
        config=None,
        use_apa=True,
        preserve_critical_layers=True
    ):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        
        # In modern mixed-precision and FP8 architectures (TransformerEngine, Megatron),
        # boundary layers (input patch embedding and output classification head) are
        # typically preserved in FP16/FP32 to ensure stable representation and lossless logits.
        use_apa_boundary = use_apa and not preserve_critical_layers
        self.patch_embed = PatchEmbedding(in_channels, patch_size, dim, config=config, use_apa=use_apa_boundary)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, config=config, use_apa=use_apa)
            for _ in range(depth)
        ])
        
        self.ln = nn.LayerNorm(dim)
        self.head = _create_linear(dim, num_classes, config=config, use_apa=use_apa_boundary)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln(x[:, 0]) # get cls token
        x = self.head(x)
        return x
