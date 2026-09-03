import sys
import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from apa import APALinear, APAConfig

def _create_linear(in_features, out_features, bias=True, config=None, use_apa=True):
    if use_apa:
        return APALinear(in_features, out_features, bias=bias, config=config)
    else:
        return nn.Linear(in_features, out_features, bias=bias)

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1, config=None, use_apa=True):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = _create_linear(n_embd, 3 * n_embd, config=config, use_apa=use_apa)
        self.c_proj = _create_linear(n_embd, n_embd, config=config, use_apa=use_apa)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                    .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # FlashAttention (PyTorch 2.0+ native fused causal attention)
        y = F.scaled_dot_product_attention(
            q, k, v, 
            is_causal=True, 
            dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.1, config=None, use_apa=True):
        super().__init__()
        self.c_fc = _create_linear(n_embd, 4 * n_embd, config=config, use_apa=use_apa)
        self.gelu = nn.GELU()
        self.c_proj = _create_linear(4 * n_embd, n_embd, config=config, use_apa=use_apa)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1, config=None, use_apa=True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout, config=config, use_apa=use_apa)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout, config=config, use_apa=use_apa)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, block_size=128, n_layer=4, n_head=4, n_embd=256, dropout=0.1, config=None, use_apa=True):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size, dropout, config, use_apa=use_apa) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = _create_linear(n_embd, vocab_size, config=config, use_apa=use_apa)

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
