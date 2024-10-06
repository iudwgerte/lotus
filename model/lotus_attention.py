import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lotus_config import LotusConfig
from model.lotus_utils import *

import math

class Attention(nn.Module):
    def __init__(self, config: LotusConfig):
        super().__init__()
        self.n_kv_heads = config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        assert config.n_heads % config.n_kv_heads == 0, f"n_heads must be divisible by n_kv_heads, but got {config.n_heads} and {config.n_kv_heads}."
        self.n_local_heads = config.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_reps = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.dim // config.n_heads
        
        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

        self.k_cache, self.v_cache = None, None

        self.dropout_val = config.dropout
        self.dropout = nn.Dropout(config.dropout)

        self.flash_attn = hasattr(F, "scaled_dot_product_attention") and config.flash_attention

        if not self.flash_attn:
            print("WARNING: Using slower attention implementation. Flash Attention requires PyTorch >= 2.0")
            mask = torch.full((1, 1, config.max_seq_len, config.max_seq_len), float("-inf"))
            mask = mask.triu(mask, diagonal=1)
            self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, pos_cis: torch.Tensor, kv_cache = False) -> torch.Tensor:
        bsz, seqlen, _ = x.size()

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rot_emb(xq, xk, pos_cis)

        if kv_cache and self.eval():
            if seqlen == 1 and all(cache is None for cache in (self.k_cache, self.v_cache)):
                xk = torch.cat([self.k_cache, xk], dim=1)
                xv = torch.cat([self.v_cache, xv], dim=1)
            self.k_cache, self.v_cache = xk, xv

        xk = repeat_kv(xk, self.n_reps)
        xv = repeat_kv(xv, self.n_reps)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if self.flash_attn:
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout_val if self.training else 0.0)
            
        else:
            score = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            score = score + self.mask[:, :, :seqlen, :seqlen]
            score = F.softmax(score.float(), dim=-1).type_as(xq)
            score = self.dropout(score)
            output = torch.matmul(score, xv)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        output = self.wo(output)
        output = self.dropout(output)

        return output
