import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lotus_utils import *
from model.lotus_attention import *
from model.lotus_feed_forward import *
from model.lotus_config import LotusConfig

import math
import numpy as np

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from typing import Optional, Tuple

class LotusBlock(nn.Module):
    def __init__(self, layer_id: int, config: LotusConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.rms_norm = RMSNorm(config.dim, eps=config.norm_eps)
        
        self.layer_id = layer_id

    def forward(self, x: torch.Tensor, pos_cis: torch.Tensor, kv_cache=False):
        h = x + self.attention(self.rms_norm(x), pos_cis, kv_cache)
        o = h + self.feed_forward(self.rms_norm(h))
        return o

class Lotus(PreTrainedModel):
    config_class = LotusConfig
    last_loss = Optional[torch.tensor]

    def __init__(self, config: LotusConfig = None):
        super().__init__(config)
        if not config:
            config = LotusConfig()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(LotusBlock(layer_id, config))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.embedding.weight = self.output.weight
        pos_cis = precompute_pos_cis(config.dim // config.n_heads, config.max_seq_len)
        self.register_buffer('pos_cis', pos_cis, persistent=False)

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

        self.last_loss = None
        self.OUT = CausalLMOutputWithPast()
        self._no_split_modules = [name for name, _ in self.named_modules()]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor] = None, kv_cache = False, **kwargs):
        current_idx = 0
        if 'input_ids' in kwargs:
            tokens = kwargs['input_ids']
        if 'attention_mask' in kwargs:
            targets = kwargs['attention_mask']
        if 'current_idx' in kwargs:
            current_idx = kwargs['current_idx']

        bsz, seq_len = tokens.size()
        h = self.embedding(tokens)
        h = self.dropout(h)
        pos_cis = self.pos_cis[current_idx:current_idx+seq_len]
        for idx, layer in enumerate(self.layers):
            h = layer(h, pos_cis, kv_cache)

        h = self.norm(h)

        if targets is not None:
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0, reduction='none')
        
        else:
            logits = self.output(h[:, [-1], :])
            self.last_loss = None

        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', self.last_loss)
        return self.OUT
    
    @torch.inference_mode()
    def generate(self, idx: torch.tensor, eos: int, max_len: int, temperature: float = 1.0, top_k: int = 10, repetition_penalty: float = 1.0, kv_cache = True, stream=False):
        index = idx.shape[1]
        init_inference = True
        while idx.shape[1] < max_len - 1:
            if init_inference or not kv_cache:
                inference_res, init_inference = self(idx, kv_cache=kv_cache), False
            else:
                inference_res = self(idx[:, -1:], kv_cache=kv_cache, current_idx=idx.shape[1] - 1)
                
            logits = inference_res.logits
            logits = logits[:, -1, :]

            for token in set(idx.tolist()[0]):
                logits[:, token] /= repetition_penalty

            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1, generator=None)
            
            if idx_next == eos:
                break

            idx = torch.cat((idx, idx_next), dim=1)
            if stream:
                yield idx[:, index:]
        
        if not stream:
            yield idx[:, index:]

    @torch.inference_mode()
    def eval_answer(self, idx: torch.tensor):
        idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
        inference_res = self(idx_cond)
        logits = inference_res.logits
        logits = logits[:, -1, :]
        return logits

