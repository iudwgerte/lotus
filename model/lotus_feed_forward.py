import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lotus_config import LotusConfig
from model.lotus_utils import *

class FeedForward(nn.Module):
    def __init__(self, config: LotusConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        if self.hidden_dim is None:
            self.hidden_dim = config.dim * 4
            self.hidden_dim = int(self.hidden_dim * 2 / 3)
            self.hidden_dim = config.multiple_of * ((self.hidden_dim + config.multiple_of - 1) // config.multiple_of)

        self.w1 = nn.Linear(config.dim, self.hidden_dim, bias=False)
        self.w2 = nn.Linear(self.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, self.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
