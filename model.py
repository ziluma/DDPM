import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from dataclasses import dataclass

@dataclass
class DiffusionConfig:
    n_t_emb = 128
    n_gp = 8
    dropout = .1
    act = nn.SiLU


class ResBlock(nn.Module):
    def __init__(self, 
        in_ch: int, 
        out_ch: int, 
        config: DiffusionConfig
    ):
        super().__init__()
        self.act = config.act 
        self.dropout = config.dropout
        self.t_mlp = nn.Sequential(
            self.act(),
            nn.Linear(config.n_t_emb, out_ch),
            Rearrange('b c -> b c 1 1')
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(config.n_gp, out_ch),
            self.act(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(config.n_gp, out_ch),
            self.act,
            nn.Dropout(config.dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch!=out_ch else nn.Identity()
    
    
    def forward(self, x, t_emb):
        h = self.block1(x)
        h = h + self.t_mlp(t_emb)
        x = self.block2(x)
        return x + self.res_conv(x)

