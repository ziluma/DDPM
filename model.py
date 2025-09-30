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
    timesteps = 100
    beta_init = 1e-4
    beta_end = .02


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
    
    
    def forward(self, x, t):
        # x: (B, C, H, W), t: (B, E)
        h = self.block1(x)
        h = h + self.t_mlp(t)   # t-> (B, C, 1, 1)
        h = self.block2(h)
        return h + self.res_conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, config: DiffusionConfig):
        super().__init__()
        self.block = ResBlock(in_ch, out_ch, config)
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)
    
    def forward(self, x, t):
        x = self.block(x, t)
        skip = x
        x = self.down(x)
        return x, skip

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, config: DiffusionConfig):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.block = ResBlock(out_ch<<1, out_ch, config)
    
    def forward(self, x, skip, t):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x, t)
        return x

        


class UNet(nn.Module):
    def __init__(
        self, 
        config: DiffusionConfig,
        ch_img: int = 1,
        ch_base: int = 64,
        ch_mults: tuple[int] = (1, 2, 2)
    ):
        super().__init__()

        E = config.n_t_emb
        self.t_mlp = nn.Sequential(
            nn.Embedding(config.timesteps, config.n_t_emb),
            nn.Linear(E, E<<2),
            config.act(),
            nn.Linear(E<<2, E)
        )

        chs = [ch_base*m for m in ch_mults]
        c0, c1, c2 = chs

        self.init_conv = nn.Conv2d(ch_img, c0, 3, padding=1)

        self.down1 = Down(c0, c0)

