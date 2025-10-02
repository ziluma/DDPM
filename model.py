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
    T = 100
    beta_1 = 1e-2
    beta_T = .45
    img_shape = [1, 28, 28]


class ResBlock(nn.Module):

    def __init__(self, in_ch, out_ch, config: DiffusionConfig):
        super().__init__()
        self.act = config.act 
        self.dropout = config.dropout
        self.t_mlp = nn.Sequential(
            self.act(),
            nn.Linear(config.n_t_emb, out_ch),
            Rearrange('b c -> b c 1 1')
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(config.n_gp, in_ch),
            self.act(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(config.n_gp, out_ch),
            self.act(),
            nn.Dropout(config.dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch!=out_ch else nn.Identity()
    
    def forward(self, x, t):
        # x: (B, Ci, H, W), t: (B, E)
        h = self.block1(x)
        h = h + self.t_mlp(t)   # t-> (B, C, 1, 1)
        h = self.block2(h)
        return h + self.res_conv(x) # (B, C, H, W)


class Down(nn.Module):

    def __init__(self, in_ch, out_ch, config: DiffusionConfig):
        super().__init__()
        self.block = ResBlock(in_ch, out_ch, config)
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)
    
    def forward(self, x, t):
        # x: (B, Ci, H, W), t: (B, E)
        x = self.block(x, t)    # (B, Co, H, W)
        skip = x            # (B, Co, H, W)
        x = self.down(x)    # (B, Co, H//2, W//2)
        return x, skip


class Up(nn.Module):

    def __init__(self, in_ch, out_ch, config: DiffusionConfig):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.block = ResBlock(out_ch+in_ch, out_ch, config)
    
    def forward(self, x, skip, t):
        # x: (B, Ci, H//2, W//2)
        # skip: (B, Ci, H, W)
        # t: (B, E)
        x = self.up(x)      # (B, Co, H, W)
        x = torch.cat([x, skip], dim=1) # (B, Ci+Co, H, W)
        x = self.block(x, t)    # (B, Co, H, W)
        return x
        

class UNet(nn.Module):

    def __init__(
        self, 
        config: DiffusionConfig,
        ch_img: int = 1,    # grayscale MNIST
        ch_base: int = 64,  
        ch_mults: tuple[int] = (1, 2, 2)
    ):
        super().__init__()

        E = config.n_t_emb
        self.t_mlp = nn.Sequential(
            nn.Embedding(config.T, config.n_t_emb),
            nn.Linear(E, E<<2),
            config.act(),
            nn.Linear(E<<2, E)
        )

        chs = [ch_base*m for m in ch_mults]
        c0, c1, c2 = chs

        self.init_conv = nn.Conv2d(ch_img, c0, 3, padding=1)

        self.down1 = Down(c0, c0, config)
        self.down2 = Down(c0, c1, config)

        self.mid1 = ResBlock(c1, c2, config)
        self.mid2 = ResBlock(c2, c1, config)

        self.up2 = Up(c1, c0, config)
        self.up1 = Up(c0, c0, config)

        self.final = nn.Sequential(
            nn.GroupNorm(config.n_gp, c0),
            config.act(),
            nn.Conv2d(c0, ch_img, 3, padding=1)
        )

    def forward(self, x, t):
        # x: (B, 1, 28, 28), t: (B,) on MNIST
        t = self.t_mlp(t)       # (B, E)
        x = self.init_conv(x)   # (B, c0, 28, 28)  

        x, skip1 = self.down1(x, t) # (B, c0, 14, 14), skip1: (B, c0, 28, 28)
        x, skip2 = self.down2(x, t) # (B, c1, 7, 7), skip2: (B, c1, 14, 14)

        x = self.mid1(x, t)     # (B, c2, 7, 7)
        x = self.mid2(x, t)     # (B, c1, 7, 7)

        x = self.up2(x, skip2, t)   # (B, c0, 14, 14)
        x = self.up1(x, skip1, t)   # (B, c0, 28, 28)

        return self.final(x)    # (B, 1, 28, 28)
    
def gather(x, ix):
    return rearrange(
        torch.gather(x, dim=0, index=ix), 
        'b -> b 1 1 1'
    )
    # return out.view([shape[0]] + [1]*(len(shape)-1))

class Diffuser(nn.Module):
    def __init__(self, model, config: DiffusionConfig, device):
        super().__init__()
        self.model = model.to(device)
        self.T = config.T
        self.shape = config.img_shape
        self.device = device
        betas = torch.linspace(config.beta_1, config.beta_T, config.T, dtype=torch.float32)
        alphas = torch.sqrt(1. - betas ** 2)
        alphas_bar = torch.cumprod(alphas, dim=0) 
        betas_bar = torch.sqrt(1. - alphas_bar ** 2)
        mu_eps_coef = betas**2 / betas_bar / alphas
        betas_til = betas / betas_bar
        betas_til[1:] *= betas_bar[:-1]

        self.register_buffer('alphas_bar', alphas_bar)
        self.register_buffer('betas_bar', betas_bar)
        self.register_buffer('alphas_recip', 1. / alphas)
        self.register_buffer('mu_eps_coef', mu_eps_coef)
        self.register_buffer('betas_til', betas_til)

        self.to(device)

    def forward(self, x0):
        ''' Algo 1 '''
        t = torch.randint(self.T, size=(x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0, dtype=torch.float32)
        alphas = gather(self.alphas_bar, t)
        betas = gather(self.betas_bar, t)
        x_t = alphas * x0 + betas * noise
        loss = F.mse_loss( self.model(x_t, t), noise, reduction='none')

        return loss

    @torch.no_grad()
    def sample_step(self, x, t): 
        ''' Sample one step: X_t -> X_{t-1}'''
        # x: (B, C, H, W), t: int 0 ... T-1
        assert 0 <= t < self.T, "time out of range!"

        t_cur = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        eps = self.model(x, t_cur)
        a = gather(self.alphas_recip, t_cur)
        b = gather(self.mu_eps_coef, t_cur)
        z = torch.randn_like(x) if t>0 else 0
        mu = a * x - b * eps
        sigma = gather(self.betas_til, t_cur)

        return mu + sigma * z

    @torch.no_grad()
    def sample(self, n_samples=2):
        ''' Algo 2 '''
        x = torch.randn([n_samples]+self.shape).to(self.device)
        for t in range(self.T-1, -1, -1):
            x = self.sample_step(x, t)
            
        return x


# import sys; sys.exit(0)

# sanity checks

if __name__ == '__main__':
    config = DiffusionConfig()

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'

    model = UNet(config).to(device)
    B = 8
    x = torch.randn(B, 1, 28, 28).to(device)
    t = torch.randint(config.T, (B,), dtype=torch.long).to(device)
    y = model(x, t)

    diffuser = Diffuser(model, config, device)
    diffuser(x)

    x = diffuser.sample()

    import matplotlib.pyplot as plt

    # assume x is your output tensor (2,1,28,28)
    imgs = x[:2].cpu().detach()  # move to CPU and detach from graph if needed

    fig, axes = plt.subplots(1, 2, figsize=(6,3))
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i,0], cmap='gray')  # take batch i, channel 0
        ax.axis('off')
    plt.show()

