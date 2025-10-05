import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils
import os

from model import UNet, Diffuser, DiffusionConfig


config = DiffusionConfig()

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
torch.manual_seed(666)

lr = 1e-3
weight_decay = 1e-4

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2.0 - 1.0),
])

train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

model = UNet(config).to(device)
diffuser = Diffuser(model, config, device)
optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

check_dir = './checkpoints'
sample_dir = './samples'

os.makedirs(sample_dir, exist_ok=True)
os.makedirs(check_dir, exist_ok=True)

n_epochs = 20
global_step = 0
for epoch in range(1, n_epochs+1):
    model.train()
    for x0, _ in train_loader:
        x0 = x0.to(device)
        loss = diffuser(x0).mean()
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if global_step % 100 == 0:
            print(f"epoch {epoch} step {global_step}: loss={loss.item():.4f}")
        global_step += 1

    with torch.no_grad():
        if epoch%5 != 0: continue
        samples = diffuser.sample(16)
        samples = (samples + 1.) / 2.
        grid = vutils.make_grid(samples, nrow=4)
        out_path = os.path.join(sample_dir, f"samples_epoch_{epoch:03d}.png")
        vutils.save_image(grid, out_path)
        print(f"Saved {out_path}")

# final checkpoint
ckpt_path = os.path.join(check_dir, "ddpm_mnist.pt")
torch.save({
    "model": model.state_dict(),
    "cfg": {
        "timesteps": config.T,
        "beta_start": config.beta_1,
        "beta_end": config.beta_T,
    },
}, ckpt_path)
print(f"Saved checkpoint to {ckpt_path}")


