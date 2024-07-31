import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_residual_block
from attention import SelfAttention



class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            VAE_residual_block(128, 128),
            VAE_residual_block(128, 128),
            nn.Conv2d(128, kernel_size=3, stride=2),
            VAE_residual_block(128, 256),
            VAE_residual_block(256, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            VAE_residual_block(256, 512),
            VAE_residual_block(512, 512),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            VAE_residual_block(512, 512),
            VAE_residual_block(512, 512),
            VAE_attention_block(512),
            VAE_residual_block(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x, noise):
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        mu, var = torch.chunk(x, 2, dim=1)

        var = torch.clamp(var, -30, 20)

        var = var.exp()

        stdev = var.sqrt()

        x = mu + stdev * noise

        x *= 0.18215

        return x
