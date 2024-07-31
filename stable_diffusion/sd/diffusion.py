import torch
from torch import nn
from torch import Functional as F
from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x):

        x = self.linear1(x)

        x = F.silu(x)

        x = self.linear2(x)

        return x


class Diffusion(nn.Module):
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent, context, time):
        time = self.time_embedding(time)

        output = self.unet(latent, contenxt, time)

        output = self.final(output)

        return output
