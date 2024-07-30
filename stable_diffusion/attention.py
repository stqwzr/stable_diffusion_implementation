import torch
from torch import nn
from torch import functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_b=True, out_proj_b=True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_b)

        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_b)

        self.n_heads = n_heads

        self.d_heads = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        in_shape = x.shape

        b_size, seq_len, d_embed = in_shape
        inter_shape = (b_size, seq_len, self.n_heads, self.d_heads)

        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        q = q.view(inter_shape).transpose(1, 2)
        k = k.view(inter_shape).transpose(1, 2)
        v = v.view(inter_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)

        weight /= math.sqrt(self.n_heads)

        weight = F.softmax(weight, dim=-1)

        output = weight @ v

        output = output.transpose(1, 2)

        output = output.reshape(in_shape)

        output = self.out_proj(output)

        return output
