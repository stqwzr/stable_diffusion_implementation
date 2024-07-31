import torch
from torch import nn
from torch import Functional as F
from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab, n_emb, n_tokens):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_emb)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_emb))

    def forward(self, tokens):
        x = self.token_embedding(tokens)

        x += self.position_embedding

        return x


class CLIPLayer(nn.Module):
    def __init__(self, n_head, n_emb):
        super().__init__()

        self.norm1 = nn.LayerNorm(n_emb)
        self.attention = SelfAttention(n_head, n_emb)
        self.norm2 = nn.LayerNorm(n_emb)
        self.linear1 = nn.Linear(n_emb, 4 * n_emb)
        self.linear2 = nn.Linear(4 * n_emb, n_emb)

    def forward(self, x):
        res = x
        x = self.norm1(x)

        x = self.attention(x)

        x += res

        res = x

        x = self.norm2(x)

        x = self.linear1(x)

        x = x * torch.sigmoid(1.702 * x)  # GiLU ya xz, found in paper

        x = self.linear2(x)

        x += res

        return x


class CLIP(nn.Module):
    def __init__(self):
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layer = nn.Module([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.norm = nn.LayerNorm(768)

    def forward(self, tokens):
        tokens = tokens.type(torch.long)

        state = self.embedding(tokens)

        for layer in self.layer:
            state = layer(state)

        output = self.norm(state)

        return output
