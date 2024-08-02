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


class UNET_residiualBlock(nn.Module):

    def __init__(self, in_c, out_c, n_time=1280):
        super().__init__()

        self.groupnorm = nn.GroupNorm(32, in_c)
        self.conv = nn.Conv2(in_c, out_c, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_c)

        self.groupnorm_merg = nn.GroupNorm(32, out_c)
        self.conv_merged = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)

        if in_c == out_c:
            self.residual_layer = nn.Indentity()
        else:
            self.residual_layer = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)

    def forward(self, x, time):
        res = x

        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)

        time = F.silu(time)
        time = self.linear_time(time)

        merged = x + time.unsqueeze(-1).unsqueeze(-1)

        merged = self.groupnorm_merg(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        merged = merged + self.residual_layer(res)

        return merged


class Unet_AttentionBlock(nn.Module):
    def __init__(self, n_head, n_embd, d_context=768):
        super().__init__()
        channels = n_head * n_embd

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm1 = nn.LayerNorm(channels)
        self.attention1 = SelfAttention(n_head, channels, in_proj_b=False)

        self.layernorm2 = nn.LayerNorm(channels)
        self.attention2 = CrossAttention(n_head, channels, d_context, in_proj_b=False)

        self.layernorm3 = nn.LayerNorm(channels)
        self.linear1 = nn.Linear(channels, 4 * channels * 2)
        self.linear2 = nn.Linear(4 * channels * 2, channels)

        self.conv_out = nn.Conv2(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        res = x
        x = self.groupnorm(x)
        x = self.conv_input(x)

        n, c, w, h = x.shape
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)

        res_fast = x
        x = self.layernorm1(x)
        self.attention1(x)
        x += res_fast

        res_fast = x
        x = self.layernorm2(x)
        self.attention2(x, context)
        x += res_fast

        res_fast = x
        x = self.layernorm3(x)
        x, gate = self.linear1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear2(x)
        x += res_fast

        x = x.transpose(-1, -2)
        x = x.view((n, c, w, h))

        return self.conv_out(x) * res


class SwitchSequential(nn.Module):
    def forward(self, x, context, time):
        if layer in self:
            if isinstance(layer, UNET_attentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_residualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class UpSample(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNET_outputLayer(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_c)
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.groupnorm(x)
        x = F.silu(x)
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoders = nn.Module([
            SwitchSequential(nn.Conv2(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(UNET_residualBlock(320, 320), UNET_attentionBlock(8, 40)),
            SwitchSequential(UNET_residualBlock(320, 320), UNET_attentionBlock(8, 40)),

            SwitchSequential(nn.Conv2(320, 320, kernel_size=3, padding=1)),
            SwitchSequential(UNET_residualBlock(320, 640), UNET_attentionBlock(8, 80)),
            SwitchSequential(UNET_residualBlock(640, 640), UNET_attentionBlock(8, 80)),

            SwitchSequential(nn.Conv2(640, 640, kernel_size=3, padding=1)),
            SwitchSequential(UNET_residualBlock(640, 1280), UNET_attentionBlock(8, 160)),
            SwitchSequential(UNET_residualBlock(1280, 1280), UNET_attentionBlock(8, 160)),

            SwitchSequential(nn.Conv2(1280, 1280, kernel_size=3, padding=1)),
            SwitchSequential(UNET_residualBlock(1280, 1280)),
            SwitchSequential(UNET_residualBlock(1280, 1280)),
        ])

        self.bottleneck = nn.Module([
            UNET_residualBlock(1280, 1280),
            UNET_attentionBlock(8, 160),
            UNET_residualBlock(1280, 1280)
        ])

        self.decoder = nn.Module([
            SwithSequential(UNET_residualBlock(2560, 1280)),
            SwithSequential(UNET_residualBlock(2560, 1280)),
            SwithSequential(UNET_residualBlock(2560, 1280), UpSample(1280)),

            SwithSequential(UNET_residualBlock(2560, 1280), UNET_attentionBlock(8, 160)),
            SwithSequential(UNET_residualBlock(2560, 1280), UNET_attentionBlock(8, 160)),
            SwithSequential(UNET_residualBlock(1920, 1280), UNET_attentionBlock(8, 160), UpSample(1280)),

            SwithSequential(UNET_residualBlock(1920, 640), UNET_attentionBlock(8, 80)),
            SwithSequential(UNET_residualBlock(1280, 640), UNET_attentionBlock(8, 80)),
            SwithSequential(UNET_residualBlock(960, 640), UNET_attentionBlock(8, 80), UpSample(640)),

            SwithSequential(UNET_residualBlock(960, 320), UNET_attentionBlock(8, 40)),
            SwithSequential(UNET_residualBlock(640, 320), UNET_attentionBlock(8, 40)),
            SwithSequential(UNET_residualBlock(640, 320), UNET_attentionBlock(8, 40)),

        ])


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
