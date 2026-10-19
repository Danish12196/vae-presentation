"""A small DDPM in plain PyTorch (no diffusers): a UNet that predicts the noise, a cosine noise
schedule, and a deterministic DDIM sampler. Enough for 64x64 cat faces on one GPU overnight."""

import math

import torch
import torch.nn.functional as F
from torch import nn


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


class TResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(32, cin), nn.Conv2d(cin, cout, 3, 1, 1)
        self.t = nn.Linear(tdim, cout)
        self.n2, self.c2 = nn.GroupNorm(32, cout), nn.Conv2d(cout, cout, 3, 1, 1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.drop = nn.Dropout(0.1)

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(F.silu(temb))[:, :, None, None]
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return h + self.skip(x)


class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n = nn.GroupNorm(32, c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.o = nn.Conv2d(c, c, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(b, 3, c, h * w).unbind(1)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return x + self.o(a.transpose(1, 2).reshape(b, c, h, w))


class UNet(nn.Module):
    """64x64x3 -> 64x64x3 noise prediction. Channels 64-128-256-256, attention at 16x16 and 8x8."""

    def __init__(self, ch=(64, 128, 256, 256), attn_at=(16, 8)):
        super().__init__()
        tdim = ch[0] * 4
        self.temb = nn.Sequential(nn.Linear(ch[0], tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inp = nn.Conv2d(3, ch[0], 3, 1, 1)
        self.downs, self.ups = nn.ModuleList(), nn.ModuleList()
        skips, c, res = [ch[0]], ch[0], 64
        for i, co in enumerate(ch):
            for _ in range(2):
                self.downs.append(nn.ModuleList([TResBlock(c, co, tdim), Attn(co) if res in attn_at else nn.Identity()]))
                c = co
                skips.append(c)
            if i < len(ch) - 1:
                self.downs.append(nn.Conv2d(c, c, 3, 2, 1))
                skips.append(c)
                res //= 2
        self.mid1, self.mida, self.mid2 = TResBlock(c, c, tdim), Attn(c), TResBlock(c, c, tdim)
        for i, co in reversed(list(enumerate(ch))):
            for _ in range(3):
                self.ups.append(nn.ModuleList([TResBlock(c + skips.pop(), co, tdim), Attn(co) if res in attn_at else nn.Identity()]))
                c = co
            if i > 0:
                self.ups.append(nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(c, c, 3, 1, 1)))
                res *= 2
        self.out = nn.Sequential(nn.GroupNorm(32, c), nn.SiLU(), nn.Conv2d(c, 3, 3, 1, 1))
        self.base = ch[0]

    def forward(self, x, t):
        temb = self.temb(timestep_embedding(t, self.base))
        h = self.inp(x)
        hs = [h]
        for m in self.downs:
            if isinstance(m, nn.ModuleList):
                h = m[1](m[0](h, temb))
            else:
                h = m(h)
            hs.append(h)
        h = self.mid2(self.mida(self.mid1(h, temb)), temb)
        for m in self.ups:
            if isinstance(m, nn.ModuleList):
                h = m[1](m[0](torch.cat([h, hs.pop()], 1), temb))
            else:
                h = m(h)
        return self.out(h)


class Cosine:
    """Cosine noise schedule (Nichol & Dhariwal 2021): alpha_bar(t) for t = 0..T-1."""

    def __init__(self, T=1000, device="cpu"):
        s = 0.008
        steps = torch.arange(T + 1, dtype=torch.float64) / T
        f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        abar = (f / f[0]).clamp(1e-5, 1)
        betas = (1 - abar[1:] / abar[:-1]).clamp(max=0.999)
        self.abar = torch.cumprod(1 - betas, 0).float().to(device)
        self.T = T

    def add_noise(self, x0, noise, t):
        a = self.abar[t][:, None, None, None]
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    @torch.no_grad()
    def ddim_sample(self, model, n, steps=50, generator=None, device="cuda", keep=()):
        """Deterministic DDIM. Returns final images in [-1, 1] and the frames at the step indices in `keep`."""
        x = torch.randn(n, 3, 64, 64, generator=generator).to(device)
        ts = torch.linspace(self.T - 1, 0, steps).long().tolist()
        frames = []
        for i, t in enumerate(ts):
            tt = torch.full((n,), t, device=device, dtype=torch.long)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                eps = model(x, tt).float()
            a = self.abar[t]
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
            a_prev = self.abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=device)
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
            if i in keep:
                frames.append(x.clone())
        return x, frames
