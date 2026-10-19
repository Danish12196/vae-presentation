"""Bigger models for the comparison slides, trained once on the desktop GPU. Their outputs are
precomputed into images (export_v2.py); they are too large to run in the browser.

    python train_v2.py gan            a DCGAN-style GAN (spectral norm, hinge loss, EMA generator)
    python train_v2.py ae vae         deeper residual autoencoder / VAE, latent 128
    python train_v2.py ddpm --until 09:30   a small diffusion model, trained until a wall-clock time
    python train_v2.py all --until 09:30    everything above, in that order

Outputs: runs/v2/<name>/model.pt (weights + history) and preview grids along the way.
Same cats and the same 512 held-out cats as train.py.
"""

import argparse
import copy
import datetime as dt
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader

import data
from models import kl_divergence, reconstruction_loss
from train import save_grid

OUT = Path(__file__).parent / "runs" / "v2"
LATENT = 128


# ------------------------------------------------------------------ residual AE / VAE

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, c, 3, 1, 1),
                                 nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, c, 3, 1, 1))

    def forward(self, x):
        return x + self.net(x)


CH = (64, 128, 192, 256)  # 64 -> 32 -> 16 -> 8 -> 4 pixels wide


class Encoder(nn.Module):
    def __init__(self, out):
        super().__init__()
        layers = [nn.Conv2d(3, CH[0], 3, 1, 1)]
        for c, nxt in zip(CH, CH[1:] + (CH[-1],)):
            layers += [ResBlock(c), nn.Conv2d(c, nxt, 4, 2, 1)]
        layers += [ResBlock(CH[-1]), nn.GroupNorm(8, CH[-1]), nn.SiLU(), nn.Flatten()]
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(CH[-1] * 16, out)

    def forward(self, x):
        return self.head(self.net(x))


class Decoder(nn.Module):
    def __init__(self, latent):
        super().__init__()
        self.fc = nn.Linear(latent, CH[-1] * 16)
        layers = [nn.Unflatten(1, (CH[-1], 4, 4)), ResBlock(CH[-1])]
        rev = CH[::-1]
        for c, nxt in zip(rev, rev[1:] + (rev[-1],)):
            layers += [nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c, nxt, 3, 1, 1), ResBlock(nxt)]
        layers += [nn.GroupNorm(8, CH[0]), nn.SiLU(), nn.Conv2d(CH[0], 3, 3, 1, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(self.fc(z))


class ProAE(nn.Module):
    def __init__(self, latent=LATENT):
        super().__init__()
        self.encoder, self.decoder = Encoder(latent), Decoder(latent)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class ProVAE(nn.Module):
    def __init__(self, latent=LATENT):
        super().__init__()
        self.latent = latent
        self.encoder, self.decoder = Encoder(2 * latent), Decoder(latent)

    def encode(self, x):
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        return mu, logvar.clamp(-12, 8)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(z), mu, logvar


# ------------------------------------------------------------------ GAN

class Generator(nn.Module):
    def __init__(self, nz=LATENT, ngf=64):
        super().__init__()
        self.fc = nn.Linear(nz, ngf * 8 * 16)
        blocks, ch = [], ngf * 8
        for out in (ngf * 4, ngf * 2, ngf, ngf):  # 4 -> 8 -> 16 -> 32 -> 64
            blocks += [nn.BatchNorm2d(ch), nn.ReLU(True), nn.Upsample(scale_factor=2), nn.Conv2d(ch, out, 3, 1, 1)]
            ch = out
        blocks += [nn.BatchNorm2d(ch), nn.ReLU(True), nn.Conv2d(ch, 3, 3, 1, 1), nn.Sigmoid()]
        self.net = nn.Sequential(nn.Unflatten(1, (ngf * 8, 4, 4)), *blocks)
        self.nz = nz

    def forward(self, z):
        return self.net(self.fc(z))


class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        sn = spectral_norm
        layers, ch = [], 3
        for out in (ndf, ndf * 2, ndf * 4, ndf * 8):  # 64 -> 32 -> 16 -> 8 -> 4
            layers += [sn(nn.Conv2d(ch, out, 3, 1, 1)), nn.LeakyReLU(0.1), sn(nn.Conv2d(out, out, 4, 2, 1)), nn.LeakyReLU(0.1)]
            ch = out
        self.net = nn.Sequential(*layers, nn.Flatten(), sn(nn.Linear(ch * 16, 1)))

    def forward(self, x):
        return self.net(x * 2 - 1).squeeze(1)


# ------------------------------------------------------------------ helpers

def loader(cats, batch, device):
    return DataLoader(data.CatDataset(cats, augment=True), batch, shuffle=True, drop_last=True,
                      num_workers=0, pin_memory=device.type == "cuda")


@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, p in zip(ema.parameters(), model.parameters()):
        pe.lerp_(p, 1 - decay)
    for be, b in zip(ema.buffers(), model.buffers()):
        be.copy_(b)


def log(name, msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] [{name}] {msg}", flush=True)


# ------------------------------------------------------------------ training runs

def train_autoencoder(kind, train_cats, held_out, device, epochs=100):
    name = f"{kind}_pro"
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = (ProVAE() if kind == "vae" else ProAE()).to(device)
    log(name, f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    dl = loader(train_cats, 128, device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * len(dl), pct_start=0.05)
    previews = torch.from_numpy(held_out[:16]).permute(0, 3, 1, 2).float().div(255).to(device)
    z_fixed = torch.randn(16, LATENT, generator=torch.Generator().manual_seed(0)).to(device)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, tot, n = time.time(), {}, 0
        for step, x in enumerate(dl):
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if kind == "vae":
                    x_hat, mu, logvar = model(x)
                    recon, kl = reconstruction_loss(x_hat.float(), x), kl_divergence(mu.float(), logvar.float())
                    warm = min(1.0, (epoch - 1 + step / len(dl)) / 10)  # KL warm-up over 10 epochs
                    loss = recon + warm * kl
                    parts = {"recon": recon.item(), "kl": kl.item()}
                else:
                    recon = reconstruction_loss(model(x)[0].float(), x)
                    loss = recon
                    parts = {"recon": recon.item()}
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            for k, v in parts.items():
                tot[k] = tot.get(k, 0) + v
            n += 1
        rec = {k: v / n for k, v in tot.items()}
        history.append(rec)
        log(name, f"epoch {epoch}/{epochs} " + " ".join(f"{k} {v:.1f}" for k, v in rec.items()) + f" ({time.time() - t0:.0f}s)")
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                rows = [previews, model(previews)[0].float()]
                if kind == "vae":
                    s = model.decode(z_fixed).float()
                    rows += [s]
            save_grid(torch.stack(rows), out / f"epoch_{epoch:03d}.png")
            torch.save({"kind": kind, "state_dict": model.state_dict(), "history": history}, out / "model.pt")
    log(name, "done")


def train_gan(train_cats, device, epochs=60):
    name = "gan"
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    G, D = Generator().to(device), Discriminator().to(device)
    G_ema = copy.deepcopy(G).eval()
    log(name, f"G {sum(p.numel() for p in G.parameters()) / 1e6:.1f}M, D {sum(p.numel() for p in D.parameters()) / 1e6:.1f}M parameters")
    dl = loader(train_cats, 128, device)
    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.0, 0.9))
    z_fixed = torch.randn(32, LATENT, generator=torch.Generator().manual_seed(0)).to(device)
    history = []
    for epoch in range(1, epochs + 1):
        t0, ld, lg, n = time.time(), 0.0, 0.0, 0
        for x in dl:
            x = x.to(device, non_blocking=True)
            z = torch.randn(len(x), LATENT, device=device)
            fake = G(z)
            # hinge loss
            d_loss = F.relu(1 - D(x)).mean() + F.relu(1 + D(fake.detach())).mean()
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()
            g_loss = -D(fake).mean()
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()
            ema_update(G_ema, G, 0.999)
            ld, lg, n = ld + d_loss.item(), lg + g_loss.item(), n + 1
        history.append({"d": ld / n, "g": lg / n})
        log(name, f"epoch {epoch}/{epochs} D {ld / n:.3f} G {lg / n:.3f} ({time.time() - t0:.0f}s)")
        if epoch % 5 == 0 or epoch == epochs:
            with torch.no_grad():
                s = G_ema(z_fixed)
            save_grid(s.view(4, 8, 3, 64, 64), out / f"epoch_{epoch:03d}.png")
            torch.save({"G": G.state_dict(), "G_ema": G_ema.state_dict(), "D": D.state_dict(), "history": history,
                        "epoch": epoch}, out / "model.pt")
            # keep each checkpoint's generator too: an early or unstable one may show mode collapse
            torch.save(G.state_dict(), out / f"G_epoch_{epoch:03d}.pt")
    log(name, "done")


def train_ddpm(train_cats, device, until):
    from tiny_diffusion import Cosine, UNet
    name = "ddpm"
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    unet = UNet().to(device)
    ema = copy.deepcopy(unet).eval()
    log(name, f"{sum(p.numel() for p in unet.parameters()) / 1e6:.1f}M parameters, training until {until:%H:%M}")
    sched = Cosine(device=device)
    dl = loader(train_cats, 64, device)
    opt = torch.optim.AdamW(unet.parameters(), lr=2e-4, weight_decay=1e-4)
    step, last_save, history = 0, time.time(), []
    done = False
    while not done:
        for x in dl:
            x = x.to(device, non_blocking=True) * 2 - 1
            noise = torch.randn_like(x)
            t = torch.randint(0, sched.T, (len(x),), device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = unet(sched.add_noise(x, noise, t), t)
            loss = F.mse_loss(pred.float(), noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            opt.step()
            ema_update(ema, unet, 0.9995 if step > 2000 else 0.99)
            step += 1
            if step % 500 == 0:
                history.append({"step": step, "loss": loss.item()})
                log(name, f"step {step} loss {loss.item():.4f}")
            if time.time() - last_save > 30 * 60 or dt.datetime.now() >= until:
                save_ddpm(ema, unet, history, step, out, sched)
                last_save = time.time()
                if dt.datetime.now() >= until:
                    done = True
                    break
    log(name, "done")


@torch.no_grad()
def save_ddpm(ema, unet, history, step, out, sched):
    torch.save({"ema": ema.state_dict(), "unet": unet.state_dict(), "history": history, "step": step}, out / "model.pt")
    x, _ = sched.ddim_sample(ema, 16, steps=50, generator=torch.Generator().manual_seed(0))
    save_grid(((x + 1) / 2).clamp(0, 1).view(2, 8, 3, 64, 64), out / f"samples_step_{step:06d}.png")
    log("ddpm", f"checkpoint + samples at step {step}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", nargs="+", choices=["gan", "ae", "vae", "ddpm", "all"])
    p.add_argument("--until", default="09:30", help="wall-clock time (HH:MM, today or tomorrow) to stop the diffusion model")
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args()
    what = ["gan", "ae", "vae", "ddpm"] if "all" in args.what else args.what
    hh, mm = map(int, args.until.split(":"))
    now = dt.datetime.now()
    until = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if until <= now:
        until += dt.timedelta(days=1)

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    train_cats, held_out = data.split(data.load_cats())
    log("v2", f"{len(train_cats)} training cats; running {what}")
    for w in what:
        if w == "gan":
            train_gan(train_cats, device, epochs=args.epochs or 60)
        elif w in ("ae", "vae"):
            train_autoencoder(w, train_cats, held_out, device, epochs=args.epochs or 100)
        elif w == "ddpm":
            train_ddpm(train_cats, device, until)


if __name__ == "__main__":
    main()
