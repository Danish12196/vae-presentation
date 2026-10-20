"""Precompute images from the bigger models (train_v2.py) for the comparison slides.

    python export_v2.py

site/data/gen_pool.jpg + .json      slide 26: a pool of VAE samples and plain-AE samples (naive and
                                    with its own code statistics), re-rolled in the browser
site/data/compare_pool.jpg + .json  slide 30: samples from the VAE, the GAN and the diffusion model
site/data/interp_v2.jpg + .json     slide 27: AE and VAE walks between every pair of the first 8
                                    showcase cats (8 frames each)
Prints how varied each GAN checkpoint's samples are (a mode-collapse check).
export.py clears site/data, so run this AFTER export.py. It adds its files to data/files.json
itself and re-stamps the site (tools/stamp_site.py).
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch

import data
from export import OUT, SITE, hashed_name, save_sprite, to_tensor, to_uint8, write_json
from train_v2 import LATENT, OUT as RUNS_V2, Generator, ProAE, ProVAE

POOL = {"vae": 120, "ae_naive": 60, "ae_fair": 60}
COMPARE = 45  # per model: five 3x3 grids
SEED = 0


def load(path, model, key="state_dict"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt[key] if key else ckpt)
    return model.eval(), ckpt


@torch.no_grad()
def decode(model, z, device):
    return torch.cat([model.decode(b.to(device)).float().cpu() for b in z.split(60)])


def save(name, images, cols, meta):
    """Write a sprite with a content-hashed name and its JSON next to it."""
    tmp = OUT / f"{name}.jpg"
    info = save_sprite(images, tmp, cols=cols, quality=84)
    final = OUT / hashed_name(f"{name}.jpg", tmp.read_bytes())
    tmp.replace(final)
    js = {"file": f"{name}.jpg", **info, **meta}
    jtmp = OUT / f"{name}.json"
    write_json(jtmp, js)
    jfinal = OUT / hashed_name(f"{name}.json", jtmp.read_bytes())
    jtmp.replace(jfinal)
    register(final, jfinal)
    print(f"wrote {final.name} ({len(images)} tiles)")


def register(*paths):
    """Add files to data/files.json (logical name -> hashed path), replacing older entries."""
    fj = OUT / "files.json"
    listing = json.loads(fj.read_text())
    for path in paths:
        rel = path.relative_to(SITE).as_posix()
        logical = re.sub(r"\.[0-9a-f]{6}(\.[^.]+)$", r"\1", rel)
        listing["files"] = [f for f in listing["files"] if f["name"] != logical]
        listing["files"].append({"name": logical, "path": rel, "bytes": path.stat().st_size})
    listing["total_bytes"] = sum(f["bytes"] for f in listing["files"])
    fj.write_text(json.dumps(listing, separators=(",", ":")))


def main():
    for old in list(OUT.glob("gen_pool.*")) + list(OUT.glob("compare_pool.*")) + list(OUT.glob("interp_v2.*")):
        old.unlink()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator().manual_seed(SEED)
    train_cats, _ = data.split(data.load_cats())

    # ---------------------------------------------------------- slide 26: VAE vs plain AE
    vae, _ = load(RUNS_V2 / "vae_pro" / "model.pt", ProVAE())
    ae, _ = load(RUNS_V2 / "ae_pro" / "model.pt", ProAE())
    vae, ae = vae.to(device), ae.to(device)
    with torch.no_grad():
        x = to_tensor(train_cats[np.random.default_rng(SEED).choice(len(train_cats), 2000, replace=False)], device)
        codes = torch.cat([ae.encode(b) for b in x.split(250)]).float().cpu()
    mean, std = codes.mean(0), codes.std(0)
    vae_s = decode(vae, torch.randn(POOL["vae"], LATENT, generator=g), device)
    ae_naive = decode(ae, torch.randn(POOL["ae_naive"], LATENT, generator=g), device)
    ae_fair = decode(ae, mean + std * torch.randn(POOL["ae_fair"], LATENT, generator=g), device)
    order = ["vae", "ae_naive", "ae_fair"]
    start, parts = 0, {}
    for k in order:
        parts[k] = [start, POOL[k]]
        start += POOL[k]
    save("gen_pool", to_uint8(torch.cat([vae_s, ae_naive, ae_fair])), 20,
         {"parts": parts, "note": "train_v2.py vae_pro / ae_pro, latent 128"})

    # ---------------------------------------------------------- slide 27: interpolation, AE vs VAE
    _, held_out = data.split(data.load_cats())
    x8 = to_tensor(held_out[:8], device)
    t = torch.linspace(0, 1, 8, device=device)[:, None]
    pairs, tiles27 = [], []
    with torch.no_grad():
        za = ae.encode(x8).float()
        mu, _ = vae.encode(x8)
        mu = mu.float()
        for a in range(8):
            for b in range(a + 1, 8):
                pairs.append([a, b])
                tiles27.append(ae.decode(za[a] * (1 - t) + za[b] * t).float().cpu())
                tiles27.append(vae.decode(mu[a] * (1 - t) + mu[b] * t).float().cpu())
    save("interp_v2", to_uint8(torch.cat(tiles27)), 8,
         {"pairs": pairs, "frames": 8, "rows": ["ae", "vae"],
          "note": "row 2k: ae_pro walk for pairs[k], row 2k+1: vae_pro walk (mu); frames from A to B"})

    # ---------------------------------------------------------- slide 30: VAE vs GAN vs diffusion
    G = Generator().to(device)
    G.load_state_dict(torch.load(RUNS_V2 / "gan" / "model.pt", map_location="cpu", weights_only=False)["G_ema"])
    G.eval()
    with torch.no_grad():
        gan_s = torch.cat([G(b.to(device)).float().cpu() for b in torch.randn(COMPARE, LATENT, generator=g).split(45)])
    vae_c = decode(vae, torch.randn(COMPARE, LATENT, generator=g), device)
    tiles = [vae_c, gan_s]
    names = ["vae", "gan"]
    ddpm = RUNS_V2 / "ddpm" / "model.pt"
    if ddpm.exists():
        from tiny_diffusion import Cosine, UNet
        unet = UNet().to(device)
        ck = torch.load(ddpm, map_location="cpu", weights_only=False)
        unet.load_state_dict(ck["ema"])
        unet.eval()
        sched = Cosine(device=device)
        out = []
        for k in range(0, COMPARE, 15):
            xk, _ = sched.ddim_sample(unet, min(15, COMPARE - k), steps=100, generator=g)
            out.append(((xk + 1) / 2).clamp(0, 1).float().cpu())
        tiles.append(torch.cat(out))
        names.append("diffusion")
        print(f"diffusion checkpoint: step {ck['step']}")
    parts, start = {}, 0
    for n, t in zip(names, tiles):
        parts[n] = [start, len(t)]
        start += len(t)
    save("compare_pool", to_uint8(torch.cat(tiles)), 15, {"parts": parts})

    # ---------------------------------------------------------- mode-collapse check per GAN checkpoint
    z = torch.randn(64, LATENT, generator=torch.Generator().manual_seed(1))
    for ck in sorted((RUNS_V2 / "gan").glob("G_epoch_*.pt")):
        Gk = Generator().to(device)
        Gk.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
        Gk.eval()
        with torch.no_grad():
            s = Gk(z.to(device)).float().cpu().flatten(1)
        spread = float(torch.cdist(s, s).mean())
        print(f"{ck.name}: mean pairwise distance {spread:.1f} (lower = less variety)")
    import subprocess, sys
    subprocess.run([sys.executable, str(SITE.parent / "tools" / "stamp_site.py")], check=True)


if __name__ == "__main__":
    main()
