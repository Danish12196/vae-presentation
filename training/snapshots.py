"""Replay the training of ae_2 and vae_2d_b1, recording what the site's "Watch it learn" slide shows.

    python snapshots.py

Same settings and seed as the real runs (train.py), but a separate copy: nothing in runs/ae_2 or
runs/vae_2d_b1 changes. After every epoch (and once before training) it records
  - the 2-D code of 600 of the latent-map cats (the same cats export.py puts on the maps),
  - the reconstruction of the first 4 showcase cats,
  - the loss parts of that epoch (the history train.py keeps anyway).
Output: runs/snapshots/<name>.npz, which export.py turns into site/data/training.json.
"""

from pathlib import Path

import numpy as np
import torch

import data
from export import N_MAP, SEED, to_tensor
from train import RUN_LIST, RUNS, train

NAMES = ["ae_2", "vae_2d_b1"]
N_CATS = 600   # dots per snapshot: enough to see the shape, small enough to ship
N_RECON = 4    # showcase cats rebuilt at every epoch
OUT = RUNS / "snapshots"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"
    train_cats, held_out = data.split(data.load_cats())
    map_cats = train_cats[np.random.default_rng(SEED).choice(len(train_cats), N_MAP, replace=False)]
    cats = to_tensor(map_cats[:N_CATS], device)
    showcase = to_tensor(held_out[:N_RECON], device)

    for name in NAMES:
        run = next(r for r in RUN_LIST if r.name == name)
        codes, recons = [], []

        @torch.no_grad()
        def record(model, epoch):
            model.eval()
            z = model.encode(cats)
            codes.append((z[0] if run.kind == "vae" else z).cpu().numpy())  # mu for the VAE
            x_hat = model(showcase)[0]
            recons.append((x_hat.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy())
            model.train()

        print(f"\n=== {name} (snapshot replay)")
        out = OUT / name
        train(run, out, data.CatDataset(train_cats, augment=True), device, 0, on_epoch=record)
        history = torch.load(out / "model.pt", map_location="cpu", weights_only=False)["history"]
        (out / "model.pt").unlink()  # the weights are not needed, only what happened on the way
        np.savez_compressed(OUT / f"{name}.npz", codes=np.stack(codes), recons=np.stack(recons),
                            originals=held_out[:N_RECON],
                            history=np.array([[h.get(k, np.nan) for k in ("loss", "recon", "kl")]
                                              for h in history], np.float32))
        print(f"wrote {OUT / name}.npz: {len(codes)} snapshots")


if __name__ == "__main__":
    main()
