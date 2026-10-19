"""Replay the training of ae_2 and vae_2d_b1, recording what the "Watch it learn" slides show.

    python snapshots.py

Same settings and seed as the real runs (train.py), but separate copies: nothing in runs/ae_2 or
runs/vae_2d_b1 changes. Most of the visible learning happens in the first few hundred steps, so
snapshots are log-spaced in optimiser steps (0, 8, 16, 32, ... the end), and the loss is kept for
every step. Each snapshot holds:
  - the 2-D code of 600 of the latent-map cats (the same cats export.py puts on the maps),
  - the reconstruction of the first 4 showcase cats.
Output: runs/snapshots/<name>.npz, which export.py turns into site/data/training.json.
"""

import numpy as np
import torch

import data
from export import N_MAP, SEED, to_tensor
from train import BATCH, RUN_LIST, RUNS, train

NAMES = ["ae_2", "vae_2d_b1"]
N_CATS = 600   # dots per snapshot: enough to see the shape, small enough to ship
N_RECON = 4    # showcase cats rebuilt at every snapshot
OUT = RUNS / "snapshots"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"
    train_cats, held_out = data.split(data.load_cats())
    map_cats = train_cats[np.random.default_rng(SEED).choice(len(train_cats), N_MAP, replace=False)]
    cats = to_tensor(map_cats[:N_CATS], device)
    showcase = to_tensor(held_out[:N_RECON], device)
    steps_per_epoch = len(train_cats) // BATCH

    for name in NAMES:
        run = next(r for r in RUN_LIST if r.name == name)
        total = run.epochs * steps_per_epoch
        # 0, then 8, 16, 32, ... doubling, plus the end and a few in between at the long tail
        marks = sorted({0, *[2 ** k for k in range(3, 20) if 2 ** k < total], 3000, 5000, total})
        codes, recons, at, per_step = [], [], [], []

        @torch.no_grad()
        def record(model, step):
            was = model.training
            model.eval()
            z = model.encode(cats)
            codes.append((z[0] if run.kind == "vae" else z).cpu().numpy())  # mu for the VAE
            x_hat = model(showcase)[0]
            recons.append((x_hat.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy())
            at.append(step)
            model.train(was)

        def on_step(model, step, parts):
            per_step.append([parts.get("recon", np.nan), parts.get("kl", np.nan)])
            if step in marks:
                record(model, step)

        print(f"\n=== {name} (snapshot replay, {len(marks)} snapshots over {total} steps)")
        out = OUT / name
        train(run, out, data.CatDataset(train_cats, augment=True), device, 0,
              on_epoch=lambda m, e: e == 0 and record(m, 0), on_step=on_step)
        (out / "model.pt").unlink()  # the weights are not needed, only what happened on the way
        np.savez_compressed(OUT / f"{name}.npz", codes=np.stack(codes), recons=np.stack(recons),
                            originals=held_out[:N_RECON], steps=np.array(at),
                            per_step=np.array(per_step, np.float32), steps_per_epoch=steps_per_epoch)
        print(f"wrote {OUT / name}.npz: {len(codes)} snapshots at steps {at}")


if __name__ == "__main__":
    main()
