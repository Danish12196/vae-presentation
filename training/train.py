"""Train every model in the presentation.

    python train.py --list                     show the runs
    python train.py --quick --epochs 2         smoke test on a small subset
    python train.py --skip-existing            full training, resumable run by run
    python train.py --only ae_64 vae_128       just some runs

Outputs go to runs/<name>/: model.pt (config, weights, loss history)
and epoch_XXX.png previews (originals, reconstructions, VAE samples).
--quick runs go to runs/quick/<name>/ so they never count as finished real runs.
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

import data
from models import VAE, AutoEncoder, CNNClassifier, kl_divergence, reconstruction_loss

RUNS = Path(__file__).parent / "runs"
BATCH = 128
LR = 1e-3
KL_WARMUP_EPOCHS = 5
PREVIEW_EVERY = 5


@dataclass
class Run:
    name: str
    kind: str  # ae, vae, dae, sparse, contractive, cls
    latent: int = 0
    beta: float = 0.0
    l1: float = 0.0
    contractive: float = 0.0
    epochs: int = 30
    purpose: str = ""


RUN_LIST = [
    Run("cnn_cls", "cls", epochs=15, purpose="cat/dog/wild classifier"),
    Run("ae_2", "ae", 2, purpose="bottleneck slider; Part 4 latent map"),
    Run("ae_16", "ae", 16, purpose="bottleneck slider"),
    Run("ae_64", "ae", 64, purpose="bottleneck slider; anomaly detector; contractive baseline"),
    Run("ae_256", "ae", 256, purpose="bottleneck slider"),
    Run("ae_128", "ae", 128, purpose="AE vs VAE sampling and interpolation"),
    Run("dae_128", "dae", 128, purpose="denoising demo"),
    Run("sparse_l1", "sparse", 1024, l1=0.5, purpose="sparse demo (ReLU code + L1)"),
    Run("sparse_none", "sparse", 1024, l1=0.0, purpose="sparse comparison (no penalty)"),
    Run("cae_64", "contractive", 64, contractive=100.0, purpose="contractive demo vs ae_64"),
    Run("vae_2d_b0", "vae", 2, beta=0.0, purpose="KL slider, beta 0"),
    Run("vae_2d_b0.1", "vae", 2, beta=0.1, purpose="KL slider, beta 0.1"),
    Run("vae_2d_b1", "vae", 2, beta=1.0, purpose="KL slider, beta 1; Part 5 map"),
    Run("vae_2d_b4", "vae", 2, beta=4.0, purpose="KL slider, beta 4"),
    Run("vae_128", "vae", 128, beta=1.0, purpose="generation, interpolation, reparameterization"),
]


# ------------------------------------------------------------------ losses

def autoencoder_loss(run, model, x, kl_weight):
    """Loss for one batch of every autoencoder flavour. Returns (total, parts for logging)."""
    if run.kind == "vae":
        # Reconstruction keeps the cat recognisable; KL keeps the clouds near N(0, 1).
        x_hat, mu, logvar = model(x)
        recon, kl = reconstruction_loss(x_hat, x), kl_divergence(mu, logvar)
        # With weight 0 (beta 0, or the first warm-up step) skip the term: an unconstrained
        # KL can overflow to inf, and 0 * inf = NaN.
        loss = recon + kl_weight * kl if kl_weight > 0 else recon
        return loss, {"recon": recon, "kl": kl}

    if run.kind == "dae":
        # Denoising: corrupt the input, but ask for the CLEAN image back.
        sigma = torch.rand(len(x), 1, 1, 1, device=x.device) * 0.5
        noisy = (x + sigma * torch.randn_like(x)).clamp(0, 1)
        recon = reconstruction_loss(model(noisy)[0], x)
        return recon, {"recon": recon}

    if run.kind == "sparse":
        # Sparse: the L1 penalty charges for every active neuron, so most stay at zero.
        x_hat, z = model(x)
        recon, l1 = reconstruction_loss(x_hat, x), z.abs().sum(1).mean()
        return recon + run.l1 * l1, {"recon": recon, "l1": l1, "active": (z > 0).float().mean()}

    if run.kind == "contractive":
        # Contractive: a small nudge to the input should barely move the code.
        # ||E(x + eps) - E(x)||^2 / eps^2 is a cheap random estimate of the Jacobian penalty.
        # Clean and nudged images share one batch so BatchNorm treats them alike.
        eps = 0.05
        z, z_nudged = model.encode(torch.cat([x, x + eps * torch.randn_like(x)])).chunk(2)
        recon = reconstruction_loss(model.decode(z), x)
        jac = ((z_nudged - z) ** 2).sum(1).mean() / eps ** 2
        # Measured relative to how spread out the codes are; otherwise the network just
        # shrinks the whole code (the decoder scales it back up) without becoming any steadier.
        jac = jac / z.var(0).sum()
        return recon + run.contractive * jac, {"recon": recon, "jacobian": jac}

    recon = reconstruction_loss(model(x)[0], x)  # plain AE: just rebuild the input
    return recon, {"recon": recon}


def colour_jitter(x, strength=0.15):
    """Random brightness, contrast and saturation per image, so colour alone can't decide the class."""
    def factor():
        return 1 + (torch.rand(len(x), 1, 1, 1, device=x.device) * 2 - 1) * strength
    x = x * factor()
    mean = x.mean((1, 2, 3), keepdim=True)
    x = (x - mean) * factor() + mean
    grey = x.mean(1, keepdim=True)
    x = (x - grey) * factor() + grey
    return x.clamp(0, 1)


# ------------------------------------------------------------------ training

def make_model(run, num_classes=3):
    if run.kind == "cls":
        return CNNClassifier(num_classes)
    if run.kind == "vae":
        return VAE(run.latent)
    return AutoEncoder(run.latent, relu_code=run.kind == "sparse")


def train(run, out, train_set, device, workers, previews=None, test=None, on_epoch=None, on_step=None):
    """on_epoch(model, epoch) runs before training (epoch 0) and after every epoch; on_step(model, step, parts)
    after every optimiser step (snapshots.py uses both)."""
    torch.manual_seed(0)
    out.mkdir(parents=True, exist_ok=True)
    num_classes = len(test["classes"]) if run.kind == "cls" else 3
    model = make_model(run, num_classes).to(device)
    loader = DataLoader(train_set, BATCH, shuffle=True, drop_last=True, num_workers=workers,
                        pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=run.epochs * len(loader))
    history = []
    if on_epoch:
        on_epoch(model, 0)

    for epoch in range(1, run.epochs + 1):
        model.train()
        start, totals = time.time(), {}
        for step, batch in enumerate(loader):
            if run.kind == "cls":
                x, y = (t.to(device, non_blocking=True) for t in batch)
                logits = model(colour_jitter(x))
                loss = F.cross_entropy(logits, y)
                parts = {"acc": (logits.argmax(1) == y).float().mean()}
            else:
                x = batch.to(device, non_blocking=True)
                # KL warm-up: the KL weight ramps linearly from 0 to beta over the first epochs,
                # so the decoder learns to draw before the latent space gets squeezed.
                progress = (epoch - 1 + step / len(loader)) / KL_WARMUP_EPOCHS
                loss, parts = autoencoder_loss(run, model, x, run.beta * min(1.0, progress))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if on_step:
                on_step(model, (epoch - 1) * len(loader) + step + 1, {k: v.item() for k, v in parts.items()})
            for k, v in {"loss": loss, **parts}.items():
                totals[k] = totals.get(k, 0.0) + v.item()

        record = {k: v / len(loader) for k, v in totals.items()}
        history.append(record)
        print(f"[{run.name}] epoch {epoch:3d}/{run.epochs}  "
              + "  ".join(f"{k} {v:.4g}" for k, v in record.items())
              + f"  ({time.time() - start:.1f}s)", flush=True)
        if on_epoch:
            on_epoch(model, epoch)
        if previews is not None and (epoch % PREVIEW_EVERY == 0 or epoch == run.epochs):
            save_preview(model, run, previews, out / f"epoch_{epoch:03d}.png")

    result = {"run": asdict(run), "state_dict": model.state_dict(), "history": history}
    if run.kind == "cls":
        result["classes"] = test["classes"]
        result["metrics"] = evaluate_classifier(model, test, device)
        (out / "metrics.json").write_text(json.dumps(result["metrics"], indent=2))
    torch.save(result, out / "model.pt")


@torch.no_grad()
def save_preview(model, run, originals, path):
    """One image: a row of held-out originals, their reconstructions, and for VAEs 32 random samples."""
    model.eval()
    rows = [originals, model(originals)[0]]
    if run.kind == "vae":
        z = torch.randn(32, run.latent, device=originals.device, generator=torch.Generator(originals.device).manual_seed(0))
        samples = model.decode(z)
        rows += [samples[:16], samples[16:]]
    save_grid(torch.stack(rows), path)
    model.train()


def save_grid(rows, path, gap=2):
    """(rows, cols, 3, 64, 64) tensor in [0, 1] -> PNG with a small gap between tiles."""
    r, c, _, h, w = rows.shape
    canvas = np.full((r * (h + gap) + gap, c * (w + gap) + gap, 3), 255, np.uint8)
    tiles = (rows.clamp(0, 1) * 255).round().byte().permute(0, 1, 3, 4, 2).cpu().numpy()
    for i in range(r):
        for j in range(c):
            y, x = gap + i * (h + gap), gap + j * (w + gap)
            canvas[y:y + h, x:x + w] = tiles[i, j]
    Image.fromarray(canvas).save(path)


@torch.no_grad()
def evaluate_classifier(model, test, device):
    model.eval()

    def predict(images):
        x = torch.from_numpy(images).permute(0, 3, 1, 2)
        return torch.cat([model(b.to(device).float() / 255).argmax(1).cpu() for b in x.split(512)]).numpy()

    classes = test["classes"]
    pred, true = predict(test["x_test"]), test["y_test"]
    confusion = np.zeros((len(classes), len(classes)), int)
    np.add.at(confusion, (true, pred), 1)  # rows: true class, columns: predicted
    per_class = {c: float(confusion[i, i] / confusion[i].sum()) for i, c in enumerate(classes)}
    held_out_cats = float((predict(test["held_out_cats"]) == 0).mean())

    print(f"[cnn_cls] test accuracy {(pred == true).mean():.3f}")
    for c, acc in per_class.items():
        print(f"  {c:5s} {acc:.3f}")
    print("  confusion (rows true, cols predicted): " + " ".join(classes))
    for c, row in zip(classes, confusion):
        print(f"  {c:5s} " + " ".join(f"{v:5d}" for v in row))
    print(f"  held-out fferlito cats called 'cat': {held_out_cats:.3f}")
    return {"classes": classes, "accuracy": float((pred == true).mean()), "per_class": per_class,
            "confusion": confusion.tolist(), "held_out_cats_accuracy": held_out_cats}


# ------------------------------------------------------------------ main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", metavar="NAME", help="train only these runs")
    p.add_argument("--list", action="store_true", help="list the runs and exit")
    p.add_argument("--epochs", type=int, help="override the number of epochs of every run")
    p.add_argument("--quick", action="store_true", help="small data subset, for testing the pipeline")
    p.add_argument("--skip-existing", action="store_true", help="skip runs that already have model.pt")
    p.add_argument("--workers", type=int, default=0, help="DataLoader worker processes")
    args = p.parse_args()

    if args.list:
        for r in RUN_LIST:
            print(f"{r.name:12s} {r.kind:12s} latent {r.latent:5d}  {r.epochs:3d} epochs  {r.purpose}")
        return

    runs = RUN_LIST
    if args.only:
        unknown = set(args.only) - {r.name for r in RUN_LIST}
        if unknown:
            p.error(f"unknown run(s): {', '.join(sorted(unknown))}")
        runs = [r for r in RUN_LIST if r.name in args.only]
    out_dir = RUNS / "quick" if args.quick else RUNS
    if args.skip_existing:
        runs = [r for r in runs if not (out_dir / r.name / "model.pt").exists()]
    if args.epochs:
        runs = [Run(**{**asdict(r), "epochs": args.epochs}) for r in runs]
    if not runs:
        print("nothing to train")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"  # pick the fastest conv kernels once
    print(f"device: {device}")

    train_cats, held_out = data.split(data.load_cats(parts=1 if args.quick else 3))
    print(f"cats: {len(train_cats)} train, {len(held_out)} held out")
    previews = torch.from_numpy(held_out[:16]).permute(0, 3, 1, 2).float().div(255).to(device)

    for run in runs:
        print(f"\n=== {run.name}: {run.purpose}")
        if run.kind == "cls":
            animals = data.load_animals(train_cats, quick=args.quick)
            animals["held_out_cats"] = held_out
            print(f"animals: {len(animals['x_train'])} train, {len(animals['x_test'])} test, "
                  f"classes {animals['classes']}")
            train_set = data.CatDataset(animals["x_train"], animals["y_train"], augment=True)
            train(run, out_dir / run.name, train_set, device, args.workers, test=animals)
        else:
            train(run, out_dir / run.name, data.CatDataset(train_cats, augment=True), device, args.workers,
                  previews=previews)


if __name__ == "__main__":
    main()
