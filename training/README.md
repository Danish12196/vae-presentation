# Training

Trains every model used by the presentation on 64x64 cat faces (plus dogs and
wild animals for the classifier). See `../SPEC.md` for the full design.

| file | what it does |
| --- | --- |
| `export.py` | writes the ONNX models and precomputed demo data into `site/` |
| `data.py` | downloads and caches the datasets, the fixed 512 held-out cats, `CatDataset` |
| `models.py` | shared conv trunk, decoder, AE, VAE, classifier, the two losses |
| `train.py` | the list of runs, one loss per model kind, the training loop |

## Setup (Windows)

```bat
cd training
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Pick the exact torch command for your CUDA version at
https://pytorch.org/get-started/locally/. Check that the GPU is visible with
`python -c "import torch; print(torch.cuda.is_available())"`.

## Quick test (a few minutes)

```bat
python train.py --quick --only ae_64 vae_2d_b1 cnn_cls --epochs 2
```

Uses only cat dataset part 1 and one AFHQ shard. Outputs go to `runs/quick/`,
so they never count as finished real runs.

## Full run

```bat
python train.py --skip-existing
```

Downloads about 270 MB of cats and 7 GB of AFHQ the first time (the AFHQ
originals are deleted once they are shrunk to 64x64). Each finished run is saved
to `runs/<name>/model.pt`, so with `--skip-existing` an interrupted run picks up
where it left off. Every 5 epochs `runs/<name>/epoch_XXX.png` shows 16 held-out
cats, their reconstructions, and for VAEs 32 random samples. The classifier's
test metrics go to `runs/cnn_cls/metrics.json`.

Other options: `--list` shows all runs, `--only <names...>` trains a subset,
`--epochs N` overrides the epoch count, `--workers N` sets DataLoader workers.

## Export for the website

```bat
python export.py
```

Writes the ONNX models and `manifest.json` to `site/models/`, and the demo data
(sprites, PCA, latent maps, anomaly statistics, ...) to `site/data/`. Untrained
models are skipped. It checks every ONNX file against PyTorch, prints a table,
and stores the biggest models' weights in float16 until the site fits in 60 MB.

## The models

| name | kind | latent | what it shows |
| --- | --- | --- | --- |
| `cnn_cls` | classifier | - | the encoder + label head twin; classifies cat/dog/wild and judges generated cats |
| `ae_2`, `ae_16`, `ae_64`, `ae_256` | plain AE | 2-256 | how the bottleneck size trades compression for detail; `ae_2` draws the latent map, `ae_64` detects anomalies |
| `ae_128` | plain AE | 128 | why an AE can't generate: random codes decode to nonsense |
| `dae_128` | denoising AE | 128 | noisy input, clean target: it learns to remove noise |
| `sparse_l1`, `sparse_none` | sparse AE | 1024 | an L1 penalty switches most code neurons off; `none` is the comparison |
| `cae_64` | contractive AE | 64 | the code barely moves when the input is jiggled (compare with `ae_64`) |
| `vae_2d_b0` ... `vae_2d_b4` | VAE | 2 | the KL weight beta, from scattered clusters (0) to packed around zero (4) |
| `vae_128` | VAE | 128 | generation, interpolation, the reparameterization trick |
