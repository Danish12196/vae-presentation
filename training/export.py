"""Turn the trained runs into the files the website loads.

    python export.py

site/models/  ONNX encoders, decoders and the classifier, plus manifest.json
site/data/    sprites and precomputed results for the demos that would be too slow
              (or too heavy) to run in the browser, plus files.json

Every file except manifest.json and files.json gets a short content hash in its name
(vae_128.decoder.3fa9c1.onnx), so the host can let browsers cache them forever: a changed
file is a new name. manifest.json names the model files; files.json maps every logical
name (data/showcase.png) to its current file. At the end this runs tools/stamp_site.py,
which gives the service worker a new cache version.

Models that haven't been trained yet are skipped; the site shows a
"this cat is still training" card for them.
"""

import hashlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from onnx import TensorProto, helper, numpy_helper
from PIL import Image
from torch import nn

import data
from train import RUN_LIST, RUNS, Run, make_model

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
MODELS = SITE / "models"
OUT = SITE / "data"

MB = 1_000_000
FILE_LIMIT = 25 * MB    # static host limit per file
TOTAL_LIMIT = 60 * MB   # whole site download, from SPEC.md
MARGIN = 1 * MB         # room for files.json and small changes
OPSET = 17
TOLERANCE = {"fp32": 1e-4, "fp16": 1e-2}

N_SHOWCASE = 64     # demo inputs everywhere on the site
N_MAP = 2000        # training cats on the latent maps and for code statistics
SPARSE = {"sparse_l1", "sparse_none"}  # too heavy to ship; their codes are precomputed instead
SEED = 0
HASH_LEN = 6
UNHASHED = {"manifest.json", "files.json", ".gitkeep"}  # fixed names the site starts from


# ------------------------------------------------------------------ helpers

def to_tensor(images, device="cpu"):
    """uint8 (N,64,64,3) -> float (N,3,64,64) in [0,1]."""
    return torch.from_numpy(images).permute(0, 3, 1, 2).float().div(255).to(device)


def to_uint8(x):
    """float (N,3,H,W) in [0,1] -> uint8 (N,H,W,3)."""
    return (x.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def save_sprite(images, path, cols=8, **save_args):
    """uint8 (N,H,W,3) -> one image grid, filled left to right, top to bottom, no gaps."""
    n, h, w, _ = images.shape
    rows = -(-n // cols)
    sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
    Image.fromarray(sheet).save(path, optimize=True, **save_args)
    return {"tile": w, "cols": cols, "count": n}


def hashed_name(name, content):
    """showcase.png + bytes -> showcase.3fa9c1.png"""
    stem, dot, suffix = name.rpartition(".")
    return f"{stem}.{hashlib.sha256(content).hexdigest()[:HASH_LEN]}.{suffix}"


def logical_name(name):
    """showcase.3fa9c1.png -> showcase.png"""
    return re.sub(rf"\.[0-9a-f]{{{HASH_LEN}}}(\.[^.]+)$", r"\1", name)


def hash_data_files():
    """Rename every file in site/data to its content-hashed name. The JSON files keep using
    logical names among themselves; the site looks them up in files.json."""
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name not in UNHASHED and logical_name(path.name) == path.name:
            path.replace(OUT / hashed_name(path.name, path.read_bytes()))


def write_json(path, obj):
    path.write_text(json.dumps(obj, separators=(",", ":")))


def rounded(x, digits=4):
    return np.round(np.asarray(x, dtype=np.float64), digits).tolist()


def load_runs(device):
    """{name: (run, model in eval mode, checkpoint)} for every trained run."""
    loaded = {}
    for run in RUN_LIST:
        path = RUNS / run.name / "model.pt"
        if not path.exists():
            print(f"skipping {run.name}: not trained yet")
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        run = Run(**ckpt["run"])
        model = make_model(run, len(ckpt.get("classes", [0, 0, 0])))
        model.load_state_dict(ckpt["state_dict"])
        loaded[run.name] = (run, model.to(device).eval(), ckpt)
    return loaded


@torch.no_grad()
def codes(run, model, x):
    """The code the site works with: z for autoencoders, mu for VAEs."""
    return torch.cat([model.encode(b)[0] if run.kind == "vae" else model.encode(b) for b in x.split(256)])


@torch.no_grad()
def recon_errors(model, x):
    """Summed squared error per image, like the training loss."""
    return torch.cat([((model(b)[0] - b) ** 2).flatten(1).sum(1) for b in x.split(256)])


# ------------------------------------------------------------------ ONNX

class Encoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode(image)  # z, or (mu, logvar) for a VAE


class Decoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, z):
        return self.model.decode(z)


def onnx_parts(run, model):
    """The ONNX graphs one model becomes: [(role, file name, module, input name, output names)]."""
    if run.kind == "cls":
        return [("model", f"{run.name}.onnx", model, "image", ["logits"])]
    code = ["mu", "logvar"] if run.kind == "vae" else ["z"]
    # eval(): the exporter restores the wrapper's mode onto the model inside it.
    return [("encoder", f"{run.name}.encoder.onnx", Encoder(model).eval(), "image", code),
            ("decoder", f"{run.name}.decoder.onnx", Decoder(model).eval(), "z", ["image"])]


def export_onnx(module, example, input_name, output_names):
    """PyTorch module -> ONNX model (float32), batch dimension left free."""
    buf = io.BytesIO()
    names = [input_name] + output_names
    torch.onnx.export(module, example, buf, opset_version=OPSET, input_names=[input_name],
                      output_names=output_names, dynamic_axes={n: {0: "batch"} for n in names})
    return onnx.load_from_string(buf.getvalue())


def float16_weights(model):
    """Store the weights as float16 and cast them back to float32 inside the graph.

    Half the download, and the maths still runs in float32, so it works on every
    onnxruntime-web backend. Inputs and outputs stay float32.
    """
    graph = model.graph
    casts = []
    for init in graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            continue
        name = init.name
        init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), name + "_fp16"))
        casts.append(helper.make_node("Cast", [name + "_fp16"], [name], to=TensorProto.FLOAT))
    nodes = casts + list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes)
    onnx.checker.check_model(model)
    return model


def max_diff(model_bytes, module, example, input_name):
    """Largest absolute difference between onnxruntime and PyTorch on the same input."""
    session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    got = session.run(None, {input_name: example.numpy()})
    with torch.no_grad():
        want = module(example)
    want = want if isinstance(want, tuple) else (want,)
    return max(float(np.abs(g - w.numpy()).max()) for g, w in zip(got, want))


def export_models(loaded, verify_images, data_bytes):
    """Export, shrink to the size budget, verify. Returns {name: {role: file}} and the table rows."""
    parts = []  # dicts: name, role, file, module, example, input, outputs, onnx
    for name, (run, model, _) in loaded.items():
        if name in SPARSE:
            continue
        model = model.cpu()
        for role, file, module, input_name, outputs in onnx_parts(run, model):
            example = verify_images if input_name == "image" else codes(run, model, verify_images)
            parts.append({"name": name, "role": role, "file": file, "module": module, "example": example,
                          "input": input_name, "proto": export_onnx(module, example, input_name, outputs),
                          "precision": "fp32"})

    # Size budget: switch the biggest files to float16 weights until everything fits,
    # except files that would then miss the float16 tolerance.
    size = lambda p: p["proto"].ByteSize()
    budget = TOTAL_LIMIT - MARGIN - data_bytes
    for p in sorted(parts, key=size, reverse=True):
        if sum(map(size, parts)) <= budget and size(p) <= FILE_LIMIT:
            break
        half = float16_weights(onnx.load_from_string(p["proto"].SerializeToString()))
        if max_diff(half.SerializeToString(), p["module"], p["example"], p["input"]) < TOLERANCE["fp16"]:
            p["proto"], p["precision"] = half, "fp16"

    rows, files = [], {}
    for old in MODELS.glob("*.onnx"):
        old.unlink()
    for p in parts:
        model_bytes = p["proto"].SerializeToString()
        file = hashed_name(p["file"], model_bytes)
        (MODELS / file).write_bytes(model_bytes)
        diff = max_diff(model_bytes, p["module"], p["example"], p["input"])
        ok = diff < TOLERANCE[p["precision"]] and len(model_bytes) <= FILE_LIMIT
        rows.append((file, len(model_bytes), p["precision"], diff, ok))
        files.setdefault(p["name"], {})[p["role"]] = {"file": file, "precision": p["precision"]}
    return files, rows


# ------------------------------------------------------------------ demo data

def export_showcase(held_out):
    """The 64 held-out cats every demo starts from."""
    meta = save_sprite(held_out[:N_SHOWCASE], OUT / "showcase.png")
    write_json(OUT / "showcase.json", {**meta, "indices": list(range(N_SHOWCASE)),
                                       "note": "indices into the 512 held-out cats"})


def export_animals(train_cats):
    """8 dogs and 8 wild animals from the classifier's test split (never trained on)."""
    animals = data.load_animals(train_cats)
    classes = animals["classes"]
    picks = np.concatenate([np.flatnonzero(animals["y_test"] == c)[:8] for c in range(1, len(classes))])
    meta = save_sprite(animals["x_test"][picks], OUT / "animals.png")
    labels = [classes[c] for c in animals["y_test"][picks]]
    write_json(OUT / "animals.json", {**meta, "labels": labels, "classes": classes})


def export_dream(cls_model, classes, device, steps=200, frames=8):
    """Ask the classifier to draw each class: gradient ascent on the input image.

    Start from grey noise and nudge the pixels to raise one class's score.
    Small random shifts and an occasional blur keep it from becoming pure static,
    but the result is still a psychedelic mess: knowing a cat is not drawing one.
    """
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    # Snapshots spaced geometrically: most of the change happens in the first few steps.
    snap_at = (np.geomspace(1, steps, frames).round().astype(int) - 1).tolist()
    blur = torch.tensor([1., 2., 1.], device=device)
    blur = (blur[:, None] * blur[None, :] / 16).expand(3, 1, 3, 3)
    rows, confidence = [], {}
    for c, cls in enumerate(classes):
        # Optimise a pre-sigmoid image so the pixels always stay in [0, 1].
        p = (0.3 * torch.randn(1, 3, 64, 64, device=device)).requires_grad_()
        opt = torch.optim.Adam([p], lr=0.05)
        shots = []
        for step in range(steps):
            if step in snap_at:
                shots.append(torch.sigmoid(p).detach())
            dx, dy = rng.integers(-2, 3, size=2)
            x = torch.roll(torch.sigmoid(p), (int(dy), int(dx)), dims=(2, 3))  # jitter
            loss = -cls_model(x)[0, c]
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 10 == 9:
                with torch.no_grad():
                    p.copy_(F.conv2d(p, blur, padding=1, groups=3))  # blur: favour smooth shapes
        final = torch.sigmoid(p).detach()
        with torch.no_grad():
            confidence[cls] = float(F.softmax(cls_model(final), 1)[0, c])
        rows.append(torch.cat(shots + [final]))
    meta = save_sprite(to_uint8(torch.cat(rows)), OUT / "cls_dream.png", cols=frames + 1)
    write_json(OUT / "cls_dream.json", {**meta, "classes": classes, "frame_steps": snap_at + [steps],
                                        "layout": "one row per class: 8 frames, then the final image",
                                        "confidence": confidence})


def export_pca(train_cats, held_out, device, ks=(2, 16, 64, 256)):
    """PCA on the training cats, the linear baseline for the autoencoders.

    A linear autoencoder trained with squared error learns this same subspace.
    """
    x = to_tensor(train_cats, device).flatten(1)
    mean = x.mean(0)
    torch.manual_seed(SEED)
    _, _, V = torch.pca_lowrank(x - mean, q=max(ks), center=False)
    show = to_tensor(held_out[:N_SHOWCASE], device).flatten(1)
    errors = {}
    for k in ks:
        basis = V[:, :k]
        recon = (mean + (show - mean) @ basis @ basis.T).clamp(0, 1)
        errors[k] = rounded(((recon - show) ** 2).sum(1).cpu(), 2)
        save_sprite(to_uint8(recon.view(-1, 3, 64, 64)), OUT / f"pca_k{k}.png")
    write_json(OUT / "pca.json", {"ks": list(ks), "files": [f"pca_k{k}.png" for k in ks],
                                  "squared_error": errors})


def export_latent_maps(loaded, map_cats, device):
    """2D codes of 2,000 training cats for the AE map (Part 4) and the VAE maps (Part 5)."""
    x = to_tensor(map_cats, device)
    colours = ["#%02x%02x%02x" % tuple(c) for c in map_cats.reshape(len(map_cats), -1, 3).mean(1).round().astype(int)]
    maps = {name: rounded(codes(run, model, x).cpu()) for name, (run, model, _) in loaded.items()
            if name == "ae_2" or name.startswith("vae_2d_b")}
    thumbs = np.stack([np.asarray(Image.fromarray(c).resize((32, 32), Image.LANCZOS)) for c in map_cats])
    # JPEG: 2,000 photos are 4.7 MB as PNG but 1 MB as JPEG, with no visible difference at 32px.
    meta = save_sprite(thumbs, OUT / "map_thumbs.jpg", cols=50, quality=90)
    write_json(OUT / "latent_maps.json", {"count": len(map_cats), "colours": colours, "maps": maps,
                                          "thumbs": {"file": "map_thumbs.jpg", **meta}})


N_SIZE_CATS = 600  # dots per map on the latent-size slide (the first 600 map cats)


def export_latent_sizes(loaded, map_cats, device):
    """The same 600 cats in ae_2, ae_16, ae_64 and ae_256's latent spaces, flattened to 2-D by PCA
    (ae_2 needs none), plus how much of the code's variance those 2 axes keep."""
    x = to_tensor(map_cats[:N_SIZE_CATS], device)
    out = {"count": N_SIZE_CATS, "maps": {}, "kept": {}}
    for name in ("ae_2", "ae_16", "ae_64", "ae_256"):
        if name not in loaded:
            continue
        run, model, _ = loaded[name]
        z = codes(run, model, x).cpu().numpy().astype(np.float64)
        if z.shape[1] > 2:
            z = z - z.mean(0)
            _, s, vt = np.linalg.svd(z, full_matrices=False)
            out["kept"][name] = round(float((s[:2] ** 2).sum() / (s ** 2).sum()), 3)
            z = z @ vt[:2].T
        else:
            out["kept"][name] = 1.0
        out["maps"][name] = rounded(z, 3)
    write_json(OUT / "latent_sizes.json", out)


def export_averages(train_cats):
    """Pixel averages of many training cats for the blur slide: the average of sharp cats is a
    blurry cat, and of thousands of them a beige ghost. (Up to 64 the site averages live.)"""
    order = np.random.default_rng(SEED).permutation(len(train_cats))
    counts = [512, 2048, len(train_cats)]
    avgs = np.stack([train_cats[order[:n]].astype(np.float64).mean(0).round().astype(np.uint8) for n in counts])
    meta = save_sprite(avgs, OUT / "averages.png", cols=len(counts))
    write_json(OUT / "averages.json", {"file": "averages.png", "counts": counts, **meta})


def export_training():
    """The snapshot replays (snapshots.py) for the "Watch it learn" slides: the loss at about 300
    log-spaced steps (each averaged over its window), and at every snapshot step the latent map
    of 600 cats and four showcase cats rebuilt."""
    out = {"models": {}}
    rows = []
    for name in ("ae_2", "vae_2d_b1"):
        path = RUNS / "snapshots" / f"{name}.npz"
        if not path.exists():
            print(f"skipping training replay of {name}: run snapshots.py first")
            continue
        npz = np.load(path)
        per_step = npz["per_step"]  # (steps, 2): recon, kl
        total = len(per_step)
        edges = np.unique(np.geomspace(1, total, 300).round().astype(int))
        lo = np.concatenate([[0], edges[:-1]])
        curve_steps = edges.tolist()
        recon = [float(per_step[a:b, 0].mean()) for a, b in zip(lo, edges)]
        kl = None if np.isnan(per_step[:, 1]).all() else [float(per_step[a:b, 1].mean()) for a, b in zip(lo, edges)]
        if not rows:
            rows.append(npz["originals"])
        out["models"][name] = {
            "steps": npz["steps"].tolist(),             # snapshot steps (0 = before training)
            "steps_per_epoch": int(npz["steps_per_epoch"]),
            "total_steps": int(total),
            "curve": {"steps": curve_steps, "recon": rounded(recon, 2), "kl": None if kl is None else rounded(kl, 3)},
            "codes": [rounded(c, 2) for c in npz["codes"]],
            "recon_row": len(rows),
        }
        rows.extend(npz["recons"])  # one row of 4 cats per snapshot
    if not out["models"]:
        return
    cols = len(rows[0])
    meta = save_sprite(np.concatenate(rows), OUT / "training_recon.jpg", cols=cols, quality=85)
    out["recons"] = {"file": "training_recon.jpg", **meta, "note": "row 0: originals; then one row per snapshot"}
    write_json(OUT / "training.json", out)


def export_anomaly(ae64, held_out, device):
    """How well ae_64 rebuilds normal (held-out) cats. Returns the (low, high) flag thresholds.

    Two-sided: an error above the 95th percentile means "I can't rebuild this, it's not a cat",
    and one below the 1st percentile means "suspiciously easy": flat or blurry images
    are simpler than any real cat face.
    """
    errors = recon_errors(ae64, to_tensor(held_out, device)).cpu().numpy()
    p1, p95, p99 = (float(np.percentile(errors, q)) for q in (1, 95, 99))
    write_json(OUT / "anomaly.json", {"model": "ae_64", "errors": rounded(errors, 2),
                                      "p1": p1, "p95": p95, "p99": p99,
                                      "flag": "error < p1 (too easy) or error > p95 (can't rebuild)"})
    return p1, p95


def verdict(error, low, high):
    if error > high:
        return "anomaly: can't rebuild it"
    if error < low:
        return "anomaly: suspiciously easy"
    return "passes as a cat"


def not_cats(held_out):
    """Six images that are not (normal) cats, for the anomaly detector."""
    rng = np.random.default_rng(SEED)
    cat, furry = held_out[0], held_out[N_SHOWCASE]
    yy, xx = np.mgrid[:64, :64]
    stripes = np.where(((xx + yy) // 8) % 2 == 0, 1, 0)[..., None] * np.array([255, 107, 107]) \
        + np.where(((xx + yy) // 8) % 2 == 1, 1, 0)[..., None] * np.array([76, 201, 240])
    zoom = np.asarray(Image.fromarray(furry[6:18, 26:38]).resize((64, 64), Image.BICUBIC))
    images = {
        "upside-down cat": cat[::-1],
        "colour-inverted cat": 255 - cat,
        "random noise": rng.integers(0, 256, (64, 64, 3)),
        "diagonal stripes": stripes,
        "beige wall": np.broadcast_to(np.array([216, 200, 168]), (64, 64, 3)),
        "extreme fur zoom": zoom,
    }
    return list(images), np.stack([np.asarray(v, dtype=np.uint8) for v in images.values()])


def export_not_cats(held_out, ae64, thresholds, device):
    """The not-cats, ae_64's error on each, and its verdict.

    Any that pass as a cat are the honest blind spot: an upside-down or colour-inverted
    cat still has fur, eyes and a face-like layout, so ae_64 rebuilds it about as well
    as a normal cat. Low reconstruction error doesn't mean "the right kind of cat".
    """
    labels, images = not_cats(held_out)
    meta = save_sprite(images, OUT / "not_cats.png", cols=6)
    out = {**meta, "labels": labels}
    if ae64 is not None:
        errors = recon_errors(ae64, to_tensor(images, device)).cpu().numpy()
        out["ae_64_errors"] = rounded(errors, 2)
        out["verdicts"] = [verdict(e, *thresholds) for e in errors]
        out["blind_spots"] = [l for l, v in zip(labels, out["verdicts"]) if v == "passes as a cat"]
    write_json(OUT / "not_cats.json", out)


@torch.no_grad()
def export_sparse_codes(loaded, held_out, device):
    """The sparse models are too heavy to ship (39 MB), so precompute what the sparse demo shows
    for the showcase cats: the 1024 codes, the exact fraction of neurons above zero, and the
    reconstructions (as sprites in the showcase layout)."""
    out = {"layout": "per model: codes (one row of 1024 per showcase cat), active (fraction > 0 per cat)",
           "recon": {}}
    x = to_tensor(held_out[:N_SHOWCASE], device)
    for name in sorted(SPARSE & loaded.keys()):
        run, model, _ = loaded[name]
        z = codes(run, model, x)
        # 4 decimals: sparse_l1's few active neurons are small, 2 decimals would zero many of them.
        out[name] = {"codes": rounded(z.cpu(), 4), "active": rounded((z > 0).float().mean(1).cpu(), 4)}
        save_sprite(to_uint8(model.decode(z)), OUT / f"{name}_recon.png")
        out["recon"][name] = f"{name}_recon.png"
    write_json(OUT / "sparse_codes.json", out)


# ------------------------------------------------------------------ manifest

def code_stats(run, model, x):
    """Per-dimension mean and std of the code over training cats (the contractive demo normalises by them)."""
    z = codes(run, model, x).cpu()
    stats = {"code_mean": rounded(z.mean(0), 5), "code_std": rounded(z.std(0), 5)}
    if run.latent == 2:
        stats["bounds"] = {"min": rounded(z.min(0).values), "max": rounded(z.max(0).values)}
    if run.kind == "sparse":
        stats["active_fraction"] = float((z > 0).float().mean())
    return stats


def write_manifest(loaded, files, map_cats, device):
    x = to_tensor(map_cats, device)
    models = {}
    for name, (run, model, ckpt) in loaded.items():
        entry = {"name": name, "kind": run.kind, "latent": run.latent, "beta": run.beta,
                 "files": files.get(name, {})}
        if run.kind == "cls":
            entry["classes"] = ckpt["classes"]
            entry["metrics"] = json.loads((RUNS / name / "metrics.json").read_text())
        else:
            entry.update(code_stats(run, model, x))
        if name in SPARSE:
            entry["codes"] = "../data/sparse_codes.json"
        models[name] = entry
    missing = [r.name for r in RUN_LIST if r.name not in loaded]
    write_json(MODELS / "manifest.json", {"models": models, "missing": missing,
                                          "input": {"name": "image", "shape": [None, 3, 64, 64],
                                                    "range": [0, 1]}})


def write_file_list():
    """Every file the site has to download (the loading screen and service worker read this),
    with its logical name: {"name": "data/showcase.png", "path": "data/showcase.3fa9c1.png"}."""
    files = [p for d in (MODELS, OUT) for p in sorted(d.iterdir())
             if p.is_file() and p.name not in (".gitkeep", "files.json")]
    entries = [{"name": p.parent.relative_to(SITE).as_posix() + "/" + logical_name(p.name),
                "path": p.relative_to(SITE).as_posix(), "bytes": p.stat().st_size} for p in files]
    total = sum(e["bytes"] for e in entries)
    write_json(OUT / "files.json", {"total_bytes": total, "files": entries})
    return total + (OUT / "files.json").stat().st_size


# ------------------------------------------------------------------ main

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True  # same files every time we export
    torch.backends.cudnn.benchmark = False
    MODELS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.iterdir():  # hashed names never overwrite each other: start clean
        if old.is_file() and old.name != ".gitkeep":
            old.unlink()

    train_cats, held_out = data.split(data.load_cats())
    map_cats = train_cats[np.random.default_rng(SEED).choice(len(train_cats), N_MAP, replace=False)]
    loaded = load_runs(device)
    model = lambda name: loaded[name][1] if name in loaded else None

    print("exporting demo data ...")
    export_showcase(held_out)
    export_pca(train_cats, held_out, device)
    thresholds = export_anomaly(model("ae_64"), held_out, device) if "ae_64" in loaded else None
    export_not_cats(held_out, model("ae_64"), thresholds, device)
    if "cnn_cls" in loaded:
        export_animals(train_cats)
        export_dream(model("cnn_cls"), loaded["cnn_cls"][2]["classes"], device)
    if any(n == "ae_2" or n.startswith("vae_2d_b") for n in loaded):
        export_latent_maps(loaded, map_cats, device)
    if "ae_2" in loaded:
        export_latent_sizes(loaded, map_cats, device)
    export_training()
    export_averages(train_cats)
    if SPARSE & loaded.keys():
        export_sparse_codes(loaded, held_out, device)
    write_manifest(loaded, {}, map_cats, device)  # placeholder so its size counts below

    print("exporting ONNX models ...")
    data_bytes = sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()) \
        + (MODELS / "manifest.json").stat().st_size
    files, rows = export_models(loaded, to_tensor(held_out[:8]), data_bytes)
    for _, m, _ in loaded.values():
        m.to(device)
    write_manifest(loaded, files, map_cats, device)
    hash_data_files()
    total = write_file_list()

    print(f"\n{'file':32s} {'size':>9s}  {'weights':7s} {'max diff':>9s}  check")
    for file, size, precision, diff, ok in rows:
        print(f"{file:32s} {size / MB:7.2f}MB  {precision:7s} {diff:9.2e}  {'OK' if ok else 'FAIL'}")
    model_bytes = sum(r[1] for r in rows)
    print(f"\nmodels {model_bytes / MB:.1f} MB + data {(total - model_bytes) / MB:.1f} MB "
          f"= {total / MB:.1f} MB (budget {TOTAL_LIMIT / MB:.0f} MB)")
    if not all(r[4] for r in rows) or total > TOTAL_LIMIT:
        raise SystemExit("export check FAILED")
    subprocess.run([sys.executable, str(ROOT / "tools" / "stamp_site.py")], check=True)


if __name__ == "__main__":
    main()
