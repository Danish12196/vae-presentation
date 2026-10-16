"""Datasets: 64x64 cat faces for the autoencoders, cat/dog/wild faces for the classifier.

Everything is downloaded once into training/data/ and cached as .npy files,
so after the first run loading takes about a second.
"""

import io
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

DATA = Path(__file__).parent / "data"
SIZE = 64

CAT_URL = "https://raw.githubusercontent.com/fferlito/Cat-faces-dataset/master/"
CAT_PARTS = ["dataset-part1.tar.gz", "dataset-part2.tar.gz", "dataset-part3.tar.gz"]

HELD_OUT = 512  # cats the models never see; the site uses them as demo inputs
SPLIT_SEED = 0

AFHQ_REPO = "huggan/AFHQv2"
AFHQ_SHARDS = 13  # the parquet shards are sorted by label; the last one has all three classes
PETS_URL = "https://thor.robots.ox.ac.uk/datasets/pets/"
ANIMAL_TEST_FRACTION = 0.1


# ---------------------------------------------------------------- cat faces

def download(parts=3):
    """Fetch and extract the fferlito cat-face archives, skipping finished steps."""
    DATA.mkdir(parents=True, exist_ok=True)
    for name in CAT_PARTS[:parts]:
        archive = DATA / name
        done = DATA / (name + ".extracted")
        if done.exists():
            continue
        if not archive.exists():
            print(f"downloading {name} ...")
            _fetch(CAT_URL + name, archive)
        print(f"extracting {name} ...")
        _safe_extract(archive, DATA / "cats" / name.removesuffix(".tar.gz"))
        done.touch()


def load_cats(parts=3):
    """All cat faces as one uint8 array of shape (N, 64, 64, 3)."""
    cache = DATA / f"cats_{parts}parts.npy"
    if cache.exists():
        return np.load(cache)
    download(parts)
    files = []
    for name in CAT_PARTS[:parts]:
        files += sorted((DATA / "cats" / name.removesuffix(".tar.gz")).rglob("*.png"))
    print(f"decoding {len(files)} cat images (only the first time) ...")
    cats = np.stack([_to_64(Image.open(f)) for f in files])
    np.save(cache, cats)
    return cats


def split(cats):
    """Fixed (train, held_out) split. The 512 held-out cats are never trained on."""
    order = np.random.default_rng(SPLIT_SEED).permutation(len(cats))
    return cats[order[HELD_OUT:]], cats[order[:HELD_OUT]]


class CatDataset(Dataset):
    """uint8 (N,64,64,3) images -> float tensors (3,64,64) in [0,1].

    With labels it yields (image, label) pairs, which the classifier uses.
    """

    def __init__(self, images, labels=None, augment=False):
        self.images = torch.from_numpy(images).permute(0, 3, 1, 2)  # NHWC -> NCHW, no copy
        self.labels = None if labels is None else torch.from_numpy(labels).long()
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        x = self.images[i].float() / 255
        if self.augment and torch.rand(()) < 0.5:
            x = x.flip(-1)  # a mirrored cat is still a cat
        return x if self.labels is None else (x, self.labels[i])


# ------------------------------------------------------- animal faces (classifier)

def load_animals(train_cats, quick=False):
    """Balanced cat/dog/wild data for the classifier.

    The cat class is half AFHQ cats and half fferlito cats, so the classifier
    has to learn "which animal", not "which dataset". `train_cats` must not
    contain the held-out cats.
    Returns a dict with x_train, y_train, x_test, y_test (uint8 / int64) and classes.
    """
    try:
        images, labels, classes = _load_afhq(quick)
    except Exception as err:  # Hugging Face down, dataset moved, ...
        print(f"AFHQ unavailable ({err}); falling back to Oxford-IIIT Pet heads")
        images, labels, classes = _load_pets()

    rng = np.random.default_rng(SPLIT_SEED)
    afhq_cats = rng.permutation(np.flatnonzero(labels == 0))
    others = [rng.permutation(np.flatnonzero(labels == c)) for c in range(1, len(classes))]

    # Same number of images in every class, as large as the data allows.
    per_class = min([len(o) for o in others] + [2 * len(afhq_cats), 2 * len(train_cats)])
    n_afhq = per_class // 2
    n_ours = per_class - n_afhq
    ours = train_cats[rng.choice(len(train_cats), n_ours, replace=False)]
    cat_images = rng.permutation(np.concatenate([images[afhq_cats[:n_afhq]], ours]))
    per_class_images = [cat_images] + [images[o[:per_class]] for o in others]

    # A fixed slice of every class is kept aside for testing.
    n_test = int(per_class * ANIMAL_TEST_FRACTION)
    out = {"x_train": [], "y_train": [], "x_test": [], "y_test": []}
    for c, imgs in enumerate(per_class_images):
        out["x_test"].append(imgs[:n_test])
        out["x_train"].append(imgs[n_test:])
        out["y_test"].append(np.full(n_test, c))
        out["y_train"].append(np.full(len(imgs) - n_test, c))
    out = {k: np.concatenate(v) for k, v in out.items()}
    out["classes"] = classes
    return out


def _load_afhq(quick):
    """AFHQv2 (cat, dog, wild) resized to 64x64, cached shard by shard."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    classes = ["cat", "dog", "wild"]
    shards = [AFHQ_SHARDS - 1] if quick else range(AFHQ_SHARDS)
    images, labels = [], []
    for s in shards:
        cache = DATA / "afhq" / f"shard{s:02d}.npz"
        if not cache.exists():
            name = f"data/train-{s:05d}-of-{AFHQ_SHARDS:05d}.parquet"
            print(f"downloading AFHQ {name} (~500 MB) ...")
            path = Path(hf_hub_download(AFHQ_REPO, name, repo_type="dataset",
                                        local_dir=DATA / "afhq" / "raw"))
            table = pq.read_table(path)
            imgs = [_to_64(Image.open(io.BytesIO(cell["bytes"])))
                    for cell in table.column("image").to_pylist()]
            np.savez(cache, images=np.stack(imgs), labels=np.array(table.column("label")))
            path.unlink()  # the 512x512 originals are no longer needed
        with np.load(cache) as f:
            images.append(f["images"])
            labels.append(f["labels"])
    return np.concatenate(images), np.concatenate(labels), classes


def _load_pets():
    """Fallback: Oxford-IIIT Pet head crops (cat vs dog) resized to 64x64."""
    import xml.etree.ElementTree as ET

    classes = ["cat", "dog"]
    cache = DATA / "pets64.npz"
    if not cache.exists():
        for name in ["images.tar.gz", "annotations.tar.gz"]:
            if not (DATA / "pets" / (name + ".extracted")).exists():
                print(f"downloading Oxford-IIIT Pet {name} ...")
                _fetch(PETS_URL + name, DATA / "pets" / name)
                _safe_extract(DATA / "pets" / name, DATA / "pets")
                (DATA / "pets" / (name + ".extracted")).touch()
        images, labels = [], []
        for line in open(DATA / "pets" / "annotations" / "list.txt"):
            if line.startswith("#"):
                continue
            stem, _, species, _ = line.split()
            xml = DATA / "pets" / "annotations" / "xmls" / f"{stem}.xml"
            if not xml.exists():  # only some images have a head box
                continue
            box = ET.parse(xml).find(".//bndbox")
            x0, y0, x1, y1 = (int(box.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
            img = Image.open(DATA / "pets" / "images" / f"{stem}.jpg").crop((x0, y0, x1, y1))
            images.append(_to_64(img))
            labels.append(int(species) - 1)  # 1 = cat, 2 = dog
        np.savez(cache, images=np.stack(images), labels=np.array(labels))
    with np.load(cache) as f:
        return f["images"], f["labels"], classes


# ------------------------------------------------------------------ helpers

def _to_64(img):
    """Any PIL image -> uint8 (64, 64, 3), resized with antialiasing."""
    img = img.convert("RGB")
    if img.size != (SIZE, SIZE):
        img = img.resize((SIZE, SIZE), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def _fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)  # only a complete download gets the real name


def _safe_extract(archive, dest):
    """Extract a tar file without letting it write outside `dest`."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        if hasattr(tarfile, "data_filter"):  # Python 3.11.4+ / 3.12+
            tar.extractall(dest, filter="data")
            return
        root = dest.resolve()
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if not (member.isfile() or member.isdir()) or not target.is_relative_to(root):
                raise RuntimeError(f"unsafe path in {archive.name}: {member.name}")
        tar.extractall(dest)
