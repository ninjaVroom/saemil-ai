"""`python -m scripts.prepare_data vision|names|tabular [--source uci|synthetic]|all` - fetch once, verify, split, cache."""
import argparse
import hashlib
import os
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np

from heads import names, tabular, vision

ROOT = Path(__file__).resolve().parent.parent
RAW, PREPARED = ROOT / "data" / "raw", ROOT / "data" / "prepared"
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
# file_hash in keras/src/datasets/mnist.py (keras-team/keras, master, read 2026-11-12); see data/SOURCES.md
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
VISION_CACHE = PREPARED / "vision.npz"
SPLIT_SEED = 17  # fixed split; `scripts.train --seed` only changes the training seed
DATASET_ID = f"mnist-sha256:{MNIST_SHA256[:12]}-split-s{SPLIT_SEED}-800/100/100"

# makemore names.txt at a pinned commit (names.txt unchanged since 8f79bd0, "first commit"); see data/SOURCES.md
NAMES_COMMIT = "988aa59e4d8fefa526d06f3b453ad116258398d4"
NAMES_URL = f"https://raw.githubusercontent.com/karpathy/makemore/{NAMES_COMMIT}/names.txt"
NAMES_SHA256 = "0a30b5557f192f32ab962680889aac5f6fda0f4cecf40a6d0b5694f58ea8cc4d"
NAMES_CACHE = PREPARED / "names.npz"
NAMES_SEED, NAMES_LIMIT = 23, 4000  # fixed split; `scripts.train names --seed` only changes the training seed
NAMES_DATASET_ID = (f"makemore-{NAMES_COMMIT[:7]}-sha256:{NAMES_SHA256[:12]}"
                    f"-clean-s{NAMES_SEED}-{NAMES_LIMIT}-80/10/10")

# UCI dataset 468 zip as downloaded 2026-11-12; see data/SOURCES.md
UCI_URL = "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"
UCI_SHA256 = "2972e6184d3ad7beaaa831d9fc2b059dc3ee29df69d1ec593c466a5cd8485d14"
UCI_ROWS, UCI_POSITIVES = 12330, 1908
TABULAR_SEED = 31  # fixed stratified split for both sources
TABULAR_CACHE = {"uci": PREPARED / "tabular.npz", "synthetic": PREPARED / "tabular_synthetic.npz"}


def tabular_dataset_id(source):
    if source == "uci":
        return f"uci-468-sha256:{UCI_SHA256[:12]}-strat-s{TABULAR_SEED}-70/15/15"
    from scripts.distill_data import GENERATOR_VERSION, SEED, SYNTHETIC_SHA256
    return f"synthetic-v{GENERATOR_VERSION}-s{SEED}-sha256:{SYNTHETIC_SHA256[:12]}-strat-s{TABULAR_SEED}-70/15/15"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, filename, expected):
    """Download once to data/raw (via .part), then verify SHA-256; a mismatch deletes the file and stops."""
    path = RAW / filename
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        print(f"downloading {url}")
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        os.replace(tmp, path)
    digest = sha256(path)
    if digest != expected:
        path.unlink()
        raise SystemExit(f"SHA-256 mismatch for {path}: got {digest}, expected {expected} (file removed)")
    return path, digest


def fetch_mnist():
    return fetch(MNIST_URL, "mnist.npz", MNIST_SHA256)


def _save(cache, **arrays):
    PREPARED.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, cache)


def _load(cache, dataset_id, seed, cmd):
    if not cache.exists():
        raise SystemExit(f"{cache} missing; run `python -m scripts.prepare_data {cmd}` first")
    with np.load(cache, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    if str(data.get("dataset_id")) != dataset_id or int(data.get("seed", -1)) != seed:
        raise SystemExit(f"{cache} is stale (dataset_id {data.get('dataset_id')!r}, seed "
                         f"{data.get('seed')!r}); rerun `python -m scripts.prepare_data {cmd}`")
    return data


def prepare_vision():
    seed = SPLIT_SEED
    path, digest = fetch_mnist()
    with np.load(path, allow_pickle=False) as z:  # one member at a time
        y_train, y_test = z["y_train"], z["y_test"]
        idx = vision.split(y_train, y_test, seed)
        x_fit, x_val = (z["x_train"][idx[k]] for k in ("fit", "val"))
        x_test = z["x_test"][idx["test"]]
    dataset_id = DATASET_ID
    arrays = {"x_fit": x_fit, "y_fit": y_train[idx["fit"]], "idx_fit": idx["fit"],
              "x_val": x_val, "y_val": y_train[idx["val"]], "idx_val": idx["val"],
              "x_test": x_test, "y_test": y_test[idx["test"]], "idx_test": idx["test"]}
    for k in ("x_fit", "x_val", "x_test"):
        assert arrays[k].dtype == np.uint8 and arrays[k].shape[1:] == (28, 28)
    for k in ("y_fit", "y_val", "y_test"):
        arrays[k] = arrays[k].astype(np.int64)
    PREPARED.mkdir(parents=True, exist_ok=True)
    tmp = VISION_CACHE.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, dataset_id=np.array(dataset_id), source_sha256=np.array(digest), seed=np.array(seed),
             pixel_convention=np.array("uint8 28x28 row-major, 255=ink on 0=background"), **arrays)
    os.replace(tmp, VISION_CACHE)
    print(f"prepared {dataset_id}: fit {len(x_fit)}, val {len(x_val)}, test {len(x_test)} -> {VISION_CACHE}")
    return VISION_CACHE


def load_vision():
    if not VISION_CACHE.exists():
        raise SystemExit(f"{VISION_CACHE} missing; run `python -m scripts.prepare_data vision` first")
    with np.load(VISION_CACHE, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    if str(data.get("dataset_id")) != DATASET_ID or int(data.get("seed", -1)) != SPLIT_SEED:
        raise SystemExit(f"{VISION_CACHE} is stale (dataset_id {data.get('dataset_id')!r}, seed "
                         f"{data.get('seed')!r}); rerun `python -m scripts.prepare_data vision`")
    return data


def prepare_names():
    path, digest = fetch(NAMES_URL, "names.txt", NAMES_SHA256)
    with open(path, encoding="utf-8") as f:
        source = names.clean(f)
    parts = names.split(source, NAMES_SEED, NAMES_LIMIT)
    assert not (set(parts["fit"]) & set(parts["val"]) or set(parts["fit"]) & set(parts["test"])
                or set(parts["val"]) & set(parts["test"]))
    arrays = {"source_names": np.array(sorted(source)), "vocab": np.array(names.VOCAB)}
    for k, v in parts.items():
        X, y = names.windows(v)
        arrays.update({f"names_{k}": np.array(v), f"X_{k}": X, f"y_{k}": y})
    _save(NAMES_CACHE, dataset_id=np.array(NAMES_DATASET_ID), source_sha256=np.array(digest),
          seed=np.array(NAMES_SEED), **arrays)
    print(f"prepared {NAMES_DATASET_ID}: {len(source)} clean names; " + ", ".join(
        f"{k} {len(v)} names / {len(arrays[f'y_{k}'])} windows" for k, v in parts.items()) + f" -> {NAMES_CACHE}")
    return NAMES_CACHE


def load_names():
    return _load(NAMES_CACHE, NAMES_DATASET_ID, NAMES_SEED, "names")


def read_tabular_source(source):
    """-> (X, y, digest). UCI: hash-verified zip. Synthetic: the distilled CSV, checked against its recorded digest."""
    if source == "uci":
        path, digest = fetch(UCI_URL, "online_shoppers_intention.zip", UCI_SHA256)
        X, y = tabular.read_uci(path)
        if (len(y), int(y.sum())) != (UCI_ROWS, UCI_POSITIVES):
            raise SystemExit(f"UCI 468: expected {UCI_ROWS} rows / {UCI_POSITIVES} positives, got {len(y)} / {y.sum()}")
        return X, y, digest
    from scripts import distill_data
    path = distill_data.OUT
    if not path.exists():
        raise SystemExit(f"{path} missing; run `python -m scripts.distill_data` first")
    digest = sha256(path)
    if digest != distill_data.SYNTHETIC_SHA256:
        raise SystemExit(f"{path}: SHA-256 {digest} != recorded {distill_data.SYNTHETIC_SHA256}; rerun distill_data")
    with open(path, encoding="utf-8", newline="") as f:
        X, y = tabular.read_csv(f)
    return X, y, digest


def prepare_tabular(source="uci"):
    X, y, digest = read_tabular_source(source)
    idx = tabular.split(X, y, TABULAR_SEED)
    dataset_id = tabular_dataset_id(source)
    arrays = {}
    for k, i in idx.items():
        arrays.update({f"X_{k}": X[i], f"y_{k}": y[i], f"idx_{k}": i})
    cache = TABULAR_CACHE[source]
    _save(cache, dataset_id=np.array(dataset_id), source_sha256=np.array(digest), seed=np.array(TABULAR_SEED),
          source=np.array(source), features=np.array(tabular.FEATURES), **arrays)
    print(f"prepared {dataset_id}: {len(y)} rows, {int(y.sum())} positives ({y.mean():.4f}); "
          + ", ".join(f"{k} {len(i)} ({y[i].mean():.4f})" for k, i in idx.items()) + f" -> {cache}")
    return cache


def load_tabular(source="uci"):
    return _load(TABULAR_CACHE[source], tabular_dataset_id(source), TABULAR_SEED, f"tabular --source {source}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m scripts.prepare_data")
    p.add_argument("head", choices=["vision", "names", "tabular", "all"])
    p.add_argument("--source", choices=["uci", "synthetic"], default="uci", help="tabular only")
    a = p.parse_args(argv)
    if a.head in ("vision", "all"):
        prepare_vision()
    if a.head in ("names", "all"):
        prepare_names()
    if a.head in ("tabular", "all"):
        prepare_tabular(a.source)


if __name__ == "__main__":
    main(sys.argv[1:])
