"""Vision head: MNIST digits, 784 -> 128 ReLU -> 10.

Pixel convention: 28x28, row-major, light ink (255) on dark background (0), uint8.
"""
import numpy as np

from saemil import Linear, ReLU, Sequential, softmax

HEAD_ID = "vision"
TRANSFORM_VERSION = 1
SIDE, BOX = 28, 20                      # frame size, longer side of the scaled digit
BLANK_INK = 100                         # raw or centered ink sum below this = blank canvas (D-010)
MIN_EXTENT = 3                          # ink bbox longer side below this (px) = blank canvas (D-010)
BLANK_MESSAGE = "Please draw a digit in the canvas first."
PER_DIGIT = {"fit": 800, "val": 100, "test": 100}


def build(rng):
    return Sequential([Linear(784, 128, "he", rng), ReLU(), Linear(128, 10, "lecun", rng)])


def to_input(pixels_uint8):
    """(B,784) / (B,28,28) / (784,) uint8 -> contiguous (B,784) float32 in [0,1]. Shared by train and serve."""
    arr = np.asarray(pixels_uint8)
    if arr.dtype != np.uint8:
        raise TypeError(f"to_input expects uint8 pixels, got {arr.dtype}")
    return np.ascontiguousarray(arr.reshape(-1, SIDE * SIDE), dtype=np.float32) / np.float32(255)


def split(y_train, y_test, seed=17):
    """Stratified index split: fit/val disjoint from the official train set, test from the official test set."""
    rng = np.random.default_rng(seed)
    out: dict[str, list] = {"fit": [], "val": [], "test": []}
    for d in range(10):
        tr = rng.permutation(np.flatnonzero(y_train == d))
        te = rng.permutation(np.flatnonzero(y_test == d))
        n_fit, n_val = PER_DIGIT["fit"], PER_DIGIT["val"]
        if tr.size < n_fit + n_val or te.size < PER_DIGIT["test"]:
            raise ValueError(f"digit {d}: not enough examples")
        out["fit"].append(tr[:n_fit])
        out["val"].append(tr[n_fit:n_fit + n_val])
        out["test"].append(te[:PER_DIGIT["test"]])
    return {k: np.concatenate(v).astype(np.int64) for k, v in out.items()}


def _resample(m, n):
    """(m, n) area-averaging matrix taking a length-n axis to length m (rows sum to 1)."""
    edges = np.arange(m + 1) * (n / m)
    j = np.arange(n)[None, :]
    overlap = np.minimum(edges[1:, None], j + 1) - np.maximum(edges[:-1, None], j)
    return np.clip(overlap, 0, None) / (n / m)


def center_by_mass(pixels):
    """Ink bbox (no padding) -> longer side to 20 px (aspect kept, area resampling) -> center of mass
    moved to (14, 14) in a 28x28 frame, placement rounded half up.

    Canonical preprocessing (decision D-011): the client canvas code must replicate this exactly.
    Accepts a flat 784 vector or any 2-D image (e.g. a full-size canvas). Returns (784,) uint8.
    """
    img = np.asarray(pixels, dtype=np.float64)
    if img.ndim == 1:
        img = img.reshape(SIDE, SIDE)
    ys, xs = np.nonzero(img)
    if ys.size == 0:
        return np.zeros(SIDE * SIDE, np.uint8)
    crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    scale = BOX / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    small = _resample(nh, h) @ crop @ _resample(nw, w).T
    mass = small.sum()
    cy = (small.sum(axis=1) @ np.arange(nh)) / mass
    cx = (small.sum(axis=0) @ np.arange(nw)) / mass
    centre = SIDE / 2  # MNIST convention: mass centre at (14, 14) in index coords (measured 13.98, 14.02)
    top = int(np.clip(np.floor(centre - cy + 0.5), 0, SIDE - nh))  # half up, like JS Math.round
    left = int(np.clip(np.floor(centre - cx + 0.5), 0, SIDE - nw))
    out = np.zeros((SIDE, SIDE))
    out[top:top + nh, left:left + nw] = small
    return np.clip(np.rint(out), 0, 255).astype(np.uint8).ravel()


def _validate(pixels):
    arr = np.asarray(pixels)
    if arr.dtype.kind not in "iuf":
        raise ValueError(f"pixels must be numeric (int or float), got dtype {arr.dtype}")
    if arr.shape not in ((SIDE * SIDE,), (SIDE, SIDE)):
        raise ValueError(f"expected shape (784,) or (28, 28), got {arr.shape}")
    arr = arr.astype(np.float64)
    if not np.isfinite(arr).all() or arr.min() < 0 or arr.max() > 255 or (arr != np.rint(arr)).any():
        raise ValueError("pixels must be integers in [0, 255]")
    return arr.astype(np.uint8).ravel()


def predict(net, pixels):
    """784 raw intensities -> {"blank": True, "message"} or {"blank": False, "digit", "top3"}."""
    px = _validate(pixels)
    if int(px.sum(dtype=np.int64)) < BLANK_INK:
        return {"blank": True, "message": BLANK_MESSAGE}
    ys, xs = np.nonzero(px.reshape(SIDE, SIDE))
    centered = center_by_mass(px)
    if max(np.ptp(ys), np.ptp(xs)) + 1 < MIN_EXTENT or int(centered.sum(dtype=np.int64)) < BLANK_INK:
        return {"blank": True, "message": BLANK_MESSAGE}
    p = softmax(net.forward(to_input(centered), training=False))[0]
    top = np.argsort(p)[::-1][:3]
    return {"blank": False, "digit": int(top[0]),
            "top3": [{"digit": int(d), "prob": float(p[d])} for d in top]}
