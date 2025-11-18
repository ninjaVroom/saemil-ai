"""Tabular head: purchase probability from six whole-session page counts/durations (UCI 468 schema).

These are completed-session totals, so the model describes finished sessions, not live ones.
"""
import csv
import io
import math
import zipfile

import numpy as np

from saemil import Linear, ReLU, Sequential, sigmoid

HEAD_ID = "tabular"
TRANSFORM_VERSION = 1
FEATURES = ["Administrative", "Administrative_Duration", "Informational",
            "Informational_Duration", "ProductRelated", "ProductRelated_Duration"]
COUNTS = {"Administrative", "Informational", "ProductRelated"}
TARGET = "Revenue"
UCI_MEMBER = "online_shoppers_intention.csv"
PERCENTILES = np.array([0, 1, 5, 25, 50, 75, 95, 99, 100], np.float64)
MIN_STD = 1e-6
SOURCE_KINDS = ("real", "synthetic")


def read_csv(text):
    """Text stream with the UCI header -> X float64 (N, 6) in FEATURES order, y int64 0/1."""
    reader = csv.DictReader(text)
    missing = [c for c in FEATURES + [TARGET] if c not in (reader.fieldnames or [])]
    if missing:
        raise ValueError(f"missing columns {missing}")
    xs, ys = [], []
    for n, row in enumerate(reader, start=2):  # line 1 is the header
        label = row[TARGET]
        if label not in ("TRUE", "FALSE"):
            raise ValueError(f"line {n}: {TARGET} must be TRUE/FALSE, got {label!r}")
        try:
            vals = [float(row[c]) for c in FEATURES]
        except (TypeError, ValueError) as e:
            raise ValueError(f"line {n}: non-numeric feature ({e})") from e
        if not all(math.isfinite(v) and v >= 0 for v in vals):
            raise ValueError(f"line {n}: features must be finite and >= 0, got {vals}")
        xs.append(vals)
        ys.append(label == "TRUE")
    if not xs:
        raise ValueError("no data rows")
    return np.array(xs, np.float64), np.array(ys, np.int64)


def read_uci(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        if UCI_MEMBER not in z.namelist():
            raise ValueError(f"{UCI_MEMBER} not in archive (members: {z.namelist()})")
        with z.open(UCI_MEMBER) as f:
            return read_csv(io.TextIOWrapper(f, encoding="utf-8", newline=""))


def split(X, y, seed=31):
    """Stratified 70/15/15 index split by class. Returns sorted int64 index arrays."""
    rng = np.random.default_rng(seed)
    out: dict[str, list] = {"fit": [], "val": [], "test": []}
    for c in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == c))
        n_fit, n_val = round(0.7 * idx.size), round(0.15 * idx.size)
        out["fit"].append(idx[:n_fit])
        out["val"].append(idx[n_fit:n_fit + n_val])
        out["test"].append(idx[n_fit + n_val:])
    return {k: np.sort(np.concatenate(v)).astype(np.int64) for k, v in out.items()}


def _log(X):
    return np.log1p(np.maximum(np.asarray(X, np.float64), 0))


def fit_preproc(X_train):
    """Train-only statistics. Standardization is on log1p(max(x,0)); percentiles/medians are raw units."""
    X = np.asarray(X_train, np.float64)
    L = _log(X)
    std = L.std(axis=0)
    return {"features": np.array(FEATURES), "means": L.mean(axis=0),
            "scales": np.where(std < MIN_STD, 1.0, std),
            "percentile_levels": PERCENTILES.copy(), "percentiles": np.percentile(X, PERCENTILES, axis=0),
            "medians": np.median(X, axis=0)}


def to_input(X_raw, preproc):
    """(N,6) or (6,) raw -> (N,6) float32 standardized log1p features. Shared by train and serve."""
    X = np.asarray(X_raw, np.float64).reshape(-1, len(FEATURES))
    return np.ascontiguousarray((_log(X) - preproc["means"]) / preproc["scales"], dtype=np.float32)


def build(rng):
    return Sequential([Linear(6, 32, "he", rng), ReLU(), Linear(32, 16, "he", rng), ReLU(),
                       Linear(16, 1, "lecun", rng)])


def preproc_from_extra(extra):
    """Artifact extra arrays ('preproc.*') -> preproc dict, checking the stored feature order."""
    pre = {k.removeprefix("preproc."): v for k, v in extra.items()}
    if pre.get("features") is None or pre["features"].tolist() != FEATURES:
        raise ValueError("artifact feature order does not match FEATURES")
    return pre


def validate(features):
    """Strict request check: exactly the six keys, real non-bool finite numbers >= 0, integral counts."""
    if not isinstance(features, dict):
        raise ValueError("features must be a dict")
    if set(features) != set(FEATURES):
        raise ValueError(f"expected exactly keys {FEATURES}, missing {sorted(set(FEATURES) - set(features))}, "
                         f"unexpected {sorted(map(str, set(features) - set(FEATURES)))}")
    row = []
    for k in FEATURES:
        v = features[k]
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
            raise ValueError(f"{k} must be a number, got {type(v).__name__}")
        try:
            v = float(v)
        except OverflowError as e:  # e.g. a huge Python int
            raise ValueError(f"{k} is out of range") from e
        if not math.isfinite(v) or v < 0:
            raise ValueError(f"{k} must be finite and >= 0, got {v}")
        if k in COUNTS and v != int(v):
            raise ValueError(f"{k} is a count and must be a whole number, got {v}")
        row.append(v)
    return np.array(row, np.float64)


def predict(net, preproc, features, source_kind):
    if source_kind not in SOURCE_KINDS:
        raise ValueError(f"source_kind must be one of {SOURCE_KINDS}")
    z = net.forward(to_input(validate(features), preproc), training=False)
    return {"probability": float(sigmoid(z)[0, 0]), "source_kind": source_kind}


def average_precision(p, y):
    """AP = sum over distinct score thresholds (ties grouped) of recall gain x precision."""
    p, y = np.asarray(p, np.float64), np.asarray(y, np.int64)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-p, kind="stable")
    p, y = p[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(p) != 0), p.size - 1]  # last index of each tie group
    tp = np.cumsum(y)[last]
    precision, recall = tp / (last + 1), tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def metrics(p, y, prevalence, bins=10):
    """Held-out scorecard vs a constant `prevalence` (the train positive rate)."""
    p, y = np.asarray(p, np.float64).ravel(), np.asarray(y, np.float64).ravel()
    eps = 1e-12

    def log_loss(q):
        q = np.clip(q, eps, 1 - eps)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    base = np.full_like(p, prevalence)
    which = np.minimum((p * bins).astype(int), bins - 1)
    rel = [{"lo": b / bins, "hi": (b + 1) / bins, "n": int((which == b).sum()),
            "mean_p": float(p[which == b].mean()) if (which == b).any() else None,
            "observed": float(y[which == b].mean()) if (which == b).any() else None} for b in range(bins)]
    return {"n": int(y.size), "positives": int(y.sum()), "observed_prevalence": float(y.mean()),
            "train_prevalence": float(prevalence),
            "log_loss": log_loss(p), "baseline_log_loss": log_loss(base),
            "brier": float(np.mean((p - y) ** 2)), "baseline_brier": float(np.mean((base - y) ** 2)),
            "average_precision": average_precision(p, y), "baseline_average_precision": float(y.mean()),
            "accuracy_at_0.5": float(((p >= 0.5) == y).mean()), "reliability": rel}
