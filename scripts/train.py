"""`python -m scripts.train vision|names|tabular [--source uci|synthetic] [--seed N]`.

Train, keep the lowest-validation-loss epoch, evaluate on test, save the archive.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from heads import names, tabular, vision
from saemil import Adam, BinaryCrossEntropy, SoftmaxCrossEntropy, sigmoid
from scripts.prepare_data import load_names, load_tabular, load_vision

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "vision.npz"
EPOCHS, BATCH, LR = 6, 64, 1e-3
NAMES_MODEL = ROOT / "models" / "names.npz"
NAMES_EPOCHS, NAMES_BATCH, NAMES_LR = 8, 128, 3e-3
NAMES_GATE_SEED, NAMES_GATE_N, NAMES_GATE_TEMP = 23, 100, 0.85
TABULAR_MODEL = {"uci": ROOT / "models" / "tabular.npz", "synthetic": ROOT / "models" / "tabular_synthetic.npz"}
TABULAR_EPOCHS, TABULAR_BATCH, TABULAR_LR = 12, 128, 1e-3
DEFAULT_SEED = {"vision": 17, "names": 23, "tabular": 31}


def evaluate(net, x_uint8, y):
    logits = net.forward(vision.to_input(x_uint8), training=False)
    loss, _ = SoftmaxCrossEntropy()(logits, y)
    pred = logits.argmax(axis=1)
    confusion = np.zeros((10, 10), np.int64)
    np.add.at(confusion, (y, pred), 1)  # rows = true digit, cols = predicted
    return loss, float((pred == y).mean()), confusion


def train_vision(seed=17, out=MODEL, log=print):
    data = load_vision()
    rng = np.random.default_rng(seed)
    net, loss_fn, opt = vision.build(rng), SoftmaxCrossEntropy(), Adam(lr=LR)
    x_fit, y_fit = data["x_fit"], data["y_fit"]
    best: tuple[float, int, dict[str, np.ndarray]] = (np.inf, 0, {})
    for epoch in range(1, EPOCHS + 1):
        order, total = rng.permutation(len(y_fit)), 0.0
        for s in range(0, len(order), BATCH):
            b = order[s:s + BATCH]
            total += net.train_step(vision.to_input(x_fit[b]), y_fit[b], loss_fn, opt) * len(b)
        val_ce, val_acc, _ = evaluate(net, data["x_val"], data["y_val"])
        log(f"epoch {epoch}: train CE {total / len(y_fit):.4f}  val CE {val_ce:.4f}  val acc {val_acc:.4f}")
        if val_ce < best[0]:
            best = (val_ce, epoch, {k: v.copy() for k, v in net.named_params().items()})
    for k, v in net.named_params().items():
        v[...] = best[2][k]
    _, test_acc, confusion = evaluate(net, data["x_test"], data["y_test"])
    dataset_id = str(data["dataset_id"])
    manifest = {"head_id": vision.HEAD_ID, "transform_version": vision.TRANSFORM_VERSION,
                "training_source_id": dataset_id, "seed": seed, "epochs_run": EPOCHS,
                "best_epoch": best[1], "val_ce": best[0], "test_accuracy": test_acc,
                "confusion_matrix": confusion.tolist(), "batch": BATCH, "lr": LR, "optimizer": "Adam",
                "pixel_convention": str(data["pixel_convention"]), "input_scaling": "uint8 / 255 float32"}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    net.save(out, manifest)
    return {"best_epoch": best[1], "val_ce": best[0], "test_accuracy": test_acc, "confusion": confusion,
            "dataset_id": dataset_id, "path": out, "rows": len(y_fit)}


def _fit(net, opt, loss_fn, x_of, y_of, n, epochs, batch, rng, val_loss, log):
    """Minibatch loop; restores the parameters of the epoch with the lowest val_loss(net)."""
    best: tuple[float, int, dict[str, np.ndarray]] = (np.inf, 0, {})
    for epoch in range(1, epochs + 1):
        order, total = rng.permutation(n), 0.0
        for s in range(0, n, batch):
            b = order[s:s + batch]
            total += net.train_step(x_of(b), y_of(b), loss_fn, opt) * len(b)
        v = val_loss(net)
        log(f"epoch {epoch}: train loss {total / n:.4f}  val loss {v:.4f}")
        if v < best[0]:
            best = (v, epoch, {k: w.copy() for k, w in net.named_params().items()})
    for k, w in net.named_params().items():
        w[...] = best[2][k]
    return best[0], best[1]


def names_ce(net, X, y):
    return SoftmaxCrossEntropy()(net.forward(X.astype(np.int64), training=False), y.astype(np.int64))[0]


def train_names(seed=23, out=NAMES_MODEL, log=print):
    data = load_names()
    rng = np.random.default_rng(seed)
    net, loss_fn, opt = names.build(rng), SoftmaxCrossEntropy(), Adam(lr=NAMES_LR)
    X, y = data["X_fit"], data["y_fit"]
    val_ce, best_epoch = _fit(net, opt, loss_fn, lambda b: X[b].astype(np.int64), lambda b: y[b].astype(np.int64),
                              len(y), NAMES_EPOCHS, NAMES_BATCH, rng,
                              lambda n: names_ce(n, data["X_val"], data["y_val"]), log)
    test_ce = names_ce(net, data["X_test"], data["y_test"])
    baseline = names.unigram_baseline_loss(y, data["y_test"], alpha=1.0)
    source = data["source_names"]
    source_set = frozenset(source.tolist())
    raw = names.candidates(net, NAMES_GATE_N, NAMES_GATE_TEMP, NAMES_GATE_SEED)
    accepted: list[str] = []
    seen: set[str] = set()
    for c in raw:
        if names.is_new(c, source_set, seen):
            seen.add(c)
            accepted.append(c)
    gate = {"seed": NAMES_GATE_SEED, "temperature": NAMES_GATE_TEMP, "candidates": len(raw),
            "source_match_rate": sum(c in source_set for c in raw) / len(raw),
            "accepted": len(accepted), "sample": accepted[:12]}
    dataset_id = str(data["dataset_id"])
    manifest = {"head_id": names.HEAD_ID, "transform_version": names.TRANSFORM_VERSION,
                "training_source_id": dataset_id, "source_sha256": str(data["source_sha256"]),
                "prepared_dataset_id": dataset_id, "prepared_cache": "data/prepared/names.npz",
                "source_kind": "real", "vocab": names.VOCAB,
                "context": names.CONTEXT, "seed": seed, "epochs_run": NAMES_EPOCHS, "best_epoch": best_epoch,
                "val_ce": val_ce, "test_ce": test_ce, "unigram_baseline_test_ce": baseline,
                "batch": NAMES_BATCH, "lr": NAMES_LR, "optimizer": "Adam", "gate_sample": gate,
                "names_counts": {k: int(len(data[f"names_{k}"])) for k in ("fit", "val", "test")},
                "source_names": int(source.size)}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    net.save(out, manifest, {"preproc.vocab": np.array(list(names.VOCAB)), "preproc.source_names": source})
    return {"best_epoch": best_epoch, "val_ce": val_ce, "test_ce": test_ce, "baseline": baseline, "gate": gate,
            "dataset_id": dataset_id, "path": out, "rows": len(y), "epochs": NAMES_EPOCHS,
            "metric": ("test CE", test_ce)}


def tabular_probs(net, pre, X):
    return sigmoid(net.forward(tabular.to_input(X, pre), training=False))[:, 0]


def train_tabular(source="uci", seed=31, out=None, log=print):
    data = load_tabular(source)
    rng = np.random.default_rng(seed)
    pre = tabular.fit_preproc(data["X_fit"])  # train split only
    net, loss_fn, opt = tabular.build(rng), BinaryCrossEntropy(), Adam(lr=TABULAR_LR)
    Xf = tabular.to_input(data["X_fit"], pre)
    yf = data["y_fit"].astype(np.float32)[:, None]
    Xv, yv = tabular.to_input(data["X_val"], pre), data["y_val"].astype(np.float32)[:, None]
    val_ll, best_epoch = _fit(net, opt, loss_fn, lambda b: Xf[b], lambda b: yf[b], len(yf), TABULAR_EPOCHS,
                              TABULAR_BATCH, rng, lambda n: loss_fn(n.forward(Xv), yv)[0], log)
    prevalence = float(data["y_fit"].mean())
    test = tabular.metrics(tabular_probs(net, pre, data["X_test"]), data["y_test"], prevalence)
    dataset_id = str(data["dataset_id"])
    kind = "real" if source == "uci" else "synthetic"
    manifest = {"head_id": tabular.HEAD_ID, "transform_version": tabular.TRANSFORM_VERSION,
                "training_source_id": dataset_id, "source_sha256": str(data["source_sha256"]),
                "prepared_dataset_id": dataset_id,
                "prepared_cache": "data/prepared/" + ("tabular.npz" if source == "uci" else "tabular_synthetic.npz"),
                "source_kind": kind, "features": tabular.FEATURES,
                "transform": "log1p(max(x,0)) then (x - mean) / scale; scale = 1 where train std < 1e-6",
                "seed": seed, "epochs_run": TABULAR_EPOCHS, "best_epoch": best_epoch, "val_log_loss": val_ll,
                "batch": TABULAR_BATCH, "lr": TABULAR_LR, "optimizer": "Adam",
                "rows": {k: int(len(data[f"y_{k}"])) for k in ("fit", "val", "test")},
                "train_prevalence": prevalence, "test_metrics": test}
    if source == "synthetic":
        from scripts import distill_data
        manifest["generator"] = {"version": distill_data.GENERATOR_VERSION, "seed": distill_data.SEED,
                                 "rows": distill_data.N, "params": distill_data.DEFAULT_PARAMS}
    out = Path(out or TABULAR_MODEL[source])
    out.parent.mkdir(parents=True, exist_ok=True)
    net.save(out, manifest, {f"preproc.{k}": v for k, v in pre.items()})
    return {"best_epoch": best_epoch, "val_log_loss": val_ll, "test": test, "dataset_id": dataset_id,
            "source_kind": kind, "path": out, "rows": len(yf), "epochs": TABULAR_EPOCHS,
            "metric": ("test log loss", test["log_loss"])}


def report_names(r):
    print(f"checkpoint: epoch {r['best_epoch']} (val CE {r['val_ce']:.4f})")
    print(f"dataset: {r['dataset_id']}")
    ok = r["test_ce"] < r["baseline"]
    print(f"test CE {r['test_ce']:.4f} nats/char vs smoothed unigram baseline {r['baseline']:.4f} "
          f"(gate model < baseline: {'PASS' if ok else 'FAIL'})")
    g = r["gate"]
    print(f"{g['candidates']} candidates (seed {g['seed']}, T {g['temperature']}): source-match rate before "
          f"filtering {g['source_match_rate']:.2f}, accepted {g['accepted']}")
    print("sample: " + ", ".join(g["sample"]))
    print(f"saved: {r['path']}")


def report_tabular(r):
    t = r["test"]
    print(f"checkpoint: epoch {r['best_epoch']} (val log loss {r['val_log_loss']:.4f})")
    print(f"dataset: {r['dataset_id']} (source_kind {r['source_kind']})")
    print(f"test n {t['n']}, positives {t['positives']}, observed prevalence {t['observed_prevalence']:.4f}, "
          f"train prevalence {t['train_prevalence']:.4f}")
    ok = t["log_loss"] < t["baseline_log_loss"]
    verdict = f"gate: {'PASS' if ok else 'FAIL'}" if r["source_kind"] == "real" else (
        f"beats baseline: {'yes' if ok else 'no'}; the gate applies to real data only")
    print(f"log loss {t['log_loss']:.4f} vs prevalence baseline {t['baseline_log_loss']:.4f} ({verdict})")
    print(f"Brier {t['brier']:.4f} vs prevalence baseline {t['baseline_brier']:.4f}")
    print(f"average precision {t['average_precision']:.4f} (baseline {t['baseline_average_precision']:.4f}), "
          f"accuracy@0.5 {t['accuracy_at_0.5']:.4f}")
    print("reliability bins (p range: n, mean p, observed):")
    for b in t["reliability"]:
        if b["n"]:
            print(f"  [{b['lo']:.1f}, {b['hi']:.1f}): {b['n']:5d}  {b['mean_p']:.3f}  {b['observed']:.3f}")
    print(f"saved: {r['path']}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m scripts.train")
    p.add_argument("head", choices=["vision", "names", "tabular"])
    p.add_argument("--seed", type=int, default=None, help="training seed (vision 17, names 23, tabular 31)")
    p.add_argument("--source", choices=["uci", "synthetic"], default="uci", help="tabular only")
    a = p.parse_args(argv)
    seed = DEFAULT_SEED[a.head] if a.seed is None else a.seed
    if a.head == "names":
        return report_names(train_names(seed))
    if a.head == "tabular":
        return report_tabular(train_tabular(a.source, seed))
    r = train_vision(seed)
    print(f"checkpoint: epoch {r['best_epoch']} (val CE {r['val_ce']:.4f})")
    print(f"dataset: {r['dataset_id']}")
    print(f"test accuracy: {r['test_accuracy']:.4f}  (gate >= 0.90: {'PASS' if r['test_accuracy'] >= 0.9 else 'FAIL'})")
    print("confusion matrix (rows true 0-9, cols predicted 0-9):")
    for d, row in enumerate(r["confusion"]):
        print(f"  {d}: " + " ".join(f"{n:3d}" for n in row))
    print(f"saved: {r['path']}")


if __name__ == "__main__":
    main(sys.argv[1:])
