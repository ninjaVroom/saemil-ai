"""`python -m scripts.distill_data` - deterministic synthetic sessions in the UCI 468 schema (fallback source).

Rows are generated locally with NumPy. The optional LLM "teacher" that may propose parameters is milestone M5
and is not implemented here.
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

from heads.tabular import FEATURES, TARGET

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "synthetic" / "online_shoppers_synthetic.csv"
GENERATOR_VERSION = 1
SEED, N = 41, 6000
DEFAULT_PARAMS = {"u_sigma": 0.6, "rate_A": 2.0, "rate_I": 0.6, "rate_P": 12.0, "dur_scale_s": 25.0,
                  "bias": -3.0, "w_logP": 0.45, "w_logA": 0.15, "w_anyI": 0.25}
# SHA-256 of generate() with the defaults above (see data/SOURCES.md); a change means the generator drifted
SYNTHETIC_SHA256 = "b299f270c6eef97a9ce4f38a3215f93455f496f9290d36335c3d38e1133f52a6"


def generate(rng_seed=SEED, n=N, params=DEFAULT_PARAMS):
    """CSV text (LF line ends, header = 6 features + Revenue). Counts as ints, durations with 4 decimals."""
    q = params
    rng = np.random.default_rng(rng_seed)
    u = rng.lognormal(0.0, q["u_sigma"], n)
    A, I, P = (rng.poisson(q[k] * u) for k in ("rate_A", "rate_I", "rate_P"))  # noqa: E741 - dataset feature initials
    dA, dI, dP = (np.where(c > 0, rng.gamma(np.maximum(c, 1), q["dur_scale_s"]), 0.0) for c in (A, I, P))
    s = q["bias"] + q["w_logP"] * np.log1p(P) + q["w_logA"] * np.log1p(A) + q["w_anyI"] * (I > 0)
    revenue = rng.random(n) < 1.0 / (1.0 + np.exp(-s))
    lines = [",".join(FEATURES + [TARGET])]
    for row in zip(A, dA, I, dI, P, dP, revenue, strict=True):
        a, da, i, di, p, dp, r = row
        lines.append(f"{a},{da:.4f},{i},{di:.4f},{p},{dp:.4f},{'TRUE' if r else 'FALSE'}")
    return "\n".join(lines) + "\n"


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m scripts.distill_data")
    ap.add_argument("--teacher", choices=["none", "cli", "cloud"], default="none")
    a = ap.parse_args(argv)
    if a.teacher != "none":
        raise SystemExit("--teacher is not implemented yet (milestone M5 bridge); use the default 'none'")
    text = generate()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".part")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.replace(tmp, OUT)
    pos = sum(line.endswith(",TRUE") for line in text.splitlines())
    print(f"generator v{GENERATOR_VERSION}, seed {SEED}, rows {N}, positives {pos} "
          f"(prevalence {pos / N:.4f}), params {DEFAULT_PARAMS}")
    print(f"sha256 {digest(text)} -> {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:])
