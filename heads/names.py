"""Names head: character-level MLP over a 3-character context (makemore style).

Vocab: 0 = boundary ".", 1-26 = a-z. Novelty is checked against the full cleaned source corpus
and within the request. Novelty is not trademark clearance.
"""
import math
import re

import numpy as np

from saemil import Embedding, Flatten, Linear, Sequential, Tanh, softmax

HEAD_ID = "names"
TRANSFORM_VERSION = 1
VOCAB = ".abcdefghijklmnopqrstuvwxyz"
CONTEXT = 3
STOI = {c: i for i, c in enumerate(VOCAB)}
MAX_LETTERS, MIN_ACCEPT = 12, 3
MAX_ATTEMPTS = 100
COUNT_RANGE, TEMP_RANGE = (1, 10), (0.2, 1.5)
_NAME = re.compile(r"[a-z]{2,16}")


def clean(lines):
    """Lowercase, strip, keep full matches of [a-z]{2,16}, dedupe keeping first occurrence."""
    seen: dict[str, None] = {}  # insertion-ordered set
    for line in lines:
        s = line.strip().lower()
        if _NAME.fullmatch(s):
            seen.setdefault(s, None)
    return list(seen)


def split(names, seed=23, limit=4000):
    """Shuffle (seed), keep <= limit, split 80/10/10 by whole name. Windows are built after this."""
    order = np.random.default_rng(seed).permutation(len(names))[:limit]
    picked = [names[i] for i in order]
    n_fit, n_val = int(0.8 * len(picked)), int(0.1 * len(picked))
    return {"fit": picked[:n_fit], "val": picked[n_fit:n_fit + n_val], "test": picked[n_fit + n_val:]}


def windows(names):
    """Every (3-ID context -> next ID) pair, boundary-padded on both sides. uint8 X (N,3), y (N,)."""
    xs, ys = [], []
    for name in names:
        ctx = [0] * CONTEXT
        for ch in name + ".":
            xs.append(ctx)
            ys.append(STOI[ch])
            ctx = ctx[1:] + [STOI[ch]]
    return np.array(xs, np.uint8).reshape(-1, CONTEXT), np.array(ys, np.uint8)


def build(rng):
    return Sequential([Embedding(27, 8, rng), Flatten(), Linear(CONTEXT * 8, 64, "lecun", rng), Tanh(),
                       Linear(64, 27, "lecun", rng)])


def unigram_baseline_loss(train_y, eval_y, alpha=1.0):
    """Mean CE (nats/char) of an add-alpha unigram over all 27 IDs (boundary included) fitted on train targets."""
    counts = np.bincount(np.asarray(train_y, np.int64), minlength=len(VOCAB)) + alpha
    p = counts / counts.sum()
    return float(-np.log(p[np.asarray(eval_y, np.int64)]).mean())


def _real(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_))


def is_new(candidate, source_set, taken):
    """Acceptance rule: 3-12 letters, not in the source corpus, not already taken in this request."""
    return MIN_ACCEPT <= len(candidate) <= MAX_LETTERS and candidate not in source_set and candidate not in taken


def _draw(net, rng, temperature):
    """One raw candidate: stop at boundary or after MAX_LETTERS letters."""
    ctx = [0] * CONTEXT
    out: list[str] = []
    while len(out) < MAX_LETTERS:
        logits = net.forward(np.array([ctx], np.int64), training=False)[0].astype(np.float64)
        i = int(rng.choice(len(VOCAB), p=softmax(logits / temperature)))
        if i == 0:
            break
        out.append(VOCAB[i])
        ctx = ctx[1:] + [i]
    return "".join(out)


def candidates(net, n, temperature=0.85, seed=None):
    """n raw (unfiltered) candidates from one seeded RNG; used for the gate report."""
    rng = np.random.default_rng(seed)
    return [_draw(net, rng, temperature) for _ in range(n)]


def sample(net, count, source_set, temperature=0.85, seed=None):
    """Up to `count` new names in <= MAX_ATTEMPTS draws from a request-local RNG.

    source_set is required (the artifact's preproc.source_names) so novelty is never skipped by accident.
    Accept length 3-12, not in source_set, not repeated in this request. Fewer than requested is normal.
    """
    if not (type(count) is int or isinstance(count, np.integer)) or not COUNT_RANGE[0] <= count <= COUNT_RANGE[1]:
        raise ValueError(f"count must be an int in {COUNT_RANGE}, got {count!r}")
    if not _real(temperature) or not math.isfinite(temperature) or not TEMP_RANGE[0] <= temperature <= TEMP_RANGE[1]:
        raise ValueError(f"temperature must be a number in {TEMP_RANGE}, got {temperature!r}")
    if seed is not None and not (type(seed) is int or isinstance(seed, np.integer)):
        raise ValueError(f"seed must be an int or None, got {seed!r}")
    rng = np.random.default_rng(seed)
    names: list[str] = []
    attempts = 0
    while len(names) < count and attempts < MAX_ATTEMPTS:
        attempts += 1
        c = _draw(net, rng, float(temperature))
        if is_new(c, source_set, names):
            names.append(c)
    return {"names": names, "accepted": len(names), "requested": int(count), "attempts": attempts}
