"""M4 names head: cleaning, whole-name split, windows, bounded novel sampling, frozen-model gate."""
import urllib.request
from pathlib import Path

import numpy as np
import pytest

import scripts.prepare_data as prep
from heads import names
from saemil import Sequential

ROOT = Path(__file__).resolve().parent.parent
PREPARED, MODEL = ROOT / "data" / "prepared" / "names.npz", ROOT / "models" / "names.npz"


def require(*paths):
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.fail(f"missing {missing}; run `python -m scripts.prepare_data names` then `python -m scripts.train names`")


def rigged(next_id):
    """Untrained net whose output always strongly prefers `next_id`."""
    net = names.build(np.random.default_rng(0))
    b = net.layers[-1].b
    b[...] = -50
    b[next_id] = 50
    return net


def test_nova_windows():
    X, y = names.windows(["nova"])
    assert X.dtype == np.uint8 and y.dtype == np.uint8
    as_text = ["".join(names.VOCAB[i] for i in row) + "->" + names.VOCAB[t] for row, t in zip(X, y, strict=True)]
    assert as_text == ["...->n", "..n->o", ".no->v", "nov->a", "ova->."]


def test_clean():
    lines = ["Emma\n", "  olivia ", "emma", "a", "x" * 17, "x" * 16, "mary-jane", "zoë", "", "Ava\r\n"]
    assert names.clean(lines) == ["emma", "olivia", "x" * 16, "ava"]


def test_split_by_whole_name_disjoint_reproducible():
    pool = [f"n{chr(97 + i % 26)}{chr(97 + i // 26 % 26)}{chr(97 + i // 676)}" for i in range(5000)]
    assert len(set(pool)) == 5000
    a, b = names.split(pool, 23, 4000), names.split(pool, 23, 4000)
    assert a == b and a != names.split(pool, 24, 4000)
    assert [len(a[k]) for k in ("fit", "val", "test")] == [3200, 400, 400]
    sets = [set(a[k]) for k in ("fit", "val", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert [len(v) for v in names.split(pool[:50], 23, 4000).values()] == [40, 5, 5]


def test_unigram_baseline_hand_computed():
    # counts over 27 IDs with alpha 1: id0 -> 2+1, id1 -> 1+1, others 1; total 3 + 27 = 30
    loss = names.unigram_baseline_loss([0, 0, 1], [0, 1], alpha=1.0)
    assert loss == pytest.approx(-(np.log(3 / 30) + np.log(2 / 30)) / 2)


def test_is_new_rule():
    src = {"emma", "ava"}
    assert names.is_new("novra", src, [])
    assert not names.is_new("emma", src, [])          # in the source corpus
    assert not names.is_new("novra", src, ["novra"])  # duplicate in this request
    assert not names.is_new("ab", src, []) and not names.is_new("a" * 13, src, [])
    assert names.is_new("abc", src, []) and names.is_new("a" * 12, src, [])


def test_sample_reproducible_and_request_local():
    net = names.build(np.random.default_rng(1))
    np.random.seed(0)
    before = np.random.random()
    np.random.seed(0)
    a = names.sample(net, 5, frozenset(), 1.0, seed=7)
    assert np.random.random() == before  # global RNG untouched
    assert a == names.sample(net, 5, frozenset(), 1.0, seed=7)
    assert a["names"] != names.sample(net, 5, frozenset(), 1.0, seed=8)["names"]
    assert a["requested"] == 5 and a["accepted"] == len(a["names"]) <= 5 and a["attempts"] <= 100
    assert len(set(a["names"])) == len(a["names"])


def test_sample_novelty_filter():
    net = names.build(np.random.default_rng(1))
    first = names.sample(net, 10, frozenset(), 1.0, seed=3)["names"]
    again = names.sample(net, 10, frozenset(first), 1.0, seed=3)["names"]
    assert first and not set(first) & set(again)


def test_sampling_is_bounded():
    r = names.sample(rigged(0), 5, frozenset(), 0.2, seed=1)  # always stops immediately: empty candidates
    assert r == {"names": [], "accepted": 0, "requested": 5, "attempts": 100}
    r = names.sample(rigged(1), 3, frozenset(), 0.2, seed=1)  # never stops: 12-letter cap, then only duplicates
    assert r == {"names": ["a" * 12], "accepted": 1, "requested": 3, "attempts": 100}


def test_source_set_is_required():
    with pytest.raises(TypeError):
        names.sample(names.build(np.random.default_rng(0)), 3)


@pytest.mark.parametrize("kw", [
    {"count": 0}, {"count": 11}, {"count": True}, {"count": 2.0}, {"count": "3"},
    {"temperature": 0}, {"temperature": 0.19}, {"temperature": 1.51}, {"temperature": -1},
    {"temperature": float("nan")}, {"temperature": float("inf")}, {"temperature": True}, {"temperature": "1"},
    {"seed": 1.5}, {"seed": "1"}, {"seed": True}])
def test_sample_validation(kw):
    args = {"count": 3, "source_set": frozenset(), "temperature": 0.85, "seed": 1} | kw
    with pytest.raises(ValueError):
        names.sample(names.build(np.random.default_rng(0)), **args)


def test_fetch_network_error_leaves_no_part(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "RAW", tmp_path)

    def boom(url, timeout):
        raise OSError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(OSError):
        prep.fetch(prep.NAMES_URL, "names.txt", prep.NAMES_SHA256)
    assert not any(tmp_path.iterdir())


def test_prepared_cache():
    require(PREPARED)
    data = prep.load_names()  # checks dataset_id and split seed
    assert str(data["source_sha256"]) == prep.NAMES_SHA256 and str(data["vocab"]) == names.VOCAB
    parts = {k: set(data[f"names_{k}"].tolist()) for k in ("fit", "val", "test")}
    assert [len(v) for v in parts.values()] == [3200, 400, 400]
    assert not (parts["fit"] & parts["val"] or parts["fit"] & parts["test"] or parts["val"] & parts["test"])
    source = set(data["source_names"].tolist())
    assert set().union(*parts.values()) <= source
    for k in parts:
        X, y = names.windows(data[f"names_{k}"].tolist())
        np.testing.assert_array_equal(X, data[f"X_{k}"])
        np.testing.assert_array_equal(y, data[f"y_{k}"])


def test_frozen_model_meets_gate():
    require(MODEL, PREPARED)
    net, manifest, extra = Sequential.load(MODEL, names.HEAD_ID)
    assert sum(p.size for p in net.named_params().values()) == 3571
    assert manifest["transform_version"] == names.TRANSFORM_VERSION
    assert manifest["training_source_id"] == prep.NAMES_DATASET_ID and manifest["vocab"] == names.VOCAB
    assert manifest["source_sha256"] == prep.NAMES_SHA256 and manifest["prepared_dataset_id"] == prep.NAMES_DATASET_ID
    assert "".join(extra["preproc.vocab"].tolist()) == names.VOCAB
    data = prep.load_names()
    np.testing.assert_array_equal(extra["preproc.source_names"], data["source_names"])
    logits = net.forward(data["X_test"].astype(np.int64))
    from saemil import SoftmaxCrossEntropy
    test_ce, _ = SoftmaxCrossEntropy()(logits, data["y_test"].astype(np.int64))
    baseline = names.unigram_baseline_loss(data["y_fit"], data["y_test"])
    assert test_ce < baseline
    assert test_ce == pytest.approx(manifest["test_ce"], rel=1e-5)
    source = frozenset(extra["preproc.source_names"].tolist())
    r = names.sample(net, 10, source, 0.85, seed=23)
    assert r["accepted"] >= 1 and not set(r["names"]) & source
    assert all(3 <= len(n) <= 12 and set(n) <= set(names.VOCAB[1:]) for n in r["names"])
