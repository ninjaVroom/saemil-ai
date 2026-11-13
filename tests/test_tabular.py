"""M4 tabular head: UCI schema, stratified split, train-only preprocessing, strict predict, metrics, gates."""
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

import scripts.prepare_data as prep
import scripts.train
from heads import tabular
from saemil import Sequential
from scripts import distill_data

ROOT = Path(__file__).resolve().parent.parent
PREPARED, MODEL = ROOT / "data" / "prepared" / "tabular.npz", ROOT / "models" / "tabular.npz"
SYN_PREPARED, SYN_MODEL = ROOT / "data" / "prepared" / "tabular_synthetic.npz", ROOT / "models" / "tabular_synthetic.npz"
F = tabular.FEATURES
GOOD = {"Administrative": 2, "Administrative_Duration": 64.0, "Informational": 0,
        "Informational_Duration": 0.0, "ProductRelated": 18, "ProductRelated_Duration": 720.5}


def require(*paths):
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.fail(f"missing {missing}; run `python -m scripts.distill_data`, `python -m scripts.prepare_data tabular "
                    f"[--source synthetic]`, then `python -m scripts.train tabular [--source synthetic]`")


def make_zip(path, text, member=tabular.UCI_MEMBER):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(member, text)
    return path


# columns deliberately out of FEATURES order, plus unused UCI columns
HEADER = ("ProductRelated,Month,Administrative_Duration,Administrative,Informational,ProductRelated_Duration,"
          "Informational_Duration,Weekend,Revenue")


def test_read_uci_fixture(tmp_path):
    text = HEADER + "\n18,Feb,64,2,0,720.5,0,FALSE,TRUE\n1,Mar,0,0,3,0,12.25,TRUE,FALSE\n"
    X, y = tabular.read_uci(make_zip(tmp_path / "u.zip", text))
    np.testing.assert_array_equal(X, [[2, 64, 0, 0, 18, 720.5], [0, 0, 3, 12.25, 1, 0]])
    assert y.tolist() == [1, 0] and y.dtype == np.int64


@pytest.mark.parametrize("text, match", [
    ("Administrative,Revenue\n1,TRUE\n", "missing columns"),
    (HEADER + "\n18,Feb,64,2,0,720.5,0,FALSE,yes\n", "TRUE/FALSE"),
    (HEADER + "\n18,Feb,-1,2,0,720.5,0,FALSE,TRUE\n", ">= 0"),
    (HEADER + "\n18,Feb,nan,2,0,720.5,0,FALSE,TRUE\n", "finite"),
    (HEADER + "\n18,Feb,inf,2,0,720.5,0,FALSE,TRUE\n", "finite"),
    (HEADER + "\n18,Feb,abc,2,0,720.5,0,FALSE,TRUE\n", "non-numeric"),
    (HEADER + "\n", "no data rows")])
def test_read_uci_rejects(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        tabular.read_uci(make_zip(tmp_path / "u.zip", text))


def test_read_uci_wrong_member(tmp_path):
    with pytest.raises(ValueError, match="not in archive"):
        tabular.read_uci(make_zip(tmp_path / "u.zip", HEADER + "\n", member="other.csv"))


def test_split_stratified_disjoint_reproducible():
    y = (np.random.default_rng(0).random(2000) < 0.155).astype(np.int64)
    X = np.zeros((2000, 6))
    a, b = tabular.split(X, y, 31), tabular.split(X, y, 31)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    allidx = np.concatenate(list(a.values()))
    assert np.array_equal(np.sort(allidx), np.arange(2000))  # disjoint and complete
    for k, frac in (("fit", 0.7), ("val", 0.15), ("test", 0.15)):
        assert abs(len(a[k]) - frac * 2000) <= 2 and abs(y[a[k]].mean() - y.mean()) < 0.01


def test_preproc_zero_variance_log1p_and_percentiles():
    X = np.array([[0, 5, 1, 0, 3, 9], [2, 5, 1, 0, 7, 1], [4, 5, 1, 0, 5, 5]], float)
    pre = tabular.fit_preproc(X)
    L = np.log1p(X)
    np.testing.assert_allclose(pre["means"], L.mean(0))
    assert (pre["scales"][[1, 2, 3]] == 1.0).all()  # zero-variance columns -> scale 1
    np.testing.assert_allclose(pre["scales"][[0, 4, 5]], L.std(0)[[0, 4, 5]])
    np.testing.assert_array_equal(pre["medians"], [2, 5, 1, 0, 5, 5])  # raw units
    assert pre["percentiles"].shape == (9, 6) and pre["percentiles"][0].tolist() == X.min(0).tolist()
    assert pre["features"].tolist() == F
    x = tabular.to_input(X, pre)
    assert x.dtype == np.float32 and x.shape == (3, 6) and np.isfinite(x).all()
    neg = X[0].copy()
    neg[0] = -5  # negative anomaly is clamped to 0 before log1p
    np.testing.assert_array_equal(tabular.to_input(neg, pre), tabular.to_input(X[0], pre))


def test_training_standardizes_on_fit_split_only(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    tiny = {"dataset_id": np.array("tiny"), "source_sha256": np.array("0" * 64)}
    for k, n, scale in (("fit", 256, 1.0), ("val", 40, 1000.0), ("test", 40, 1000.0)):
        tiny[f"X_{k}"] = rng.random((n, 6)) * 50 * scale  # val/test on a wildly different scale
        tiny[f"y_{k}"] = (rng.random(n) < 0.3).astype(np.int64)
    monkeypatch.setattr(scripts.train, "load_tabular", lambda source: tiny)
    r = scripts.train.train_tabular("uci", 31, out=tmp_path / "t.npz", log=lambda *_: None)
    _, manifest, extra = Sequential.load(r["path"], tabular.HEAD_ID)
    pre = tabular.fit_preproc(tiny["X_fit"])
    for k in ("means", "scales", "percentiles", "medians"):
        np.testing.assert_array_equal(extra[f"preproc.{k}"], pre[k])
    assert manifest["train_prevalence"] == pytest.approx(tiny["y_fit"].mean())


def test_synthetic_csv_reproducible():
    a, b = distill_data.generate(), distill_data.generate()
    assert a == b and distill_data.digest(a) == distill_data.SYNTHETIC_SHA256
    assert distill_data.generate(rng_seed=42, n=50) != distill_data.generate(n=50)
    X, y = tabular.read_csv(io.StringIO(a, newline=""))  # same schema as UCI
    assert X.shape == (distill_data.N, 6) and 0.05 < y.mean() < 0.4
    assert (X[:, [0, 2, 4]] == np.rint(X[:, [0, 2, 4]])).all()
    assert ((X[:, [1, 3, 5]] > 0) == (X[:, [0, 2, 4]] > 0)).all()  # duration > 0 iff count > 0


def test_distill_teacher_not_implemented():
    with pytest.raises(SystemExit, match="not implemented"):
        distill_data.main(["--teacher", "cli"])


@pytest.fixture
def net_pre():
    X = np.random.default_rng(2).random((50, 6)) * 100
    return tabular.build(np.random.default_rng(0)), tabular.fit_preproc(X)


def test_predict_ok(net_pre):
    net, pre = net_pre
    r = tabular.predict(net, pre, GOOD, "real")
    assert set(r) == {"probability", "source_kind"} and 0 < r["probability"] < 1 and r["source_kind"] == "real"
    np_typed = {k: (np.int64(v) if isinstance(v, int) else np.float32(v)) for k, v in GOOD.items()}
    assert tabular.predict(net, pre, np_typed, "synthetic")["probability"] == pytest.approx(r["probability"])
    with pytest.raises(ValueError):
        tabular.predict(net, pre, GOOD, "made-up")


@pytest.mark.parametrize("change", [
    {"drop": "Informational"}, {"Extra": 1}, {"Administrative": True}, {"Administrative_Duration": "64"},
    {"Administrative": None}, {"ProductRelated_Duration": float("nan")}, {"Informational_Duration": float("inf")},
    {"ProductRelated": -1}, {"Administrative_Duration": -0.5}, {"ProductRelated": 2.5}, {"Informational": [1]},
    {"Administrative": np.bool_(True)}, {"Administrative": 10 ** 400}, {"ProductRelated_Duration": 10 ** 400}])
def test_predict_rejects(net_pre, change):
    net, pre = net_pre
    bad = dict(GOOD)
    if "drop" in change:
        del bad[change["drop"]]
    else:
        bad.update(change)
    with pytest.raises(ValueError):
        tabular.predict(net, pre, bad, "real")


def test_predict_rejects_non_dict(net_pre):
    with pytest.raises(ValueError):
        tabular.predict(*net_pre, list(GOOD.values()), "real")


def test_average_precision_groups_ties():
    assert tabular.average_precision([0.9, 0.5, 0.5, 0.1], [1, 0, 1, 0]) == pytest.approx(0.5 + 0.5 * 2 / 3)
    assert tabular.average_precision([0.5, 0.5], [0, 1]) == pytest.approx(0.5)  # order-independent
    assert tabular.average_precision([0.5, 0.5], [1, 0]) == pytest.approx(0.5)
    assert tabular.average_precision([0.9, 0.1, 0.2], [1, 0, 0]) == pytest.approx(1.0)


def test_metrics_baseline_and_bins():
    y = np.array([1, 0, 0, 0])
    m = tabular.metrics(np.full(4, 0.25), y, 0.25)
    assert m["log_loss"] == pytest.approx(m["baseline_log_loss"]) and m["brier"] == pytest.approx(m["baseline_brier"])
    assert m["brier"] == pytest.approx((0.75 ** 2 + 3 * 0.25 ** 2) / 4)
    m = tabular.metrics(np.array([0.95, 0.05, 1.0, 0.0]), y, 0.25)
    bins = m["reliability"]
    assert len(bins) == 10 and sum(b["n"] for b in bins) == 4
    assert bins[9]["n"] == 2 and bins[9]["observed"] == 0.5 and bins[0]["n"] == 2 and bins[0]["observed"] == 0.0


def check_artifact(model, prepared, source, kind):
    require(model, prepared)
    net, manifest, extra = Sequential.load(model, tabular.HEAD_ID)
    assert sum(p.size for p in net.named_params().values()) == 769
    assert manifest["transform_version"] == tabular.TRANSFORM_VERSION and manifest["source_kind"] == kind
    assert manifest["training_source_id"] == prep.tabular_dataset_id(source) and manifest["features"] == F
    assert manifest["prepared_dataset_id"] == manifest["training_source_id"]
    assert manifest["source_sha256"] == (prep.UCI_SHA256 if source == "uci" else distill_data.SYNTHETIC_SHA256)
    pre = tabular.preproc_from_extra(extra)
    data = prep.load_tabular(source)
    ref = tabular.fit_preproc(data["X_fit"])
    for k in ("means", "scales", "percentile_levels", "percentiles", "medians"):
        np.testing.assert_allclose(pre[k], ref[k])
    p = scripts.train.tabular_probs(net, pre, data["X_test"])
    m = tabular.metrics(p, data["y_test"], float(data["y_fit"].mean()))
    assert m["log_loss"] == pytest.approx(manifest["test_metrics"]["log_loss"], rel=1e-6)
    row = dict(zip(F, data["X_test"][0].tolist(), strict=True))
    assert tabular.predict(net, pre, row, kind)["probability"] == pytest.approx(float(p[0]), rel=1e-6)
    return m, manifest


def test_frozen_uci_model_meets_gate():
    m, _ = check_artifact(MODEL, PREPARED, "uci", "real")
    assert m["n"] == 1850 and m["log_loss"] < m["baseline_log_loss"] and m["brier"] < m["baseline_brier"]


def test_frozen_synthetic_model_has_scorecard():
    m, manifest = check_artifact(SYN_MODEL, SYN_PREPARED, "synthetic", "synthetic")
    assert manifest["generator"]["seed"] == distill_data.SEED and m["n"] == 900
    assert manifest["training_source_id"] != prep.tabular_dataset_id("uci")
