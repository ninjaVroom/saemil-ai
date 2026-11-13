"""M3 gate: vision head splits, train==serve transform, centering, blank handling, frozen model accuracy."""
import io
import urllib.request
from pathlib import Path

import numpy as np
import pytest

import scripts.prepare_data as prep
import scripts.train
from heads import vision
from saemil import Sequential, softmax

ROOT = Path(__file__).resolve().parent.parent
PREPARED, MODEL = ROOT / "data" / "prepared" / "vision.npz", ROOT / "models" / "vision.npz"
BLANK = {"blank": True, "message": "Please draw a digit in the canvas first."}


def require(*paths):
    """Gate artifacts are mandatory: fail loudly instead of skipping."""
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.fail(f"missing {missing}; run `python -m scripts.prepare_data vision` "
                    f"then `python -m scripts.train vision --seed 17`")


def fake_labels(n, seed):
    return np.random.default_rng(seed).integers(0, 10, n)


def test_split_disjoint_stratified_reproducible():
    y_train, y_test = fake_labels(12000, 1), fake_labels(2000, 2)
    a, b = vision.split(y_train, y_test, 17), vision.split(y_train, y_test, 17)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.array_equal(a["fit"], vision.split(y_train, y_test, 18)["fit"])
    assert not set(a["fit"]) & set(a["val"])
    for k, src in (("fit", y_train), ("val", y_train), ("test", y_test)):
        assert len(set(a[k])) == len(a[k])
        assert (np.bincount(src[a[k]], minlength=10) == vision.PER_DIGIT[k]).all()


def test_split_rejects_short_digit():
    with pytest.raises(ValueError):
        vision.split(np.zeros(1000, int), np.zeros(100, int))


def test_prepared_cache():
    require(PREPARED)
    data = prep.load_vision()  # also checks dataset_id and split seed
    assert str(data["source_sha256"]) == prep.MNIST_SHA256
    assert not set(data["idx_fit"]) & set(data["idx_val"])
    for k, n in vision.PER_DIGIT.items():
        assert data[f"x_{k}"].dtype == np.uint8 and data[f"x_{k}"].shape == (10 * n, 28, 28)
        assert (np.bincount(data[f"y_{k}"], minlength=10) == n).all()


def test_hash_mismatch_deletes_download(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "RAW", tmp_path)
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: io.BytesIO(b"not mnist"))
    with pytest.raises(SystemExit) as e:
        prep.fetch_mnist()
    assert e.value.code not in (0, None) and "mismatch" in str(e.value.code)
    assert not any(tmp_path.iterdir())  # neither mnist.npz nor the .part file is left behind


def test_to_input_is_the_training_transform():
    assert scripts.train.vision.to_input is vision.to_input  # one function for train and serve
    px = np.random.default_rng(0).integers(0, 256, (5, 28, 28), dtype=np.uint8)
    x = vision.to_input(px)
    assert x.dtype == np.float32 and x.shape == (5, 784) and x.flags.c_contiguous
    np.testing.assert_array_equal(x, px.reshape(5, 784).astype(np.float32) / np.float32(255))
    np.testing.assert_array_equal(vision.to_input(px[0].ravel()), x[:1])
    with pytest.raises(TypeError):
        vision.to_input(x)  # already scaled floats must not be scaled twice


def test_best_checkpoint_is_restored(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    tiny = {"dataset_id": np.array("tiny"), "pixel_convention": np.array("test")}
    for k, n in (("fit", 64), ("val", 10), ("test", 10)):
        tiny[f"x_{k}"] = rng.integers(0, 256, (n, 28, 28), dtype=np.uint8)
        tiny[f"y_{k}"] = rng.integers(0, 10, n)
    monkeypatch.setattr(scripts.train, "load_vision", lambda: tiny)
    real, snaps, scripted = scripts.train.evaluate, [], iter([0.5, 0.2, 0.9, 0.3, 0.8, 0.7])

    def fake_evaluate(net, x, y):
        loss, acc, conf = real(net, x, y)
        if len(snaps) < scripts.train.EPOCHS:  # the per-epoch validation calls
            snaps.append({k: v.copy() for k, v in net.named_params().items()})
            return next(scripted), acc, conf
        return loss, acc, conf

    monkeypatch.setattr(scripts.train, "evaluate", fake_evaluate)
    r = scripts.train.train_vision(17, out=tmp_path / "v.npz", log=lambda *_: None)
    assert r["best_epoch"] == 2 and r["val_ce"] == 0.2
    net, manifest, _ = Sequential.load(tmp_path / "v.npz", vision.HEAD_ID)
    assert manifest["best_epoch"] == 2
    for k, v in net.named_params().items():
        np.testing.assert_array_equal(v, snaps[1][k])
    assert not np.array_equal(snaps[1]["0.W"], snaps[-1]["0.W"])  # the last epoch really differs


def test_center_by_mass_frozen_rectangle():
    img = np.zeros((28, 28), np.uint8)
    img[1:11, 0:5] = 255  # 10x5 block in the top-left corner
    out = vision.center_by_mass(img.ravel()).reshape(28, 28)
    expected = np.zeros((28, 28), np.uint8)
    expected[5:25, 10:20] = 255  # longer side -> 20 px, aspect kept, mass centre (14.5, 14.5) rounded half up
    np.testing.assert_array_equal(out, expected)


def test_center_by_mass_off_center_stroke():
    img = np.zeros((28, 28), np.uint8)
    for t in range(12):  # thin diagonal "7"-ish stroke near the bottom-right edge
        img[14 + t, 26 - t // 2] = 255
    img[14, 18:27] = 200
    out = vision.center_by_mass(img).reshape(28, 28).astype(float)
    ys, xs = np.nonzero(out)
    assert max(np.ptp(ys), np.ptp(xs)) + 1 == 20
    r = np.arange(28)
    assert abs(out.sum(1) @ r / out.sum() - 14) <= 0.5 and abs(out.sum(0) @ r / out.sum() - 14) <= 0.5
    big = np.kron(img, np.ones((10, 10), np.uint8))  # same drawing on a 280x280 canvas
    big_out = vision.center_by_mass(big).astype(int)
    assert np.abs(big_out - out.ravel()).max() <= 60 and np.abs(big_out - out.ravel()).mean() < 3


def test_center_by_mass_empty():
    assert not vision.center_by_mass(np.zeros(784)).any()


@pytest.fixture
def net():
    return vision.build(np.random.default_rng(0))


@pytest.mark.parametrize("bad", [
    ["a"] * 784, np.full(784, "1"), np.ones(784, bool), np.ones(784, complex), np.ones(784, object),
    np.ones((2, 392)), np.ones((784, 1)), np.zeros(783), np.full(784, np.nan), np.full(784, np.inf),
    np.full(784, 256), np.full(784, -1), np.full(784, 0.5)])
def test_invalid_pixels_rejected(net, bad):
    with pytest.raises(ValueError):
        vision.predict(net, bad)


def test_blank_rules(net):
    line = np.zeros((28, 28), int)
    line[4:24, 14] = 5  # 20 px tall, ink sum exactly 100, survives centering unchanged
    assert vision.predict(net, line.ravel())["blank"] is False
    line[4, 14] = 4  # ink sum 99
    assert vision.predict(net, line.ravel()) == BLANK
    assert vision.predict(net, np.zeros(784, int)) == BLANK
    dot = np.zeros(784, int)
    dot[400] = 255  # plenty of ink but bbox longer side 1 px
    assert vision.predict(net, dot) == BLANK
    faint = np.zeros((28, 28), int)
    faint[::2, ::3] = 1  # scattered ink: raw sum 140, centered sum < 100
    assert faint.sum() >= 100 and vision.center_by_mass(faint).sum() < 100
    assert vision.predict(net, faint) == BLANK


def test_predict_top3(net):
    assert sum(p.size for p in net.named_params().values()) == 101_770
    px = np.zeros((28, 28), int)
    px[4:24, 12:16] = 255
    r = vision.predict(net, px.ravel().tolist())
    assert r["blank"] is False and len(r["top3"]) == 3
    probs = [t["prob"] for t in r["top3"]]
    assert r["digit"] == r["top3"][0]["digit"] and len({t["digit"] for t in r["top3"]}) == 3
    assert probs == sorted(probs, reverse=True) and 0 < sum(probs) <= 1 + 1e-6
    assert vision.predict(net, px)["digit"] == r["digit"]  # (28, 28) accepted too


def test_frozen_model_meets_gate():
    require(MODEL, PREPARED)
    net, manifest, _ = Sequential.load(MODEL, vision.HEAD_ID)
    assert manifest["transform_version"] == vision.TRANSFORM_VERSION
    assert manifest["training_source_id"] == prep.DATASET_ID
    data = prep.load_vision()
    x, y = data["x_test"], data["y_test"]
    logits = net.forward(vision.to_input(x))
    assert (logits.argmax(1) == y).mean() >= 0.90
    # serve path (validate -> blank rules -> center_by_mass -> to_input) agrees with the train path on MNIST
    for i in range(0, 1000, 50):
        r = vision.predict(net, x[i].ravel())
        assert r["digit"] == logits[i].argmax()
        assert r["top3"][0]["prob"] == pytest.approx(float(softmax(logits[i:i + 1])[0].max()), rel=1e-5)
