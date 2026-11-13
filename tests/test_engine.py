"""M1 gate: forward pass, init, caches, dtype, save/load and manifest validation."""
import json
import os

import numpy as np
import pytest

from saemil import (
    Adam,
    BinaryCrossEntropy,
    Embedding,
    Flatten,
    Linear,
    ReLU,
    Sequential,
    Sigmoid,
    SoftmaxCrossEntropy,
    Tanh,
)


def names_net(seed, dtype=np.float32):
    rng = np.random.default_rng(seed)
    return Sequential([Embedding(27, 8, rng, dtype), Flatten(), Linear(24, 64, "lecun", rng, dtype), Tanh(),
                       Linear(64, 27, "lecun", rng, dtype)])


def small_net(seed=0, dtype=np.float32):
    rng = np.random.default_rng(seed)
    return Sequential([Linear(3, 4, "he", rng, dtype), ReLU(), Linear(4, 2, "lecun", rng, dtype)])


def test_dense_hand_computed():
    layer = Linear(2, 3, "he", np.random.default_rng(0), np.float64)
    layer.W[...] = [[1, 0, -1], [2, 1, 0]]
    layer.b[...] = [0.5, -1, 2]
    y = layer.forward(np.array([[1.0, 2.0], [0.0, -1.0]]), training=False)
    np.testing.assert_array_equal(y, [[5.5, 1.0, 1.0], [-1.5, -2.0, 2.0]])


def test_embedding_gathers_duplicate_ids():
    emb = Embedding(5, 2, np.random.default_rng(0), np.float64)
    emb.E[...] = np.arange(10.0).reshape(5, 2)
    y = emb.forward(np.array([[1, 1, 3]]), training=False)
    np.testing.assert_array_equal(y, [[[2, 3], [2, 3], [6, 7]]])


def test_embedding_rejects_bad_ids():
    emb = Embedding(5, 2, np.random.default_rng(0))
    with pytest.raises(ValueError):
        emb.forward(np.array([[5]]), training=False)
    with pytest.raises(ValueError):
        emb.forward(np.array([[-1]]), training=False)
    with pytest.raises(TypeError):
        emb.forward(np.array([[1.0]]), training=False)


def test_names_shape_walk():
    net = names_net(1)
    x = np.random.default_rng(2).integers(0, 27, size=(64, 3))
    shapes = []
    for layer in net.layers:
        x = layer.forward(x, training=False)
        shapes.append(x.shape)
    assert shapes == [(64, 3, 8), (64, 24), (64, 64), (64, 64), (64, 27)]


def test_init_reproducible_and_scaled():
    a, b, c = names_net(5), names_net(5), names_net(6)
    for k, v in a.named_params().items():
        np.testing.assert_array_equal(v, b.named_params()[k])
    assert not np.array_equal(a.named_params()["0.E"], c.named_params()["0.E"])
    rng = np.random.default_rng(0)
    he = Linear(784, 128, "he", rng, np.float64)
    lecun = Linear(784, 128, "lecun", rng, np.float64)
    emb = Embedding(1000, 50, rng, np.float64)
    assert abs(he.W.std() / np.sqrt(2 / 784) - 1) < 0.02
    assert abs(lecun.W.std() / np.sqrt(1 / 784) - 1) < 0.02
    assert abs(emb.E.std() / 0.1 - 1) < 0.02
    assert not he.b.any()
    with pytest.raises(ValueError):
        Linear(2, 2, "xavier", rng)


def test_inference_leaves_caches_untouched():
    net = Sequential([Embedding(6, 2, np.random.default_rng(0)), Flatten(),
                      Linear(4, 3, "he", np.random.default_rng(1)), ReLU(), Linear(3, 3, "lecun", np.random.default_rng(2)),
                      Sigmoid(), Linear(3, 2, "lecun", np.random.default_rng(3)), Tanh()])
    net.forward(np.array([[1, 2]]), training=False)
    assert all(layer.cache is None for layer in net.layers)
    net.forward(np.array([[1, 2], [3, 4]]), training=True)
    before = [layer.cache for layer in net.layers]
    net.forward(np.array([[5, 0]]), training=False)
    assert all(layer.cache is c for layer, c in zip(net.layers, before, strict=True))


def test_float32_preserved_through_forward_and_training():
    net = small_net(dtype=np.float32)
    x = np.random.default_rng(0).standard_normal((5, 3))  # float64 input
    assert net.forward(x).dtype == np.float32
    opt = Adam(0.01)
    net.train_step(x, np.array([0, 1, 0, 1, 1]), SoftmaxCrossEntropy(), opt)
    assert all(v.dtype == np.float32 for v in net.named_params().values())
    assert all(v.dtype == np.float32 for v in net.named_grads().values())
    assert all(v.dtype == np.float32 for v in opt.m.values())
    _, d = BinaryCrossEntropy()(np.zeros((2, 1), np.float32), np.array([[0], [1]]))
    assert d.dtype == np.float32


def save_small(tmp_path, dtype=np.float32):
    net = small_net(3, dtype)
    path = tmp_path / "m.npz"
    preproc = {"preproc.means": np.arange(3.0), "preproc.scales": np.ones(3)}
    net.save(path, {"head_id": "demo", "transform_version": 1}, preproc)
    return net, path


def test_save_load_round_trip(tmp_path):
    net, path = save_small(tmp_path)
    assert os.listdir(tmp_path) == ["m.npz"]  # no temp file left behind
    loaded, manifest, extra = Sequential.load(path, "demo")
    x = np.random.default_rng(9).standard_normal((4, 3))
    np.testing.assert_array_equal(loaded.forward(x), net.forward(x))
    assert manifest["head_id"] == "demo" and manifest["transform_version"] == 1
    assert [s["type"] for s in manifest["layers"]] == ["Linear", "ReLU", "Linear"]
    np.testing.assert_array_equal(extra["preproc.means"], np.arange(3.0))
    assert all(v.dtype == np.float32 for v in loaded.named_params().values())
    for arr in list(loaded.named_params().values()) + list(extra.values()):
        assert not arr.flags.writeable
    with np.load(path, allow_pickle=False) as z:
        assert z["manifest"].shape == () and z["manifest"].dtype.kind == "U"


def test_embedding_net_round_trip(tmp_path):
    net = names_net(4)
    net.save(tmp_path / "n.npz", {"head_id": "names"})
    loaded, _, _ = Sequential.load(tmp_path / "n.npz", "names")
    x = np.array([[0, 1, 1], [26, 3, 0]])
    np.testing.assert_array_equal(loaded.forward(x), net.forward(x))


def rewrite(path, mutate):
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    original = arrays["manifest"]
    manifest = json.loads(str(original))
    mutate(arrays, manifest)
    if arrays.get("manifest") is original:  # mutation edited the dict, not the raw array
        arrays["manifest"] = np.array(json.dumps(manifest), dtype=np.str_)
    with open(path, "wb") as f:
        np.savez(f, **arrays)


def _set(key, value):
    return lambda a, m: m.__setitem__(key, value)


def _cfg(i, key, value, w_shape=None):
    def mutate(a, m):
        m["layers"][i]["config"][key] = value
        if w_shape is not None:
            m["layers"][i]["shapes"]["W"] = list(w_shape)
            a[f"{i}.W"] = np.zeros(w_shape, np.float32)
    return mutate


# case -> (mutation of (arrays, manifest), expected error text)
CORRUPTIONS = {
    "future_version": (_set("format_version", 2), "unsupported format_version"),
    "version_as_string": (_set("format_version", "1"), "unsupported format_version"),
    "unknown_layer": (lambda a, m: m["layers"][1].__setitem__("type", "Evil"), "unknown layer type"),
    "layer_entry_not_object": (lambda a, m: m["layers"].__setitem__(1, "ReLU"), "corrupted manifest layer entry"),
    "missing_array": (lambda a, m: a.pop("2.b"), r"missing \['2.b'\]"),
    "extra_array": (lambda a, m: a.__setitem__("9.W", np.zeros(2)), r"unexpected \['9.W'\]"),
    "wrong_shape": (lambda a, m: a.__setitem__("0.W", np.zeros((4, 3), np.float32)), "0.W: shape"),
    "nan_param": (lambda a, m: a["0.b"].__setitem__(0, np.nan), "0.b: non-finite"),
    "inf_preproc": (lambda a, m: a["preproc.scales"].__setitem__(1, np.inf), "preproc.scales: non-finite"),
    "config_disagrees": (_cfg(0, "d_in", 5), "shapes disagree"),
    "bad_config_key": (_cfg(0, "evil", 1), "bad Linear config"),
    "zero_d_in": (_cfg(0, "d_in", 0, (0, 4)), "positive int"),
    "bool_d_out": (lambda a, m: m["layers"][2]["config"].__setitem__("d_out", True), "positive int"),
    "null_config": (lambda a, m: m["layers"][1].__setitem__("config", None), "bad ReLU config"),
    "chain_mismatch": (_cfg(2, "d_in", 5, (5, 2)), "does not connect"),
    "layers_missing": (lambda a, m: m.pop("layers"), "malformed"),
    "extra_not_preproc": (lambda a, m: (m["extra_arrays"].__setitem__("x", [1]), a.__setitem__("x", np.zeros(1))),
                          "preproc"),
    "manifest_missing": (lambda a, m: a.pop("manifest"), "manifest missing"),
    "manifest_not_json": (lambda a, m: a.__setitem__("manifest", np.array("{not json", dtype=np.str_)),
                          "corrupted manifest"),
    "manifest_not_object": (lambda a, m: a.__setitem__("manifest", np.array("[1, 2]", dtype=np.str_)),
                            "not an object"),
    "manifest_bytes": (lambda a, m: a.__setitem__("manifest", np.frombuffer(b"{}", np.uint8)), "0-d string"),
    "manifest_pickled": (lambda a, m: a.__setitem__("manifest", np.array(dict(m), dtype=object)),
                         "unreadable archive"),
    "preproc_object": (lambda a, m: a.__setitem__("preproc.means", np.array([1, "x", None], dtype=object)),
                       "unreadable archive"),
    "mixed_dtypes": (lambda a, m: a.__setitem__("0.W", a["0.W"].astype(np.float64)), "share one"),
}


def test_identity_rewrite_still_loads(tmp_path):
    """Control: the rewrite helper alone must not break a valid archive."""
    _, path = save_small(tmp_path)
    rewrite(path, lambda a, m: None)
    Sequential.load(path, "demo")


@pytest.mark.parametrize("case", sorted(CORRUPTIONS))
def test_corrupted_archive_rejected(tmp_path, case):
    _, path = save_small(tmp_path)
    mutate, message = CORRUPTIONS[case]
    rewrite(path, mutate)
    with pytest.raises(ValueError, match=message):
        Sequential.load(path, "demo")


def _shift_layers(a, m):
    """Prepend a Tanh so the Embedding is no longer first; renumber the arrays to match."""
    m["layers"].insert(0, {"type": "Tanh", "config": {}, "shapes": {}})
    moved = {}
    for k in [k for k in a if k[0].isdigit()]:
        i, name = k.split(".")
        moved[f"{int(i) + 1}.{name}"] = a.pop(k)
    a.update(moved)


@pytest.mark.parametrize("mutate, message", [
    (_cfg(2, "d_in", 25, (25, 64)), "does not connect"),  # 25 is not a multiple of embedding dim 8
    (_shift_layers, "Embedding must be the first layer"),
])
def test_embedding_chain_rejected(tmp_path, mutate, message):
    path = tmp_path / "n.npz"
    names_net(4).save(path, {"head_id": "names"})
    rewrite(path, mutate)
    with pytest.raises(ValueError, match=message):
        Sequential.load(path, "names")


def test_save_rejects_unregistered_and_mismatched_nets(tmp_path):
    class Custom(ReLU):
        pass

    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="unregistered layer type Custom"):
        Sequential([Linear(2, 2, "he", rng), Custom()]).save(tmp_path / "c.npz", {"head_id": "demo"})
    with pytest.raises(ValueError, match="does not connect"):
        Sequential([Linear(2, 3, "he", rng), ReLU(), Linear(4, 1, "lecun", rng)]).save(
            tmp_path / "c.npz", {"head_id": "demo"})
    assert os.listdir(tmp_path) == []


def test_layer_configs_are_per_instance():
    a, b = ReLU(), Tanh()
    a.config["x"] = 1
    assert b.config == {} and ReLU().config == {}


def test_head_mismatch_and_garbage_rejected(tmp_path):
    _, path = save_small(tmp_path)
    with pytest.raises(ValueError, match="head mismatch"):
        Sequential.load(path, "vision")
    junk = tmp_path / "junk.npz"
    junk.write_bytes(b"this is not a zip archive")
    with pytest.raises(ValueError):
        Sequential.load(junk, "demo")


def test_failed_save_keeps_target_and_cleans_tmp(tmp_path):
    net, path = save_small(tmp_path)
    original = path.read_bytes()
    net.layers[0].W[0, 0] = np.nan
    with pytest.raises(ValueError):
        net.save(path, {"head_id": "demo"})
    with pytest.raises(ValueError):
        net.save(path, {"head_id": "demo"}, {"not_preproc": np.zeros(1)})
    with pytest.raises(ValueError):
        net.save(path, {})
    assert path.read_bytes() == original
    assert os.listdir(tmp_path) == ["m.npz"]
