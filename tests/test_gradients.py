"""M2 gate: float64 central-difference gradient checks, edge cases and the Adam reference."""
import warnings

import numpy as np
import pytest

from saemil import (
    MSE,
    SGD,
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
    sigmoid,
    softmax,
)

H = 1e-5
STRICT = dict(over="raise", divide="raise", invalid="raise")  # underflow to 0 is expected and fine


def numeric_grad(f, q):
    """Central differences of scalar f() w.r.t. every coordinate of q (perturbed in place, restored exactly)."""
    g = np.zeros_like(q)
    for idx in np.ndindex(q.shape):
        orig = q[idx]
        q[idx] = orig + H
        fp = f()
        q[idx] = orig - H
        fm = f()
        q[idx] = orig
        g[idx] = (fp - fm) / (2 * H)
    return g


def assert_close(name, g_num, g_ana):
    err = np.abs(g_num - g_ana)
    tol = 1e-7 + 1e-5 * np.maximum(np.abs(g_num), np.abs(g_ana))
    if not (err <= tol).all():
        i = np.unravel_index(np.argmax(err - tol), err.shape)
        pytest.fail(f"{name}: worst index {i}, numeric {g_num[i]:.10g}, analytic {g_ana[i]:.10g}, err {err[i]:.3g}")


def away_from_zero(x):
    return np.sign(x) * (np.abs(x) + 0.1) + (x == 0) * 0.1


def check_layer(layer, x):
    """F = sum(forward(X) * G_fixed); check every param and (float) input."""
    g_fixed = np.random.default_rng(99).standard_normal(layer.forward(x, training=False).shape)

    def f():
        return float(np.sum(layer.forward(x, training=True) * g_fixed))

    layer.forward(x, training=True)
    dx = layer.backward(g_fixed)
    analytic = {k: v.copy() for k, v in layer.grads().items()}
    for k, p in layer.params().items():
        assert_close(f"{type(layer).__name__}.{k}", numeric_grad(f, p), analytic[k])
    if x.dtype.kind == "f":
        assert_close(f"{type(layer).__name__}.input", numeric_grad(f, x), dx)
    else:
        assert dx is None


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("kind", ["linear_he", "linear_lecun", "relu", "sigmoid", "tanh", "flatten", "embedding"])
def test_layer_gradients(kind, batch):
    rng = np.random.default_rng(batch)
    x = rng.standard_normal((batch, 3))
    layer = {
        "linear_he": lambda: Linear(3, 5, "he", rng, np.float64),
        "linear_lecun": lambda: Linear(3, 2, "lecun", rng, np.float64),
        "relu": ReLU, "sigmoid": Sigmoid, "tanh": Tanh, "flatten": Flatten,
        "embedding": lambda: Embedding(7, 4, rng, np.float64),
    }[kind]()
    if kind == "relu":
        x = away_from_zero(x)
    if kind == "sigmoid":
        x = x * 4  # cover both sign branches with real magnitude
    if kind == "flatten":
        x = rng.standard_normal((batch, 3, 2))
    if kind == "embedding":
        x = rng.integers(0, 6, size=(batch, 3))  # row 6 never used; repeats likely
        x[0, :2] = 2  # guarantee a repeat
    check_layer(layer, x)
    if kind == "linear_he":
        layer.forward(x, training=True)
        layer.backward(np.ones((batch, 5)))
        np.testing.assert_allclose(layer.db, np.full(5, batch))  # db sums over batch, no division


def test_embedding_repeated_ids_accumulate_exactly():
    emb = Embedding(6, 3, np.random.default_rng(0), np.float64)
    g = np.array([0.1, -0.7, 1.3])
    emb.forward(np.array([[4, 4, 4]]), training=True)
    emb.backward(np.tile(g, (1, 3, 1)))
    np.testing.assert_array_equal(emb.dE[4], 3 * g)
    assert not np.delete(emb.dE, 4, axis=0).any()  # every unused row is exactly 0


def test_relu_at_exactly_zero_has_zero_derivative():
    relu = ReLU()
    relu.forward(np.array([[0.0, -1.0, 2.0, -0.0]]), training=True)
    np.testing.assert_array_equal(relu.backward(np.ones((1, 4))), [[0, 0, 1, 0]])


@pytest.mark.parametrize("batch", [1, 5])
def test_loss_gradients(batch):
    rng = np.random.default_rng(batch + 10)
    z = rng.standard_normal((batch, 4)) * 3
    y = rng.integers(0, 4, size=batch)
    ce = SoftmaxCrossEntropy()
    assert_close("CE", numeric_grad(lambda: ce(z, y)[0], z), ce(z, y)[1])

    zb = rng.standard_normal((batch, 1)) * 3
    yb = rng.integers(0, 2, size=(batch, 1))
    bce = BinaryCrossEntropy()
    assert_close("BCE", numeric_grad(lambda: bce(zb, yb)[0], zb), bce(zb, yb)[1])

    p = rng.standard_normal((batch, 2))
    t = rng.standard_normal((batch, 2))
    mse = MSE()
    assert_close("MSE", numeric_grad(lambda: mse(p, t)[0], p), mse(p, t)[1])


def test_loss_values_match_definitions():
    z = np.array([[1.0, 2.0, 0.5], [0.0, -1.0, 3.0]])
    y = np.array([1, 0])
    p = softmax(z)
    loss, _ = SoftmaxCrossEntropy()(z, y)
    assert np.isclose(loss, -np.mean(np.log(p[[0, 1], y])))
    zb, yb = np.array([[0.3], [-2.0]]), np.array([[1], [0]])
    s = 1 / (1 + np.exp(-zb))
    assert np.isclose(BinaryCrossEntropy()(zb, yb)[0], -np.mean(yb * np.log(s) + (1 - yb) * np.log(1 - s)))
    assert np.isclose(MSE()(np.array([[1.0, 3.0]]), np.array([[0.0, 1.0]]))[0], 2.5)


@pytest.mark.parametrize("y", [[0, 3], [-1, 0], [0, 2.0]])
def test_cross_entropy_rejects_bad_targets(y):
    with pytest.raises(ValueError):
        SoftmaxCrossEntropy()(np.zeros((2, 3)), np.array(y))


def composed_nets(rng):
    d = np.float64
    yield ("embed_tanh_ce",
           Sequential([Embedding(6, 3, rng, d), Flatten(), Linear(9, 5, "lecun", rng, d), Tanh(),
                       Linear(5, 4, "lecun", rng, d)]),
           lambda b: rng.integers(0, 5, size=(b, 3)), lambda b: rng.integers(0, 4, size=b), SoftmaxCrossEntropy())
    yield ("relu_bce",
           Sequential([Linear(3, 6, "he", rng, d), ReLU(), Linear(6, 4, "he", rng, d), ReLU(),
                       Linear(4, 1, "lecun", rng, d)]),
           lambda b: rng.standard_normal((b, 3)), lambda b: rng.integers(0, 2, size=(b, 1)), BinaryCrossEntropy())
    yield ("sigmoid_mse",
           Sequential([Linear(2, 3, "lecun", rng, d), Sigmoid(), Linear(3, 2, "lecun", rng, d)]),
           lambda b: rng.standard_normal((b, 2)), lambda b: rng.standard_normal((b, 2)), MSE())


# B=1, B>1 and an odd B; losses average over the actual rows, so a partial final batch is just another B.
@pytest.mark.parametrize("batch", [1, 3, 5])
def test_composed_net_gradients(batch):
    rng = np.random.default_rng(batch)
    for name, net, make_x, make_y, loss_fn in composed_nets(rng):
        x, y = make_x(batch), make_y(batch)

        def f(net=net, x=x, y=y, loss_fn=loss_fn):  # bind this iteration's values
            return loss_fn(net.forward(x, training=True), y)[0]

        net.zero_grad()
        _, d = loss_fn(net.forward(x, training=True), y)
        dx = net.backward(d)
        analytic = {k: v.copy() for k, v in net.named_grads().items()}
        for k, p in net.named_params().items():
            assert_close(f"{name}[{k}]", numeric_grad(f, p), analytic[k])
        if x.dtype.kind == "f":
            assert_close(f"{name}[input]", numeric_grad(f, x), dx)


def test_extreme_logits_are_finite():
    with np.errstate(**STRICT), warnings.catch_warnings():
        warnings.simplefilter("error")
        z = np.array([[1000.0, -1000.0, 0.0], [-1000.0, 1000.0, 1000.0]])
        loss, d = SoftmaxCrossEntropy()(z, np.array([1, 0]))
        assert np.isfinite(loss) and np.isfinite(d).all()
        assert np.isclose(loss, 1000.0 + (2000.0 + np.log(2)) / 2)
        zb = np.array([[1000.0], [-1000.0], [1000.0], [-1000.0]])
        loss, d = BinaryCrossEntropy()(zb, np.array([[0], [1], [1], [0]]))
        assert np.isfinite(loss) and np.isfinite(d).all()
        assert np.isclose(loss, 500.0)
        for dtype in (np.float32, np.float64):
            loss, d = SoftmaxCrossEntropy()(z.astype(dtype), np.array([1, 0]))
            assert np.isfinite(loss) and np.isfinite(d).all() and d.dtype == dtype


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_stable_sigmoid_no_warnings(dtype):
    x = np.array([-1000.0, -100.0, -30.0, 0.0, 30.0, 1000.0], dtype=dtype)
    with np.errstate(**STRICT), warnings.catch_warnings():
        warnings.simplefilter("error")
        s = sigmoid(x)
        a = Sigmoid().forward(x, training=False)
        p = softmax(np.array([[1000.0, -1000.0]], dtype=dtype))
    assert s.dtype == dtype and a.dtype == dtype
    np.testing.assert_array_equal(s, a)
    assert s[0] == 0 and s[3] == 0.5 and s[-1] == 1 and np.isfinite(s).all()
    np.testing.assert_allclose(s[2], np.exp(-30.0), rtol=1e-5)  # -100 is subnormal in float32
    np.testing.assert_array_equal(p, [[1.0, 0.0]])


def test_adam_two_steps_match_scalar_reference():
    lr, b1, b2, eps = 0.1, 0.9, 0.999, 1e-8
    params = {"a": np.array([1.0, -2.0]), "b": np.array([[0.5, 0.0], [3.0, -1.0]])}
    grad_steps = [{"a": np.array([0.3, -1.0]), "b": np.array([[2.0, -0.5], [0.1, 0.0]])},
                  {"a": np.array([-0.2, 4.0]), "b": np.array([[1.0, 0.5], [-0.3, 0.7]])}]
    expected = {k: v.copy() for k, v in params.items()}
    for name in expected:
        for idx in np.ndindex(expected[name].shape):
            theta, m, v = float(expected[name][idx]), 0.0, 0.0
            for t, grads in enumerate(grad_steps, start=1):
                g = float(grads[name][idx])
                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * g * g
                theta -= lr * (m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)) ** 0.5 + eps)
            expected[name][idx] = theta
    opt = Adam(lr, b1, b2, eps)
    for grads in grad_steps:
        opt.step(params, grads)
    assert opt.t == 2
    for name in params:
        np.testing.assert_allclose(params[name], expected[name], rtol=1e-12, atol=1e-15)


def test_sgd_step():
    p = {"w": np.array([1.0, 2.0])}
    SGD(0.5).step(p, {"w": np.array([2.0, -2.0])})
    np.testing.assert_array_equal(p["w"], [0.0, 3.0])
