"""M2 gate: XOR end to end, then save -> load round trip."""
import numpy as np

from saemil import Adam, BinaryCrossEntropy, Linear, Sequential, Tanh, sigmoid

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
Y = np.array([[0], [1], [1], [0]], dtype=np.float32)
MAX_UPDATES = 3000


def test_xor_solved_and_round_trips(tmp_path):
    rng = np.random.default_rng(7)
    net = Sequential([Linear(2, 8, "lecun", rng, np.float32), Tanh(), Linear(8, 1, "lecun", rng, np.float32)])
    loss_fn, opt = BinaryCrossEntropy(), Adam(lr=0.03)
    for _ in range(MAX_UPDATES):
        net.train_step(X, Y, loss_fn, opt)
        logits = net.forward(X)  # evaluate the post-update weights
        loss, _ = loss_fn(logits, Y)
        correct = int(((sigmoid(logits) >= 0.5) == Y).sum())
        if correct == 4 and loss < 0.05:
            break
    assert correct == 4 and loss < 0.05, f"not solved in {opt.t} updates: {correct}/4, BCE {loss}"

    path = tmp_path / "xor.npz"
    net.save(path, {"head_id": "xor", "transform_version": 1})
    loaded, manifest, _ = Sequential.load(path, "xor")
    np.testing.assert_allclose(loaded.forward(X), logits, rtol=1e-6, atol=1e-7)
    assert manifest["format_version"] == 1
