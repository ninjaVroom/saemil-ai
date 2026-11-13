"""T-601b: the browser's centering (studio/static/centering.js) must match heads.vision.center_by_mass byte for byte
(D-011). Runs the JS under Node; a missing Node is a failure, not a skip.

Exact ties: the server computes the scaled crop with BLAS matmul/dot (OpenBLAS, DYNAMIC_ARCH, FMA kernels), so when
the true centre of mass sits exactly on a half pixel (e.g. a straight, point-symmetric stroke) or a scaled pixel is
exactly x.5, the server's own placement is decided by last-ulp rounding that differs between CPUs. No JS port can
reproduce that bit for bit. Such inputs are detected and must match up to that one tie-break (bbox shift <= 1 px,
pixel values within 1); every other input must be byte-identical. The server re-centres whatever the client sends,
so predictions are unaffected. Proposed backend fix (coordinator): round the offsets/pixels before floor/rint."""
import json
import shutil
import subprocess
from itertools import pairwise
from pathlib import Path

import numpy as np

from heads.vision import BOX, SIDE, _resample, center_by_mass

ROOT = Path(__file__).resolve().parent.parent
JS = ROOT / "studio" / "static" / "centering.js"
RUNNER = """
const {centerByMass} = require(process.argv[1]);
let buf = ""; process.stdin.on("data", d => buf += d).on("end", () => {
  const out = JSON.parse(buf).map(f => centerByMass(f.px, f.h, f.w));
  process.stdout.write(JSON.stringify(out));
});
"""


def stroke(h, w, pts, width):
    """Anti-aliased-ish polyline: integer intensities, like a canvas alpha channel."""
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w))
    for (y0, x0), (y1, x1) in pairwise(pts):
        d = np.array([y1 - y0, x1 - x0], float)
        t = np.clip(((yy - y0) * d[0] + (xx - x0) * d[1]) / max(d @ d, 1e-9), 0, 1)
        dist = np.hypot(yy - (y0 + t * d[0]), xx - (x0 + t * d[1]))
        img = np.maximum(img, np.clip(width / 2 + 0.5 - dist, 0, 1))
    return np.rint(img * 255).astype(np.int64)


def fixtures():
    z = np.load(ROOT / "data" / "prepared" / "vision.npz")
    rng = np.random.default_rng(5)
    out = [img.astype(np.int64) for img in z["x_test"][:40]]                    # MNIST test images, 28x28
    out += [np.kron(img, np.ones((10, 10), np.int64)) for img in z["x_test"][:5]]  # upscaled to a 280 canvas
    out.append(stroke(280, 280, [(10, 220), (90, 270)], 20))                   # off-centre stroke, top right
    out.append(stroke(280, 280, [(180, 5), (270, 40), (200, 60)], 18))         # off-centre, bottom left
    thin = np.zeros((280, 280), np.int64); thin[40:240, 137] = 255; out.append(thin)  # one-pixel "1"
    out.append(stroke(280, 280, [(40, 140), (240, 140)], 20))                  # drawn "1"
    out.append(stroke(200, 300, [(20, 30), (180, 280), (150, 120)], 16))       # non-square canvases
    out.append(stroke(400, 50, [(10, 25), (390, 20)], 6))
    out.append(stroke(37, 91, [(3, 3), (30, 88), (5, 60)], 3))
    for h, w in ((5, 40), (3, 40), (1, 40), (40, 5)):                          # half-way scaled sizes (2.5 -> 2)
        a = np.zeros((h, w), np.int64); a[:, :] = 200; out.append(a)
    dot = np.zeros((28, 28), np.int64); dot[27, 27] = 9; out.append(dot)       # corner single pixel
    out.append(np.zeros((28, 28), np.int64))                                   # empty
    for _ in range(30):                                                        # random sparse ink, odd sizes
        h, w = rng.integers(20, 320, 2)
        a = rng.integers(0, 256, (h, w)) * (rng.random((h, w)) < rng.uniform(0.01, 0.3))
        out.append(a.astype(np.int64))
    return out


def run_js(fx):
    node = shutil.which("node")
    assert node, "Node.js is required for the client centering parity test (D-011); install node and rerun"
    payload = json.dumps([{"px": f.ravel().tolist(), "h": f.shape[0], "w": f.shape[1]} for f in fx])
    r = subprocess.run([node, "-e", RUNNER, str(JS)], input=payload, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout)
    assert len(got) == len(fx)
    return got


def is_tie(img, eps=1e-7):
    """True when the server's result hinges on last-ulp rounding (see module docstring)."""
    img = np.asarray(img, np.float64)
    ys, xs = np.nonzero(img)
    if ys.size == 0:
        return False
    crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    nh, nw = max(1, round(h * BOX / max(h, w))), max(1, round(w * BOX / max(h, w)))
    small = _resample(nh, h) @ crop @ _resample(nw, w).T
    cy = small.sum(axis=1) @ np.arange(nh) / small.sum()
    cx = small.sum(axis=0) @ np.arange(nw) / small.sum()
    near_int = lambda v: abs(v - np.rint(v)) < eps  # noqa: E731
    return bool(near_int(SIDE / 2 - cy + 0.5) or near_int(SIDE / 2 - cx + 0.5)
                or near_int(small - 0.5).any())


def bbox(a):
    ys, xs = np.nonzero(a)
    return ys.min(), xs.min(), a[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(int)


def test_js_centering_matches_server():
    """Curated fixtures (MNIST test images, upscaled canvases, off-centre strokes, thin "1", non-square canvases,
    half-way scaled sizes, single pixel, empty): byte-identical, including the few exact ties among them."""
    fx = fixtures()
    for i, (f, js) in enumerate(zip(fx, run_js(fx), strict=True)):
        want = center_by_mass(f).tolist()
        assert js == want, f"fixture {i} shape {f.shape}: {sum(a != b for a, b in zip(js, want, strict=True))} pixels differ"


def test_js_centering_stroke_sweep():
    """300 random strokes on random canvas sizes: byte-identical unless the input is an exact tie."""
    rng = np.random.default_rng(9)
    fx = []
    for _ in range(300):
        h, w = rng.integers(60, 400, 2)
        pts = [tuple(rng.integers(0, [h, w])) for _ in range(rng.integers(2, 5))]
        fx.append(stroke(h, w, pts, int(rng.integers(2, 25))))
    exact = ties = 0
    for i, (f, js) in enumerate(zip(fx, run_js(fx), strict=True)):
        want = center_by_mass(f)
        js = np.array(js, np.uint8)
        if not is_tie(f):
            assert js.tolist() == want.tolist(), f"sweep {i} shape {f.shape} differs and is not a tie"
            exact += 1
            continue
        ties += 1
        (ty, tx, a), (wy, wx, b) = bbox(js.reshape(SIDE, SIDE)), bbox(want.reshape(SIDE, SIDE))
        assert abs(ty - wy) <= 1 and abs(tx - wx) <= 1 and a.shape == b.shape, f"sweep {i}: tie off by > 1 px"
        assert np.abs(a - b).max() <= 1, f"sweep {i}: tie pixel values differ by > 1"
    assert exact >= 200, (exact, ties)


def test_js_tiny_stroke_rule():
    """D-023: raw ink bbox longer side < 3/28 of the canvas side -> client posts zeros (server blank)."""
    node = shutil.which("node")
    assert node, "Node.js is required for this check; install node and rerun"
    def img(h, w, y0, y1, x0, x1):
        a = np.zeros((h, w), np.int64); a[y0:y1, x0:x1] = 255; return a
    cases = [(np.zeros((280, 280), np.int64), True),
             (img(280, 280, 100, 129, 100, 110), True),    # 29 px < 30
             (img(280, 280, 100, 130, 100, 110), False),   # 30 px = 3/28 of 280
             (img(28, 28, 5, 7, 5, 7), True), (img(28, 28, 5, 8, 5, 6), False),
             (img(200, 400, 10, 20, 10, 52), True),        # longer side 42 < 3/28 * 400
             (img(200, 400, 10, 20, 10, 53), False)]
    runner = """const {isTinyStroke} = require(process.argv[1]); let b = "";
process.stdin.on("data", d => b += d).on("end", () =>
  process.stdout.write(JSON.stringify(JSON.parse(b).map(f => isTinyStroke(f.px, f.h, f.w)))));"""
    payload = json.dumps([{"px": a.ravel().tolist(), "h": a.shape[0], "w": a.shape[1]} for a, _ in cases])
    r = subprocess.run([node, "-e", runner, str(JS)], input=payload, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout) == [want for _, want in cases]
