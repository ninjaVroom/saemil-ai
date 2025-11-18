"""Sequential container and the pickle-free .npz artifact format."""
import json
import os
import tempfile
import zipfile

import numpy as np

from .activations import ReLU, Sigmoid, Tanh
from .layers import Embedding, Flatten, Linear

LAYER_REGISTRY = {"Linear": Linear, "Embedding": Embedding, "Flatten": Flatten,
                  "ReLU": ReLU, "Sigmoid": Sigmoid, "Tanh": Tanh}
FORMAT_VERSION = 1
_REVERSE = {cls: name for name, cls in LAYER_REGISTRY.items()}


def _param_shapes(kind, cfg):
    """Expected parameter shapes from a layer config, checked before anything is allocated."""
    sizes = {"Linear": ("d_in", "d_out"), "Embedding": ("vocab", "dim")}.get(kind, ())
    keys = set(sizes) | ({"init"} if kind == "Linear" else set())
    if not isinstance(cfg, dict) or set(cfg) != keys:
        raise ValueError(f"bad {kind} config {cfg!r}")
    for k in sizes:
        if type(cfg[k]) is not int or cfg[k] <= 0:
            raise ValueError(f"{kind}.{k} must be a positive int, got {cfg[k]!r}")
    if kind == "Linear":
        return {"W": [cfg["d_in"], cfg["d_out"]], "b": [cfg["d_out"]]}
    if kind == "Embedding":
        return {"E": [cfg["vocab"], cfg["dim"]]}
    return {}


def _check_chain(specs):
    """Reject layer stacks whose sizes cannot connect. Input width is unknown until the first Linear."""
    width = None  # int: 2-D feature width; ("seq", k): (B,T,k); ("flat", k): (B,T*k), T unknown
    for i, spec in enumerate(specs):
        kind, cfg = spec["type"], spec["config"]
        if kind == "Embedding":
            if i != 0:
                raise ValueError(f"layer {i}: Embedding must be the first layer (it takes integer IDs)")
            width = ("seq", cfg["dim"])
        elif kind == "Flatten":
            if isinstance(width, tuple) and width[0] == "seq":
                width = ("flat", width[1])
        elif kind == "Linear":
            d_in = cfg["d_in"]
            ok = (width is None or width == d_in or (width == ("seq", d_in))
                  or (isinstance(width, tuple) and width[0] == "flat" and d_in % width[1] == 0))
            if not ok:
                raise ValueError(f"layer {i}: Linear d_in={d_in} does not connect to incoming width {width}")
            width = ("seq", cfg["d_out"]) if width == ("seq", d_in) else cfg["d_out"]


class Sequential:
    def __init__(self, layers):
        self.layers = list(layers)

    def forward(self, x, training=False):
        for layer in self.layers:
            x = layer.forward(x, training)
        return x

    def backward(self, dy):
        for layer in reversed(self.layers):
            dy = layer.backward(dy)
        return dy

    def named_params(self):
        return {f"{i}.{k}": v for i, layer in enumerate(self.layers) for k, v in layer.params().items()}

    def named_grads(self):
        return {f"{i}.{k}": v for i, layer in enumerate(self.layers) for k, v in layer.grads().items()}

    def zero_grad(self):
        for g in self.named_grads().values():
            g.fill(0)

    def train_step(self, x, y, loss_fn, opt):
        self.zero_grad()
        loss, dlogits = loss_fn(self.forward(x, training=True), y)
        self.backward(dlogits)
        opt.step(self.named_params(), self.named_grads())
        return loss

    def save(self, path, manifest, extra_arrays=None):
        """Write tmp file in the target directory, validate it by loading, then os.replace."""
        extra = dict(extra_arrays or {})
        if not isinstance(manifest.get("head_id"), str):
            raise ValueError("manifest needs a string 'head_id'")
        for name in extra:
            if not name.startswith("preproc."):
                raise ValueError(f"extra array {name!r} must be named 'preproc.*'")
        params = self.named_params()
        layers = []
        for layer in self.layers:
            kind = _REVERSE.get(type(layer))
            if kind is None:
                raise ValueError(f"cannot save unregistered layer type {type(layer).__name__}")
            layers.append({"type": kind, "config": layer.config,
                           "shapes": {k: list(v.shape) for k, v in layer.params().items()}})
        full = dict(manifest, format_version=FORMAT_VERSION, layers=layers,
                    extra_arrays={k: list(np.shape(v)) for k, v in extra.items()})
        text = json.dumps(full, allow_nan=False)

        path = os.fspath(path)
        fd, tmp = tempfile.mkstemp(suffix=".npz.tmp", dir=os.path.dirname(os.path.abspath(path)))
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez(f, manifest=np.array(text, dtype=np.str_), **params, **extra)
                f.flush()
                os.fsync(f.fileno())
            Sequential.load(tmp, manifest["head_id"])
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    @classmethod
    def load(cls, path, expect_head):
        """Validate and restore an archive. Returns (net, manifest, extra_arrays).

        Arrays are read-only, so a loaded net is for inference only (an optimizer step raises).
        """
        try:
            with np.load(path, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
        except (OSError, ValueError, EOFError, zipfile.BadZipFile) as e:
            raise ValueError(f"unreadable archive {path}: {e}") from e

        raw = arrays.pop("manifest", None)
        if raw is None or raw.shape != () or raw.dtype.kind != "U":
            raise ValueError("manifest missing or not a 0-d string array")
        try:
            manifest = json.loads(str(raw))
        except ValueError as e:
            raise ValueError(f"corrupted manifest: {e}") from e
        if not isinstance(manifest, dict):
            raise ValueError("corrupted manifest: not an object")
        if manifest.get("format_version") != FORMAT_VERSION or type(manifest.get("format_version")) is not int:
            raise ValueError(f"unsupported format_version {manifest.get('format_version')!r}")
        if manifest.get("head_id") != expect_head:
            raise ValueError(f"head mismatch: expected {expect_head!r}, got {manifest.get('head_id')!r}")

        specs, extra_spec = manifest.get("layers"), manifest.get("extra_arrays")
        if not isinstance(specs, list) or not isinstance(extra_spec, dict):
            raise ValueError("corrupted manifest: 'layers' / 'extra_arrays' malformed")
        expected = {}
        try:
            for i, spec in enumerate(specs):
                if spec["type"] not in LAYER_REGISTRY:
                    raise ValueError(f"unknown layer type {spec['type']!r}")
                shapes = _param_shapes(spec["type"], spec["config"])
                if shapes != spec["shapes"]:
                    raise ValueError(f"layer {i} shapes disagree with its config")
                expected.update({f"{i}.{k}": v for k, v in shapes.items()})
            _check_chain(specs)
        except (KeyError, TypeError) as e:
            raise ValueError(f"corrupted manifest layer entry: {e!r}") from e
        if not all(name.startswith("preproc.") for name in extra_spec):
            raise ValueError("extra arrays must be named 'preproc.*'")
        expected.update(extra_spec)

        if set(arrays) != set(expected):
            missing, unexpected = set(expected) - set(arrays), set(arrays) - set(expected)
            raise ValueError(f"array mismatch: missing {sorted(missing)}, unexpected {sorted(unexpected)}")
        for name, arr in arrays.items():
            if list(arr.shape) != expected[name]:
                raise ValueError(f"{name}: shape {list(arr.shape)} != expected {expected[name]}")
            if arr.dtype.kind in "fc" and not np.isfinite(arr).all():
                raise ValueError(f"{name}: non-finite values")
            arr.flags.writeable = False

        param_dtypes = {arrays[n].dtype for n in expected if n not in extra_spec}
        if len(param_dtypes) > 1 or not all(d in (np.float32, np.float64) for d in param_dtypes):
            raise ValueError(f"parameters must share one float32/float64 dtype, got {param_dtypes}")
        dtype = param_dtypes.pop() if param_dtypes else np.float32

        layers, dummy = [], np.random.default_rng(0)
        for i, spec in enumerate(specs):
            kind = LAYER_REGISTRY[spec["type"]]
            if spec["type"] in ("Linear", "Embedding"):
                try:
                    layer = kind(**spec["config"], rng=dummy, dtype=dtype)
                except (TypeError, ValueError) as e:
                    raise ValueError(f"layer {i}: bad config: {e}") from e
                for k in layer.params():
                    setattr(layer, k, arrays[f"{i}.{k}"])
            else:
                layer = kind()
            layers.append(layer)
        extra = {k: arrays[k] for k in extra_spec}
        return cls(layers), manifest, extra
