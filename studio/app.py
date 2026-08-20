"""Saemil Studio backend (M6, T-501): one FastAPI app over the three frozen heads and the AI bridge.

Run: python -m uvicorn studio.app:app --host 127.0.0.1 --port 8000   (one worker, loopback only)

- Lifespan loads models/{vision,names,tabular}.npz once. A missing or broken archive becomes a named diagnostic for
  that head (its routes answer 503); the other heads keep serving. SAEMIL_MODEL_DIR overrides the model directory.
- Prediction routes are plain `def` (FastAPI threadpool) and stateless; /api/report is async and the bridge runs the
  provider in a worker thread, so a slow provider never blocks predictions (D-007).
- `.env` in the project root is read at startup (T-502): KEY=VALUE, # comments, optional quotes; variables already
  set in the environment win; values are never logged or served.
- Tabular input cap (T-302 note): each feature must be <= TABULAR_CAP_FACTOR x its raw train maximum (stored p100).
  10x keeps every plausible session (train max ProductRelated_Duration is ~12 h) while refusing absurd values the
  model never saw; above it the route answers 422.
- D-019: Host header must be 127.0.0.1 or localhost (any port), bodies over 256 KB get 413, no /docs, /redoc or
  /openapi.json (they would load a CDN). Only the .env keys in ENV_KEYS are applied.
- No UI yet: GET / returns a JSON notice until studio/static/index.html exists; /static is mounted only if the
  directory exists.
"""
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

import saemil_bridge
from heads import names as names_head
from heads import tabular as tabular_head
from heads import vision as vision_head
from saemil import Sequential
from studio.schemas import (
    HeadStatus,
    NamesRequest,
    NamesResponse,
    ReportRequest,
    ReportResponse,
    StatusResponse,
    TabularFeature,
    TabularRequest,
    TabularResponse,
    VisionRequest,
    VisionResponse,
)

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
HEADS = ("vision", "names", "tabular")
TABULAR_CAP_FACTOR = 10.0
ALLOWED_HOSTS = ("127.0.0.1", "localhost")  # D-019: Host allowlist against DNS rebinding (port is ignored)
MAX_BODY = 256 * 1024                       # D-019: request body cap, 413 beyond
# .env keys the studio applies: mirrors .env.example, plus OLLAMA_HOST (read by the bridge) and SAEMIL_MODEL_DIR.
ENV_KEYS = frozenset({"ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "GEMINI_API_KEY", "GEMINI_MODEL", "OPENAI_API_KEY",
                      "OPENAI_MODEL", "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "SAEMIL_CLI_TOOLS", "SAEMIL_CLI_TIMEOUT_S",
                      "SAEMIL_OLLAMA_MODEL", "SAEMIL_LLM_MODEL", "SAEMIL_LLM_ON_DEVICE", "OLLAMA_HOST",
                      "SAEMIL_MODEL_DIR"})
_ENV_LINE = re.compile(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*")
_QUOTED = re.compile(r"""(["'])(.*?)\1(?:\s+#.*)?""")


# ---------------------------------------------------------------- .env

def parse_env(text):
    """[export ]KEY=VALUE lines -> dict. Blank lines and # comments skipped. A quoted value keeps everything inside
    its quotes and may be followed by ' # comment'; an unquoted value ends at ' #'. Malformed lines are ignored
    (never echoed: they may hold a secret)."""
    out = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _ENV_LINE.fullmatch(line)
        if not m:
            continue
        key, val = m.groups()
        if val[:1] in ("'", '"'):
            q = _QUOTED.fullmatch(val)
            if not q:
                continue
            val = q.group(2)
        else:
            val = re.split(r"\s+#", val, maxsplit=1)[0]
        out[key] = val
    return out


def load_env(path, environ=None):
    """Apply ENV_KEYS from a .env file without overriding variables already set; other keys are ignored.
    Returns the keys it set (names only)."""
    environ = os.environ if environ is None else environ
    try:
        text = Path(path).read_text(encoding="utf-8-sig")  # -sig: PowerShell-written files carry a BOM
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        return []
    added = []
    for key, val in parse_env(text).items():
        if key in ENV_KEYS and key not in environ:
            environ[key] = val
            added.append(key)
    return added


# ---------------------------------------------------------------- model loading

_BAR = np.zeros((28, 28), np.uint8)
_BAR[4:24, 13:16] = 255  # a drawn "1": non-blank smoke input


def _smoke(name, head):
    """Run the route's own code path once, so a broken archive fails at startup, not on a request."""
    net = head["net"]
    shape, dtype, width = {"vision": ((1, 784), np.float32, 10),
                           "names": ((1, names_head.CONTEXT), np.int64, len(names_head.VOCAB)),
                           "tabular": ((1, len(tabular_head.FEATURES)), np.float32, 1)}[name]
    out = net.forward(np.zeros(shape, dtype), training=False)
    if out.shape != (1, width):
        raise ValueError(f"network output shape {out.shape} != expected {(1, width)}")
    if name == "vision":
        if not vision_head.predict(net, np.zeros(784, np.uint8))["blank"]:
            raise ValueError("smoke: empty canvas not reported blank")
        r = vision_head.predict(net, _BAR.ravel())
        if r["blank"] or len(r["top3"]) != 3 or not np.isfinite([t["prob"] for t in r["top3"]]).all():
            raise ValueError("smoke: drawn digit gave no finite top-3")
    elif name == "names":
        names_head.sample(net, 1, head["source_set"], 0.85, 0)
    else:
        p = tabular_head.predict(net, head["preproc"], dict.fromkeys(tabular_head.FEATURES, 0),
                                 head["manifest"]["source_kind"])["probability"]
        if not 0 <= p <= 1:
            raise ValueError(f"smoke: probability {p} outside [0, 1]")
        _features(head)


def _diagnostic(label, e, path):
    """Error text with no filesystem path. OS errors carry the path in their repr (backslashes doubled on Windows),
    so they are reduced to their type; anything else is scrubbed of the archive and directory paths as a belt."""
    chain: list[BaseException] = []
    x = e
    while x is not None and not any(x is c for c in chain):
        chain.append(x)
        x = x.__cause__ or x.__context__
    for x in chain:
        if isinstance(x, OSError):
            return f"{label}: unreadable archive ({type(x).__name__})"
    text = f"{label}: {e}"
    for p, repl in ((path, label), (path.parent, "<models>")):
        for form in (str(p), str(p).replace("\\", "\\\\"), p.as_posix()):
            text = text.replace(form, repl)
    return text


def load_head(model_dir, name):
    """-> dict(net, manifest, ...) or dict(error=diagnostic). Diagnostics name the file, never the full path."""
    path = Path(model_dir) / f"{name}.npz"
    label = f"{name}.npz"
    if not path.is_file():
        return {"error": f"{label}: archive not found"}
    try:
        net, manifest, extra = Sequential.load(str(path), name)
        head = {"net": net, "manifest": manifest}
        if name == "names":
            if "preproc.source_names" not in extra:
                raise ValueError("missing preproc.source_names (novelty check needs the source corpus)")
            head["source_set"] = frozenset(extra["preproc.source_names"].tolist())
        elif name == "tabular":
            pre = tabular_head.preproc_from_extra(extra)
            missing = {"means", "scales", "percentile_levels", "percentiles", "medians"} - set(pre)
            if missing:
                raise ValueError(f"missing preproc arrays {sorted(missing)}")
            if manifest.get("source_kind") not in tabular_head.SOURCE_KINDS:
                raise ValueError(f"manifest source_kind must be one of {tabular_head.SOURCE_KINDS}")
            p100 = np.asarray(pre["percentiles"], np.float64)[-1]
            head.update(preproc=pre, caps=dict(zip(tabular_head.FEATURES, (TABULAR_CAP_FACTOR * p100).tolist(), strict=True)))
        if not isinstance(manifest.get("training_source_id"), str):
            raise ValueError("manifest has no training_source_id")
        _smoke(name, head)
        return head
    except Exception as e:  # any failure is that head's diagnostic; the others still load
        return {"error": _diagnostic(label, e, path)}


def _features(head):
    pre = head["preproc"]
    levels = np.asarray(pre["percentile_levels"]).tolist()
    pct = np.asarray(pre["percentiles"], np.float64)
    return [TabularFeature(name=f, kind="count" if f in tabular_head.COUNTS else "duration",
                           median=float(pre["medians"][i]),
                           percentiles={f"p{lv:g}": float(pct[j, i]) for j, lv in enumerate(levels)},
                           input_max=head["caps"][f])
            for i, f in enumerate(tabular_head.FEATURES)]


# ---------------------------------------------------------------- app

class BodyCap:
    """ASGI middleware: 413 for a body over MAX_BODY (by Content-Length, or by counting a chunked body) before the
    app parses it. The body (<= 256 KB) is buffered and replayed to the app."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        length = dict(scope["headers"]).get(b"content-length")
        too_big = length is not None and (not length.isdigit() or int(length) > MAX_BODY)
        chunks, size, more = [], 0, True
        while more and not too_big:
            msg = await receive()
            if msg["type"] == "http.disconnect":
                return
            chunks.append(msg.get("body", b""))
            size += len(chunks[-1])
            more, too_big = msg.get("more_body", False), size > MAX_BODY
        if too_big:
            return await JSONResponse({"detail": f"request body over {MAX_BODY} bytes"}, 413)(scope, receive, send)
        body, sent = b"".join(chunks), False

        async def replay():
            nonlocal sent
            if sent:
                return await receive()  # after the body: pass through (e.g. disconnect)
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        return await self.app(scope, replay, send)


def create_app(model_dir=None, env_file=ROOT / ".env"):
    """model_dir None -> SAEMIL_MODEL_DIR or <root>/models, resolved at startup (after .env is applied)."""

    @asynccontextmanager
    async def lifespan(app):
        if env_file is not None:
            load_env(env_file)
        mdir = model_dir or os.environ.get("SAEMIL_MODEL_DIR") or ROOT / "models"
        app.state.heads = {name: load_head(mdir, name) for name in HEADS}
        yield

    app = FastAPI(title="Saemil Studio", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(BodyCap)  # added first = inner: the Host check runs before any body is read
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(ALLOWED_HOSTS))
    if STATIC.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.exception_handler(RequestValidationError)
    async def invalid(request, exc):
        # The default handler echoes each bad `input`: a NaN/Infinity there fails JSON encoding (500) and echoes
        # whole payloads back. Keep only where and why.
        return JSONResponse(status_code=422, content={"detail": [
            {"type": e.get("type"), "loc": list(e.get("loc", ())), "msg": e.get("msg")} for e in exc.errors()]})

    def ready(name):
        head = app.state.heads[name]
        if "error" in head:
            raise HTTPException(503, detail={"head": name, "error": head["error"]})
        return head

    @app.get("/", response_model=None)
    def index():
        page = STATIC / "index.html"
        if page.is_file():
            return FileResponse(page)
        return {"ui": "not built yet", "message": "Saemil Studio UI is not built yet. The API is under /api.",
                "endpoints": ["GET /api/status", "POST /api/vision", "POST /api/names", "POST /api/tabular",
                              "POST /api/report"]}

    @app.get("/api/status")
    def status() -> StatusResponse:  # plain def: the bridge registry may run `ollama list` (up to 3 s)
        heads = {}
        for name, head in app.state.heads.items():
            if "error" in head:
                heads[name] = HeadStatus(ready=False, error=head["error"])
                continue
            m = head["manifest"]
            heads[name] = HeadStatus(ready=True, source_id=m["training_source_id"],
                                     source_kind=m.get("source_kind", "real" if name == "vision" else None),
                                     features=_features(head) if name == "tabular" else None)
        providers = saemil_bridge.status(saemil_bridge.registry(saemil_bridge.load_config()))
        return StatusResponse(ui_built=(STATIC / "index.html").is_file(), heads=heads, providers=providers)

    @app.post("/api/vision")
    def vision(req: VisionRequest) -> VisionResponse:
        head = ready("vision")
        try:
            return vision_head.predict(head["net"], np.asarray(req.pixels, np.int64))
        except ValueError as e:
            raise HTTPException(422, detail=str(e)) from None

    @app.post("/api/names")
    def names(req: NamesRequest) -> NamesResponse:
        head = ready("names")
        try:
            return names_head.sample(head["net"], req.count, head["source_set"], req.temperature, req.seed)
        except ValueError as e:
            raise HTTPException(422, detail=str(e)) from None

    @app.post("/api/tabular")
    def tabular(req: TabularRequest) -> TabularResponse:
        head = ready("tabular")
        features = req.model_dump()
        for k, v in features.items():
            if v > head["caps"][k]:
                raise HTTPException(422, detail=f"{k} must be <= {head['caps'][k]:g} "
                                                f"({TABULAR_CAP_FACTOR:g}x the largest training value)")
        try:
            out = tabular_head.predict(head["net"], head["preproc"], features, head["manifest"]["source_kind"])
        except ValueError as e:
            raise HTTPException(422, detail=str(e)) from None
        return TabularResponse(**out, source_id=head["manifest"]["training_source_id"])

    @app.post("/api/report")
    async def report(req: ReportRequest) -> ReportResponse:
        fields = {"vision": req.vision, "names": req.names, "tabular": req.tabular}
        r = await saemil_bridge.run_report(req.mode, req.vendor, saemil_bridge.ProviderRequest("report", fields))
        return ReportResponse(text=r.text, provider_used=r.provider_id, mode_used=r.mode_used,
                              fallback_reason=r.fallback_reason, latency_ms=r.latency_ms)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)
