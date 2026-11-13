"""T-501 studio route tests: real models, no network, no real CLIs or vendors."""
import json
import shutil
import socket
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pytest
import uvicorn
from starlette.exceptions import StarletteDeprecationWarning

with warnings.catch_warnings():  # starlette now prefers httpx2 (not installed; install nothing new): httpx still works
    warnings.simplefilter("ignore", StarletteDeprecationWarning)
    from fastapi.testclient import TestClient

import saemil_bridge as B
from heads import names as names_head
from heads import tabular as tabular_head
from heads import vision as vision_head
from saemil import Sequential
from studio import app as studio
from studio.app import create_app, load_env, parse_env

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
SECRET_KEY, SECRET_MODEL = "sk-FAKE-SECRET-studio-0123456789", "fake-model-id-studio-xyz"
SESSION = {"Administrative": 2, "Administrative_Duration": 64.0, "Informational": 0,
           "Informational_Duration": 0.0, "ProductRelated": 18, "ProductRelated_Duration": 720.5}


@pytest.fixture(autouse=True)
def offline_bridge(monkeypatch):
    """No CLI probing and no HTTP: any cloud call fails inside the bridge and falls back."""
    monkeypatch.setenv("SAEMIL_CLI_TOOLS", "none")

    def no_network(*a, **k):
        raise B.BridgeError("http_error")
    monkeypatch.setattr(B, "_post_json", no_network)


def client(model_dir=MODELS, env_file=None):
    # base_url host must pass the strict Host allowlist (D-019); TestClient's default "testserver" would get 400
    return TestClient(create_app(model_dir=model_dir, env_file=env_file), base_url="http://127.0.0.1")


@pytest.fixture(scope="module")
def c():
    with client() as tc:
        yield tc


@pytest.fixture(scope="module")
def digits():
    z = np.load(ROOT / "data" / "prepared" / "vision.npz")
    return z["x_test"][:20].reshape(20, 784), z["y_test"][:20]


@pytest.fixture(scope="module")
def nets():
    return {h: Sequential.load(str(MODELS / f"{h}.npz"), h) for h in ("vision", "names", "tabular")}


# ---------------------------------------------------------------- happy paths

def test_index_serves_ui_offline(c):
    """T-601b: / serves the UI; its assets load with the right types; nothing references another host (offline)."""
    import re
    r = c.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    html = r.text
    assert "<title>Saemil Studio</title>" in html
    types = {"/static/styles.css": "text/css", "/static/centering.js": "text/javascript",
             "/static/app.js": "text/javascript"}
    for path in types:
        assert f'"{path}"' in html, path
    bodies = {"/": html}
    for path, ctype in types.items():
        a = c.get(path)
        assert a.status_code == 200 and a.headers["content-type"].split(";")[0] in (ctype, "application/javascript")
        bodies[path] = a.text
    assert sorted(p.name for p in studio.STATIC.iterdir()) == ["app.js", "centering.js", "index.html", "styles.css"]
    for path, text in bodies.items():
        assert not re.search(r"(?i)(https?:)?//[a-z0-9.-]+\.[a-z]{2,}", text), f"external URL in {path}"
        assert not re.search(r"(?i)@import|url\(\s*['\"]?(?!data:)[a-z]+:", text), f"external import in {path}"
    for js in ("/static/app.js", "/static/centering.js"):  # server text only via textContent (T-502)
        for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
            assert sink not in bodies[js], f"{sink} in {js}"


def test_status(c, nets):
    s = c.get("/api/status").json()
    assert s["ui_built"] is True  # studio/static/index.html exists since Section 6
    for h in ("vision", "names", "tabular"):
        assert s["heads"][h]["ready"] is True and s["heads"][h]["error"] is None
        assert s["heads"][h]["source_id"] == nets[h][1]["training_source_id"]
    assert s["heads"]["tabular"]["source_kind"] == "real"
    feats = s["heads"]["tabular"]["features"]
    pre = tabular_head.preproc_from_extra(nets["tabular"][2])
    assert [f["name"] for f in feats] == tabular_head.FEATURES
    for i, f in enumerate(feats):
        assert f["median"] == pytest.approx(pre["medians"][i])
        assert f["percentiles"]["p100"] == pytest.approx(pre["percentiles"][-1, i])
        assert f["percentiles"]["p0"] == pytest.approx(pre["percentiles"][0, i])
        assert f["input_max"] == pytest.approx(10 * pre["percentiles"][-1, i])
        assert f["kind"] == ("count" if f["name"] in tabular_head.COUNTS else "duration")
    p = s["providers"]
    assert p["offline"]["available"] is True and p["cli"] == []
    assert {x["id"] for x in p["cloud"]} == {"claude", "gemini", "openai", "deepseek"}


def test_vision_real_mnist(c, digits, nets):
    xs, ys = digits
    correct = 0
    for x, y in zip(xs, ys, strict=True):
        r = c.post("/api/vision", json={"pixels": x.tolist()})
        assert r.status_code == 200
        body = r.json()
        assert body == vision_head.predict(nets["vision"][0], x)  # head shape, unchanged (D-015)
        assert set(body) == {"blank", "digit", "top3"} and set(body["top3"][0]) == {"digit", "prob"}
        correct += body["digit"] == y
    assert correct >= 17


def test_vision_blank_is_200(c):
    for px in ([0] * 784, [0] * 783 + [255]):  # empty and a stray tap (D-010)
        r = c.post("/api/vision", json={"pixels": px})
        assert r.status_code == 200
        assert r.json() == {"blank": True, "message": vision_head.BLANK_MESSAGE}


def test_names_seeded_and_novel(c, nets):
    body = {"count": 5, "temperature": 0.85, "seed": 7}
    a, b = c.post("/api/names", json=body), c.post("/api/names", json=body)
    assert a.status_code == 200 and a.json() == b.json()
    src = frozenset(nets["names"][2]["preproc.source_names"].tolist())
    assert a.json() == names_head.sample(nets["names"][0], 5, src, 0.85, 7)
    assert a.json()["requested"] == 5 and all(n not in src for n in a.json()["names"])
    assert c.post("/api/names", json={"count": 3}).status_code == 200  # seed + temperature optional


def test_tabular(c, nets):
    r = c.post("/api/tabular", json=SESSION)
    assert r.status_code == 200
    body = r.json()
    net, man, extra = nets["tabular"]
    want = tabular_head.predict(net, tabular_head.preproc_from_extra(extra), SESSION, "real")["probability"]
    assert body == {"probability": pytest.approx(want), "source_id": man["training_source_id"], "source_kind": "real"}
    assert 0 < body["probability"] < 1


def test_report_offline(c):
    v = c.post("/api/vision", json={"pixels": np.load(ROOT / "data/prepared/vision.npz")["x_test"][0].ravel().tolist()})
    fields = {"vision": v.json(), "names": c.post("/api/names", json={"count": 2, "seed": 1}).json(),
              "tabular": c.post("/api/tabular", json=SESSION).json()}
    r = c.post("/api/report", json={"mode": "offline", **fields})
    assert r.status_code == 200
    body = r.json()
    assert body["provider_used"] == "offline" and body["mode_used"] == "offline" and body["fallback_reason"] is None
    assert body["text"] == B.offline_report(fields)


@pytest.mark.parametrize("req, reason", [({"mode": "cloud", "vendor": "claude"}, "missing_key"),
                                         ({"mode": "cli"}, "no_on_device_cli"),
                                         ({"mode": "cli", "vendor": "ollama"}, "unknown_provider")])
def test_report_fallback(c, req, reason):
    r = c.post("/api/report", json={**req, "tabular": {"probability": 0.24, "source_kind": "real"}})
    assert r.status_code == 200
    body = r.json()
    assert body["provider_used"] == "offline" and body["mode_used"] == "offline"
    assert body["fallback_reason"] == reason and "24.0%" in body["text"]


def test_report_cloud_failure_falls_back(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET_KEY)
    monkeypatch.setenv("ANTHROPIC_MODEL", SECRET_MODEL)
    with client() as tc:
        body = tc.post("/api/report", json={"mode": "cloud", "vendor": "claude"}).json()
    assert body["provider_used"] == "offline" and body["fallback_reason"] == "http_error"


# ---------------------------------------------------------------- validation

def px(n=784, v=0):
    return [v] * n


@pytest.mark.parametrize("payload", [
    {"pixels": px(783)}, {"pixels": px(785)}, {"pixels": []}, {},
    {"pixels": px(783) + [256]}, {"pixels": px(783) + [-1]}, {"pixels": px(783) + [1.5]},
    {"pixels": px(783) + ["7"]}, {"pixels": px(783) + [True]}, {"pixels": px(783) + [None]},
    {"pixels": "0" * 784}, {"pixels": px(), "extra": 1},
])
def test_vision_validation(c, payload):
    assert c.post("/api/vision", json=payload).status_code == 422


@pytest.mark.parametrize("payload", [
    {"count": 0}, {"count": 11}, {"count": 2.0}, {"count": "3"}, {"count": True}, {},
    {"count": 3, "temperature": 0.19}, {"count": 3, "temperature": 1.51}, {"count": 3, "temperature": "1"},
    {"count": 3, "seed": 1.5}, {"count": 3, "seed": -1}, {"count": 3, "seed": "4"}, {"count": 3, "seed": 2**64},
    {"count": 3, "x": 1},
])
def test_names_validation(c, payload):
    assert c.post("/api/names", json=payload).status_code == 422


@pytest.mark.parametrize("temperature", ["NaN", "Infinity", "-Infinity"])
def test_names_nan_temperature(c, temperature):  # Python's JSON parser accepts these literals
    r = c.post("/api/names", content=f'{{"count": 3, "temperature": {temperature}}}',
               headers={"content-type": "application/json"})
    assert r.status_code == 422


def _tab(**change):
    d = dict(SESSION, **change)
    return {k: v for k, v in d.items() if v is not ...}


@pytest.mark.parametrize("payload", [
    _tab(Administrative=...), _tab(Extra=1), {}, _tab(Administrative=-1), _tab(Administrative_Duration=-0.5),
    _tab(Administrative=2.5), _tab(Administrative=2.0), _tab(Informational="3"), _tab(ProductRelated=True),
    _tab(ProductRelated_Duration=None), _tab(ProductRelated_Duration="720"),
    _tab(ProductRelated=7051), _tab(Administrative_Duration=27205.1), _tab(ProductRelated=10**400),
])
def test_tabular_validation(c, payload):
    assert c.post("/api/tabular", json=payload).status_code == 422


def test_tabular_cap_boundary(c):
    assert c.post("/api/tabular", json=_tab(ProductRelated=7050)).status_code == 200  # 10 x train max 705
    r = c.post("/api/tabular", json=_tab(ProductRelated=7051))
    assert r.status_code == 422 and "ProductRelated" in r.json()["detail"]


@pytest.mark.parametrize("value", ["NaN", "Infinity"])
def test_tabular_nan(c, value):
    body = json.dumps(SESSION).replace("720.5", value)
    r = c.post("/api/tabular", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422


@pytest.mark.parametrize("payload", [{"mode": "online"}, {"mode": "cloud", "vendor": "acme"},
                                     {"mode": "offline", "vision": [1]}, {"mode": "offline", "x": 1}])
def test_report_validation(c, payload):
    assert c.post("/api/report", json=payload).status_code == 422


def test_head_valueerror_maps_to_422(c, monkeypatch):
    def boom(*a, **k):
        raise ValueError("bad input from head")
    monkeypatch.setattr(vision_head, "predict", boom)
    r = c.post("/api/vision", json={"pixels": px()})
    assert r.status_code == 422 and r.json()["detail"] == "bad input from head"


# ---------------------------------------------------------------- secrets

def test_no_secrets_in_any_response(monkeypatch, digits):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET_KEY)
    monkeypatch.setenv("ANTHROPIC_MODEL", SECRET_MODEL)
    monkeypatch.setenv("OPENAI_API_KEY", SECRET_KEY + "-openai")
    with client() as tc:
        status = tc.get("/api/status")
        assert {p["id"]: p["available"] for p in status.json()["providers"]["cloud"]}["claude"] is True
        responses = [status, tc.get("/"),
                     tc.post("/api/vision", json={"pixels": digits[0][0].tolist()}),
                     tc.post("/api/names", json={"count": 3, "seed": 1}),
                     tc.post("/api/tabular", json=SESSION),
                     tc.post("/api/report", json={"mode": "cloud", "vendor": "claude"}),
                     tc.post("/api/report", json={"mode": "cloud", "vendor": "openai"}),
                     tc.post("/api/report", json={"mode": "offline"})]
    for r in responses:
        assert r.status_code == 200
        for s in (SECRET_KEY, SECRET_MODEL, str(ROOT), str(MODELS)):
            assert s not in r.text


# ---------------------------------------------------------------- missing / broken artifacts

@pytest.mark.parametrize("damage, expect", [("delete", "archive not found"), ("garbage", "unreadable archive"),
                                            ("wrong_head", "head mismatch")])
def test_missing_or_broken_artifact(tmp_path, damage, expect, digits):
    for h in ("vision", "names", "tabular"):
        shutil.copy(MODELS / f"{h}.npz", tmp_path / f"{h}.npz")
    target = tmp_path / "tabular.npz"
    if damage == "delete":
        target.unlink()
    elif damage == "garbage":
        target.write_bytes(b"not a zip archive")
    else:
        shutil.copy(MODELS / "vision.npz", target)
    with client(tmp_path) as tc:
        r = tc.post("/api/tabular", json=SESSION)
        assert r.status_code == 503
        detail = r.json()["detail"]
        assert detail["head"] == "tabular" and detail["error"].startswith("tabular.npz: ") and expect in detail["error"]
        assert str(tmp_path) not in r.text
        assert tc.post("/api/vision", json={"pixels": digits[0][0].tolist()}).status_code == 200
        assert tc.post("/api/names", json={"count": 2, "seed": 3}).status_code == 200
        assert tc.post("/api/report", json={"mode": "offline"}).status_code == 200
        s = tc.get("/api/status")
        assert s.status_code == 200 and str(tmp_path) not in s.text
        h = s.json()["heads"]
        assert h["tabular"] == {"ready": False, "source_id": None, "source_kind": None,
                                "error": detail["error"], "features": None}
        assert h["vision"]["ready"] and h["names"]["ready"]


def test_names_archive_without_source_set_is_rejected(tmp_path):
    net, man, extra = Sequential.load(str(MODELS / "names.npz"), "names")
    net.save(tmp_path / "names.npz", {k: v for k, v in man.items() if k not in ("layers", "extra_arrays")},
             {"preproc.vocab": extra["preproc.vocab"]})
    with client(tmp_path) as tc:  # vision + tabular absent too
        r = tc.post("/api/names", json={"count": 2})
        assert r.status_code == 503 and "source_names" in r.json()["detail"]["error"]
        assert tc.post("/api/vision", json={"pixels": px()}).json()["detail"]["error"] == "vision.npz: archive not found"


# ---------------------------------------------------------------- .env

def test_parse_env():
    text = ("# comment\n\nA=1\n B = two words \nC=\"quoted # not comment\"\nD='single'\nE=val # trailing\n"
            "F=\nnot a line\n=nokey\nG=a=b\n  # indented comment\nH=\"v\" # comment\nexport I='x y' #c\n"
            "J=\"unterminated\nK=\"a\"junk\nL=x#not-a-comment\n")
    assert parse_env(text) == {"A": "1", "B": "two words", "C": "quoted # not comment", "D": "single",
                               "E": "val", "F": "", "G": "a=b", "H": "v", "I": "x y", "L": "x#not-a-comment"}


def test_env_allowlist_matches_example():
    example = set(parse_env((ROOT / ".env.example").read_text(encoding="utf-8")))
    assert example <= studio.ENV_KEYS and studio.ENV_KEYS - example == {"OLLAMA_HOST", "SAEMIL_MODEL_DIR"}


def test_load_env_never_overrides(tmp_path):
    f = tmp_path / ".env"
    f.write_text("\ufeffGEMINI_MODEL=fromfile\nexport GEMINI_API_KEY=\"fresh\" # c\nPATH=evil\nOTHER=x\n",
                 encoding="utf-8")  # BOM, as PowerShell writes
    env = {"GEMINI_MODEL": "already"}
    assert load_env(f, env) == ["GEMINI_API_KEY"]  # PATH / OTHER are not allowlisted
    assert env == {"GEMINI_MODEL": "already", "GEMINI_API_KEY": "fresh"}
    assert load_env(tmp_path / "missing.env", env) == []


def test_env_file_applied_at_startup(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    (tmp_path / ".env").write_text(f"DEEPSEEK_API_KEY={SECRET_KEY}\nDEEPSEEK_MODEL={SECRET_MODEL}\n")
    try:
        with client(env_file=tmp_path / ".env") as tc:
            r = tc.get("/api/status")
        cloud = {p["id"]: p["available"] for p in r.json()["providers"]["cloud"]}
        assert cloud["deepseek"] is True and SECRET_KEY not in r.text and SECRET_MODEL not in r.text
    finally:  # load_env wrote os.environ directly
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)


# ---------------------------------------------------------------- live server: concurrency + slow provider

@pytest.fixture(scope="module")
def live():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(model_dir=MODELS, env_file=None), log_level="warning"))
    t = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    t.start()
    deadline = time.monotonic() + 30
    while not server.started:
        assert time.monotonic() < deadline and t.is_alive(), "server did not start"
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        t.join(10)
        sock.close()
        assert not t.is_alive()


def _jobs(digits):
    xs, _ = digits
    jobs = [("/api/vision", {"pixels": x.tolist()}) for x in xs[:8]]
    jobs += [("/api/names", {"count": 4, "temperature": 0.9, "seed": s}) for s in range(8)]
    jobs += [("/api/tabular", dict(SESSION, ProductRelated=n)) for n in range(0, 80, 10)]
    return jobs


def test_concurrent_predictions_identical(live, digits):
    jobs = _jobs(digits)
    with httpx.Client(base_url=live, timeout=30) as h:
        want = [h.post(p, json=b).json() for p, b in jobs]
        with ThreadPoolExecutor(16) as pool:
            for _ in range(3):
                got = list(pool.map(lambda j: h.post(j[0], json=j[1]), jobs * 2))
                assert all(r.status_code == 200 for r in got)
                assert [r.json() for r in got] == want * 2


class SlowCLI:
    mode, on_device, id, label, timeout_s = "cli", True, "slowcli", "Slow CLI (test)", 30

    def __init__(self, seconds):
        self.seconds = seconds

    def available(self):
        return True

    def generate(self, req):
        time.sleep(self.seconds)
        return "slow provider text"


def test_slow_provider_does_not_block_predictions(live, digits, monkeypatch):
    monkeypatch.setattr(B, "registry", lambda config: {"cli:slowcli": SlowCLI(4.0)})
    out = {}

    def run_report():
        with httpx.Client(base_url=live, timeout=30) as h:
            out["r"] = h.post("/api/report", json={"mode": "cli"})

    t = threading.Thread(target=run_report)
    t0 = time.monotonic()
    t.start()
    time.sleep(0.5)
    with httpx.Client(base_url=live, timeout=30) as h:
        for path, body in _jobs(digits):
            s = time.monotonic()
            assert h.post(path, json=body).status_code == 200
            assert time.monotonic() - s < 1.5
        assert h.get("/api/status").status_code == 200
    assert t.is_alive(), "report finished before predictions; the test proved nothing"
    t.join(30)
    r = out["r"].json()
    assert r["text"] == "slow provider text" and r["provider_used"] == "slowcli" and r["fallback_reason"] is None
    assert time.monotonic() - t0 >= 4.0


# ---------------------------------------------------------------- T-503 rework (D-019)

def _no_paths(text, *extra):
    home = str(Path.home())
    for s in (str(ROOT), ROOT.as_posix(), str(ROOT).replace("\\", "\\\\"), home, home.replace("\\", "\\\\"),
              Path.home().name, *extra):
        assert s not in text, s


def test_oserror_diagnostic_has_no_path(tmp_path, monkeypatch, digits):
    for h in ("vision", "names", "tabular"):
        shutil.copy(MODELS / f"{h}.npz", tmp_path / f"{h}.npz")
    real = np.load

    def denied(path, *a, **k):
        if "tabular" in str(path):
            raise PermissionError(13, "Permission denied", str(path))
        return real(path, *a, **k)
    monkeypatch.setattr(np, "load", denied)
    with client(tmp_path) as tc:
        r, s = tc.post("/api/tabular", json=SESSION), tc.get("/api/status")
        assert tc.post("/api/vision", json={"pixels": digits[0][0].tolist()}).status_code == 200
    assert r.status_code == 503 and r.json()["detail"]["error"] == "tabular.npz: unreadable archive (PermissionError)"
    assert s.status_code == 200 and s.json()["heads"]["tabular"]["error"] == r.json()["detail"]["error"]
    for text in (r.text, s.text):
        _no_paths(text, str(tmp_path), str(tmp_path).replace("\\", "\\\\"), tmp_path.as_posix())


def test_diagnostic_scrubs_both_path_forms(tmp_path):
    path = tmp_path / "vision.npz"
    for form in (str(path), str(path).replace("\\", "\\\\"), path.as_posix(), str(tmp_path)):
        out = studio._diagnostic("vision.npz", ValueError(f"bad thing at {form!s} here"), path)
        assert str(tmp_path) not in out and tmp_path.as_posix() not in out and out.startswith("vision.npz: ")


@pytest.mark.parametrize("drop", ["preproc.medians", "preproc.means"])
def test_tabular_missing_preproc_array(tmp_path, drop, digits):
    for h in ("vision", "names"):
        shutil.copy(MODELS / f"{h}.npz", tmp_path / f"{h}.npz")
    net, man, extra = Sequential.load(str(MODELS / "tabular.npz"), "tabular")
    net.save(tmp_path / "tabular.npz", {k: v for k, v in man.items() if k not in ("layers", "extra_arrays")},
             {k: v for k, v in extra.items() if k != drop})
    with client(tmp_path) as tc:
        s = tc.get("/api/status")
        assert s.status_code == 200
        err = s.json()["heads"]["tabular"]["error"]
        assert err.startswith("tabular.npz: ") and drop.removeprefix("preproc.") in err
        assert tc.post("/api/tabular", json=SESSION).status_code == 503
        assert tc.post("/api/vision", json={"pixels": digits[0][0].tolist()}).status_code == 200
        assert tc.post("/api/names", json={"count": 2}).status_code == 200
        _no_paths(s.text, str(tmp_path))


def test_smoke_runs_route_paths(monkeypatch):
    calls = []
    for mod, fn in ((vision_head, "predict"), (names_head, "sample"), (tabular_head, "predict")):
        orig = getattr(mod, fn)
        monkeypatch.setattr(mod, fn, lambda *a, _o=orig, _n=f"{mod.HEAD_ID}.{fn}", **k: calls.append(_n) or _o(*a, **k))
    heads = {h: studio.load_head(MODELS, h) for h in ("vision", "names", "tabular")}
    assert all("error" not in h for h in heads.values())
    assert calls.count("vision.predict") == 2 and calls.count("names.sample") == 1 and calls.count("tabular.predict") == 1


def test_foreign_host_rejected(c):
    for host in ("evil.example", "attacker.test:8000", "127.0.0.1.nip.io"):
        assert c.get("/api/status", headers={"host": host}).status_code == 400
    for host in ("127.0.0.1", "127.0.0.1:59999", "localhost", "localhost:8000"):
        assert c.get("/api/status", headers={"host": host}).status_code == 200


def test_foreign_host_rejected_live(live):
    port = live.rsplit(":", 1)[1]
    with httpx.Client(timeout=10) as h:
        assert h.get(f"{live}/api/status").status_code == 200
        assert h.get(f"http://localhost:{port}/api/status").status_code == 200
        assert h.get(f"{live}/api/status", headers={"host": f"rebind.example:{port}"}).status_code == 400


def test_docs_disabled(c):
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert c.get(path).status_code == 404


def test_body_cap(c):
    big = b'{"pixels": [' + b"0," * (512 * 1024) + b'0]}'  # ~1 MB
    assert len(big) > 1_000_000
    r = c.post("/api/vision", content=big, headers={"content-type": "application/json"})
    assert r.status_code == 413
    ok = json.dumps({"pixels": [0] * 784}).encode()
    assert c.post("/api/vision", content=ok, headers={"content-type": "application/json"}).status_code == 200


def test_body_cap_chunked(live):
    def chunks(n):
        for _ in range(n):
            yield b" " * 65536
        yield b'{"count": 2}'
    with httpx.Client(base_url=live, timeout=30) as h:
        r = h.post("/api/names", content=chunks(16), headers={"content-type": "application/json"})  # ~1 MB, chunked
        assert r.status_code == 413
        r = h.post("/api/names", content=chunks(1), headers={"content-type": "application/json"})  # 64 KB, chunked
        assert r.status_code == 200
