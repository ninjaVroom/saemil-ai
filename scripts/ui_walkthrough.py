"""T-601b browser walkthrough: real models, real server, Edge via Playwright (global Python).

Run from the project root:  python scripts/ui_walkthrough.py
Starts uvicorn (.venv Python) on 127.0.0.1 with an ephemeral port, drives the UI, saves screenshots to
docs/screenshots/, and always stops the browser and the server. Exits non-zero on the first failed check.
"""
import json
import socket
import subprocess
import sys
import time
import urllib.request
from itertools import pairwise
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "docs" / "screenshots"
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
VIEWS = ("vision", "names", "tabular", "results", "providers")
LOG: list[tuple[str, str]] = []


def check(cond, what):
    LOG.append(("PASS" if cond else "FAIL", what))
    print(("PASS  " if cond else "FAIL  ") + what, flush=True)
    if not cond:
        raise SystemExit(f"check failed: {what}")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(port):
    proc = subprocess.Popen([str(VENV_PY), "-m", "uvicorn", "studio.app:app", "--host", "127.0.0.1", "--port", str(port)],
                            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=2) as r:
                st = json.load(r)
            return proc, st
        except OSError:
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else b""
                raise SystemExit("server exited: " + out.decode(errors="replace")) from None
            time.sleep(0.5)
    proc.kill()
    raise SystemExit("server did not start")


def stroke(page, pts):
    """Mouse-draw a polyline given in 0..1 canvas coordinates."""
    box = page.locator("#pad").bounding_box()
    xy = [(box["x"] + x * box["width"], box["y"] + y * box["height"]) for x, y in pts]
    page.mouse.move(*xy[0])
    page.mouse.down()
    for _, b in pairwise(xy):
        page.mouse.move(*b, steps=8)
    page.mouse.up()


def shot(page, name):
    """Whole page in one tall viewport, so the pinned toolbar sits at its natural place instead of over content."""
    vp = page.viewport_size
    page.set_viewport_size({"width": vp["width"], "height": page.evaluate("document.documentElement.scrollHeight")})
    page.screenshot(path=SHOTS / name)
    page.set_viewport_size(vp)


def no_overflow(page, label):
    w = page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
    check(w[0] <= w[1], f"{label}: no horizontal overflow (scrollWidth {w[0]} <= {w[1]})")


def go(page, view):
    """Switch view with the in-page toolbar/task controls (the same buttons a user presses)."""
    btn = {"vision": "Draw digit", "names": "Generate names", "tabular": "Edit session"}.get(view)
    if btn:
        page.locator(".toolbar").get_by_role("button", name=btn).click()
    elif view == "results":
        page.locator('[data-view="tabular"] [data-go="results"]').evaluate("b => b.click()")
    elif view == "providers":
        page.locator('[data-go="providers"]').evaluate("b => b.click()")
    expect(page.locator(f'[data-view="{view}"]')).to_be_visible()


def new_page(ctx, errors, foreign, allow_failed_loads=False):
    """Page with console/page errors and non-localhost requests recorded. Fault-injection pages expect the
    browser's own "Failed to load resource" line for the mocked 503s/aborts; every other error still counts."""
    page = ctx.new_page()

    def on_console(m):
        if m.type == "error" and not (allow_failed_loads and "Failed to load resource" in m.text):
            errors.append(m.text)
    page.on("console", on_console)
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("request", lambda r: urlparse(r.url).hostname not in ("127.0.0.1", None)
            and not r.url.startswith("data:") and foreign.append(r.url))
    return page


def seeded(browser, base, value, **kw):
    """Context whose localStorage history starts as `value` (a JSON string)."""
    ctx = browser.new_context(**({"viewport": {"width": 1280, "height": 900}} | kw))
    ctx.add_init_script(f"try {{ if (!sessionStorage.getItem('seeded')) {{ sessionStorage.setItem('seeded', '1');"
                        f" localStorage.setItem('saemil.history.v1', {json.dumps(value)}); }} }} catch (e) {{}}")
    return ctx


def main():
    SHOTS.mkdir(parents=True, exist_ok=True)
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    proc, status = start_server(port)
    print(f"server on {base}; heads ready: " + ", ".join(h for h, s in status["heads"].items() if s["ready"]))
    errors: list[str] = []
    foreign: list[str] = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(channel="msedge", headless=True)
            try:
                ctx = browser.new_context(viewport={"width": 1280, "height": 900})
                saved = run(new_page(ctx, errors, foreign), base, ctx)
                ctx.close()
                storage_unavailable(browser, base, errors, foreign)
                faults(browser, base, saved, errors, foreign)
                touch_drawing(browser, base, errors, foreign)
            finally:
                browser.close()
        check(not errors, f"no console errors ({errors[:3]})")
        check(not foreign, f"no non-localhost requests ({foreign[:3]})")
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
        print(f"server stopped (exit {proc.poll()})")
    print(f"\n{sum(1 for s, _ in LOG if s == 'PASS')} checks passed; screenshots in {SHOTS}")
    for p in sorted(SHOTS.glob("*.png")):
        print("  " + p.relative_to(ROOT).as_posix())


def run(page, base, ctx):
    page.goto(base + "/")
    for h in ("Vision", "Names", "Tabular"):
        expect(page.locator(f"#badge-{h.lower()}")).to_have_text(f"{h}: ready")
    check(True, "status badges ready for all three heads")
    expect(page.get_by_role("heading", level=1)).to_have_text("What would you like to try?")
    shot(page, "desktop-tasks.png")
    for width in (768, 390):
        page.set_viewport_size({"width": width, "height": 900})
        no_overflow(page, f"{width}px tasks")
    shot(page, "mobile-tasks.png")
    page.set_viewport_size({"width": 1280, "height": 900})
    no_overflow(page, "1280px tasks")

    # ---- vision: draw a 7 with the mouse
    page.get_by_role("button", name="Recognize a digit").click()
    expect(page.locator("#h-vision")).to_be_focused()
    check(True, "task switch moves focus to the view h1")
    expect(page.locator("#pad")).to_have_attribute("aria-label", "Drawing area, empty")
    stroke(page, [(0.28, 0.25), (0.72, 0.25), (0.45, 0.82)])
    expect(page.locator("#vision-out .digit")).to_be_visible()
    digit = page.locator("#vision-out .digit").inner_text()
    conf = page.locator("#vision-out .confidence").inner_text()
    check(page.locator("#vision-out .bar-row").count() == 3, f"drawn digit predicted: {digit} ({conf}), 3 bars")
    expect(page.locator("#pad")).to_have_attribute("aria-label", "Drawing area, contains a drawing")
    shot(page, "desktop-vision.png")

    page.get_by_role("button", name="Clear drawing").click()
    expect(page.locator("#vision-out")).to_have_text("Your result appears here.")
    check(True, "Clear drawing resets canvas and result")

    # A single tap leaves no ink; the server's blank rule (D-010) answers and the UI shows its message verbatim.
    stroke(page, [(0.5, 0.5), (0.5, 0.5)])
    expect(page.locator("#vision-out")).to_contain_text("Please draw a digit in the canvas first.")
    expect(page.locator("#vision-out")).to_contain_text("Waiting for a drawing")
    check(page.locator("#vision-out .digit").count() == 0, "blank canvas: server message shown verbatim, no digit")
    shot(page, "desktop-vision-blank.png")
    page.get_by_role("button", name="Clear drawing").click()
    # D-023: a short dash (2% of the canvas; with the 20/280 pen its bbox stays under 3/28 of the side) is blank
    stroke(page, [(0.49, 0.5), (0.51, 0.5)])
    expect(page.locator("#vision-out")).to_contain_text("Please draw a digit in the canvas first.")
    check(page.locator("#vision-out .digit").count() == 0, "short dash: server blank message, no digit (D-023)")

    # example-digit picker, keyboard only
    page.locator("#example-digit").focus()
    page.keyboard.press("End")  # last option: 9
    page.keyboard.press("Tab")
    expect(page.locator("#use-example")).to_be_focused()
    page.keyboard.press("Enter")
    expect(page.locator("#vision-out .digit")).to_be_visible()
    ex = page.locator("#vision-out .digit").inner_text()
    check(ex == "9", f"keyboard example digit 9 classified as {ex}")

    # drawing after an example replaces it: only the user's stroke is in the buffer that is centred and posted
    stroke(page, [(0.5, 0.15), (0.5, 0.85)])
    expect(page.locator("#vision-out .digit")).to_have_text("1")
    outside = page.evaluate("""(() => { const c = document.getElementById('pad'), d = c.getContext('2d')
        .getImageData(0, 0, c.width, c.height).data; let n = 0;
      for (let y = 0; y < c.height; y++) for (let x = 0; x < c.width; x++)
        if ((x < 120 || x > 160) && d[(y * c.width + x) * 4 + 3]) n++;
      return n; })()""")
    check(outside == 0, f"example 9 then a drawn 1: no example ink left in the canvas buffer ({outside} px), predicted 1")
    shot(page, "desktop-vision-after-example.png")
    page.get_by_role("button", name="Clear drawing").click()

    # ---- names
    go(page, "names")
    page.fill("#n-count", "5")
    page.fill("#n-temp", "0.85")
    page.fill("#n-seed", "42")
    page.locator("#n-seed").press("Enter")
    expect(page.locator("#names-out .meta")).to_be_visible()
    meta = page.locator("#names-out .meta").inner_text()
    first = page.locator("#names-out .name").all_inner_texts()
    check(meta.endswith("/ temperature 0.85 / seed 42"), f"names generated: {first} — {meta}")
    shot(page, "desktop-names.png")
    page.get_by_role("button", name="Generate names", exact=True).last.click()
    expect(page.locator("#names-submit")).to_have_text("Generate names")
    expect(page.locator("#names-submit")).to_be_enabled()
    check(page.locator("#names-out .name").all_inner_texts() == first, "same seed and settings reproduce the list")
    page.fill("#n-count", "11")
    page.locator("#n-count").press("Enter")
    expect(page.locator("#n-count-err")).to_have_text("Enter a whole number from 1 to 10.")
    check(page.locator("#n-count").get_attribute("aria-invalid") == "true", "count 11 shows the field error")
    page.fill("#n-count", "5")
    expect(page.locator("#n-count-err")).to_be_visible()  # stays until the next submit validates
    page.fill("#n-seed", "")
    page.locator("#n-seed").press_sequentially("e")  # badInput: the box reports "" but must not send seed=null
    page.locator("#n-seed").press("Enter")
    expect(page.locator("#n-seed-err")).to_be_visible()
    check(page.locator("#n-count-err").is_hidden(), "bad seed text shows the seed error; valid count clears its error")
    page.fill("#n-seed", "42")

    # ---- tabular
    go(page, "tabular")
    expect(page.locator("#tabular-out .metric")).to_contain_text("%")
    p0 = page.locator("#tabular-out .metric").inner_text()
    badge = page.locator("#tabular-out .badge").inner_text()
    check(page.locator("input[type=range]").count() == 6, f"six sliders; initial estimate {p0}, badge '{badge}'")
    rng = page.locator("#f-ProductRelated-range")
    rng.focus()
    for _ in range(15):
        page.keyboard.press("ArrowRight")
    metric = page.locator("#tabular-out .metric")
    expect(metric).not_to_have_text(p0)
    expect(page.locator("#tabular-out .updating")).to_have_text("")
    p1 = metric.inner_text()
    vt = rng.get_attribute("aria-valuetext")
    check(vt.endswith(" pages") and p1 != p0, f"slider moved: aria-valuetext '{vt}', estimate changed {p0} -> {p1}")
    cap = float(page.locator("#f-ProductRelated_Duration").get_attribute("max"))
    page.fill("#f-ProductRelated_Duration", str(int(cap) + 10))
    expect(page.locator("#f-ProductRelated_Duration-err")).to_contain_text("Enter a value from 0 to")
    expect(page.locator("#tabular-out .updating")).to_have_text("")
    check(True, "number box above input_max shows the field error, no stuck 'Updating…'")
    page.fill("#f-ProductRelated_Duration", "3000")
    expect(page.locator("#f-ProductRelated_Duration-val")).to_have_text("3000.0 seconds")
    expect(page.locator("#f-ProductRelated_Duration-err")).to_be_hidden()
    expect(metric).not_to_have_text(p1)
    expect(page.locator("#tabular-out .updating")).to_have_text("")
    p2 = metric.inner_text()
    check(p2 != p1, f"typed 3000 seconds: error hidden, estimate changed {p1} -> {p2}")
    big = page.locator("#f-ProductRelated-range")
    page.fill("#f-ProductRelated", str(int(float(big.get_attribute("max"))) + 5))
    check(big.input_value() == big.get_attribute("max"), "value above p99 pins the slider thumb at its max")
    expect(page.locator("#tabular-out .updating")).to_have_text("")
    shot(page, "desktop-tabular.png")

    # ---- explanation (offline)
    check(page.locator("#provider").input_value() == "offline:", "provider defaults to Offline Template")
    page.get_by_role("button", name="Explain results").click()
    expect(page.locator("#explanation .ex-text")).not_to_be_empty()
    prov = page.locator("#explanation .provenance").inner_text()
    check(prov == "via Offline Template", f"offline explanation rendered, provenance '{prov}'")
    check(page.locator("#cards .card").count() == 3, "three result cards on the overview")
    shot(page, "desktop-report.png")
    go(page, "names")
    page.fill("#n-seed", "7")
    page.locator("#n-seed").press("Enter")
    expect(page.locator("#names-out .meta")).to_contain_text("seed 7")
    go(page, "results")
    expect(page.locator("#explanation")).to_contain_text("Based on earlier results")
    check(True, "changing a head result marks the explanation 'Based on earlier results'")
    go(page, "providers")
    expect(page.locator(".pcard").first).to_contain_text("Selected")
    shot(page, "desktop-providers.png")

    # ---- viewports: 1280, 768, 390
    for width in (1280, 768, 390):
        page.set_viewport_size({"width": width, "height": 900})
        for v in VIEWS:
            go(page, v)
            no_overflow(page, f"{width}px {v}")
    page.set_viewport_size({"width": 768, "height": 1000})
    go(page, "results")
    shot(page, "tablet-report.png")

    page.set_viewport_size({"width": 390, "height": 844})
    go(page, "vision")
    w = page.locator("#pad").bounding_box()["width"]
    check(abs(w - 350) <= 1, f"mobile canvas width = viewport - 40 ({w:.0f}px)")
    page.get_by_role("button", name="Clear drawing").click()
    stroke(page, [(0.5, 0.15), (0.5, 0.85)])
    expect(page.locator("#vision-out .digit")).to_be_visible()
    shot(page, "mobile-vision.png")
    for v in ("names", "tabular", "results"):
        go(page, v)
        shot(page, f"mobile-{'report' if v == 'results' else v}.png")
    hb = page.get_by_role("button", name="New session").last.bounding_box()["height"]
    check(hb >= 48, f"mobile buttons have a 48 px hit area ({hb:.0f}px)")
    tall = page.evaluate("[...document.querySelectorAll('.toolbar > *:not(label)')].map(e => e.getBoundingClientRect().height)")
    check(all(48 <= h <= 64 for h in tall), f"390 px toolbar controls are 48-64 px tall ({[round(h) for h in tall]})")
    page.get_by_role("button", name="Open sessions").click()
    expect(page.locator("#sidebar")).to_be_visible()
    expect(page.locator("#new-session")).to_be_focused()
    page.screenshot(path=SHOTS / "mobile-drawer.png")
    page.keyboard.press("Shift+Tab")
    expect(page.locator("#clear-history")).to_be_focused()
    page.keyboard.press("Tab")
    expect(page.locator("#new-session")).to_be_focused()
    check(True, "drawer: Tab and Shift+Tab wrap inside the drawer")
    page.keyboard.press("Escape")
    expect(page.locator("#sidebar")).to_be_hidden()
    expect(page.get_by_role("button", name="Open sessions")).to_be_focused()
    check(True, "mobile drawer opens with focus inside and Esc closes it, focus returns to the menu button")
    no_overflow(page, "390px results")

    # ---- history persists after reload; clear works
    page.set_viewport_size({"width": 1280, "height": 900})
    titles_before = page.locator("#history .h-title").all_inner_texts()
    page.reload()
    expect(page.locator("#badge-vision")).to_have_text("Vision: ready")
    titles_after = page.locator("#history .h-title").all_inner_texts()
    check(titles_after == titles_before and titles_after[0] == "Three-head evaluation",
          f"history persists after reload: {titles_after}")
    expect(page.locator('[data-view="results"]')).to_be_visible()
    expect(page.locator("#cards .card").nth(1)).to_contain_text("accepted")
    page.get_by_role("button", name="+ New session").click()
    expect(page.locator("#h-tasks")).to_be_focused()
    check(page.locator("#history li").count() == 2, "+ New session adds an entry to the list")
    shot(page, "desktop-history.png")
    saved = page.evaluate("localStorage.getItem('saemil.history.v1')")
    page.get_by_role("button", name="Clear history").click()
    expect(page.locator("#confirm-clear")).to_contain_text("Delete all saved sessions from this browser?")
    page.get_by_role("button", name="Delete history").click()
    expect(page.locator("#history li")).to_have_count(0)
    expect(page.locator("#history-note")).to_have_text("No saved sessions yet.")
    check(page.evaluate("localStorage.getItem('saemil.history.v1')") is None, "Clear history empties storage and list")
    page.reload()
    expect(page.locator("#history li")).to_have_count(0)
    return saved


def storage_unavailable(browser, base, errors, foreign):
    ctx = browser.new_context(viewport={"width": 1280, "height": 900})
    ctx.add_init_script("""Object.defineProperty(window, 'localStorage',
        {get() { throw new DOMException('blocked', 'SecurityError'); }});""")
    page = new_page(ctx, errors, foreign)
    page.goto(base + "/")
    expect(page.locator("#history-note")).to_have_text("History is not available in this browser.")
    page.get_by_role("button", name="Generate name ideas").click()
    page.fill("#n-seed", "3")
    page.locator("#n-seed").press("Enter")
    expect(page.locator("#names-out .meta")).to_be_visible()
    check(True, "localStorage throwing: history notice shown and the app still works")
    ctx.close()


MSG_UNREACHABLE = "Studio is not responding. Check that the server is running, then try again."


def faults(browser, base, saved, errors, foreign):
    """Mocked failures (route interception) against the real server; labelled as mocks in the output."""
    valid = next(x for x in json.loads(saved) if x["results"]["vision"])  # the three-head session
    saved = json.dumps([valid])
    corrupt = [5, "x", {"id": "bad", "version": "x"},
               {"id": "c1", "version": 1, "summary": {"vision": 7},
                "results": {"vision": {"resp": {"blank": False, "digit": "7", "top3": "nope"}},
                            "names": {"resp": {"names": [1, 2]}}, "tabular": {"resp": {"probability": "high"}}},
                "explanation": {"resp": {"text": 5}}},
               valid]
    ctx = seeded(browser, base, json.dumps(corrupt))
    page = new_page(ctx, errors, foreign)
    page.goto(base + "/")
    expect(page.locator("#badge-names")).to_have_text("Names: ready")
    expect(page.locator("#history li")).to_have_count(3)
    expect(page.locator('[data-view="tasks"]')).to_be_visible()
    page.locator("#history li button").last.click()
    expect(page.locator("#cards .card").first).to_contain_text("confidence")
    check(True, "corrupted-but-valid-JSON history: bad entries dropped, app starts, valid session restores")
    ctx.close()

    # status unreachable while opening on the results view (mock: /api/status aborted)
    ctx = seeded(browser, base, saved)
    page = new_page(ctx, errors, foreign, allow_failed_loads=True)
    page.route("**/api/status", lambda r: r.abort())
    page.goto(base + "/")
    expect(page.locator('[data-view="results"]')).to_be_visible()
    notice = page.locator("#status-notice")
    expect(notice).to_contain_text(MSG_UNREACHABLE)
    shot(page, "desktop-status-unreachable.png")
    page.unroute("**/api/status")
    notice.get_by_role("button", name="Try again").click()
    expect(page.locator("#badge-vision")).to_have_text("Vision: ready")
    expect(notice).to_be_hidden()
    check(True, "MOCK status failure on the results view: notice + Try again shown; retry recovers")

    # network failure on a head + Try again (mock: /api/tabular aborted once)
    page.route("**/api/tabular", lambda r: r.abort())
    go(page, "tabular")
    page.locator("#f-Administrative-range").focus()
    page.keyboard.press("ArrowRight")
    out = page.locator("#tabular-out")
    expect(out).to_contain_text(MSG_UNREACHABLE)
    page.unroute("**/api/tabular")
    out.get_by_role("button", name="Try again").click()
    expect(page.locator("#tabular-out .metric")).to_contain_text("%")
    check(True, "MOCK tabular network failure: unreachable notice, Try again recovers")

    # session switch while a names request is in flight: the button must not stay busy
    held = []
    page.route("**/api/names", lambda r: held.append(r))
    go(page, "names")
    page.locator("#n-count").press("Enter")
    btn = page.locator("#names-submit")
    expect(btn).to_have_attribute("aria-busy", "true")
    page.get_by_role("button", name="+ New session").click()
    go(page, "names")
    expect(btn).to_have_text("Generate names")
    expect(btn).to_be_enabled()
    check(btn.get_attribute("aria-busy") is None, "new session mid-request restores the Generate names button")
    held[0].continue_()
    page.unroute("**/api/names")
    expect(page.locator("#names-out")).to_have_text("Generated names appear here.")
    check(True, "late names response for the previous session is discarded")

    # 503 from a head (mock: /api/vision answers 503 with a diagnostic)
    page.route("**/api/vision", lambda r: r.fulfill(status=503, content_type="application/json",
               body=json.dumps({"detail": {"head": "vision", "error": "vision.npz: archive not found"}})))
    go(page, "vision")
    stroke(page, [(0.3, 0.2), (0.6, 0.8)])
    expect(page.locator("#badge-vision")).to_have_text("Vision: unavailable")
    expect(page.locator("#h-vision")).to_have_text("Vision is unavailable. Other heads are ready.")
    expect(page.locator('[data-view="vision"] .notice')).to_contain_text("Diagnostic: vision.npz: archive not found")
    shot(page, "desktop-head-unavailable.png")
    check(True, "MOCK vision 503: unavailable badge, frame-11 notice with diagnostic; other heads usable")
    ctx.close()


def touch_drawing(browser, base, errors, foreign):
    """390 px touch device: draw a 1 with real touch input (CDP touch events -> pointerType 'touch')."""
    ctx = browser.new_context(viewport={"width": 390, "height": 844}, has_touch=True, is_mobile=True)
    page = new_page(ctx, errors, foreign)
    page.goto(base + "/")
    expect(page.locator("#badge-vision")).to_have_text("Vision: ready")
    page.get_by_role("button", name="Recognize a digit").tap()
    page.evaluate("window.__pt = []; document.getElementById('pad').addEventListener('pointerdown',"
                  " e => window.__pt.push(e.pointerType))")
    page.locator("#pad").scroll_into_view_if_needed()
    box = page.locator("#pad").bounding_box()
    cdp = ctx.new_cdp_session(page)
    x = box["x"] + box["width"] / 2
    ys = [box["y"] + box["height"] * f for f in (0.15, 0.3, 0.45, 0.6, 0.75, 0.85)]
    cdp.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": x, "y": ys[0]}]})
    for y in ys[1:]:
        cdp.send("Input.dispatchTouchEvent", {"type": "touchMove", "touchPoints": [{"x": x, "y": y}]})
    cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
    expect(page.locator("#vision-out .digit")).to_be_visible()
    d = page.locator("#vision-out .digit").inner_text()
    pts = page.evaluate("window.__pt")
    check(pts == ["touch"] and d == "1", f"touch drawing at 390 px: pointerType {pts}, predicted {d}")
    shot(page, "mobile-vision-touch.png")
    ctx.close()


if __name__ == "__main__":
    sys.exit(main())
