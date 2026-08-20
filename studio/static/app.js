/* Saemil Studio UI. Vanilla JS, no network beyond this origin.
   Server text (report prose, messages, diagnostics, labels) is only ever rendered with textContent. */
(function () {
  "use strict";

  // Example digits for the keyboard path (D-021): data/prepared/vision.npz test split, rows 0, 100, ..., 900
  // (digit d is row 100*d); each is 784 uint8 pixels, base64. Comment gives the official MNIST test-set index.
  var EXAMPLES = [
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAbm3U/f9iAAAAAAAAAAAAAAAAAAAAAAAAAAALm/38/Pz9yD8AAAAAAAAAAAAAAAAAAAAAAAAglvz9/Pz8/fzXAAAAAAAAAAAAAAAAAAAAAAE/7fz8/fz8/P381wAAAAAAAAAAAAAAAAAAABC//fz8/P38/Pz9/NcAAAAAAAAAAAAAAAAAALve/P38/Pz9/Pz8/edHAAAAAAAAAAAAAAAAFbXy/Pz93PH8/ef8/P1hAAAAAAAAAAAAAAAAFc78/Pz8vw+1/Gwu0/z9swAAAAAAAAAAAAAASf39/f3tPgAqfQAAAHj9/9MfAAAAAAAAAAAAAHP8/PznRwAAAAAAAAAl/P3nMwAAAAAAAAAAAADZ/Pz8TAAAAAAAAAALlvz9ngAAAAAAAAAAAAAA2fz8/CMAAAAAAAAASfz8/SMAAAAAAAAAAAAAANr9/c4UAAAAAAAAAIf9/f8jAAAAAAAAAAAAAADZ/PyuCgAAAAAAACDj/PyfBQAAAAAAAAAAAAAA2fz8/CMAAAAAG3jU/PznKQAAAAAAAAAAAAAAANn8/PwjAAAAKt38/fz8bAAAAAAAAAAAAAAAAABJ/f39/eht//39/f/eKQAAAAAAAAAAAAAAAAAANOf8/Pz8/P38/Py0LQAAAAAAAAAAAAAAAAAAAABI7Pz8/Pz9/OLXAAAAAAAAAAAAAAAAAAAAAAAAAD5s6Pz8v2wfAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==", // digit 0: prepared test row 0, MNIST test index 6181
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAH2aPQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABXH1/Z8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAACD9/f2fAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMrv39nwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAhP39/TwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAYf7+/l8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAFdf9/bkYAAAAAAAAAAAAAAAAAAAAAAAAAAAAIc7//b0QAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGj9/vVfAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADf/f7dAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA2v/v7cKAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAg/f39nwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAcf39/Z8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAL/9/dUUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD3m/dUjAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACc//++AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/v39qQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAUf79vRAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJ/+/X4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA8/ukvAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==", // digit 1: prepared test row 100, MNIST test index 1646
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHF2dnZ2dnZ2dnYkAAAAAAAAAAAAAAAAAAAAcKv9/f39/f39/f393V4AAAAAAAAAAAAAAAAAAP79/f3915WVlZWVld36iQAAAAAAAAAAAAAAAACc+/1tGRAAAAAAAAA7/fpYAAAAAAAAAAAAAAAAAFyiDAAAAAAAAAAALv39YQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJz94Q8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFTw8VYAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADDv/dYAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADOw/dBcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFXv8GUUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFTv1VgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFPw/ToAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJFfxyVQTAAAAAAAAAAAAAAAAAAAAAAAAAAAAVfT9/VsAAAghISEyo6Ojo6MAAAAAAAAAAAAAAOv9/f2tnJyz/f39/e3ZqpxhAAAAAAAAAAAAAADr/f39/f346urq6tkoAAAAAAAAAAAAAAAAAAAA3/X98oRuUgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAtdRIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==", // digit 2: prepared test row 200, MNIST test index 1076
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAfg9b+/v7+nBEAAAAAAAAAAAAAAAAAAAAAAAAp5Prz87W99v2/OAAAAAAAAAAAAAAAAAAAAAAU5dpPAAAAABnR/eMRAAAAAAAAAAAAAAAAAAAABSIQAAAAAAAAG9v+cAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFX3m/YYAAAAAAAAAAAAAAAAAAAAAAAAAAAAITMv9/asHAAAAAAAAAAAAAAAAAAAAAAAAAAACff399WMhAAAAAAAAAAAAAAAAAAAAAAAAAAAs3P37uhkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlf39zSAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD641/37lw8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAw3sf11AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABrr/RQAAAAAAAAAAAAAAAAAAAJLQgAAAAAAAAAF4v0UAAAAAAAAAAAAAAAAAABX6TMAAAAAAAAAYv3nDgAAAAAAAAAAAAAAAAAq4akAAAAAAAAAArL+fQAAAAAAAAAAAAAAAAAAX/2VAAAAAAAAAC/9/UoAAAAAAAAAAAAAAAAAAB/e2ikAAAAAAC7k/aMGAAAAAAAAAAAAAAAAAAAAVf3qWgUANoPm/eUTAAAAAAAAAAAAAAAAAAAAAAF5/f319fr9850HAAAAAAAAAAAAAAAAAAAAAAAAEGq7/f39rU8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==", // digit 3: prepared test row 300, MNIST test index 574
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB3/jcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAjv1QAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJ39UAAAAAAAAAAAAAAAAAAAAAAAAEkMAAAAAADe/TwAAAAAAAAAAAAAAAAAAAAAAA/5kgAAAABG+/0AAAAAAAAAAAAAAAAAAAAAAAAV/rAAAAAAhP6QAAAAAAAAAAAAAAAAAAAAAAAAjP11AAAAGOP9PAAAAAAAAAAAAAAAAAAAAAAAN/79TQAAAH/91TMeAAAAAAAAAAAAAAAAAAAAJO3+/ffe3t/1/f39vAAAAAAAAAAAAAAAAAAAAED9+a2t2v3+/dWBXwUAAAAAAAAAAAAAAAAAAAAUmjIAABRu//xFAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWv6hAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIOf+ZwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFT9wwgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/j/W4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABT/scAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAO4/2LAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIP39RgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACD9/V8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAW7ulBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==", // digit 4: prepared test row 400, MNIST test index 5027
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAZImBgj8n9/c1yAAAAAAAAAAAAAAAAAAZDmLCw6fz9/Pz8/Pz83BIAAAAAAAAAAAAAAADN/Pz8/Pz8/fz8/Pz8oYYMAAAAAAAAAAAAAAAA8vz8/Pz86efmi3kxCgEAAAAAAAAAAAAAAAAAAPL8/K1YWAwAAAAAAAAAAAAAAAAAAAAAAAAAAE/6/PwtAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACY/Pz8FQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAd5Pz8/G4lAAAAAAAAAAAAAAAAAAAAAAAAAAAAAK/8/Pz88WosAAAAAAAAAAAAAAAAAAAAAAAAAAAN9Pz8/Pz87p9SAAAAAAAAAAAAAAAAAAAAAAAAAABJb3Tc3Pn//f1YAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABDi+j89mkPAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA86/z8tAkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADTt/PxZAAAAAAAAAAAAAAAAAAAAJaVBFgAAAAAAOOn8+EEAAAAAAAAAAAAAAAAAAAzS/Nt1DAAAI5Lq/PyIAAAAAAAAAAAAAAAAAAAAXvv8/Oro5/D8/Pz8+TwAAAAAAAAAAAAAAAAAAABh/Pz8/fz8/Pz8/PU2AAAAAAAAAAAAAAAAAAAAA1Xo/P38/Pz8/PlhAAAAAAAAAAAAAAAAAAAAAAAAGICP2/zzjo4oAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==", // digit 5: prepared test row 500, MNIST test index 9176
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAVN6//BCQAAAAAAAAAAAAAAAAAAAAAAAAAAAEK3/vPl+hgAAAAAAAAAAAAAAAAAAAAAAAAACGjp/sMzB98YAAAAAAAAAAAAAAAAAAAAAAAACI7+/nwNABbCEwAAAAAAAAAAAAAAAAAAAAAAPdP+7l0BAAAAAAAAAAAAAAAAAAAAAAAAAAAAWPP+rRgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAVvz9hgQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOfz4hgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAN+T9cgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGN3+rAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACKj+rw4AAAAcZLfTgA0AAAAAAAAAAAAAAAAAADr+2xIAACW48f7+/v7FHAAAAAAAAAAAAAAAAADB/l0ANLz18W1hFUbR/jAAAAAAAAAAAAAAAAAQ6e0Xm/37li4AAAAAnP0vAAAAAAAAAAAAAAAAAL/wrf7VRQAAAAAAZ+m3AAAAAAAAAAAAAAAAAABS/v7mGQAAABJPo/3gIgAAAAAAAAAAAAAAAAAAM/7+/s/Oms7n/uKADgAAAAAAAAAAAAAAAAAAAKjzf6XF7ffFtEcOAAAAAAAAAAAAAAAAAAAAAAB2zQAAABEVAAAAAAAAAAAAAAAAAAAAAAAAAAAAQt0LAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==", // digit 6: prepared test row 600, MNIST test index 6693
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAR6P/UnC0AIxUAAAAAAAAAAAAAAAAAAAAAAAAbo/67sO/8+vzNAAAAAAAAAAAAAAAAAAAAAABU+f3+JwATX2/+7AAAAAAAAAAAAAAAAAAAAAAAkv7+uQIAAAAa/rcAAAAAAAAAAAAAAAAAAAAAAOP+/igAAAAAMP6AAAAAAAAAAAAAAAAAAAAAAADp/vEaAAAAAKL+QAAAAAAAAAAAAAAAAAAAAAAP7f6GAAAAAADG/kAAAAAAAAAAAAAAAAAAAAAAQf7+HgAAAAAH0usIAAAAAAAAAAAAAAAAAAAAADX6rQQAAAAAH/7sDAAAAAAAAAAAAAAAAAAAAAAAIwQAAAhWgZD+/qbYagAAAAAAAAAAAAAAAAAAAAAAAABn/v7+/v7+/pEAAAAAAAAAAAAAAAAMAAAAAAAACFqF5f6aNYAOAAAAAAAAAAAAAAAAEgAAAAAAAAAAAMv+PAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABzz6Q0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAj/uQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAZP6jAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHr+bAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAC2/jgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAz/45AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAM/+UwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==", // digit 7: prepared test row 700, MNIST test index 5914
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9lMn90ZRFJwAAAAAAAAAAAAAAAAAAAAAAABat/fz8/Pz9/O5AAAAAAAAAAAAAAAAAAAAAAAvK/P38+6BSkO7897w+AAAAAAAAAAAAAAAAAAAW/OLj/KAAAAAcjvz873EAAAAAAAAAAAAAAAAAFvw7lPx+AAAAAAQq2v2oAAAAAAAAAAAAAAAAABLeKpT9vAAAAAAAAKj+qAAAAAAAAAAAAAAAAAAAEAtF/OcAAAAACJvy/YQAAAAAAAAAAAAAAAAAAAAAFtT7ggAWiOr89JgFAAAAAAAAAAAAAAAAAAAAAABV/Oep/fz3ojEAAAAAAAAAAAAAAAAAAAAAAAAADuz8/P3ZWwAAAAAAAAAAAAAAAAAAAAAAAAAACJ39/f2WAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHv8/Pz8jQAAAAAAAAAAAAAAAAAAAAAAAAAAAAn1/Ksv6OgpAAAAAAAAAAAAAAAAAAAAAAAAAACw/cQHALn9qAAAAAAAAAAAAAAAAAAAAAAAAAAA0/1QAAAS/csJAAAAAAAAAAAAAAAAAAAAAAAAJuyWAAAAAP/9FQAAAAAAAAAAAAAAAAAAAAAAABDdagAAAHT94A4AAAAAAAAAAAAAAAAAAAAAAAAA084WS8X5/Z8AAAAAAAAAAAAAAAAAAAAAAAAAANP9/Pz8/JoVAAAAAAAAAAAAAAAAAAAAAAAAAABP/fz8tpMSAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==", // digit 8: prepared test row 800, MNIST test index 1268
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABEl7cpO2TCwAAAAAAAAAAAAAAAAAAAAAAAAAAlv7+/v7+/sQAAAAAAAAAAAAAAAAAAAAAAAAASfv+/v7+/v7zAAAAAAAAAAAAAAAAAAAAAAAAAGz+1rdk0/7+mQAAAAAAAAAAAAAAAAAAAAAAAABs/nEAAMT+/t0AAAAAAAAAAAAAAAAAAAAAAAAAnf6sWmj4/v74LQAAAAAAAAAAAAAAAAAAAAAAAN/+/unY/v7+/mgAAAAAAAAAAAAAAAAAAAAAAAAw+P7+/v7+/twAAAAAAAAAAAAAAAAAAAAAAAAAAJGk/v7+/v4FAAAAAAAAAAAAAAAAAAAAAAAAAAAACGDDeuz+BQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKn+/gUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADE//4FAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAxP7MAwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA8f+iAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADX4/hcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMz/4tAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPP7+jQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACnr/mcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAY26wEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIONoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==", // digit 9: prepared test row 900, MNIST test index 2916
  ];

  var HEADS = ["vision", "names", "tabular"];
  var HEAD_LABEL = { vision: "Vision", names: "Names", tabular: "Tabular" };
  var FEATURE_LABEL = {
    Administrative: "Administrative pages", Administrative_Duration: "Administrative duration",
    Informational: "Informational pages", Informational_Duration: "Informational duration",
    ProductRelated: "Product pages", ProductRelated_Duration: "Product duration"
  };
  var CLI_NAME = { ollama: "Ollama", llm: "llm", claude: "Claude", gemini: "Gemini" };
  var VENDORS = ["claude", "gemini", "openai", "deepseek", "ollama", "llm"];
  var REASON = {
    timeout: "timed out", missing_executable: "not found", no_on_device_cli: "not found", queue_full: "was busy",
    unknown_provider: "is not available", unknown_mode: "is not available"
  };
  var MSG_UNREACHABLE = "Studio is not responding. Check that the server is running, then try again.";
  var MSG_TOO_LARGE = "That request was too large. Clear the input and try again.";
  var HISTORY_KEY = "saemil.history.v1";
  var MAX_HISTORY = 20; // fixed cap keeps localStorage small; make it a setting if anyone asks
  var INK = [243, 244, 237]; // --ink
  var PAD = 280, STROKE = 20;

  // ------------------------------------------------------------------ tiny DOM helpers
  function $(sel, root) { return (root || document).querySelector(sel); }
  function $all(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  }
  function add(parent) { for (var i = 1; i < arguments.length; i++) if (arguments[i]) parent.appendChild(arguments[i]); return parent; }
  function pct(p) { return (p * 100).toFixed(1) + "%"; }
  function pctWords(p) { return (p * 100).toFixed(1) + " percent"; }
  function plural(n, word) { return n + " " + word + (Number(n) === 1 ? "" : "s"); }
  function pad2(i) { return (i < 10 ? "0" : "") + i; }
  function clone(o) { return JSON.parse(JSON.stringify(o)); }
  function announce(head, text) { var n = $("#live-" + head); n.textContent = ""; setTimeout(function () { n.textContent = text; }, 50); }

  // ------------------------------------------------------------------ API
  function api(path, body) {
    var opts = body === undefined ? {} : {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body)
    };
    return fetch(path, opts).then(function (r) {
      return r.json().catch(function () { return null; }).then(function (data) {
        if (r.ok) return { kind: "ok", data: data };
        if (r.status === 503) return { kind: "unavailable", data: data };
        if (r.status === 422) return { kind: "invalid", data: data };
        if (r.status === 413) return { kind: "too_large" };
        return { kind: "error", status: r.status };
      });
    }, function () { return { kind: "network" }; });
  }

  // ------------------------------------------------------------------ state
  var S = {
    status: null,
    unavail: { vision: null, names: null, tabular: null }, // diagnostic text when a head is unavailable
    session: null,
    seq: { vision: 0, names: 0, tabular: 0, report: 0 },
    explainPending: false,
    explainError: null,
    view: "tasks",
    features: [],
    values: {},
    providers: []
  };
  var history = [], historyOk = true;

  function newSession() {
    return {
      id: "s" + Date.now().toString(36) + Math.random().toString(36).slice(2, 7), created: Date.now(),
      results: { vision: null, names: null, tabular: null },
      summary: { vision: "", names: "", tabular: "" }, explanation: null, version: 0
    };
  }
  // Stored history is untrusted input: keep only well-formed parts, so a bad entry never breaks startup.
  function isNum(v) { return typeof v === "number" && isFinite(v); }
  function isInt(v) { return Number.isInteger(v); }
  function obj(v) { return v !== null && typeof v === "object" && !Array.isArray(v); }
  var CHECK = {
    vision: function (r) {
      return obj(r) && obj(r.resp) && r.resp.blank === false && isInt(r.resp.digit) && Array.isArray(r.resp.top3) &&
        r.resp.top3.length > 0 && r.resp.top3.every(function (t) { return obj(t) && isInt(t.digit) && isNum(t.prob); }) &&
        Array.isArray(r.pixels) && r.pixels.length === 784 && r.pixels.every(isNum);
    },
    names: function (r) {
      return obj(r) && obj(r.resp) && Array.isArray(r.resp.names) &&
        r.resp.names.every(function (n) { return typeof n === "string"; }) && isInt(r.resp.accepted) &&
        isInt(r.resp.requested) && isNum(r.temperature) && (r.seed === null || isInt(r.seed));
    },
    tabular: function (r) {
      return obj(r) && obj(r.resp) && isNum(r.resp.probability) && typeof r.resp.source_kind === "string" &&
        obj(r.inputs) && Object.keys(r.inputs).every(function (k) { return isNum(r.inputs[k]); });
    }
  };
  function validExplanation(x) {
    return obj(x) && obj(x.resp) && typeof x.resp.text === "string" && typeof x.resp.mode_used === "string" &&
      typeof x.resp.provider_used === "string" &&
      (x.resp.fallback_reason === null || typeof x.resp.fallback_reason === "string") &&
      typeof x.requested === "string" && isNum(x.version) && Array.isArray(x.missing) &&
      x.missing.every(function (m) { return typeof m === "string"; });
  }
  function sanitizeSession(s) {
    if (!obj(s) || typeof s.id !== "string" || !s.id) return null;
    var out = newSession();
    out.id = s.id;
    out.created = isNum(s.created) ? s.created : out.created;
    out.version = isInt(s.version) ? s.version : 0;
    HEADS.forEach(function (h) {
      var r = obj(s.results) ? s.results[h] : null, sum = obj(s.summary) ? s.summary[h] : "";
      if (r && CHECK[h](r)) { out.results[h] = r; out.summary[h] = typeof sum === "string" ? sum : ""; }
    });
    out.explanation = validExplanation(s.explanation) ? s.explanation : null;
    return out;
  }

  // ------------------------------------------------------------------ history (localStorage only, never sent)
  function loadHistory() {
    var raw = null;
    try { raw = window.localStorage.getItem(HISTORY_KEY); } catch (e) { historyOk = false; return; }
    try { history = raw ? JSON.parse(raw) : []; } catch (e) { history = []; }
    if (!Array.isArray(history)) history = [];
    history = history.map(sanitizeSession).filter(Boolean);
  }
  function saveHistory() {
    if (!historyOk) return;
    for (;;) {
      try { window.localStorage.setItem(HISTORY_KEY, JSON.stringify(history)); return; } catch (e) {
        var quota = e && (e.name === "QuotaExceededError" || e.name === "NS_ERROR_DOM_QUOTA_REACHED" || e.code === 22);
        if (quota && history.length > 1) { history.pop(); continue; } // drop the oldest session and retry
        historyOk = false;
        return;
      }
    }
  }
  function upsertSession() {
    if (!historyOk) return renderHistory();
    var i = history.findIndex(function (h) { return h.id === S.session.id; });
    if (i >= 0) history.splice(i, 1);
    history.unshift(clone(S.session));
    if (history.length > MAX_HISTORY) history.length = MAX_HISTORY;
    saveHistory();
    renderHistory();
  }
  function sessionTitle(s) {
    var have = HEADS.filter(function (h) { return s.results[h]; });
    if (have.length === 3) return "Three-head evaluation";
    if (have.length === 2) return "Two-head session";
    if (have.length === 1) return { vision: "Digit recognition", names: "Name exploration", tabular: "Shopper session" }[have[0]];
    return "New session";
  }
  function runSummary(s) {
    return HEADS.map(function (h) { return s.results[h] ? s.summary[h] : ""; }).filter(Boolean).join(" · ");
  }
  function renderHistory() {
    var list = $("#history"), note = $("#history-note"), clear = $("#clear-history");
    list.replaceChildren();
    if (!historyOk) {
      note.textContent = "History is not available in this browser.";
      note.hidden = false; clear.hidden = true;
      return;
    }
    clear.hidden = history.length === 0;
    note.hidden = history.length > 0;
    note.textContent = "No saved sessions yet.";
    history.forEach(function (h) {
      var b = el("button");
      b.type = "button";
      if (S.session && h.id === S.session.id) b.setAttribute("aria-current", "true");
      add(b, el("span", "h-title", sessionTitle(h)), el("span", "h-sub", runSummary(h) || "No results yet"));
      b.addEventListener("click", function () { openSession(h.id); });
      list.appendChild(add(el("li"), b));
    });
  }
  function invalidatePending() {
    HEADS.concat("report").forEach(function (k) { S.seq[k]++; });
    S.explainPending = false; S.explainError = null;
    clearTimeout(tabTimer);
    setBusy($("#names-submit"), false);
    setExplainBusy(false);
  }
  function openSession(id) {
    var h = history.find(function (x) { return x.id === id; });
    if (!h) return;
    invalidatePending();
    S.session = clone(h);
    clearPad();
    resetForSession();
    renderHistory();
    show(HEADS.some(function (k) { return S.session.results[k]; }) ? "results" : "tasks");
  }
  function startNewSession() {
    invalidatePending();
    S.session = newSession();
    clearPad();
    resetSliders();
    resetForSession();
    upsertSession();
    show("tasks");
  }
  function resetForSession() {
    S.visionOut = S.namesOut = S.tabOut = null;
    S.tabPending = false;
    renderVisionOut();
    renderNamesOut();
    if (S.session.results.tabular) setSliderValues(S.session.results.tabular.inputs);
    renderTabularOut();
    refreshExplainButton();
  }

  // result changed -> new version (makes an older explanation stale), saved to history
  function touch() {
    S.session.version++;
    upsertSession();
    refreshExplainButton();
    if (S.view === "results") renderResults();
  }

  // ------------------------------------------------------------------ status and badges
  function loadStatus() {
    var slot = $("#status-notice");
    slot.replaceChildren();
    return api("/api/status").then(function (r) {
      if (r.kind !== "ok") {
        slot.appendChild(dangerNotice(r.kind === "too_large" ? MSG_TOO_LARGE : MSG_UNREACHABLE, loadStatus));
        return;
      }
      S.status = r.data;
      HEADS.forEach(function (h) {
        var st = r.data.heads && r.data.heads[h];
        if (!st || !st.ready) S.unavail[h] = (st && st.error) || "status not reported";
      });
      renderBadges();
      buildProviders(r.data.providers || {});
      var tab = r.data.heads && r.data.heads.tabular;
      if (tab && tab.ready && Array.isArray(tab.features)) buildSliders(tab.features);
      $all("[data-go]").forEach(function (b) { b.disabled = false; });
      if (S.session.results.tabular) setSliderValues(S.session.results.tabular.inputs);
      if (S.view !== "tasks" && S.view !== "results" && S.view !== "providers") show(S.view, false);
      if (S.view === "results") renderResults();
    });
  }
  function renderBadges() {
    HEADS.forEach(function (h) {
      var b = $("#badge-" + h);
      b.className = "badge " + (S.unavail[h] ? "unavailable" : "ready");
      b.textContent = HEAD_LABEL[h] + ": " + (S.unavail[h] ? "unavailable" : "ready");
    });
  }
  function markUnavailable(head, data) {
    var d = data && data.detail;
    S.unavail[head] = (d && typeof d.error === "string" && d.error) || "the model did not answer";
    renderBadges();
    if (S.view === head) renderUnavailable(head);
    refreshExplainButton();
  }

  // ------------------------------------------------------------------ notices
  function dangerNotice(text, retry) {
    var n = el("div", "notice danger");
    n.setAttribute("role", "alert");
    n.appendChild(el("p", "", text));
    if (retry) {
      var b = el("button", "btn secondary", "Try again");
      b.type = "button";
      b.addEventListener("click", retry);
      n.appendChild(add(el("div", "row"), b));
    }
    return n;
  }
  function failureNotice(r, retry) {
    if (r.kind === "too_large") return dangerNotice(MSG_TOO_LARGE, null);
    if (r.kind === "network") return dangerNotice(MSG_UNREACHABLE, retry);
    return dangerNotice("The request failed (" + (r.status || r.kind) + "). Try again.", retry);
  }
  var OTHER_ACTIONS = {
    vision: [["names", "Generate names"], ["tabular", "Edit shopping session"]],
    names: [["vision", "Draw digit"], ["tabular", "Edit shopping session"]],
    tabular: [["vision", "Draw digit"], ["names", "Generate names"]]
  };
  function renderUnavailable(head) {
    var view = $('[data-view="' + head + '"]');
    var label = HEAD_LABEL[head];
    $(".eyebrow", view).textContent = "HEAD STATUS / RECOVERY";
    $("h1", view).textContent = label + " is unavailable. Other heads are ready.";
    $(".lede", view).textContent = "A missing model affects only its own task. The other tools still work.";
    $(".editor", view).hidden = true;
    var slot = $(".unavail-slot", view);
    if (slot.childNodes.length) return;
    var n = el("div", "notice danger");
    add(n, el("span", "badge unavailable", label + " unavailable"),
      el("p", "n-head title", "The " + head + " model could not be loaded."),
      el("p", "muted", "Diagnostic: " + S.unavail[head]),
      el("p", "muted", "Create or restore the artifact, then restart Studio to load it."));
    var row = el("div", "row");
    OTHER_ACTIONS[head].forEach(function (a, i) {
      var b = el("button", "btn " + (i === 0 ? "primary" : "secondary"), a[1]);
      b.type = "button";
      b.disabled = !!S.unavail[a[0]];
      b.addEventListener("click", function () { show(a[0]); });
      row.appendChild(b);
    });
    n.appendChild(row);
    add(slot, n, el("p", "note muted", "When a head is unavailable, its last result is labeled as previous output. " +
      "A new explanation identifies which inputs are missing."));
  }

  // ------------------------------------------------------------------ views
  function show(view, focus) {
    S.view = view;
    $all(".view").forEach(function (v) { v.hidden = v.getAttribute("data-view") !== view; });
    $all(".tool").forEach(function (t) {
      if (t.getAttribute("data-go") === view) t.setAttribute("aria-current", "page"); else t.removeAttribute("aria-current");
    });
    if (S.unavail[view]) renderUnavailable(view);
    if (view === "vision") renderVisionOut();
    if (view === "names") renderNamesOut();
    if (view === "tabular") {
      renderTabularOut();
      if (!S.session.results.tabular && !S.unavail.tabular && S.features.length) sendTabular();
    }
    if (view === "results") renderResults();
    if (view === "providers") renderProviderCards();
    closeDrawer(false);
    if (focus !== false) {
      var h = $('[data-view="' + view + '"] h1');
      h.focus();
      window.scrollTo(0, 0);
    }
  }

  // ------------------------------------------------------------------ vision
  var pad = $("#pad"), ctx = pad.getContext("2d", { willReadFrequently: true });
  var drawing = false, activePointer = null, last = null, hasInk = false, exampleShown = false; // the canvas is the ink buffer that is posted

  function setPadLabel() { pad.setAttribute("aria-label", hasInk ? "Drawing area, contains a drawing" : "Drawing area, empty"); }
  function clearPad() { ctx.clearRect(0, 0, PAD, PAD); hasInk = false; exampleShown = false; setPadLabel(); }
  function padPoint(e) {
    var r = pad.getBoundingClientRect();
    return { x: (e.clientX - r.left) * PAD / r.width, y: (e.clientY - r.top) * PAD / r.height };
  }
  function strokeTo(p) {
    ctx.strokeStyle = "rgb(" + INK.join(",") + ")";
    ctx.lineWidth = STROKE; ctx.lineCap = "round"; ctx.lineJoin = "round";
    ctx.beginPath(); ctx.moveTo(last.x, last.y); ctx.lineTo(p.x, p.y); ctx.stroke();
    last = p;
  }
  pad.addEventListener("pointerdown", function (e) {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    if (S.unavail.vision || drawing) return; // one pointer at a time; others are ignored until it ends
    e.preventDefault();
    activePointer = e.pointerId;
    if (exampleShown) clearPad(); // a new drawing replaces a loaded example, so only the user's ink is classified
    pad.setPointerCapture(e.pointerId);
    drawing = true; last = padPoint(e); // a tap alone leaves no ink: the server answers it as blank (D-010)
  });
  pad.addEventListener("pointermove", function (e) {
    if (!drawing || e.pointerId !== activePointer) return;
    var evs = e.getCoalescedEvents ? e.getCoalescedEvents() : [];
    (evs.length ? evs : [e]).forEach(function (ev) { strokeTo(padPoint(ev)); });
    if (!hasInk) { hasInk = true; setPadLabel(); }
  });
  function endStroke(e) {
    if (!drawing || e.pointerId !== activePointer) return;
    drawing = false; activePointer = null;
    var data = ctx.getImageData(0, 0, PAD, PAD).data, alpha = new Uint8Array(PAD * PAD);
    for (var i = 0; i < alpha.length; i++) alpha[i] = data[i * 4 + 3];
    var px = window.Centering.isTinyStroke(alpha, PAD, PAD) ? new Array(784).fill(0)  // D-023: server says blank
      : window.Centering.centerByMass(alpha, PAD, PAD);
    submitVision(px, "Drawing classified");
  }
  pad.addEventListener("pointerup", endStroke);
  pad.addEventListener("pointercancel", endStroke);
  pad.addEventListener("lostpointercapture", endStroke); // capture lost without pointerup: never leave drawing locked

  function exampleDigit(d) {
    var s = atob(EXAMPLES[d]), px = new Uint8Array(784);
    for (var i = 0; i < 784; i++) px[i] = s.charCodeAt(i);
    return px;
  }
  $("#use-example").addEventListener("click", function () {
    var d = Number($("#example-digit").value), px = exampleDigit(d), cell = PAD / 28;
    clearPad();
    for (var i = 0; i < 784; i++) if (px[i]) {
      ctx.fillStyle = "rgba(" + INK.join(",") + "," + (px[i] / 255) + ")";
      ctx.fillRect((i % 28) * cell, Math.floor(i / 28) * cell, cell, cell);
    }
    hasInk = true; exampleShown = true; setPadLabel();
    submitVision(window.Centering.centerByMass(px, 28, 28), "Example digit " + d + " classified");
  });
  $("#clear-drawing").addEventListener("click", function () {
    S.seq.vision++;
    clearPad();
    var had = !!S.session.results.vision;
    S.session.results.vision = null;
    S.visionOut = null;
    renderVisionOut();
    if (had) touch();
  });

  function submitVision(pixels, summary) {
    var seq = ++S.seq.vision, sess = S.session;
    S.visionOut = { kind: "pending" };
    renderVisionOut();
    api("/api/vision", { pixels: Array.prototype.slice.call(pixels) }).then(function (r) {
      if (seq !== S.seq.vision || sess !== S.session) return; // stale
      if (r.kind === "ok" && r.data.blank) {
        S.visionOut = { kind: "blank", message: r.data.message };
        var had = !!sess.results.vision;
        sess.results.vision = null;
        announce("vision", r.data.message);
        renderVisionOut();
        if (had) touch();
      } else if (r.kind === "ok") {
        S.visionOut = null;
        sess.results.vision = { resp: r.data, pixels: Array.prototype.slice.call(pixels) };
        sess.summary.vision = summary;
        announce("vision", "Prediction " + r.data.digit + ", " + pctWords(r.data.top3[0].prob) + " confidence");
        renderVisionOut();
        touch();
      } else if (r.kind === "unavailable") {
        S.visionOut = null;
        markUnavailable("vision", r.data);
      } else {
        S.visionOut = { kind: "error", r: r, retry: function () { submitVision(pixels, summary); } };
        renderVisionOut();
      }
    });
  }
  function bars(top3) {
    var wrap = el("div", "bars");
    top3.forEach(function (t) {
      var row = el("div", "bar-row");
      var bar = el("div", "bar"), fill = el("span");
      fill.style.width = Math.max(0, Math.min(100, t.prob * 100)) + "%";
      bar.setAttribute("aria-hidden", "true");
      add(row, el("span", "num", t.digit), add(bar, fill), el("span", "pct", pct(t.prob)));
      wrap.appendChild(row);
    });
    return wrap;
  }
  function renderVisionOut() {
    var out = $("#vision-out"), st = S.visionOut, res = S.session.results.vision;
    out.replaceChildren();
    if (st && st.kind === "pending") return add(out, el("p", "muted", "Reading the drawing…"));
    if (st && st.kind === "error") return add(out, failureNotice(st.r, st.retry));
    if (st && st.kind === "blank") {
      return add(out, el("p", "heading", "Draw a digit."), el("p", "", st.message),
        el("span", "badge example chip", "Waiting for a drawing"), el("p", "muted small", "No prediction has been made."));
    }
    if (!res) return add(out, el("p", "muted", "Your result appears here."));
    var v = res.resp, other = 1 - v.top3.reduce(function (a, t) { return a + t.prob; }, 0);
    add(out, el("p", "digit num", v.digit), el("p", "confidence num", pct(v.top3[0].prob) + " confidence"), bars(v.top3),
      el("p", "small muted", "Other digits account for the remaining " + pct(Math.max(0, other)) + "."));
  }

  // ------------------------------------------------------------------ names
  var nf = { count: $("#n-count"), temperature: $("#n-temp"), seed: $("#n-seed") };
  var NAME_ERR = {
    count: "Enter a whole number from 1 to 10.", temperature: "Enter a number from 0.2 to 1.5.",
    seed: "Enter a whole number, 0 or larger, or leave it empty."
  };
  function fieldError(input, msg) {
    var err = document.getElementById(input.id + "-err");
    if (msg) { input.setAttribute("aria-invalid", "true"); err.textContent = msg; err.hidden = false; }
    else { input.removeAttribute("aria-invalid"); err.textContent = ""; err.hidden = true; }
  }
  function readNames() {
    var c = nf.count.value.trim(), t = nf.temperature.value.trim(), s = nf.seed.value.trim(), ok = true;
    var count = Number(c), temp = Number(t), seed = s === "" ? null : Number(s);
    function check(input, good, key) { good = good && !input.validity.badInput; fieldError(input, good ? null : NAME_ERR[key]); ok = ok && good; }
    check(nf.count, c !== "" && Number.isInteger(count) && count >= 1 && count <= 10, "count");
    check(nf.temperature, t !== "" && isFinite(temp) && temp >= 0.2 && temp <= 1.5, "temperature");
    check(nf.seed, seed === null || (Number.isSafeInteger(seed) && seed >= 0), "seed");
    return ok ? { count: count, temperature: temp, seed: seed } : null;
  }
  function setBusy(btn, busy, label) {
    if (busy) {
      btn.dataset.label = btn.textContent;
      btn.style.minWidth = btn.offsetWidth + "px";
      btn.textContent = label; btn.setAttribute("aria-busy", "true"); btn.disabled = true;
    } else if (btn.dataset.label) {
      btn.textContent = btn.dataset.label; btn.removeAttribute("aria-busy"); btn.disabled = false; btn.style.minWidth = "";
      delete btn.dataset.label;
    }
  }
  $("#names-form").addEventListener("submit", function (e) {
    e.preventDefault();
    var req = readNames();
    if (req) submitNames(req);
  });
  function submitNames(req) {
    var seq = ++S.seq.names, sess = S.session, btn = $("#names-submit");
    setBusy(btn, true, "Generating…");
    S.namesOut = null;
    api("/api/names", req).then(function (r) {
      if (seq !== S.seq.names || sess !== S.session) return;
      setBusy(btn, false);
      if (r.kind === "ok") {
        sess.results.names = { resp: r.data, temperature: req.temperature, seed: req.seed };
        sess.summary.names = plural(req.count, "name") + " requested" + (req.seed === null ? "" : " (seed " + req.seed + ")");
        announce("names", plural(r.data.accepted, "name") + " generated");
        renderNamesOut();
        touch();
      } else if (r.kind === "invalid") {
        var detail = (r.data && Array.isArray(r.data.detail)) ? r.data.detail : [];
        var hit = false;
        detail.forEach(function (d) {
          var f = d.loc && d.loc[1];
          if (nf[f]) { fieldError(nf[f], NAME_ERR[f]); hit = true; }
        });
        if (!hit) fieldError(nf.count, NAME_ERR.count);
      } else if (r.kind === "unavailable") {
        markUnavailable("names", r.data);
      } else {
        S.namesOut = { r: r, retry: function () { submitNames(req); } };
        renderNamesOut();
      }
    });
  }
  function namesMeta(n) {
    return n.resp.accepted + " of " + n.resp.requested + " accepted / temperature " + n.temperature +
      (n.seed === null || n.seed === undefined ? "" : " / seed " + n.seed);
  }
  function renderNamesOut() {
    var out = $("#names-out"), n = S.session.results.names;
    out.replaceChildren();
    if (S.namesOut) return add(out, failureNotice(S.namesOut.r, S.namesOut.retry));
    if (!n) return add(out, el("p", "muted", "Generated names appear here."));
    var r = n.resp;
    if (r.accepted > 0) {
      var ol = el("ol");
      r.names.forEach(function (name, i) { ol.appendChild(add(el("li", "cand"), el("span", "idx", pad2(i + 1)), el("span", "name", name))); });
      out.appendChild(ol);
    }
    if (r.accepted < r.requested) {
      var w = el("div", "notice warning");
      add(w, el("p", "n-head", r.accepted + " of " + r.requested + " accepted"),
        el("p", "", "The sampler reached its attempt limit. Try another seed or a different temperature to explore more candidates."));
      out.appendChild(w);
    }
    add(out, el("p", "meta num", namesMeta(n)),
      el("p", "muted", "Filtered against the source list and this request. Candidate names have not been checked for trademark availability."));
  }

  // ------------------------------------------------------------------ tabular
  var tabTimer = null;
  function fmtCap(f) { var v = f.kind === "count" ? Math.floor(f.input_max) : Math.floor(f.input_max * 10) / 10; return String(v); }
  function fmtVal(f, v) {
    return f.kind === "count" ? plural(Math.round(v), "page") : Number(v).toFixed(1) + (Number(v) === 1 ? " second" : " seconds");
  }
  function snap(f, v) { return f.kind === "count" ? Math.round(v) : Math.round(v * 10) / 10; }

  function buildSliders(features) {
    S.features = features.filter(function (f) { return FEATURE_LABEL[f.name]; });
    var box = $("#sliders");
    box.replaceChildren();
    S.features.forEach(function (f) {
      var id = "f-" + f.name, step = f.kind === "count" ? 1 : 0.1;
      var lo = snap(f, f.percentiles.p0 || 0), hi = snap(f, f.percentiles.p99 || 0);
      if (hi <= lo) hi = lo + step;
      var row = el("div", "slider-row"), head = el("div", "slider-head");
      var lab = el("label", "label", FEATURE_LABEL[f.name]);
      lab.htmlFor = id; lab.id = id + "-label";
      var val = el("span", "label num"); val.id = id + "-val";
      add(head, lab, val);
      var range = el("input"); range.type = "range"; range.min = lo; range.max = hi; range.step = step;
      range.setAttribute("aria-labelledby", lab.id); range.id = id + "-range";
      var num = el("input", "field"); num.type = "number"; num.id = id; num.min = 0; num.max = fmtCap(f); num.step = step;
      num.inputMode = f.kind === "count" ? "numeric" : "decimal";
      num.setAttribute("aria-describedby", id + "-err");
      var err = el("p", "small error-text"); err.id = id + "-err"; err.hidden = true;
      add(row, head, add(el("div", "slider-line"), range, num), err);
      box.appendChild(row);
      range.addEventListener("input", function () {
        setValue(f, Number(range.value), "range");
        scheduleTabular();
      });
      num.addEventListener("input", function () {
        var t = num.value.trim(), v = Number(t);
        var ok = t !== "" && isFinite(v) && v >= 0 && v <= f.input_max && (f.kind !== "count" || Number.isInteger(v));
        fieldError(num, ok ? null : "Enter a value from 0 to " + fmtCap(f) + ".");
        if (!ok) { clearTimeout(tabTimer); S.tabPending = false; renderTabularOut(); return; }
        setValue(f, v, "num");
        scheduleTabular();
      });
    });
    resetSliders();
  }
  function setValue(f, v, from) {
    var id = "f-" + f.name, range = document.getElementById(id + "-range"), num = document.getElementById(id);
    S.values[f.name] = v;
    if (from !== "range") range.value = Math.min(v, Number(range.max)); // above p99: thumb pinned at max
    if (from !== "num") { num.value = String(v); fieldError(num, null); }
    var lo = Number(range.min), hi = Number(range.max);
    range.style.setProperty("--fill", ((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo) * 100) + "%");
    range.setAttribute("aria-valuetext", fmtVal(f, v));
    document.getElementById(id + "-val").textContent = fmtVal(f, v);
  }
  function resetSliders() { S.features.forEach(function (f) { setValue(f, snap(f, f.median)); }); }
  function setSliderValues(inputs) {
    S.features.forEach(function (f) { if (typeof inputs[f.name] === "number") setValue(f, inputs[f.name]); });
  }
  function scheduleTabular() {
    clearTimeout(tabTimer);
    S.seq.tabular++; // inputs changed: a response for the older inputs is dropped
    S.tabPending = true;
    renderTabularOut();
    tabTimer = setTimeout(sendTabular, 300);
  }
  function sendTabular() {
    var seq = ++S.seq.tabular, sess = S.session, body = {};
    S.features.forEach(function (f) { body[f.name] = f.kind === "count" ? Math.round(S.values[f.name]) : S.values[f.name]; });
    S.tabPending = true;
    S.tabOut = null;
    renderTabularOut();
    api("/api/tabular", body).then(function (r) {
      if (seq !== S.seq.tabular || sess !== S.session) return;
      S.tabPending = false;
      if (r.kind === "ok") {
        sess.results.tabular = { resp: r.data, inputs: body };
        sess.summary.tabular = "Session estimated";
        announce("tabular", "Estimated probability " + pctWords(r.data.probability));
        renderTabularOut();
        touch();
      } else if (r.kind === "invalid") {
        var d = r.data && r.data.detail;
        S.features.forEach(function (f) {
          var hit = Array.isArray(d) ? d.some(function (x) { return x.loc && x.loc[1] === f.name; })
            : typeof d === "string" && d.indexOf(f.name + " ") === 0;
          if (hit) fieldError(document.getElementById("f-" + f.name), "Enter a value from 0 to " + fmtCap(f) + ".");
        });
        renderTabularOut();
      } else if (r.kind === "unavailable") {
        markUnavailable("tabular", r.data);
      } else {
        S.tabOut = { r: r, retry: sendTabular };
        renderTabularOut();
      }
    });
  }
  function sourceBadge(kind) {
    return kind === "real" ? el("span", "badge ready", "Real data (UCI)") : el("span", "badge warning", "Synthetic data");
  }
  function renderTabularOut() {
    var out = $("#tabular-out"), t = S.session.results.tabular;
    out.replaceChildren();
    if (S.tabOut) return add(out, failureNotice(S.tabOut.r, S.tabOut.retry));
    var upd = el("p", "updating", S.tabPending ? "Updating…" : "");
    upd.setAttribute("aria-hidden", "true");
    if (!t) return add(out, el("p", "metric muted", "–"), upd);
    var p = t.resp.probability, bar = el("div", "bar prob-bar"), fill = el("span");
    fill.style.width = Math.max(0, Math.min(100, p * 100)) + "%";
    bar.setAttribute("aria-hidden", "true");
    add(out, el("p", "metric", pct(p)), upd, add(bar, fill), sourceBadge(t.resp.source_kind),
      el("p", "muted", t.resp.source_kind === "real"
        ? "An estimate for one completed session from a model trained on the UCI online shoppers data. This is not a measured conversion rate."
        : "An example estimate from a model trained on generated sessions. This is not a measured conversion rate."));
  }

  // ------------------------------------------------------------------ providers
  function buildProviders(p) {
    var opts = [{ value: "offline:", label: "Offline Template", enabled: true, status: "Available",
      desc: "Deterministic text from your results. No key, model download, or network needed." }];
    var cli = Array.isArray(p.cli) ? p.cli.filter(function (c) { return VENDORS.indexOf(c.id) >= 0; }) : [];
    var onDev = cli.filter(function (c) { return c.on_device && c.available; });
    opts.push({ value: "cli:", label: "Auto-Detect Local CLI", enabled: onDev.length > 0,
      status: onDev.length ? "Available" : "Not detected",
      desc: "Only confirmed on-device profiles can run offline. Hosted tools are labeled." +
        (onDev.length ? " Detected: " + onDev.map(function (c) { return CLI_NAME[c.id]; }).join(", ") + "." : "") });
    cli.forEach(function (c) {
      if (c.on_device || !c.available) return; // hosted CLIs only as an explicit choice (D-016)
      opts.push({ value: "cli:" + c.id, label: CLI_NAME[c.id] + " CLI", enabled: true, status: "Hosted",
        desc: "Runs the installed " + CLI_NAME[c.id] + " command-line tool, which sends the results to its hosted service." });
    });
    (Array.isArray(p.cloud) ? p.cloud : []).forEach(function (c) {
      if (VENDORS.indexOf(c.id) < 0) return;
      opts.push({ value: "cloud:" + c.id, label: String(c.label) + " API", enabled: !!c.available,
        status: c.available ? "Available" : "Not configured",
        desc: "Use an explicitly configured provider. Credentials stay on the server." });
    });
    S.providers = opts;
    var sel = $("#provider"), cur = sel.value;
    sel.replaceChildren();
    opts.forEach(function (o) {
      var op = el("option", "", o.label + (o.status === "Available" ? "" : " (" + o.status + ")"));
      op.value = o.value; op.disabled = !o.enabled;
      sel.appendChild(op);
    });
    sel.value = opts.some(function (o) { return o.value === cur && o.enabled; }) ? cur : "offline:";
  }
  function providerFor(value) { return S.providers.find(function (o) { return o.value === value; }) || S.providers[0]; }
  function renderProviderCards() {
    var box = $("#provider-cards"), sel = $("#provider");
    box.replaceChildren();
    S.providers.forEach(function (o) {
      var card = el("label", "pcard"), input = el("input");
      input.type = "radio"; input.name = "provider-card"; input.value = o.value;
      input.checked = sel.value === o.value; input.disabled = !o.enabled;
      var selected = input.checked;
      var badge = el("span", "badge " + (selected ? "ready" : o.status === "Hosted" ? "warning" : "example"),
        selected ? "Selected" : o.status);
      add(card, input, el("span", "heading p-title", o.label), badge, el("span", "p-desc", o.desc));
      input.addEventListener("change", function () { sel.value = o.value; renderProviderCards(); $('input[value="' + o.value + '"]', box).focus(); });
      box.appendChild(card);
    });
  }
  $("#provider").addEventListener("change", function () { if (S.view === "providers") renderProviderCards(); });
  function usedLabel(resp) {
    if (resp.mode_used === "cli") return (CLI_NAME[resp.provider_used] || resp.provider_used) + " CLI";
    if (resp.mode_used === "cloud") {
      var o = providerFor("cloud:" + resp.provider_used);
      return o && o.value === "cloud:" + resp.provider_used ? o.label : resp.provider_used + " API";
    }
    return "Offline Template";
  }

  // ------------------------------------------------------------------ explanation
  function available(h) { return !!S.session.results[h] && !S.unavail[h]; }
  function refreshExplainButton() {
    var b = $("#explain");
    if (b.getAttribute("aria-busy") === "true") return;
    b.disabled = !HEADS.some(available);
  }
  function setExplainBusy(busy) {
    var b = $("#explain");
    if (busy) setBusy(b, true, "Explaining…"); else { setBusy(b, false); refreshExplainButton(); }
  }
  function explain() {
    var sess = S.session, heads = HEADS.filter(available);
    if (!heads.length) return;
    var parts = $("#provider").value.split(":"), opt = providerFor($("#provider").value);
    var body = { mode: parts[0], vendor: parts[1] || null };
    heads.forEach(function (h) { body[h] = sess.results[h].resp; });
    var missing = HEADS.filter(function (h) { return heads.indexOf(h) < 0; }).map(function (h) {
      return HEAD_LABEL[h] + " (" + (S.unavail[h] ? "unavailable" : "no result yet") + ")";
    });
    var seq = ++S.seq.report, version = sess.version;
    S.explainPending = true; S.explainError = null;
    setExplainBusy(true);
    show("results");
    api("/api/report", body).then(function (r) {
      if (seq !== S.seq.report || sess !== S.session) return; // superseded: discard the late response
      S.explainPending = false;
      setExplainBusy(false);
      if (r.kind === "ok") {
        sess.explanation = { resp: r.data, requested: opt.label, version: version, missing: missing };
        var fb = r.data.fallback_reason;
        announce("explain", fb ? "Explanation used Offline Template. " + opt.label + " " + (REASON[fb] || "failed") + "."
          : "Explanation ready via " + usedLabel(r.data) + ".");
        upsertSession();
      } else {
        S.explainError = { r: r, retry: explain };
      }
      if (S.view === "results") renderExplanation();
    });
  }
  $("#explain").addEventListener("click", explain);

  function renderExplanation() {
    var box = $("#explanation"), ex = S.session.explanation;
    box.replaceChildren();
    box.hidden = false;
    if (S.explainPending) {
      return add(box, el("p", "eyebrow", "EXPLANATION"),
        el("p", "muted", "Preparing an explanation from the current results… You can keep drawing or generate another set of names."));
    }
    if (S.explainError) return add(box, el("p", "eyebrow", "EXPLANATION"), failureNotice(S.explainError.r, S.explainError.retry));
    if (!ex) { box.hidden = true; return; }
    var fb = ex.resp.fallback_reason, head = el("div", "ex-head");
    add(head, el("p", "eyebrow", fb ? "OFFLINE FALLBACK" : "EXPLANATION"),
      el("p", "provenance" + (fb ? " fallback" : ""), fb
        ? ex.requested + " " + (REASON[fb] || "failed") + " / Offline Template used"
        : "via " + usedLabel(ex.resp)));
    box.appendChild(head);
    if (ex.version !== S.session.version) box.appendChild(el("p", "small muted", "Based on earlier results"));
    box.appendChild(el("p", "ex-text", ex.resp.text));
    if (ex.missing && ex.missing.length) box.appendChild(el("p", "small muted", "Not included: " + ex.missing.join(", ")));
  }

  // ------------------------------------------------------------------ results overview
  function thumb(pixels) {
    var c = el("canvas", "thumb"); c.width = 28; c.height = 28;
    c.setAttribute("role", "img"); c.setAttribute("aria-label", "Centred 28 by 28 input");
    var g = c.getContext("2d"), img = g.createImageData(28, 28);
    for (var i = 0; i < 784; i++) {
      img.data[i * 4] = INK[0]; img.data[i * 4 + 1] = INK[1]; img.data[i * 4 + 2] = INK[2]; img.data[i * 4 + 3] = pixels[i] || 0;
    }
    g.putImageData(img, 0, 0);
    return c;
  }
  function card(i, title, model, head) {
    var c = el("article", "card"), h = el("div", "card-head");
    add(h, el("span", "idx", pad2(i)), el("h2", "heading", title));
    add(c, h, el("p", "model", model));
    if (S.unavail[head]) {
      add(c, add(el("div", "row"), el("span", "badge unavailable", HEAD_LABEL[head] + ": unavailable"),
        S.session.results[head] ? el("span", "small prev-label", "Previous output") : null));
    }
    if (!S.session.results[head]) c.appendChild(el("p", "muted", "No result yet."));
    return c;
  }
  function renderResults() {
    var s = S.session, cards = $("#cards");
    $("#run-summary").textContent = runSummary(s) || "No results yet. Choose a task to start.";
    cards.replaceChildren();
    var v = s.results.vision, n = s.results.names, t = s.results.tabular;
    var cv = card(1, "Digit recognition", "MNIST / 101,770 parameters", "vision");
    if (v) {
      var side = add(el("div"), el("p", "number", v.resp.digit), el("p", "small accent", pct(v.resp.top3[0].prob) + " confidence"));
      add(cv, add(el("div", "thumb-row"), thumb(v.pixels), side), bars(v.resp.top3));
    }
    var cn = card(2, "Name generation", "Character model / 3,571 parameters", "names");
    if (n) {
      var ol = el("ol", "mini-list");
      n.resp.names.forEach(function (name, i) { ol.appendChild(add(el("li"), el("span", "idx", pad2(i + 1)), el("span", "", name))); });
      add(cn, ol, add(el("div", "card-foot"), el("span", "small accent", n.resp.accepted + " of " + n.resp.requested + " accepted"),
        el("span", "small muted num", "T " + n.temperature + (n.seed === null || n.seed === undefined ? "" : " / seed " + n.seed))));
    }
    var ct = card(3, "Purchase probability", "Session model / 769 parameters", "tabular");
    if (t) {
      add(ct, el("p", "metric", pct(t.resp.probability)), el("p", "small muted", "Estimated probability for this session"),
        sourceBadge(t.resp.source_kind),
        el("p", "small muted", plural(t.inputs.ProductRelated || 0, "product page") + " / " +
          fmtVal({ kind: "duration" }, t.inputs.ProductRelated_Duration || 0)), el("p", "small muted", "Completed-session inputs"));
    }
    add(cards, cv, cn, ct);
    renderExplanation();
  }

  // ------------------------------------------------------------------ drawer (≤767) and dialog
  var sidebar = $("#sidebar"), menu = $("#menu"), mainCol = $("#main-col"), backdrop = null;
  var mobile = window.matchMedia("(max-width: 767px)");
  function openDrawer() {
    sidebar.classList.add("open");
    sidebar.setAttribute("role", "dialog"); sidebar.setAttribute("aria-modal", "true");
    mainCol.inert = true;
    backdrop = el("div", "backdrop");
    backdrop.addEventListener("click", function () { closeDrawer(true); });
    document.body.appendChild(backdrop);
    menu.setAttribute("aria-expanded", "true");
    $("#new-session").focus();
  }
  function closeDrawer(restoreFocus) {
    if (!sidebar.classList.contains("open")) return;
    sidebar.classList.remove("open");
    sidebar.removeAttribute("role"); sidebar.removeAttribute("aria-modal");
    mainCol.inert = false;
    if (backdrop) backdrop.remove();
    backdrop = null;
    menu.setAttribute("aria-expanded", "false");
    if (restoreFocus) menu.focus();
  }
  menu.addEventListener("click", openDrawer);
  sidebar.addEventListener("keydown", function (e) { // modal drawer: Tab wraps inside it
    if (e.key !== "Tab" || !sidebar.classList.contains("open")) return;
    var f = $all("button, select, input, a[href]", sidebar).filter(function (n) { return !n.disabled && n.offsetParent !== null; });
    if (!f.length) return;
    var first = f[0], lastF = f[f.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); lastF.focus(); }
    else if (!e.shiftKey && document.activeElement === lastF) { e.preventDefault(); first.focus(); }
  });
  document.addEventListener("keydown", function (e) { if (e.key === "Escape" && sidebar.classList.contains("open")) closeDrawer(true); });
  mobile.addEventListener("change", function () { if (!mobile.matches) closeDrawer(false); });

  var dialog = $("#confirm-clear");
  $("#clear-history").addEventListener("click", function () { dialog.returnValue = ""; dialog.showModal(); });
  dialog.addEventListener("close", function () {
    if (dialog.returnValue === "ok") {
      try { window.localStorage.removeItem(HISTORY_KEY); } catch (e) { historyOk = false; }
      history = [];
      invalidatePending();
      S.session = newSession();
      clearPad(); resetSliders(); resetForSession();
      renderHistory();
      show("tasks"); // the trigger (Clear history) is hidden once the list is empty: focus the view heading
    } else {
      $("#clear-history").focus();
    }
  });

  // ------------------------------------------------------------------ wiring and start
  $all("[data-go]").forEach(function (b) { b.addEventListener("click", function () { show(b.getAttribute("data-go")); }); });
  $("#new-session").addEventListener("click", startNewSession);
  $("#new-session-bottom").addEventListener("click", startNewSession);

  try {
    loadHistory();
    S.session = history.length ? clone(history[0]) : newSession();
    renderHistory();
    resetForSession();
    show(HEADS.some(function (h) { return S.session.results[h]; }) ? "results" : "tasks", false);
  } catch (e) { // never let a stored session stop the app
    history = []; S.session = newSession();
    try { renderHistory(); resetForSession(); show("tasks", false); } catch (e2) { /* status still loads below */ }
  } finally {
    loadStatus();
  }
})();
