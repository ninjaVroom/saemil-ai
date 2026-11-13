"""AI bridge: optional prose explanation of the three head results.

The bridge never predicts. Every path ends in some report: any provider failure becomes the deterministic offline
template plus a `fallback_reason` code; nothing raises to the caller. One provider per report (D-007).
Stdlib only (urllib, subprocess, shutil, asyncio, json).

Rules enforced here:
- Keys live only in the server-side config dict. They are sent only in vendor request headers and never appear in
  ProviderResult, status(), exception text or logs (fallback reasons are fixed codes).
- A local CLI failure never escalates to a cloud or hosted provider: the fallback is always the offline template.
  Auto-detect (mode "cli", vendor None) only picks on-device profiles; hosted CLIs must be named explicitly.
- CLI: resolved path + argv list, shell=False, prompt via UTF-8 stdin only, cwd = a fresh empty temp dir,
  child env without *_API_KEY, stdout to a temp file polled for size (killed past CLI_MAX_BYTES), child process
  tree killed on timeout, one CLI job at a time. The one-job slot is a threading.Lock taken inside the worker
  thread and released only after the child is reaped, so a cancelled or timed-out await cannot free it early
  (D-016; replaces amendment 4.1's asyncio.Semaphore(1) with the same one-job meaning). Busy => "queue_full".
- CLI timeout defaults to 15 seconds.
- Hosted CLIs run as text-only, single-turn calls (D-016). Flags checked 2026-11-12 against:
  Claude Code CLI reference https://code.claude.com/docs/en/cli-reference
    -p (print mode, reads piped stdin), --tools "" (disable all built-in tools), --disallowedTools "*" (deny rule
    that "removes every tool", MCP tools included), --strict-mcp-config (only MCP
    servers from --mcp-config, none given => none), --setting-sources "" (load no user/project/local settings),
    --disable-slash-commands, --no-session-persistence, --max-turns 1. All eight are listed in that reference.
  Gemini CLI reference https://geminicli.com/docs/cli/cli-reference/ , headless https://geminicli.com/docs/cli/headless/ ,
  plan mode https://geminicli.com/docs/cli/plan-mode/ , policy engine https://geminicli.com/docs/reference/policy-engine/
    Piped stdin in a non-TTY runs headless. --approval-mode plan was NOT used: the plan-mode page says leaving plan
    mode non-interactively switches to YOLO. --approval-mode default is used instead: the policy engine treats
    "ask_user" as deny in non-interactive mode, so write/shell tools are refused; read-only tools stay allowed
    (they see only the empty temp cwd). --allowed-mcp-server-names with a name no server has, so no MCP server loads.
    Gemini has no documented CLI flag that disables every tool; a deny-all policy file is the upgrade path.
- ollama counts as on-device only after `ollama list` shows the exact configured model, its tag has no
  "cloud" in it, and OLLAMA_HOST is unset or loopback (a remote daemon is hosted, not on-device). llm is on-device only when SAEMIL_LLM_ON_DEVICE=1 (no plugin check is made).

Vendor request/response shapes checked against official docs on 2026-11-12:
- Anthropic Messages: https://platform.claude.com/docs/en/api/messages
  POST https://api.anthropic.com/v1/messages, headers x-api-key + anthropic-version: 2023-06-01,
  body {model, max_tokens, messages}; text = content[] blocks with type "text".
- Gemini generateContent: https://aistudio.google.com/api-reference/generate-content and
  https://aistudio.google.com/docs/text-generation (x-goog-api-key header, host generativelanguage.googleapis.com)
  POST /v1beta/models/{model}:generateContent, body {contents:[{role, parts:[{text}]}]};
  text = candidates[0].content.parts[].text; no candidates => promptFeedback.blockReason.
- OpenAI Responses: https://developers.openai.com/api/reference/resources/responses/methods/create and
  https://developers.openai.com/api/reference/overview (Authorization: Bearer)
  POST https://api.openai.com/v1/responses, body {model, input, max_output_tokens};
  text = output[] items type "message" -> content[] type "output_text"; "refusal" content => refused.
- DeepSeek: https://api-docs.deepseek.com/ and https://api-docs.deepseek.com/api/create-chat-completion
  POST https://api.deepseek.com/chat/completions, Authorization: Bearer, body {model, messages, max_tokens};
  text = choices[0].message.content.
Model IDs come only from env; an empty key or model leaves that provider unavailable.
"""
import asyncio
import ipaddress
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

DEFAULT_CLI_TIMEOUT_S = 15.0
HTTP_TIMEOUT_S = 15.0
HTTP_MAX_BYTES = 256 * 1024
CLI_MAX_BYTES = 64 * 1024
DEFAULT_MAX_CHARS = 1200
CLI_TOOLS = ("ollama", "llm", "claude", "gemini")
CLOUD_KEYS = {"claude": "ANTHROPIC", "gemini": "GEMINI", "openai": "OPENAI", "deepseek": "DEEPSEEK"}
_CLI_LOCK = threading.Lock()  # one CLI job at a time; held by the worker thread (see docstring)
OLLAMA_LIST_TIMEOUT_S = 3.0
OFFLINE_UNAVAILABLE = "A report could not be produced from these results."
CLAUDE_FLAGS = ["-p", "--tools", "", "--disallowedTools", "*", "--strict-mcp-config", "--setting-sources", "", "--disable-slash-commands",
                "--no-session-persistence", "--max-turns", "1"]
GEMINI_FLAGS = ["--approval-mode", "default", "--allowed-mcp-server-names", "saemil-no-mcp"]


class BridgeError(Exception):
    """Carries a fixed reason code only - never vendor text, URLs or keys."""


@dataclass
class ProviderRequest:
    task_type: str
    fields: dict | None  # untrusted; clean_fields() filters it
    max_chars: int = DEFAULT_MAX_CHARS


@dataclass
class ProviderResult:
    text: str
    provider_id: str
    mode_used: str
    latency_ms: int
    fallback_reason: str | None


# ---------------------------------------------------------------- permitted fields + offline template

NAME_RE = re.compile(r"[a-z]{3,12}")  # the names head's alphabet and length bounds; anything else is dropped


def _prob(x):
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        return None
    try:  # range check first: comparing 10**400 is exact, converting it would overflow; NaN fails the range
        return float(x) if 0 <= x <= 1 else None
    except (OverflowError, TypeError, ValueError):
        return None


def _digit(x):
    return int(x) if isinstance(x, int) and not isinstance(x, bool) and 0 <= x <= 9 else None


def _count(x):
    return int(x) if isinstance(x, int) and not isinstance(x, bool) and 0 <= x <= 1000 else None


def clean_fields(fields):
    """Keep only the permitted, well-formed head outputs; anything else becomes None (= unavailable)."""
    fields = fields if isinstance(fields, dict) else {}
    out: dict[str, dict | None] = {"vision": None, "names": None, "tabular": None}
    v = fields.get("vision")
    if isinstance(v, dict):
        if v.get("blank") is True:
            out["vision"] = {"blank": True}
        else:
            top3 = [{"digit": _digit(t.get("digit")), "prob": _prob(t.get("prob"))}
                    for t in (v.get("top3") or [])[:3] if isinstance(t, dict)] if isinstance(v.get("top3"), list) else []
            if _digit(v.get("digit")) is not None and top3 and all(None not in t.values() for t in top3):
                out["vision"] = {"digit": _digit(v["digit"]), "top3": top3}
    n = fields.get("names")
    if isinstance(n, dict) and isinstance(n.get("names"), list):
        names = [s for s in n["names"][:10] if isinstance(s, str) and NAME_RE.fullmatch(s)]
        if names or not n["names"]:  # names sent that all fail the alphabet are invalid, not "none were found"
            out["names"] = {"names": names, "accepted": _count(n.get("accepted")),
                            "requested": _count(n.get("requested"))}
    t = fields.get("tabular")
    if isinstance(t, dict) and _prob(t.get("probability")) is not None and t.get("source_kind") in ("real", "synthetic"):
        out["tabular"] = {"probability": _prob(t["probability"]), "source_kind": t["source_kind"]}
    return out


def offline_report(fields):
    """Deterministic template (amendment 4.2) that names any missing or blank head instead of failing."""
    f = clean_fields(fields)
    v, n, t = f["vision"], f["names"], f["tabular"]
    if v is None:
        parts = ["No valid digit result was available."]
    elif v.get("blank"):
        parts = ["The digit canvas was blank, so there is no digit prediction."]
    else:
        parts = [f"The drawn digit most resembles {v['digit']} ({v['top3'][0]['prob'] * 100:.1f}% confidence)."]
    if n is None:
        parts.append("No valid name result was available.")
    else:
        tally = (f" ({n['accepted']} of {n['requested']} requested)"
                 if n["accepted"] is not None and n["requested"] is not None and n["accepted"] < n["requested"] else "")
        parts.append(f"Proposed brand name candidates: {', '.join(n['names'][:2])}{tally}." if n["names"]
                     else f"The name generator returned no novel names{tally}.")
    if t is None:
        parts.append("No valid purchase-probability result was available.")
    else:
        parts.append(f"Session features indicate an estimated {t['probability'] * 100:.1f}% "
                     f"purchase probability (Source: {t['source_kind']}).")
    return " ".join(parts)


def safe_offline_report(fields):
    try:
        return offline_report(fields)
    except Exception:  # last resort; clean_fields should make this unreachable
        return OFFLINE_UNAVAILABLE


def build_prompt(req):
    data = json.dumps(clean_fields(req.fields), sort_keys=True)
    return ("Explain these local model outputs to a non-expert in two or three plain sentences. "
            "Use only the numbers given; do not invent results; say so if a result is missing. "
            f"Reply with plain text under {req.max_chars} characters.\nData: {data}\n")


def _finish(text, max_chars):
    text = "".join(c for c in (text or "") if c in "\n\t" or c.isprintable()).strip()
    if not text:
        raise BridgeError("empty_output")
    return text[:max_chars]


# ---------------------------------------------------------------- cloud adapters

def _post_json(url, headers, body, timeout, max_bytes):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(max_bytes + 1)
    except urllib.error.HTTPError as e:
        e.close()
        raise BridgeError(f"http_{e.code}") from None
    except TimeoutError:
        raise BridgeError("timeout") from None
    except (urllib.error.URLError, OSError, ValueError):
        raise BridgeError("network_error") from None
    if len(raw) > max_bytes:
        raise BridgeError("oversized_response")
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise BridgeError("bad_json") from None


def _tokens(max_chars):
    return max(1024, max_chars // 3)  # generous allowance; the text is cut to max_chars afterwards


def claude_request(prompt, key, model, max_chars):
    return ("https://api.anthropic.com/v1/messages", {"x-api-key": key, "anthropic-version": "2023-06-01"},
            {"model": model, "max_tokens": _tokens(max_chars), "messages": [{"role": "user", "content": prompt}]})


def claude_parse(data):
    return "".join(b["text"] for b in data["content"] if b.get("type") == "text" and isinstance(b.get("text"), str))


def gemini_request(prompt, key, model, max_chars):
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"{urllib.parse.quote(model, safe='')}:generateContent")
    return (url, {"x-goog-api-key": key},
            {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
             "generationConfig": {"maxOutputTokens": _tokens(max_chars)}})


def gemini_parse(data):
    if not data.get("candidates"):
        raise BridgeError("blocked" if (data.get("promptFeedback") or {}).get("blockReason") else "malformed_response")
    finish = data["candidates"][0].get("finishReason")
    if finish not in (None, "STOP"):  # MAX_TOKENS, SAFETY, RECITATION, ... -> their own fixed codes
        raise BridgeError("gemini_" + finish.lower() if re.fullmatch(r"[A-Z_]{1,32}", str(finish)) else "gemini_other")
    parts = (data["candidates"][0].get("content") or {}).get("parts") or []
    return "".join(p["text"] for p in parts if isinstance(p.get("text"), str))


def openai_request(prompt, key, model, max_chars):
    return ("https://api.openai.com/v1/responses", {"Authorization": f"Bearer {key}"},
            {"model": model, "input": prompt, "max_output_tokens": _tokens(max_chars)})


def openai_parse(data):
    if data.get("status") == "incomplete":
        raise BridgeError("incomplete")
    content = [c for item in data["output"] if item.get("type") == "message" for c in item.get("content") or []]
    text = "".join(c["text"] for c in content if c.get("type") == "output_text" and isinstance(c.get("text"), str))
    if not text and any(c.get("type") == "refusal" for c in content):
        raise BridgeError("refused")
    return text


def deepseek_request(prompt, key, model, max_chars):
    return ("https://api.deepseek.com/chat/completions", {"Authorization": f"Bearer {key}"},
            {"model": model, "max_tokens": _tokens(max_chars), "stream": False,
             "messages": [{"role": "user", "content": prompt}]})


def deepseek_parse(data):
    if not data.get("choices"):
        raise BridgeError("malformed_response")
    return data["choices"][0]["message"]["content"] or ""


VENDORS = {"claude": ("Claude", claude_request, claude_parse), "gemini": ("Gemini", gemini_request, gemini_parse),
           "openai": ("OpenAI", openai_request, openai_parse),
           "deepseek": ("DeepSeek", deepseek_request, deepseek_parse)}


class CloudAPI:
    mode = "cloud"
    on_device = False

    def __init__(self, vid, key, model, timeout_s=HTTP_TIMEOUT_S):
        self.id = vid
        self.label, self._build, self._parse = VENDORS[vid]
        self._key, self._model, self.timeout_s = key or "", model or "", timeout_s

    def __repr__(self):
        return f"CloudAPI({self.id!r}, available={self.available()})"

    def available(self):
        return bool(self._key and self._model)

    def generate(self, req):
        if not self._key:
            raise BridgeError("missing_key")
        if not self._model:
            raise BridgeError("missing_model")
        url, headers, body = self._build(build_prompt(req), self._key, self._model, req.max_chars)
        data = _post_json(url, headers, body, self.timeout_s, HTTP_MAX_BYTES)
        try:
            text = self._parse(data)
        except BridgeError:
            raise
        except (KeyError, IndexError, TypeError, AttributeError):
            raise BridgeError("malformed_response") from None
        return _finish(text, req.max_chars)


# ---------------------------------------------------------------- local CLI

def _child_env():
    return {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}  # no silent paid-API use


if sys.platform == "win32":
    _NO_WINDOW = subprocess.CREATE_NO_WINDOW
else:
    _NO_WINDOW = 0


def _kill_tree(proc):
    if os.name == "nt":  # TerminateProcess on the parent alone would orphan e.g. node under a .cmd shim
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, shell=False,
                       creationflags=_NO_WINDOW)
    proc.kill()
    proc.wait()


class LocalCLI:
    """One instance per configured profile. argv None means the executable or its model is not usable."""
    mode = "cli"

    def __init__(self, name, argv, on_device, timeout_s=DEFAULT_CLI_TIMEOUT_S):
        self.id, self.argv, self.on_device, self.timeout_s = name, argv, on_device, timeout_s

    @property
    def label(self):
        return f"{self.id} ({'not available' if self.argv is None else 'on-device' if self.on_device else 'hosted'})"

    def available(self):
        return self.argv is not None

    def generate(self, req):
        if self.argv is None:
            raise BridgeError("missing_executable")
        if not _CLI_LOCK.acquire(blocking=False):
            raise BridgeError("queue_full")
        try:  # released only after the child has been reaped, whatever happened to the awaiting caller
            with tempfile.TemporaryDirectory(prefix="saemil-cli-", ignore_cleanup_errors=True) as cwd, \
                    tempfile.TemporaryFile() as out:
                raw = self._run(build_prompt(req).encode("utf-8"), cwd, out)
        finally:
            _CLI_LOCK.release()
        return _finish(raw.decode("utf-8", errors="replace"), req.max_chars)

    def _run(self, prompt, cwd, out):
        try:
            proc = subprocess.Popen(self.argv, stdin=subprocess.PIPE, stdout=out, stderr=subprocess.DEVNULL,
                                    shell=False, cwd=cwd, env=_child_env(), creationflags=_NO_WINDOW)
        except OSError:
            raise BridgeError("missing_executable") from None
        with proc:  # closes stdin and reaps the child on every path
            try:  # the prompt is about 1 KB (names are capped), well under a pipe buffer, so this cannot block
                assert proc.stdin is not None  # stdin=PIPE above
                proc.stdin.write(prompt)
                proc.stdin.close()
            except OSError:
                pass  # child exited without reading; its exit code decides
            deadline = time.monotonic() + self.timeout_s
            while True:
                try:
                    proc.wait(timeout=0.05)
                    break
                except subprocess.TimeoutExpired:
                    if os.fstat(out.fileno()).st_size > CLI_MAX_BYTES:
                        _kill_tree(proc)
                        raise BridgeError("oversized_output") from None
                    if time.monotonic() > deadline:
                        _kill_tree(proc)
                        raise BridgeError("timeout") from None
        if proc.returncode != 0:
            raise BridgeError("nonzero_exit")
        if os.fstat(out.fileno()).st_size > CLI_MAX_BYTES:
            raise BridgeError("oversized_output")
        out.seek(0)
        return out.read(CLI_MAX_BYTES)


def _ollama_list(path):
    """stdout of `ollama list`, or "" on any failure. Same isolation as a report job."""
    try:
        with tempfile.TemporaryDirectory(prefix="saemil-cli-", ignore_cleanup_errors=True) as cwd:
            r = subprocess.run([path, "list"], capture_output=True, timeout=OLLAMA_LIST_TIMEOUT_S, shell=False,
                               cwd=cwd, env=_child_env(), stdin=subprocess.DEVNULL, creationflags=_NO_WINDOW)
        return r.stdout.decode("utf-8", errors="replace") if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def ollama_has_local_model(listing, model):
    """True only for the exact configured NAME in `ollama list` output, and never for a *cloud* tag."""
    tag = model.split(":", 1)[1] if ":" in model else ""
    if "cloud" in tag.lower():
        return False
    return model in [line.split()[0] for line in listing.splitlines()[1:] if line.split()]


def is_loopback_host(value):
    """OLLAMA_HOST unset/empty, localhost, 127.0.0.0/8 or ::1 (optional scheme and port). Anything else is remote."""
    value = (value or "").strip()
    if not value:
        return True
    try:
        host = urllib.parse.urlsplit(value if "://" in value else f"http://{value}").hostname
    except ValueError:
        return False
    if not host:  # e.g. ":11434" - the client then uses its default, 127.0.0.1
        return True
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _cli_profile(name, config):
    """Fixed invocation profiles; the prompt is never an argument. Returns (argv or None, on_device)."""
    path = shutil.which(name)
    if name == "ollama":
        model = config.get("ollama_model")
        ok = bool(path and model) and ollama_has_local_model(_ollama_list(path), model)
        return ([path, "run", model] if ok else None), ok and is_loopback_host(config.get("ollama_host"))
    if name == "llm":  # llm's default model may be a paid API, so a model is required; hosted unless declared
        model = config.get("llm_model")
        return ([path, "-m", model] if path and model else None), bool(config.get("llm_on_device"))
    if name == "claude":
        return ([path, *CLAUDE_FLAGS] if path else None), False
    if name == "gemini":
        return ([path, *GEMINI_FLAGS] if path else None), False
    return None, False


# ---------------------------------------------------------------- config, registry, status, run

def load_config(env=None):
    env = os.environ if env is None else env
    try:
        timeout = float(env.get("SAEMIL_CLI_TIMEOUT_S") or DEFAULT_CLI_TIMEOUT_S)
        timeout = timeout if 1 <= timeout <= 120 else DEFAULT_CLI_TIMEOUT_S  # NaN fails the range too
    except ValueError:
        timeout = DEFAULT_CLI_TIMEOUT_S
    tools = [t.strip() for t in (env.get("SAEMIL_CLI_TOOLS") or ",".join(CLI_TOOLS)).split(",")]
    cfg = {"cli_tools": [t for t in CLI_TOOLS if t in tools], "cli_timeout_s": timeout,
           "ollama_model": env.get("SAEMIL_OLLAMA_MODEL", "").strip(),
           "ollama_host": env.get("OLLAMA_HOST", "").strip(),
           "llm_model": env.get("SAEMIL_LLM_MODEL", "").strip(),
           "llm_on_device": env.get("SAEMIL_LLM_ON_DEVICE", "").strip() == "1"}
    for vid, prefix in CLOUD_KEYS.items():
        cfg[f"{vid}_key"] = env.get(f"{prefix}_API_KEY", "").strip()
        cfg[f"{vid}_model"] = env.get(f"{prefix}_MODEL", "").strip()
    return cfg


def registry(config):
    """Keyed "mode:id": vendor "claude" means the Claude CLI in mode cli and the Claude API in mode cloud.
    runs `ollama list` on every call; cache the registry in the studio if status polling gets hot."""
    providers: dict[str, LocalCLI | CloudAPI] = {}
    for name in config["cli_tools"]:
        argv, on_device = _cli_profile(name, config)
        providers[f"cli:{name}"] = LocalCLI(name, argv, on_device, config["cli_timeout_s"])
    for vid in VENDORS:
        providers[f"cloud:{vid}"] = CloudAPI(vid, config[f"{vid}_key"], config[f"{vid}_model"])
    return providers


def status(providers):
    """What /api/status may show: ids, labels, availability booleans. No keys, models, paths or env values."""
    return {"offline": {"available": True, "label": "Offline Template"},
            "cli": [{"id": p.id, "label": p.label, "available": p.available(), "on_device": p.on_device}
                    for p in providers.values() if p.mode == "cli"],
            "cloud": [{"id": p.id, "label": p.label, "available": p.available()}
                      for p in providers.values() if p.mode == "cloud"]}


def _pick(mode, vendor, providers):
    """Resolve exactly one provider for the mode, or a reason code. Never crosses from cli to cloud."""
    if mode == "cli" and vendor is None:
        ready = [p for p in providers.values() if p.mode == "cli" and p.on_device and p.available()]
        return (ready[0], None) if ready else (None, "no_on_device_cli")
    p = providers.get(f"{mode}:{vendor}") if isinstance(vendor, str) else None
    if p is None or p.mode != mode:
        return None, "unknown_provider"
    return p, None


async def run_report(mode, vendor, req, providers=None):
    """Always returns a ProviderResult; failures become the offline template with fallback_reason set."""
    t0 = time.perf_counter()
    fields = getattr(req, "fields", None)

    def result(text, pid, used, reason):
        return ProviderResult(text, pid, used, int((time.perf_counter() - t0) * 1000), reason)

    def fallback(reason):
        return result(safe_offline_report(fields), "offline", "offline", reason)

    try:
        max_chars = getattr(req, "max_chars", DEFAULT_MAX_CHARS)
        max_chars = min(max(max_chars, 200), 4000) if type(max_chars) is int else DEFAULT_MAX_CHARS
        req = ProviderRequest(str(getattr(req, "task_type", "report")), fields, max_chars)
        if mode == "offline":
            return result(safe_offline_report(fields), "offline", "offline", None)
        if mode not in ("cli", "cloud"):
            return fallback("unknown_mode")
        if providers is None:  # registry may run `ollama list` (up to 3 s): keep it off the event loop
            providers = await asyncio.to_thread(registry, load_config())
        p, reason = _pick(mode, vendor, providers)
        if p is None:
            return fallback(reason)
        if not p.available() and mode == "cli":
            return fallback("missing_executable")
        # the CLI's own deadline kills the child; this outer bound only stops the caller waiting forever
        text = await asyncio.wait_for(asyncio.to_thread(p.generate, req), timeout=p.timeout_s + 2)
        return result(text, p.id, mode, None)
    except BridgeError as e:
        return fallback(str(e))
    except (asyncio.TimeoutError, TimeoutError):
        return fallback("timeout")
    except Exception:  # never raise to the caller (amendment 4.1)
        return fallback("provider_error")


if __name__ == "__main__":  # python saemil_bridge.py -> offline report for a fixed sample
    sample = {"vision": {"blank": False, "digit": 7, "top3": [{"digit": 7, "prob": 0.914}]},
              "names": {"names": ["narela", "vimo"], "accepted": 2, "requested": 2},
              "tabular": {"probability": 0.24, "source_kind": "synthetic"}}
    print(asyncio.run(run_report("offline", None, ProviderRequest("report", sample))).text, file=sys.stdout)
