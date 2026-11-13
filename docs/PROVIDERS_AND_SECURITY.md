# Explanation providers and security

## Providers

The Studio's explanation is written by one provider per request. It never makes predictions; it only describes the three heads' results.

| Mode | What runs | Needs |
|---|---|---|
| `offline` (default) | Deterministic template in `saemil_bridge.py` | Nothing |
| `cli` | A local command-line tool: `ollama`, `llm`, `claude` or `gemini` | The tool on `PATH` |
| `cloud` | Anthropic, Gemini, OpenAI or DeepSeek HTTP API | An API key and a model ID |

Configuration lives in `.env` in the project root (copy `.env.example`; fill only what you use). Names: `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL`, `GEMINI_API_KEY` / `GEMINI_MODEL`, `OPENAI_API_KEY` / `OPENAI_MODEL`, `DEEPSEEK_API_KEY` / `DEEPSEEK_MODEL`, `SAEMIL_CLI_TOOLS`, `SAEMIL_CLI_TIMEOUT_S` (default 15, range 1–120), `SAEMIL_OLLAMA_MODEL`, `SAEMIL_LLM_MODEL`, `SAEMIL_LLM_ON_DEVICE`. An empty key or model leaves that provider unavailable. Variables already set in the environment win over `.env`; keys outside this list are ignored.

Rules:
- **On-device vs hosted.** `cli` mode with no vendor picks only on-device tools. `ollama` counts as on-device only if `ollama list` shows the exact `SAEMIL_OLLAMA_MODEL` name, the tag does not contain "cloud", and `OLLAMA_HOST` is unset or loopback. `llm` counts only with `SAEMIL_LLM_ON_DEVICE=1`. `claude` and `gemini` are hosted and run only when named explicitly.
- **Never auto-escalate.** Any provider failure (missing key, timeout, bad output, busy) falls back to the offline template with a `fallback_reason` code. A CLI failure never moves on to a cloud or hosted provider.
- **Keys stay on the server.** They are sent only in vendor request headers and never appear in responses, `/api/status`, error text or logs. The browser never sees them.
- The page shows which provider actually wrote the text (`provider_used`, `mode_used`).

Real CLI tools and real vendor APIs were not tested; adapters were tested against fixtures shaped like the vendor docs.

## Security

- Binds to `127.0.0.1`; the Host header must be `127.0.0.1` or `localhost` (otherwise 400), which blocks DNS rebinding. Run one worker. There is no authentication.
- Request bodies over 262,144 bytes (256 KB) get 413. `/docs`, `/redoc` and `/openapi.json` are disabled (they would load a CDN). 422 errors do not echo input.
- Models load with `np.load(..., allow_pickle=False)`; weights plus a JSON manifest, no pickle. Every shape, dtype and layer connection is checked before the network is built.
- Local CLIs run with an argv list (`shell=False`), the prompt on stdin, a fresh empty temp directory as working directory, `*_API_KEY` variables removed from the child environment, output capped, the process tree killed on timeout, and one CLI job at a time. `claude` runs with tools, MCP servers, settings and slash commands disabled and one turn. `gemini` runs with `--approval-mode default` (write and shell tools are refused when non-interactive) and no MCP servers; Gemini has no documented flag that disables every tool.
- `/api/report` trusts the result snapshots the browser sends, so a client can ask for an explanation of made-up numbers. Acceptable for a local demo; the alternative is server-side recomputation or signed result IDs.
