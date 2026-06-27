# LTEngine — AGENTS.md (Cheat Sheet)

**What:** Local, self-hosted AI machine translation API (Rust), LibreTranslate-compatible, powered by an external inference backend via the OpenAI `/chat/completions` API (e.g., Ollama, vLLM, llama.cpp server).  
**License:** AGPL-3.0. **Edition:** Rust 2024. **Default port:** 5050.

---

## Build / Run

```bash
git clone <repo> && cd LTEngine
cargo build --release   # static musl binary, no GPU features needed
./target/release/ltengine --backend-host localhost:11434 -m gemma3-4b
```

**CLI args:**
- `--backend-host` — inference server address (default `localhost:11434` for Ollama)
- `-m MODEL_ID` — model name (passed to backend, any string accepted)
- `--host/--port` — bind address (default `0.0.0.0:5050`)
- `--api-key` — require this key on incoming translation requests (server-side auth)
- `--backend-api-key` — API key sent to backend as Bearer token (backend auth)
- `--retry-attempts` — max retry attempts (default 5)
- `--retry-delay` — base backoff delay in ms (default 1000)
- `--model-recheck-interval` — periodic model availability recheck interval in seconds (0 to disable, default 30)
- `--char-limit` — max request length (default 5000)

The service retries with exponential backoff if the backend is unreachable. Models are NOT auto-downloaded — the backend must have the model pre-loaded.

---

## Project Structure

```
LTEngine/
├── Cargo.toml              # workspace root, lints (pedantic, missing_docs)
├── AGENTS.md               # this file
├── ltengine/               # the actual crate
│   ├── Cargo.toml          # dependencies (no GPU features)
│   ├── build.rs            # embeds `./resources` via `static_files` into binary
│   ├── resources/          # LibreTranslate frontend (Vue2 + Materialize)
│   └── src/
│       ├── main.rs         # CLI args, HTTP routes, health endpoints
│       ├── backend.rs      # TranslateProvider trait, OpenAiProvider, ProviderManager with retry/recheck
│       ├── prompt.rs       # PromptBuilder → system+user prompt templates
│       ├── languages.rs    # supported language list, whatlang detection, aliases
│       ├── error_response.rs  # ErrorResponse wrapper for JSON errors
│       ├── banner.rs       # ASCII art banner
│       └── tests.rs        # mock provider tests
├── Dockerfile              # static musl build → lean runtime, exposes 5050
├── docker-compose.yml      # Ollama + ltengine setup
└── .github/workflows/      # build (Rust, no GPU) + Docker (push to ghcr.io)
```

---

## API Endpoints (LibreTranslate-compatible)

All endpoints accept `application/json`, `application/x-www-form-urlencoded`, or `multipart/form-data`.

| Method | Path              | Notes |
|--------|-------------------|-------|
| POST   | `/translate`      | Body: `{q, source, target, format? (text/html), api_key?, alternatives?}`. Returns `{translatedText, detectedLanguage?}` |
| POST   | `/detect`         | Body: `{q}`. Returns `[{language, confidence}]` |
| GET    | `/languages`      | Returns full language list |
| GET    | `/health`         | Service health check (always returns 200) |
| GET    | `/health/backend` | Backend connectivity check (200 OK or 503 error) |
| GET    | `/frontend/settings` | Returns settings JSON (API key mode, char limit, etc.) |
| POST   | `/translate_file` | **501 Not Implemented** |
| POST   | `/suggest`        | **501 Not Implemented** |

Root `/` serves the embedded frontend (Vue2 + Materialize CSS).

**Translation request validation:** `q`, `source`, `target` required. `source` can be `"auto"` (uses `whatlang-rs` for detection). `target` must match a supported language. Character limit enforced. API key check (if configured) returns 403. Backend auth failure (401) returns 403. Backend model not found (404) returns 404. Transient errors (connection, timeout) are retried.

---

## Architecture — Backend Layer

- `backend.rs` provides a `TranslateProvider` trait with `translate` and `ping` methods.
- `OpenAiProvider` sends system+user prompts to an OpenAI-compatible `/v1/chat/completions` endpoint using reqwest.
- `ProviderManager` wraps the provider with:
  - **Retry with exponential backoff** on transient errors (timeouts, connection issues, HTTP 5xx)
  - **Automatic provider recreation** when backend becomes unreachable
  - **Periodic model availability rechecking** (default every 30s) that recreates provider if backend comes back online
- **Concurrent requests** — no mutex; multiple translation requests can run simultaneously.

---

## Prompting

`prompt.rs` builds a system/user pair:
- **System prompt** instructs the model to be an expert translator, capture nuances, preserve meaning, never add explanations. For `format=html`, additionally says to preserve HTML tags.
- **User prompt** format depends on source language:
  - `auto`: `"Translate the text below to {target}.\n\nText: {q}\n\n{target}:\n"`
  - specified: `"Translate the text below from {source} to {target}.\n\n{source}: {q}\n\n{target}:\n"`

The final input is wrapped as a single user message: `system + "\n\n" + user`.

---

## Languages

- 50+ languages defined in `languages.rs` with aliases (e.g., `zh` → Chinese simplified, `zt` → traditional, `pb` → Brazilian Portuguese).
- Detection via `whatlang-rs` with an allowlist of supported languages. Confidence scaled to 0-100.

---

## Post-processing (`improve_formatting`)

After translation, the function:
- Trims trailing/leading punctuation to match the source (if source ends with `! ? . , ; 。` and translation differs, replace or remove mismatched punctuation).
- Mirrors source casing: if all-lowercase → lowercase result; if all-uppercase → uppercase. If first char case differs → flip result's first char to match source.

---

## Error Handling

- **BackendError enum** with `Http` (HTTP status code), `Other` (parsing/empty), and `Retryable` (network issues) variants.
- ProviderManager detects `Retryable` errors by type (no string matching), recreates the provider, and retries the translation.
- Client errors (401 → 403 Forbidden, 404 → 404 Model not found) are returned immediately.
- Transient connection/timeout errors trigger provider recreation + retry loop.

---

## Roadmap / Known Limitations (from README)

- **No `/translate_file`** — files not supported.
- **No sentence splitting** — long texts sent as-is; should chunk/translate/merge.
- **Language detection weak for short texts** — plan to port LexiLang to Rust.
- **No CLI inference mode** — currently only server mode.
- **Not available as a library** — no Python bindings yet.
- **No benchmarks** — need comparison with proprietary software.

---

## Docker

- **Image:** `ghcr.io/libretranslate/ltengine:main` (built from `Dockerfile`, fully static musl binary).
- **Environment:** `MODEL` (model ID), `BACKEND_HOST` (default `localhost:11434`), `HF_HOME` (unused, legacy).
- **docker-compose.yml** includes an optional Ollama service with GPU reservation.

---

## Dependencies

| Crate | Purpose |
|-------|---------|
| `actix-web` | HTTP server, routing, multipart |
| `clap` | CLI argument parsing |
| `serde`/`serde_json` | JSON serialization |
| `reqwest` | HTTP client for backend communication |
| `async-trait` | Async trait support |
| `tokio` | Async runtime (time, rt-multi-thread) |
| `whatlang` | Language detection |
| `actix-web-static-files` | Serve frontend from embedded resources |
| `static-files` | Build-time embedding of `./resources` |
| `encoding_rs` | UTF-8 decoding during token generation |
| `anyhow`/`thiserror` | Error handling |
| `actix-multipart` | Multipart form handling |

---

## Testing

Test file at `ltengine/src/tests.rs` includes:
- Mock `TranslateProvider` implementations for unit testing
- Basic translation and prompt builder tests
- Retry mock to verify call counts
- OpenAiProvider connection-refused test (no server)

---

## Contributing & Coding Style

- Workspace-level lints enforce `missing_docs = "warn"` and `missing_debug_implementations = "warn"` with `clippy::pedantic`.
- Add missing documentation to public items. All error enums/structs need `Debug`.
- Use `Result`/`anyhow` for error propagation; `thiserror` for custom error types.
- Frontend is LibreTranslate's original Vue2 + Materialize CSS (static files, not a separate app).
