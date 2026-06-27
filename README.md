# LTEngine

Free and Open Source Local AI Machine Translation API, written in Rust, entirely self-hosted and compatible with [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate). Its translation capabilities are powered by large language models (LLMs) running on an external inference server that supports the OpenAI chat completions API (e.g., [Ollama](https://ollama.com/), [vLLM](https://github.com/vllm-project/vllm), [llama.cpp](https://github.com/ggml-org/llama.cpp) server).

![Translation](https://github.com/user-attachments/assets/37dd4e20-382b-459d-bcc1-5de3ed4b4c18)

The LLMs in LTEngine are much larger than the lightweight transformer models in [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate). Thus memory usage and speed are traded off for quality of outputs, which for some languages has been reported as being [on par or better than DeepL](https://community.libretranslate.com/t/ltengine-llm-powered-local-machine-translation/1862/5).

> ⚠️ LTEngine is in active development. Check the [Roadmap](#roadmap) for current limitations.

> 💡 **Migration from llama.cpp version:** Previous versions of LTEngine ran llama.cpp directly (with GPU layer offloading). This version uses an external backend (Ollama, vLLM, etc.) via the OpenAI-compatible API. Models must now be loaded on the backend side. To upgrade:
> 1. Set up an external inference server (e.g., `ollama pull gemma3-4b`)
> 2. Run `ltengine --backend-host localhost:11434 -m gemma3-4b`
> 3. The old `--model-file` and `--cpu` options are removed — the backend handles GPU/CPU decisions.

## Requirements

 * [Rust](https://www.rust-lang.org/)
 * A compatible backend server (Ollama recommended for local use):
   - [Ollama](https://ollama.com/) — easiest for local GGUF models (GPU support built-in)
   - [vLLM](https://github.com/vllm-project/vllm) — high-throughput serving for HuggingFace models
   - llama.cpp server — compatible with any GGUF model

## Build

```bash
git clone https://github.com/LibreTranslate/LTEngine
cd LTEngine
cargo build --release
```

## Run

```bash
./target/release/ltengine --backend-host localhost:11434 -m gemma3-4b
```

To use a different backend or model:

```bash
./target/release/ltengine --backend-host http://localhost:8000 -m meta-llama/Llama-3-8b
```

**Error handling:**
- **Retry logic:** LTEngine automatically retries failed requests with exponential backoff (default: 5 attempts) for transient errors (timeouts, connection issues, HTTP 5xx). Client errors (401 auth, 404 model not found) are returned immediately with clear messages.
- **Model validation:** On provider initialization (lazy, triggered by first request or periodic recheck), the provider checks if the specified model is available on the backend (queries Ollama's `/api/tags` or `/v1/models`). If the model is not found, it lists available models and fails gracefully without crashing the service.
- **API keys:** Use `--api-key` to require authentication on incoming translation requests. Use `--backend-api-key` to authenticate with the external backend (passed as Bearer token). The keys are never logged; only status codes appear in error messages.

## Models

LTEngine now delegates model loading to the backend server. You can use any model supported by your inference backend. For Ollama, common choices include:

| Model | Notes |
|-------|-------|
| `gemma3-4b` (default) | Good balance of quality/speed |
| `gemma3-12b` | Higher quality, slower |
| `llama3.3-8b` | Well-rounded general model |
| `gemma3-27b` | Best quality, slowest |

See your backend's documentation for the full model list and how to pull/load models.

> **Tip for Docker users:** Run `ollama pull gemma3-4b` in your Ollama container before starting the service, or pull it from the host. The docker-compose setup below handles this automatically.

### Simple

Request:

```javascript
const res = await fetch("http://0.0.0.0:5050/translate", {
  method: "POST",
  body: JSON.stringify({
    q: "Hello!",
    source: "en",
    target: "es",
  }),
  headers: { "Content-Type": "application/json" },
});

console.log(await res.json());
```

Response:

```javascript
{
    "translatedText": "¡Hola!"
}
```

List of language codes: https://0.0.0.0:5000/languages

### Auto Detect Language

Request:

```javascript
const res = await fetch("http://0.0.0.0:5000/translate", {
  method: "POST",
  body: JSON.stringify({
    q: "Ciao!",
    source: "auto",
    target: "en",
  }),
  headers: { "Content-Type": "application/json" },
});

console.log(await res.json());
```

Response:

```javascript
{
    "detectedLanguage": {
        "confidence": 83,
        "language": "it"
    },
    "translatedText": "Bye!"
}
```

### Health Check Endpoints

**Service health:** `GET /health`
```javascript
const res = await fetch("http://localhost:5050/health");
console.log(await res.json());
// { "status": "ok", "service": "ltengine", "version": "0.1.1" }
```

**Backend health:** `GET /health/backend` — checks if the translation backend is reachable and returns the backend status (200 or 503).
```javascript
const res = await fetch("http://localhost:5050/health/backend");
console.log(await res.json());
// { "status": "ok", "backend": "Backend is reachable" }
// or { "status": "error", "backend": "Backend not reachable: ..." }
```

These endpoints are useful for health probes in load balancers and monitoring tools.

## API Notes

### Source equals Target Behavior

If the `source` and `target` parameters are the same (e.g., both `"en"`), the API returns the original text **without** calling the translation backend. This is a performance optimization to avoid unnecessary LLM calls, but it means that the text is not "improved" or "corrected" in the same language. If you need text improvement in the same language, you should use a different approach (e.g., set the source to `"auto"` or use a different API).

### Retry and Reliability

The service will not crash if the backend is initially unreachable. During startup:
- It attempts to connect to the backend with exponential backoff (default: 5 attempts with base delay of 1s)
- If all attempts fail, it logs the error but the service continues running
- The first translation request will retry and attempt to recreate the provider

If the backend becomes unreachable during operation:
- Translation requests will automatically retry with exponential backoff
- If the retry fails after the maximum attempts, a 503 error is returned to the client
- Periodic background rechecking (default: every 30 seconds) will detect when the backend comes back online and recreate the provider

### CLI Options

```bash
./target/release/ltengine \
  --host 0.0.0.0 \
  --port 5050 \
  --backend-host localhost:11434 \
  --model gemma3-4b \
  --api-key your-secret-key \
  --backend-api-key your-backend-key \
  --retry-attempts 5 \
  --retry-delay 1000 \
  --model-recheck-interval 30 \
  --char-limit 5000
```

- `--api-key`: Require this key on incoming translation requests (for server-side authentication)
- `--backend-api-key`: API key to send to the external backend (as Bearer token, for backend authentication)
- `--retry-attempts`: Maximum number of retry attempts for provider creation and translation requests (default: 5)
- `--retry-delay`: Base delay in milliseconds for exponential backoff (default: 1000ms → 1s, 2s, 4s, 8s...)
- `--model-recheck-interval`: Interval in seconds for periodic model availability rechecks. Set to 0 to disable (default: 30).

## Language Bindings

You can use the LTEngine API using the following bindings:

- Rust: <https://github.com/DefunctLizard/libretranslate-rs>
- Node.js: <https://github.com/franciscop/translate>
- TypeScript: <https://github.com/tderflinger/libretranslate-ts>
- .Net: <https://github.com/sigaloid/LibreTranslate.Net>
- Go: <https://github.com/SnakeSel/libretranslate>
- Python: <https://github.com/argosopentech/LibreTranslate-py>
- PHP: <https://github.com/jefs42/libretranslate>
- C++: <https://github.com/argosopentech/LibreTranslate-cpp>
- Swift: <https://github.com/wacumov/libretranslate>
- Unix: <https://github.com/argosopentech/LibreTranslate-sh>
- Shell: <https://github.com/Hayao0819/Hayao-Tools/tree/master/libretranslate-sh>
- Java: <https://github.com/suuft/libretranslate-java>
- Ruby: <https://github.com/noesya/libretranslate>
- R: <https://github.com/myanesp/libretranslateR>

## Roadmap

 - [x] Remove llama.cpp mutex — concurrent translation requests are now supported via the HTTP provider
 - [x] Cancel inference — HTTP connection drops are now handled by the request cancellation in the provider
 - [x] Retry logic with exponential backoff for transient backend errors
 - [x] Model availability validation at startup
 - [x] Health check endpoints (`/health`, `/health/backend`)
 - [x] Fully static Docker image (musl, no dynamic dependencies)
 - [ ] Add support for `/translate_file` (ability to translate files).
 - [ ] Add support for sentence splitting. Currently text is sent to the LLM as-is, but longer texts (like documents) should be split into chunks, translated and merged back.
 - [ ] Better language detection for short texts (port [LexiLang](https://github.com/LibreTranslate/LexiLang) to Rust)
 - [ ] Create comparative benchmarks between LTEngine and proprietary software.
 - [ ] Add support for command line inference (run `./ltengine translate` as a command line app separate from `./ltengine server`)
 - [ ] Make ltengine available as a library, possibly creating bindings for other languages like Python.
 - [x] Automated builds / CI
 - [x] Configurable retry logic with exponential backoff
 - [x] Periodic model availability re-checking with auto-recovery
 - [x] Resilience against initial backend unreachability (no crash on startup)
 - [ ] Your ideas? We welcome contributions.

## Contributing

We welcome contributions! Just open a pull request.

## Credits

This project uses Rust crates for HTTP communication (reqwest) and language detection (whatlang-rs). The underlying language models are trained by various open-source communities (Gemma, Llama, Mistral, etc.).

## License

[GNU Affero General Public License v3](https://www.gnu.org/licenses/agpl-3.0.en.html)

## Trademark

See [Trademark Guidelines](https://github.com/LibreTranslate/LibreTranslate/blob/main/TRADEMARK.md)
