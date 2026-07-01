use actix_multipart::form::{MultipartForm, text::Text as MPText};
use actix_web::{
    App, FromRequest, HttpRequest, HttpResponse, HttpServer, Responder, get, http::header, post,
    web,
};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use std::time::Instant;

mod backend;
mod banner;
#[cfg(feature = "benchmark")]
mod benchmark;
mod cleanup;
mod detection;
mod error_response;
mod languages;
mod prompt;
mod static_handler;
#[cfg(test)]
mod tests;

use backend::{BackendError, BackendTimings, ProviderConfig, ProviderManager};
use banner::print_banner;
use detection::create_detector;
use error_response::ErrorResponse;
use languages::{LANGUAGES, get_language_from_code};
use prompt::PromptBuilder;

#[derive(Parser, Debug, Clone, Default, ValueEnum)]
enum BenchmarkMode {
    /// Print benchmark results as plain text (default)
    #[default]
    Text,
    /// Generate a self-contained HTML report with charts and expandable details
    Html,
}

#[derive(Parser, Debug, Clone)]
#[command(version, about, long_about = None)]
struct Args {
    /// Hostname to bind to (env var: LTE_HOST)
    #[arg(long, env = "LTE_HOST", default_value = "0.0.0.0")]
    host: String,

    /// Port to bind to (env var: LTE_PORT)
    #[arg(short, long, env = "LTE_PORT", default_value_t = 5050)]
    port: u16,

    /// Maximum total character length of all source text; rejects if exceeded (env var: LTE_CHAR_LIMIT)
    #[arg(long, env = "LTE_CHAR_LIMIT", default_value_t = 50000)]
    char_limit: usize,

    /// Maximum size of individual translation chunks (env var: LTE_CHUNK_SIZE)
    #[arg(long, env = "LTE_CHUNK_SIZE", default_value_t = 5000)]
    chunk_size: usize,

    /// Maximum number of concurrent translation requests per input text (env var: LTE_MAX_PARALLEL)
    #[arg(long, env = "LTE_MAX_PARALLEL", default_value_t = 2)]
    max_parallel: usize,

    /// Maximum number of strings accepted in JSON q arrays (env var: LTE_BATCH_LIMIT)
    #[arg(long, env = "LTE_BATCH_LIMIT", default_value_t = 50)]
    batch_limit: usize,

    /// Model to use (passed to the backend provider) (env var: LTE_MODEL)
    #[arg(short = 'm', long, env = "LTE_MODEL", default_value = "gemma3-4b")]
    model: String,

    /// Backend host (e.g., "localhost:11434" for Ollama, or full URL like "http://ollama:11434") (env var: LTE_BACKEND_HOST)
    #[arg(long, env = "LTE_BACKEND_HOST", default_value = "localhost:11434")]
    backend_host: String,

    /// Set an API key (optional, for authenticating translation requests to the API) (env var: LTE_API_KEY)
    #[arg(long, env = "LTE_API_KEY", default_value = "")]
    api_key: String,

    /// Set a backend API key (optional, for authenticating with the external backend) (env var: LTE_BACKEND_API_KEY)
    #[arg(
        long = "backend-api-key",
        env = "LTE_BACKEND_API_KEY",
        default_value = ""
    )]
    backend_api_key: String,

    /// Maximum number of retry attempts for provider creation and translation requests (env var: LTE_RETRY_ATTEMPTS)
    #[arg(long, env = "LTE_RETRY_ATTEMPTS", default_value_t = 5)]
    retry_attempts: usize,

    /// Base delay in milliseconds for exponential backoff (1000ms → 1s, 2s, 4s...) (env var: LTE_RETRY_DELAY)
    #[arg(long, env = "LTE_RETRY_DELAY", default_value_t = 1000)]
    retry_delay: u64,

    /// Interval in seconds for periodic model availability rechecks (0 to disable) (env var: LTE_MODEL_RECHECK_INTERVAL)
    #[arg(long, env = "LTE_MODEL_RECHECK_INTERVAL", default_value_t = 30)]
    model_recheck_interval: u64,

    /// Enable language detection via LLM (env var: LTE_LLM_DETECT)
    #[arg(long = "llm-detect", env = "LTE_LLM_DETECT", action)]
    llm_detect: bool,

    /// Run a benchmark against a dataset and exit (env var: LTE_BENCHMARK). Options: "text" or "html"
    #[arg(long, env = "LTE_BENCHMARK")]
    benchmark: Option<BenchmarkMode>,

    /// Path to the benchmark dataset JSON file (env var: LTE_DATASET)
    #[arg(long, env = "LTE_DATASET")]
    dataset: Option<String>,

    /// Disable Unicode cleanup of LLM output (cleanup is ON by default; use --disable-cleanup to turn it off) (env var: LTE_DISABLE_CLEANUP)
    #[arg(long, env = "LTE_DISABLE_CLEANUP", action)]
    disable_cleanup: bool,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[serde(untagged)]
enum QueryText {
    Single(String),
    Batch(Vec<String>),
}

impl QueryText {
    fn as_slice(&self) -> &[String] {
        match self {
            QueryText::Single(q) => std::slice::from_ref(q),
            QueryText::Batch(q) => q.as_slice(),
        }
    }

    fn is_batch(&self) -> bool {
        matches!(self, QueryText::Batch(_))
    }

    fn is_empty_or_blank(&self) -> bool {
        let items = self.as_slice();
        items.is_empty() || items.iter().any(|q| q.trim().is_empty())
    }

    fn total_len(&self) -> usize {
        self.as_slice().iter().map(String::len).sum()
    }

    fn shaped_translated_text(&self, translated_texts: Vec<String>) -> Value {
        if self.is_batch() {
            serde_json::json!(translated_texts)
        } else {
            // Defensive: empty input should return null, not an empty string (which could be ambiguous)
            serde_json::json!(
                translated_texts
                    .into_iter()
                    .next()
                    .map_or(Value::Null, Value::String)
            )
        }
    }

    fn shaped_empty_alternatives(&self) -> Value {
        if self.is_batch() {
            Value::Array(
                (0..self.as_slice().len())
                    .map(|_| Value::Array(Vec::new()))
                    .collect(),
            )
        } else {
            Value::Array(Vec::new())
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct TranslateRequest {
    q: Option<QueryText>,
    source: Option<String>,
    target: Option<String>,
    format: Option<String>,
    api_key: Option<String>,
    alternatives: Option<u32>,
    enable_cleanup_reporting: Option<bool>,
    enable_performance_reporting: Option<bool>,
}

#[derive(MultipartForm)]
struct MPTranslateRequest {
    q: Option<MPText<String>>,
    source: Option<MPText<String>>,
    target: Option<MPText<String>>,
    format: Option<MPText<String>>,
    api_key: Option<MPText<String>>,
    alternatives: Option<MPText<u32>>,
    enable_cleanup_reporting: Option<MPText<String>>,
    enable_performance_reporting: Option<MPText<String>>,
}

/// Helper function to convert a string to bool.
/// Returns `true` for "true", "yes", "1" (case-insensitive), `false` for anything else.
fn str_to_bool(s: &str) -> bool {
    matches!(s.trim().to_lowercase().as_str(), "true" | "yes" | "1")
}

impl MPTranslateRequest {
    fn into_translate_request(self) -> TranslateRequest {
        TranslateRequest {
            q: self.q.map(|v| QueryText::Single(v.into_inner())),
            source: self.source.map(|v| v.into_inner()),
            target: self.target.map(|v| v.into_inner()),
            format: self.format.map(|v| v.into_inner()),
            api_key: self.api_key.map(|v| v.into_inner()),
            alternatives: self.alternatives.map(|v| v.into_inner()),
            enable_cleanup_reporting: self
                .enable_cleanup_reporting
                .map(|v| str_to_bool(&v.into_inner())),
            enable_performance_reporting: self
                .enable_performance_reporting
                .map(|v| str_to_bool(&v.into_inner())),
        }
    }
}

async fn parse_payload(
    req: HttpRequest,
    payload: web::Payload,
) -> Result<TranslateRequest, ErrorResponse> {
    let content_type = req
        .headers()
        .get(header::CONTENT_TYPE)
        .map(|h| h.to_str().unwrap_or(""))
        .unwrap_or("");
    let body: TranslateRequest;

    if content_type.starts_with("application/json") {
        let json =
            actix_web::web::Json::<TranslateRequest>::from_request(&req, &mut payload.into_inner())
                .await?;
        body = json.into_inner()
    } else if content_type.starts_with("application/x-www-form-urlencoded") {
        let form =
            actix_web::web::Form::<TranslateRequest>::from_request(&req, &mut payload.into_inner())
                .await?;
        body = form.into_inner()
    } else if content_type.starts_with("multipart/form-data") {
        let form =
            MultipartForm::<MPTranslateRequest>::from_request(&req, &mut payload.into_inner())
                .await?;
        body = form.into_inner().into_translate_request();
    } else {
        return Err(ErrorResponse {
            error: "Unsupported content-type".to_string(),
            status: 400,
        });
    }

    Ok(body)
}

fn check_params(
    body: &TranslateRequest,
    args: &Args,
    required_params: &[(&str, &Option<String>)],
) -> Result<bool, ErrorResponse> {
    // Validate q separately because JSON can provide either a scalar string or a batch array.
    let q = body.q.as_ref().ok_or_else(|| ErrorResponse {
        error: "Invalid request: missing q parameter".to_string(),
        status: 400,
    })?;

    if q.is_empty_or_blank() {
        return Err(ErrorResponse {
            error: "Invalid request: missing q parameter".to_string(),
            status: 400,
        });
    }

    // Validate required scalar params.
    for (key, value) in required_params {
        if value.as_ref().is_none_or(|v| v.trim().is_empty()) {
            return Err(ErrorResponse {
                error: format!("Invalid request: missing {} parameter", key),
                status: 400,
            });
        }
    }

    // Check API key: if the server has an API key configured, the request must include
    // it. Returns 403 if the key is missing or doesn't match. Uses constant-time comparison
    // to prevent timing attacks.
    if !args.api_key.is_empty() {
        let key_matches = body
            .api_key
            .as_ref()
            .map(|key| {
                // Use constant-time comparison to prevent timing attacks
                subtle::ConstantTimeEq::ct_eq(key.as_bytes(), args.api_key.as_bytes()).into()
            })
            .unwrap_or(false);
        if !key_matches {
            return Err(ErrorResponse {
                error: "Invalid API key".to_string(),
                status: 403,
            });
        }
    }

    if q.is_batch() && q.as_slice().len() > args.batch_limit {
        return Err(ErrorResponse {
            error: format!(
                "Invalid request: batch size ({}) exceeds limit ({})",
                q.as_slice().len(),
                args.batch_limit
            ),
            status: 400,
        });
    }

    let request_len = q.total_len();
    if request_len > args.char_limit {
        return Err(ErrorResponse {
            error: format!(
                "Invalid request: total text length ({}) exceeds maximum allowed ({})",
                request_len, args.char_limit
            ),
            status: 400,
        });
    }

    Ok(true)
}

/// Chunk a long text into smaller pieces, preferring paragraph boundaries,
/// then sentence boundaries (". "), and finally fixed-size fallback.
fn chunk_text(text: &str, max_size: usize) -> Vec<String> {
    // If text fits in one chunk, just return it.
    if text.len() <= max_size {
        return vec![text.to_string()];
    }
    // First try to split on paragraph boundaries (two or more newlines).
    let paragraphs: Vec<&str> = text
        .split("\n\n")
        .filter(|p| !p.trim().is_empty())
        .collect();

    if paragraphs.len() > 1 {
        // Group paragraphs without exceeding max_size.
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        for para in paragraphs.iter() {
            let para_str = para.trim();
            if current_chunk.is_empty() {
                current_chunk = para_str.to_string();
            } else if (format!("{0}\n\n{1}", current_chunk, para_str)).len() <= max_size {
                current_chunk = format!("{0}\n\n{1}", current_chunk, para_str);
            } else {
                chunks.push(current_chunk);
                current_chunk = para_str.to_string();
            }
        }
        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }

        // Recursively handle any oversized chunks.
        let mut final_chunks = Vec::new();
        for chunk in chunks {
            if chunk.len() > max_size {
                final_chunks.extend(chunk_text(&chunk, max_size));
            } else {
                final_chunks.push(chunk);
            }
        }
        return final_chunks;
    }

    // No paragraph boundaries. Try sentence splitting on ". " or ".\n".
    let sentences: Vec<String> = text
        .split(". ")
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            if s.trim().ends_with('.') {
                s.trim().to_string()
            } else {
                format!("{}.", s.trim())
            }
        })
        .collect();

    if sentences.len() > 1 {
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        for sent in &sentences {
            if current_chunk.is_empty() {
                current_chunk = sent.clone();
            } else if (format!("{0} {1}", current_chunk, sent)).len() <= max_size {
                current_chunk = format!("{0} {1}", current_chunk, sent);
            } else {
                chunks.push(current_chunk);
                current_chunk = sent.clone();
            }
        }
        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }

        let mut final_chunks = Vec::new();
        for chunk in chunks {
            if chunk.len() > max_size {
                final_chunks.extend(chunk_text(&chunk, max_size));
            } else {
                final_chunks.push(chunk);
            }
        }
        return final_chunks;
    }

    // No sentence boundaries either. Fixed-size character chunks, trying to break at sentence boundary.
    let mut chunks = Vec::new();
    let mut remaining = text;
    loop {
        if remaining.len() <= max_size {
            chunks.push(remaining.trim().to_string());
            break;
        }
        // Find the last ". " within the first max_size characters
        let window: String = remaining.chars().take(max_size).collect();
        if let Some(pos) = window.rfind(". ") {
            let cut_point = pos + 2;
            chunks.push(window[..cut_point].trim().to_string());
            remaining = &remaining[cut_point..];
        } else {
            // No sentence boundary found, just cut at max_size but avoid mid-word cuts
            let cut_window: String = remaining.chars().take(max_size).collect();
            // Find the last space to avoid cutting a word in half
            let last_space = cut_window.rfind(' ').unwrap_or(cut_window.chars().count());
            let mut trimmed = cut_window[..last_space].trim().to_string();
            if trimmed.is_empty() {
                trimmed = cut_window
                    .split_whitespace()
                    .next()
                    .unwrap_or(&cut_window)
                    .to_string();
            }
            chunks.push(trimmed);
            let consumed = last_space; // consume up to the last space (not including the space itself)
            remaining = &remaining[consumed..];
        }
    }
    chunks.retain(|c| !c.is_empty());
    chunks
}

fn improve_formatting(q: &str, translation: &str) -> String {
    let t = translation.trim().to_string();

    if q.is_empty() {
        return String::new();
    }

    if t.is_empty() {
        return q.to_owned();
    }

    let q_last_char = q.chars().next_back().unwrap();
    let translation_last_char = t.chars().next_back().unwrap();
    let mut result = t.clone();

    const PUNCTUATION_CHARS: [char; 6] = ['!', '?', '.', ',', ';', '。'];
    if PUNCTUATION_CHARS.contains(&q_last_char) {
        if q_last_char != translation_last_char {
            if PUNCTUATION_CHARS.contains(&translation_last_char) {
                result.pop();
            }

            result.push(q_last_char);
        }
    } else if PUNCTUATION_CHARS.contains(&translation_last_char) {
        result.pop();
    }

    if q.chars().all(|c| c.is_lowercase()) {
        result = result.to_lowercase();
    }

    if q.chars().all(|c| c.is_uppercase()) {
        result = result.to_uppercase();
    }

    if let (Some(q0), Some(r0)) = (q.chars().next(), result.chars().next()) {
        if q0.is_lowercase() && r0.is_uppercase() {
            result.replace_range(0..r0.len_utf8(), &r0.to_lowercase().to_string());
        } else if q0.is_uppercase() && r0.is_lowercase() {
            result.replace_range(0..r0.len_utf8(), &r0.to_uppercase().to_string());
        }
    }

    result.trim().to_string()
}

#[post("/detect")]
async fn detect(
    req: HttpRequest,
    payload: web::Payload,
    args: web::Data<Arc<Args>>,
    provider_manager: Option<web::Data<Arc<ProviderManager>>>,
) -> Result<HttpResponse, ErrorResponse> {
    let body = parse_payload(req, payload).await?;
    check_params(&body, &args, &[])?;

    let q = body.q.as_ref().expect("q was validated by check_params");

    let detector = if args.llm_detect {
        if let Some(pm) = provider_manager {
            create_detector(&args, &pm)
        } else {
            return Err(ErrorResponse {
                error: "Provider manager not available; cannot use LLM detection".to_string(),
                status: 500,
            });
        }
    } else {
        Box::new(detection::WhatlangDetector)
    };

    let mut results = Vec::new();
    for text in q.as_slice() {
        let d = detector.detect(text).await;
        results.push(serde_json::json!({
            "language": d.language.code,
            "confidence": d.confidence,
        }));
    }

    Ok(HttpResponse::Ok().json(serde_json::Value::Array(results)))
}

fn check_format(format: &str) -> Result<(), ErrorResponse> {
    match format {
        "text" | "html" => Ok(()),
        _ => Err(ErrorResponse {
            error: "Invalid format. Supported formats: text, html".to_string(),
            status: 400,
        }),
    }
}

fn map_translation_error(e: anyhow::Error) -> ErrorResponse {
    eprintln!("translation error: {}", e);

    // Check if the error is a BackendError with an HTTP status code or retryable.
    let (msg, status) = if let Some(backend_err) = e.downcast_ref::<BackendError>() {
        match backend_err {
            BackendError::Http(status_code) => {
                // Don't leak backend details, use generic message.
                let http_status = match *status_code {
                    401 => 403, // Forbidden (authentication failed)
                    404 => 404, // Model not found
                    _ => 503,   // Other server errors → 503
                };
                (format!("Backend returned {}", status_code), http_status)
            }
            BackendError::ModelNotFound(msg) => {
                // Model not available on the backend — return 404.
                (msg.to_string(), 404)
            }
            BackendError::Retryable(_) => {
                // ProviderManager dropped the provider after max retries.
                // The next request will trigger a new creation attempt.
                ("Backend temporarily unavailable".to_string(), 503)
            } // No Other variant — generic handling below covers remaining cases.
        }
    } else {
        // Generic error (e.g., creation failed after max retries) — 503.
        (format!("{}", e), 503)
    };

    ErrorResponse { error: msg, status }
}

async fn translate_one_with_cleanup(
    q: &String,
    source: &str,
    target: &str,
    prompt_builder: &PromptBuilder,
    provider_manager: &ProviderManager,
    disable_cleanup: bool,
) -> Result<(String, usize, usize, Option<BackendTimings>), ErrorResponse> {
    // If source equals target, return the original text without translating.
    // This is a performance optimization but may be semantically unexpected.
    let (translated_text, backend_timings) = if source == target {
        (q.clone(), None)
    } else {
        let prompt = prompt_builder.build(q);
        let result = provider_manager
            .translate(&prompt.system, &prompt.user)
            .await
            .map_err(map_translation_error)?;
        (result.text, result.backend_timings)
    };

    // Apply cleanup unless disabled via CLI/env
    let (translated_text, removed, replaced) = if disable_cleanup {
        (translated_text, 0, 0)
    } else {
        let result = cleanup::cleanup_output(&translated_text);
        if result.removed > 0 || result.replaced > 0 {
            eprintln!(
                "cleanup: {} characters removed, {} characters replaced from LLM output",
                result.removed, result.replaced
            );
        }
        (result.cleaned, result.removed, result.replaced)
    };

    Ok((
        improve_formatting(q, &translated_text),
        removed,
        replaced,
        backend_timings,
    ))
}

#[get("/health")]
async fn health() -> impl Responder {
    HttpResponse::Ok().json(serde_json::json!({
        "status": "ok",
        "service": "ltengine",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

#[get("/health/backend")]
async fn backend_health(provider_manager: web::Data<Arc<ProviderManager>>) -> impl Responder {
    let result = provider_manager.ping().await;

    match result {
        Ok(_) => HttpResponse::Ok().json(serde_json::json!({
            "status": "ok",
            "backend": "Backend is reachable"
        })),
        Err(e) => {
            let err_msg = format!("{}", e);
            HttpResponse::ServiceUnavailable().json(serde_json::json!({
                "status": "error",
                "backend": err_msg
            }))
        }
    }
}

#[post("/translate")]
async fn translate(
    req: HttpRequest,
    payload: web::Payload,
    args: web::Data<Arc<Args>>,
    provider_manager: web::Data<Arc<ProviderManager>>,
) -> Result<HttpResponse, ErrorResponse> {
    let body = parse_payload(req, payload).await?;
    check_params(
        &body,
        &args,
        &[("source", &body.source), ("target", &body.target)],
    )?;

    let q = body.q.as_ref().expect("q was validated by check_params");
    let source = body
        .source
        .as_ref()
        .expect("source was validated by check_params");
    let target = body
        .target
        .as_ref()
        .expect("target was validated by check_params");
    let format = body.format.as_deref().unwrap_or("text");
    check_format(format)?;

    let tgt_lang = get_language_from_code(target).ok_or_else(|| ErrorResponse {
        error: format!("{} is not supported", target),
        status: 400,
    })?;

    let mut pb = PromptBuilder::new();
    pb.set_format(format);
    pb.set_target_language(tgt_lang.name);

    let mut translated_texts = Vec::with_capacity(q.as_slice().len());
    let mut detected_languages = Vec::with_capacity(q.as_slice().len());
    let mut total_cleanup_removed = 0usize;
    let mut total_cleanup_replaced = 0usize;

    let mut detect_per_text_ms: Vec<u64> = Vec::with_capacity(q.as_slice().len());
    let mut translate_per_text_ms: Vec<u64> = Vec::with_capacity(q.as_slice().len());
    let mut backend_timings: Vec<Option<BackendTimings>> = Vec::with_capacity(q.as_slice().len());

    let semaphore = Arc::new(tokio::sync::Semaphore::new(args.max_parallel));

    if source == "auto" {
        // Use LLM-based detection if enabled, otherwise fall back to whatlang.
        let detector = create_detector(&args, &provider_manager);

        for text in q.as_slice() {
            let detected = detector.detect(text).await;
            detect_per_text_ms.push(detected.detect_time_ms);

            pb.set_source_language(detected.language.name);
            detected_languages.push(detected.clone());

            let start = Instant::now();
            let chunks = chunk_text(text, args.chunk_size);
            if chunks.len() == 1 {
                // Single chunk, translate directly (no chunking)
                let (translated, removed, replaced, timing) = translate_one_with_cleanup(
                    text,
                    detected.language.internal_code,
                    target,
                    &pb,
                    &provider_manager,
                    args.disable_cleanup,
                )
                .await?;
                translate_per_text_ms.push(start.elapsed().as_millis() as u64);
                backend_timings.push(timing);
                translated_texts.push(translated);
                total_cleanup_removed += removed;
                total_cleanup_replaced += replaced;
            } else {
                // Chunked translation — dispatch in parallel
                let detected_code = detected.language.internal_code;
                let disable_cleanup = args.disable_cleanup;
                let pm_for_chunks = provider_manager.clone();
                let pb_for_chunks = pb.clone();
                let futures: Vec<_> = chunks
                    .into_iter()
                    .map(|chunk| {
                        let permit = Arc::clone(&semaphore);
                        let pm = pm_for_chunks.clone();
                        let pb = pb_for_chunks.clone();
                        async move {
                            let _permit = permit.acquire().await;
                            translate_one_with_cleanup(
                                &chunk,
                                detected_code,
                                target,
                                &pb,
                                &pm,
                                disable_cleanup,
                            )
                            .await
                        }
                    })
                    .collect();

                let results = futures::future::join_all(futures).await;
                let mut translated_parts = Vec::new();
                let mut chunk_timings: Vec<Option<BackendTimings>> = Vec::new();
                let mut chunk_removed = 0usize;
                let mut chunk_replaced = 0usize;

                for result in results {
                    let (translated, removed, replaced, timing) = result?;
                    translated_parts.push(translated);
                    chunk_timings.push(timing);
                    chunk_removed += removed;
                    chunk_replaced += replaced;
                }

                let combined = translated_parts.join("\n");
                // Don't apply improve_formatting to chunked text — just join.
                translate_per_text_ms.push(start.elapsed().as_millis() as u64);
                // Use the timing from the last chunk (approximate), flatten the Option<Option<T>>
                let last_timing = chunk_timings.last().cloned().flatten();
                backend_timings.push(last_timing);
                translated_texts.push(combined);
                total_cleanup_removed += chunk_removed;
                total_cleanup_replaced += chunk_replaced;
            }
        }
    } else {
        // Source is explicitly specified, no detection is performed.
        let src_lang = get_language_from_code(source).ok_or_else(|| ErrorResponse {
            error: format!("{} is not supported", source),
            status: 400,
        })?;
        pb.set_source_language(src_lang.name);

        for text in q.as_slice() {
            let start = Instant::now();
            let chunks = chunk_text(text, args.chunk_size);
            if chunks.len() == 1 {
                // Single chunk
                let (translated, removed, replaced, timing) = translate_one_with_cleanup(
                    text,
                    source,
                    target,
                    &pb,
                    &provider_manager,
                    args.disable_cleanup,
                )
                .await?;
                translate_per_text_ms.push(start.elapsed().as_millis() as u64);
                backend_timings.push(timing);
                translated_texts.push(translated);
                total_cleanup_removed += removed;
                total_cleanup_replaced += replaced;
            } else {
                // Chunked translation — dispatch in parallel
                let disable_cleanup = args.disable_cleanup;
                let pm_for_chunks = provider_manager.clone();
                let pb_for_chunks = pb.clone();
                let futures: Vec<_> = chunks
                    .into_iter()
                    .map(|chunk| {
                        let permit = Arc::clone(&semaphore);
                        let pm = pm_for_chunks.clone();
                        let pb = pb_for_chunks.clone();
                        let source_clone = source.clone();
                        async move {
                            let _permit = permit.acquire().await;
                            translate_one_with_cleanup(
                                &chunk,
                                &source_clone,
                                target,
                                &pb,
                                &pm,
                                disable_cleanup,
                            )
                            .await
                        }
                    })
                    .collect();

                let results = futures::future::join_all(futures).await;
                let mut translated_parts = Vec::new();
                let mut chunk_timings: Vec<Option<BackendTimings>> = Vec::new();
                let mut chunk_removed = 0usize;
                let mut chunk_replaced = 0usize;

                for result in results {
                    let (translated, removed, replaced, timing) = result?;
                    translated_parts.push(translated);
                    chunk_timings.push(timing);
                    chunk_removed += removed;
                    chunk_replaced += replaced;
                }

                let combined = translated_parts.join("\n");
                translate_per_text_ms.push(start.elapsed().as_millis() as u64);
                let last_timing = chunk_timings.last().cloned().flatten();
                backend_timings.push(last_timing);
                translated_texts.push(combined);
                total_cleanup_removed += chunk_removed;
                total_cleanup_replaced += chunk_replaced;
            }
        }
    }

    let mut response =
        serde_json::json!({"translatedText": q.shaped_translated_text(translated_texts)});

    // For API compatibility with the original LibreTranslate project.
    if body.alternatives.is_some_and(|v| v > 0) {
        response["alternatives"] = q.shaped_empty_alternatives();
    }

    if source == "auto" {
        // Build detectedLanguage from our detected languages (LLM or whatlang fallback)
        let detected_lang_array = if q.is_batch() {
            serde_json::Value::Array(
                detected_languages
                    .iter()
                    .map(|d| {
                        serde_json::json!({
                            "language": d.language.code,
                            "confidence": d.confidence
                        })
                    })
                    .collect(),
            )
        } else {
            serde_json::json!({
                "language": detected_languages[0].language.code,
                "confidence": detected_languages[0].confidence
            })
        };
        response["detectedLanguage"] = detected_lang_array;
    }

    // Add cleanup report if requested
    let mut reports = serde_json::Map::new();
    if body.enable_cleanup_reporting == Some(true) {
        reports.insert(
            "cleanup".to_string(),
            serde_json::json!({
                "removed": total_cleanup_removed,
                "replaced": total_cleanup_replaced
            }),
        );
    }

    // Add performance report if requested
    if body.enable_performance_reporting == Some(true) {
        let total_detect_ms: u64 = detect_per_text_ms.iter().sum();
        let total_translate_ms: u64 = translate_per_text_ms.iter().sum();

        // Serialize backend timings as an array (null for missing timings)
        let backend_timings_json: Vec<Value> = backend_timings
            .iter()
            .map(|t| match t {
                Some(bt) => serde_json::to_value(bt).unwrap_or(Value::Null),
                None => Value::Null,
            })
            .collect();

        reports.insert(
            "performance".to_string(),
            serde_json::json!({
                "detect": {
                    "total_ms": total_detect_ms,
                    "per_text_ms": detect_per_text_ms
                },
                "translate": {
                    "total_ms": total_translate_ms,
                    "per_text_ms": translate_per_text_ms
                },
                "backend_timings": backend_timings_json
            }),
        );
    }

    if !reports.is_empty() {
        response["reports"] = Value::Object(reports);
    }

    Ok(HttpResponse::Ok().json(response))
}

#[post("/translate_file")]
async fn translate_file() -> Result<HttpResponse, ErrorResponse> {
    Err(ErrorResponse {
        error: "Not implemented".to_string(),
        status: 501,
    })
}

#[post("/suggest")]
async fn suggest() -> Result<HttpResponse, ErrorResponse> {
    Err(ErrorResponse {
        error: "Not implemented".to_string(),
        status: 501,
    })
}

#[get("/languages")]
async fn get_languages() -> impl Responder {
    HttpResponse::Ok().json(&*LANGUAGES)
}

#[get("/frontend/settings")]
async fn get_frontend_settings(args: web::Data<Arc<Args>>) -> impl Responder {
    HttpResponse::Ok().json(serde_json::json!({
        "apiKeys": false,
        "charLimit": args.char_limit,
        "chunkSize": args.chunk_size,
        "filesTranslation": false,
        "frontendTimeout": 1000,
        "keyRequired": false,
        "language": {
            "source": {
                "code": "auto",
                "name": "Auto Detect"
            },
            "target": {
                "code": "en",
                "name": "English"
            }
        },
        "suggestions": false,
        "supportedFilesFormat": []
    }))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let args = Arc::new(Args::parse());

    let host = args.host.clone();
    let port = args.port;

    // Create translation provider (OpenAI-compatible backend) with retry logic
    // Server-side API key is used for authenticating incoming translation requests.
    // Backend API key is sent to the external backend for authentication.
    let backend_api_key = if args.backend_api_key.is_empty() {
        None
    } else {
        Some(args.backend_api_key.as_str())
    };
    let provider_config = ProviderConfig {
        max_attempts: args.retry_attempts,
        base_delay_ms: args.retry_delay,
        recheck_interval_secs: args.model_recheck_interval,
    };
    let provider_manager = Arc::new(ProviderManager::new(
        &args.backend_host,
        &args.model,
        backend_api_key,
        provider_config,
    ));

    // Initialize the provider with retry (non-blocking if backend isn't ready yet)
    provider_manager.initialize().await;

    // If benchmark mode, run the benchmark and exit
    if let Some(benchmark_mode) = &args.benchmark {
        #[cfg(feature = "benchmark")]
        {
            benchmark::run_benchmark(&args, benchmark_mode, &provider_manager).await;
        }
        #[cfg(not(feature = "benchmark"))]
        {
            eprintln!("Benchmark feature not enabled. Rebuild with `--features benchmark`.");
        }
        return Ok(());
    }

    // Start periodic rechecker if enabled
    if args.model_recheck_interval > 0 {
        provider_manager.start_rechecker();
    }

    print_banner();

    let backend_host = args.backend_host.clone();
    let model = args.model.clone();
    let retry_attempts = args.retry_attempts;
    let retry_delay = args.retry_delay;

    let args = Arc::clone(&args);

    // Clone provider_manager before moving into closure
    let provider_manager_for_shutdown = provider_manager.clone();

    let server = HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(provider_manager.clone()))
            .app_data(web::Data::new(args.clone()))
            .service(get_languages)
            .service(get_frontend_settings)
            .service(health)
            .service(backend_health)
            .service(translate)
            .service(translate_file)
            .service(detect)
            .service(suggest)
            // Catch-all: any unmatched path goes to the static handler
            .default_service(web::to(static_handler::serve_static))
    })
    .bind((host.clone(), port))?
    .run();

    println!("Running on: http://{}:{}", host, port);
    println!("Using backend: {} (model: {})", backend_host, model);
    println!("Health endpoints: GET /health (service), GET /health/backend (backend status)");
    println!(
        "Provider will retry {} times with base delay {}ms for transient errors",
        retry_attempts, retry_delay
    );

    // Run server with graceful shutdown on SIGINT/SIGTERM
    tokio::select! {
        _ = server => {},
        _ = tokio::signal::ctrl_c() => {
            eprintln!("Shutting down...");
            provider_manager_for_shutdown.shutdown();
        }
    }

    Ok(())
}

#[cfg(test)]
mod chunk_tests {
    use super::chunk_text;

    #[test]
    fn test_chunk_text_empty() {
        let chunks = chunk_text("", 100);
        assert_eq!(chunks, vec!["".to_string()]);
    }

    #[test]
    fn test_chunk_text_small() {
        let chunks = chunk_text("Hello, world!", 100);
        assert_eq!(chunks, vec!["Hello, world!".to_string()]);
    }

    #[test]
    fn test_chunk_text_exact_size() {
        let text = "a".repeat(21);
        assert_eq!(text.len(), 21);
        let chunks = chunk_text(&text, 21);
        assert_eq!(chunks, vec![text]);
    }

    #[test]
    fn test_chunk_text_over_limit_fixed_fallback() {
        // Long string with no paragraph or sentence boundaries
        let text = "a".repeat(120);
        let chunks = chunk_text(&text, 50);
        assert_eq!(chunks.len(), 3); // 50 + 50 + 20
        assert_eq!(chunks[0].len(), 50);
        assert_eq!(chunks[1].len(), 50);
        assert_eq!(chunks[2].len(), 20);
        assert_eq!(chunks.join(""), text);
    }

    #[test]
    fn test_chunk_text_by_paragraphs() {
        let text = "First paragraph line 1. Line 2.\n\nSecond paragraph. More text.";
        let chunks = chunk_text(text, 31); // first para is 30 bytes, so 31 should fit
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], "First paragraph line 1. Line 2.");
        assert_eq!(chunks[1], "Second paragraph. More text.");
    }

    #[test]
    fn test_chunk_text_by_paragraphs_merging() {
        // Two paragraphs that fit together under the limit
        let text = "Short para.\n\nAnother short.";
        let chunks = chunk_text(text, 100);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Short para.\n\nAnother short.");
    }

    #[test]
    fn test_chunk_text_by_paragraphs_over_limit_fallback() {
        // Two paragraphs, but first one exceeds the limit when combined with second
        let text = "Very long paragraph with lots of content that exceeds limit.\n\nAnother short.";
        let chunks = chunk_text(text, 60);
        // First paragraph alone is 55 chars, plus the newline and second para (13 chars) = 68 chars > 60
        // So first para should be its own chunk, second para its own chunk
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].len() <= 60);
        assert!(chunks[1].len() <= 60);
    }

    #[test]
    fn test_chunk_text_by_sentences() {
        let text = "First sentence. Second sentence. Third sentence.";
        let chunks = chunk_text(text, 20);
        // "First sentence." is 15 chars, " Second sentence." is 16 chars, etc.
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "First sentence.");
        assert_eq!(chunks[1], "Second sentence.");
        assert_eq!(chunks[2], "Third sentence.");
    }

    #[test]
    fn test_chunk_text_mixed() {
        let text = "Para 1. Short.\n\nPara 2 is a very long paragraph that will be split into sentences because it exceeds the chunk size when combined with other paragraphs, and then further split if needed.";
        let chunks = chunk_text(text, 50);
        assert!(chunks.len() >= 2);
        for chunk in &chunks {
            assert!(
                chunk.len() <= 50,
                "Chunk exceeds limit: {} chars",
                chunk.len()
            );
        }
        // The first chunk should be the short paragraph
        assert_eq!(chunks[0], "Para 1. Short.");
    }
}
