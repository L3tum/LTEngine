use actix_web::{
    get, post, web, App, HttpRequest, HttpResponse, 
    HttpServer, Responder, http::header, FromRequest
};
use actix_multipart::form::{MultipartForm, text::Text as MPText};
use actix_web_static_files::ResourceFiles;
use std::sync::Arc;
use clap::Parser;
use serde::{Deserialize, Serialize};

mod error_response;
mod languages;
mod backend;
mod banner;
mod prompt;
#[cfg(test)]
mod tests;

use languages::{detect_lang, get_language_from_code, LANGUAGES};
use error_response::ErrorResponse;
use backend::{BackendError, ProviderManager, ProviderConfig};
use banner::print_banner;
use prompt::PromptBuilder;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

#[derive(Parser, Debug, Clone)]
#[command(version, about, long_about = None)]
struct Args {
    /// Hostname to bind to (env var: LTE_HOST)
    #[arg(long, env = "LTE_HOST", default_value = "0.0.0.0")]
    host: String,

    /// Port to bind to (env var: LTE_PORT)
    #[arg(short, long, env = "LTE_PORT", default_value_t = 5050)]
    port: u16,

    /// Character limit for translation requests (env var: LTE_CHAR_LIMIT)
    #[arg(long, env = "LTE_CHAR_LIMIT", default_value_t = 5000)]
    char_limit: usize,

    /// Model to use (passed to the backend provider) (env var: LTE_MODEL)
    #[arg(short='m', long, env = "LTE_MODEL", default_value = "gemma3-4b")]
    model: String,

    /// Backend host (e.g., "localhost:11434" for Ollama, or full URL like "http://ollama:11434") (env var: LTE_BACKEND_HOST)
    #[arg(long, env = "LTE_BACKEND_HOST", default_value = "localhost:11434")]
    backend_host: String,

    /// Set an API key (optional, for authenticating translation requests to the API) (env var: LTE_API_KEY)
    #[arg(long, env = "LTE_API_KEY", default_value = "")]
    api_key: String,

    /// Set a backend API key (optional, for authenticating with the external backend) (env var: LTE_BACKEND_API_KEY)
    #[arg(long = "backend-api-key", env = "LTE_BACKEND_API_KEY", default_value = "")]
    backend_api_key: String,

    /// Maximum number of retry attempts for provider creation and translation requests (env var: LTE_RETRY_ATTEMPTS)
    #[arg(long, env = "LTE_RETRY_ATTEMPTS", default_value_t = 5)]
    retry_attempts: usize,

    /// Base delay in milliseconds for exponential backoff (1000ms → 1s, 2s, 4s...) (env var: LTE_RETRY_DELAY)
    #[arg(long, env = "LTE_RETRY_DELAY", default_value_t = 1000)]
    retry_delay: u64,

    /// Interval in seconds for periodic model availability rechecks (0 to disable) (env var: LTE_MODEL_RECHECK_INTERVAL)
    #[arg(long, env = "LTE_MODEL_RECHECK_INTERVAL", default_value_t = 30)]
    model_recheck_interval: u64
}

#[derive(Debug, Deserialize, Serialize)]
struct TranslateRequest {
    q: Option<String>,
    source: Option<String>,
    target: Option<String>,
    format: Option<String>,
    api_key: Option<String>,
    alternatives: Option<u32>
}

#[derive(MultipartForm)]
struct MPTranslateRequest {
    q: Option<MPText<String>>,
    source: Option<MPText<String>>,
    target: Option<MPText<String>>,
    format: Option<MPText<String>>,
    api_key: Option<MPText<String>>,
    alternatives: Option<MPText<u32>>
}
impl MPTranslateRequest {
    fn into_translate_request(self) -> TranslateRequest {
        TranslateRequest {
            q: self.q.map(|v| v.into_inner()),
            source: self.source.map(|v| v.into_inner()),
            target: self.target.map(|v| v.into_inner()),
            format: self.format.map(|v| v.into_inner()),
            api_key: self.api_key.map(|v| v.into_inner()),
            alternatives: self.alternatives.map(|v| v.into_inner()),
        }
    }
}

async fn parse_payload(req: HttpRequest, payload: web::Payload) -> Result<TranslateRequest, ErrorResponse>{
    let content_type = req.headers().get(header::CONTENT_TYPE).map(|h| h.to_str().unwrap_or("")).unwrap_or("");
    let body: TranslateRequest;

    if content_type.starts_with("application/json") {
        let json = actix_web::web::Json::<TranslateRequest>::from_request(&req, &mut payload.into_inner()).await?;
        body = json.into_inner()
    } else if content_type.starts_with("application/x-www-form-urlencoded") {
        let form = actix_web::web::Form::<TranslateRequest>::from_request(&req, &mut payload.into_inner()).await?;
        body = form.into_inner()
    } else if content_type.starts_with("multipart/form-data") {
        let form = MultipartForm::<MPTranslateRequest>::from_request(&req, &mut payload.into_inner()).await?;
        body = form.into_inner().into_translate_request();
    } else {
        return Err(ErrorResponse{ error: "Unsupported content-type".to_string(), status: 400 });
    }

    return Ok(body);
}

fn check_params(body: &TranslateRequest, args: &Args, required_params: &[(&str, &Option<String>)]) -> Result<bool, ErrorResponse> {
    // Validate required params
    for (key, value) in required_params {
        if value.as_ref().is_none_or(|v| v.trim().is_empty()) {
            return Err(ErrorResponse {
                error: format!("Invalid request: missing {} parameter", key),
                status: 400,
            });
        }
    }
    
    // Check API key: if the server has an API key configured, the request must include
    // it. Returns 403 if the key is missing or doesn't match.
    if !args.api_key.is_empty() && body.api_key.as_ref().is_none_or(|key| *key != args.api_key) {
        return Err(ErrorResponse {
            error: format!("Invalid API key"),
            status: 403,
        });
    }

    let q = body.q.as_ref().unwrap();
    if q.len() > args.char_limit {
        return Err(ErrorResponse {
            error: format!("Invalid request: request ({}) exceeds text limit ({})", q.len(), args.char_limit),
            status: 400,
        });
    }

    Ok(true)
}

fn improve_formatting(q: &String, translation: &String) -> String {
    let t = translation.trim().to_string();

    if q.len() == 0 {
        return String::new();
    }

    if t.len() == 0 {
        return q.clone();
    }

    let q_last_char = q.chars().rev().next().unwrap();
    let translation_last_char = t.chars().rev().next().unwrap();
    let mut result = t.clone();

    const PUNCTUATION_CHARS: [char; 6] = ['!', '?', '.', ',', ';', '。'];
    if PUNCTUATION_CHARS.contains(&q_last_char){
        if q_last_char != translation_last_char{
            if PUNCTUATION_CHARS.contains(&translation_last_char){
                result.pop();
            }

            result.push(q_last_char);
        }
    }else if PUNCTUATION_CHARS.contains(&translation_last_char) {
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
        }else if q0.is_uppercase() && r0.is_lowercase() {
            result.replace_range(0..r0.len_utf8(), &r0.to_uppercase().to_string());
        }
    }

    result.trim().to_string()
}

#[post("/detect")]
async fn detect(req: HttpRequest, payload: web::Payload, args: web::Data<Arc<Args>>) -> Result<HttpResponse, ErrorResponse> {
    let body = parse_payload(req, payload).await?;
    check_params(&body, &args, &[
        ("q", &body.q)
    ])?;

    let q = body.q.unwrap();
    let d = detect_lang(&q);

    Ok(HttpResponse::Ok().json(serde_json::json!([{
        "language": d.language.code,
        "confidence": d.confidence
    }])))
}

fn check_format(format: &str) -> Result<bool, ErrorResponse> {
    match format {
        "text" | "html" => Ok(true),
        _ => Err(ErrorResponse {
            error: "Invalid format. Supported formats: text, html".to_string(),
            status: 400,
        })
    }
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
async fn backend_health(provider: web::Data<Arc<ProviderManager>>) -> impl Responder {
    let result = provider.ping().await;
    
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
    check_params(&body, &args, &[
        ("q", &body.q),
        ("source", &body.source),
        ("target", &body.target),
    ])?;

    let q = body.q.unwrap();
    let source = body.source.unwrap();
    let target = body.target.unwrap();
    let format = body.format.unwrap_or("text".to_string());
    check_format(&format)?;
    
    let mut pb = PromptBuilder::new();
    pb.set_format(&format);

    if source == "auto"{
        pb.set_source_language("auto");
    }else{
        let src_lang = get_language_from_code(&source).ok_or_else(|| ErrorResponse {
            error: format!("{} is not supported", source),
            status: 400,
        })?;
        pb.set_source_language(src_lang.name);
    }

    let tgt_lang = get_language_from_code(&target).ok_or_else(|| ErrorResponse {
        error: format!("{} is not supported", target),
        status: 400,
    })?;
    pb.set_target_language(tgt_lang.name);

    let prompt = pb.build(&q);
    
    // If source equals target, return the original text without translating.
    // This is a performance optimization but may be semantically unexpected.
    let translated_text = if source == target {
        q.clone()
    } else {
        provider_manager.translate(&prompt.system, &prompt.user).await.map_err(|e| {
            eprintln!("translation error: {}", e);

            // Check if the error is a BackendError with an HTTP status code or retryable
            let (msg, status) = if let Some(backend_err) = e.downcast_ref::<BackendError>() {
                match backend_err {
                    BackendError::Http { status_code, detail: _ } => {
                        // Don't leak backend details, use generic message
                        let http_status = match *status_code {
                            401 => 403,  // Forbidden (authentication failed)
                            404 => 404,  // Model not found
                            _ => 503,    // Other server errors → 503
                        };
                        (format!("Backend returned {}", status_code), http_status)
                    }
                    BackendError::Retryable(_) => {
                        // Should not happen in normal operation (ProviderManager handles retries),
                        // but if it does, return a service unavailable error
                        ("Backend temporarily unavailable".to_string(), 503)
                    }
                    BackendError::Other(_) => {
                        // Other backend errors (parsing, empty response) — 500
                        ("Backend error".to_string(), 500)
                    }
                }
            } else {
                // Generic error (e.g., model not available)
                (format!("{:#}", e), 500)
            };

            ErrorResponse { error: msg, status }
        })?
    };
    
    let mut response = serde_json::json!({"translatedText": improve_formatting(&q, &translated_text)});

    // For compatibility with LibreTranslate API
    if body.alternatives.is_some_and(|v| v > 0) {
        response["alternatives"] = serde_json::json!([]);
    }

    if source == "auto" {
        let d = detect_lang(&q);
        response["detectedLanguage"] = serde_json::json!({
            "language": d.language.code,
            "confidence": d.confidence
        });
    }

    Ok(HttpResponse::Ok().json(response))
}

#[post("/translate_file")]
async fn translate_file() -> Result<HttpResponse, ErrorResponse> {
    Err(ErrorResponse{
        error: "Not implemented".to_string(),
        status: 501
    })
}

#[post("/suggest")]
async fn suggest() -> Result<HttpResponse, ErrorResponse> {
    Err(ErrorResponse{
        error: "Not implemented".to_string(),
        status: 501
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
    let backend_api_key = if args.backend_api_key.is_empty() { None } else { Some(args.backend_api_key.as_str()) };
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

    let server = HttpServer::new(move || {
        let generated = generate();

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
            .service(ResourceFiles::new("/", generated))
    })
    .bind((host.clone(), port))?
    .run();

    println!("Running on: http://{}:{}", host, port);
    println!("Using backend: {} (model: {})", backend_host, model);
    println!("Health endpoints: GET /health (service), GET /health/backend (backend status)");
    println!("Provider will retry {} times with base delay {}ms for transient errors",
             retry_attempts, retry_delay);

    return server.await;
}