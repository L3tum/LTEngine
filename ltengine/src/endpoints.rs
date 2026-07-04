//! Miscellaneous endpoints: health, languages, settings, stubs.

use actix_web::{HttpResponse, Responder, get, post, web};
use std::sync::Arc;

use crate::backend::ProviderManager;
use crate::cli::Args;
use crate::error_response::ErrorResponse;
use crate::languages::LANGUAGES;

/// GET /health — service health check.
#[get("/health")]
pub async fn health() -> impl Responder {
    HttpResponse::Ok().json(serde_json::json!({
        "status": "ok",
        "service": "ltengine",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

/// GET /health/backend — backend connectivity check.
#[get("/health/backend")]
pub async fn backend_health(provider_manager: web::Data<Arc<ProviderManager>>) -> impl Responder {
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

/// POST /translate_file — not implemented.
#[post("/translate_file")]
pub async fn translate_file() -> Result<HttpResponse, ErrorResponse> {
    Err(ErrorResponse {
        error: "Not implemented".to_string(),
        status: 501,
    })
}

/// POST /suggest — not implemented.
#[post("/suggest")]
pub async fn suggest() -> Result<HttpResponse, ErrorResponse> {
    Err(ErrorResponse {
        error: "Not implemented".to_string(),
        status: 501,
    })
}

/// GET /languages — list all supported languages.
#[get("/languages")]
pub async fn get_languages() -> impl Responder {
    HttpResponse::Ok().json(&*LANGUAGES)
}

/// GET /frontend/settings — frontend configuration.
#[get("/frontend/settings")]
pub async fn get_frontend_settings(args: web::Data<Arc<Args>>) -> impl Responder {
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
        "tone": ["neutral", "formal", "informal"],
        "suggestions": false,
        "supportedFilesFormat": []
    }))
}
