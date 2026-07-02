//! Language detection endpoint.

use actix_web::{HttpRequest, HttpResponse, post, web};
use std::sync::Arc;

use crate::cli::Args;
use crate::detection::create_detector;
use crate::error_response::ErrorResponse;
use crate::request::{check_params, parse_payload};

/// POST /detect — detect language(s) of input text.
#[post("/detect")]
pub async fn detect(
    req: HttpRequest,
    payload: web::Payload,
    args: web::Data<Arc<Args>>,
    provider_manager: Option<web::Data<Arc<crate::backend::ProviderManager>>>,
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
        Box::new(crate::detection::WhatlangDetector)
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
