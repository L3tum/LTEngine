//! Request models, parsing, and validation.

use actix_multipart::form::{MultipartForm, text::Text as MPText};
use actix_web::{FromRequest, HttpRequest, http::header, web};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::cli::Args;
use crate::error_response::ErrorResponse;

/// Represents either a single query string or a batch of strings.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[serde(untagged)]
pub enum QueryText {
    Single(String),
    Batch(Vec<String>),
}

impl QueryText {
    /// View the underlying strings as a slice.
    pub fn as_slice(&self) -> &[String] {
        match self {
            QueryText::Single(q) => std::slice::from_ref(q),
            QueryText::Batch(q) => q.as_slice(),
        }
    }

    /// True if this is a batch (array).
    pub fn is_batch(&self) -> bool {
        matches!(self, QueryText::Batch(_))
    }

    /// True if the query is missing or any item is blank.
    pub fn is_empty_or_blank(&self) -> bool {
        let items = self.as_slice();
        items.is_empty() || items.iter().any(|q| q.trim().is_empty())
    }

    /// Total character length of all items.
    pub fn total_len(&self) -> usize {
        self.as_slice().iter().map(String::len).sum()
    }

    /// Build the response shape for translated text (string or array).
    pub fn shaped_translated_text(&self, translated_texts: Vec<String>) -> Value {
        if self.is_batch() {
            serde_json::json!(translated_texts)
        } else {
            serde_json::json!(
                translated_texts
                    .into_iter()
                    .next()
                    .map_or(Value::Null, Value::String)
            )
        }
    }

    /// Build an empty alternatives array with the correct shape.
    pub fn shaped_empty_alternatives(&self) -> Value {
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

/// Full translation request body (JSON, form, or multipart).
#[derive(Debug, Deserialize, Serialize)]
pub struct TranslateRequest {
    pub q: Option<QueryText>,
    pub source: Option<String>,
    pub target: Option<String>,
    pub format: Option<String>,
    pub api_key: Option<String>,
    pub alternatives: Option<u32>,
    pub enable_cleanup_reporting: Option<bool>,
    pub enable_performance_reporting: Option<bool>,
}

/// Multipart form variant of TranslateRequest.
#[derive(MultipartForm)]
pub struct MPTranslateRequest {
    pub q: Option<MPText<String>>,
    pub source: Option<MPText<String>>,
    pub target: Option<MPText<String>>,
    pub format: Option<MPText<String>>,
    pub api_key: Option<MPText<String>>,
    pub alternatives: Option<MPText<u32>>,
    pub enable_cleanup_reporting: Option<MPText<String>>,
    pub enable_performance_reporting: Option<MPText<String>>,
}

/// Convert a string to bool ("true", "yes", "1" → true).
fn str_to_bool(s: &str) -> bool {
    matches!(s.trim().to_lowercase().as_str(), "true" | "yes" | "1")
}

impl MPTranslateRequest {
    /// Convert a multipart form into a normal TranslateRequest.
    pub fn into_translate_request(self) -> TranslateRequest {
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

/// Parse the HTTP request payload into a TranslateRequest, handling JSON,
/// form-urlencoded, and multipart/form-data content types.
pub async fn parse_payload(
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

/// Validate translation request parameters: required fields, API key, batch limit, and total char limit.
pub fn check_params(
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

/// Validate the format parameter (text or html).
pub fn check_format(format: &str) -> Result<(), ErrorResponse> {
    match format {
        "text" | "html" => Ok(()),
        _ => Err(ErrorResponse {
            error: "Invalid format. Supported formats: text, html".to_string(),
            status: 400,
        }),
    }
}
