//! Translation pipeline and /translate endpoint.

use actix_web::{HttpRequest, HttpResponse, post, web};
use futures::future;
use std::sync::Arc;
use std::time::Instant;

use crate::backend::{BackendError, BackendTimings, ProviderManager};
use crate::chunking::chunk_text;
use crate::cli::Args;
use crate::detection::create_detector;
use crate::error_response::ErrorResponse;
use crate::languages::get_language_from_code;
use crate::postprocess::improve_formatting;
use crate::prompt::PromptBuilder;
use crate::request::{check_format, check_params, parse_payload};

/// Map a translation backend error to an HTTP ErrorResponse.
fn map_translation_error(e: anyhow::Error) -> ErrorResponse {
    eprintln!("translation error: {}", e);

    let (msg, status) = if let Some(backend_err) = e.downcast_ref::<BackendError>() {
        match backend_err {
            BackendError::Http(status_code) => {
                let http_status = match *status_code {
                    401 => 403,
                    404 => 404,
                    _ => 503,
                };
                (format!("Backend returned {}", status_code), http_status)
            }
            BackendError::ModelNotFound(msg) => (msg.to_string(), 404),
            BackendError::Retryable(_) => ("Backend temporarily unavailable".to_string(), 503),
        }
    } else {
        (format!("{}", e), 503)
    };

    ErrorResponse { error: msg, status }
}

/// Translate a single text chunk with cleanup and post-processing.
pub async fn translate_one_with_cleanup(
    q: &String,
    source: &str,
    target: &str,
    prompt_builder: &PromptBuilder,
    provider_manager: &ProviderManager,
    disable_cleanup: bool,
) -> Result<(String, usize, usize, Option<BackendTimings>), ErrorResponse> {
    // If source equals target, return the original text without translating.
    let (translated_text, backend_timings) = if source == target {
        (q.to_string(), None)
    } else {
        let prompt = prompt_builder.build(q);
        let result = provider_manager
            .translate(&prompt.system, &prompt.user)
            .await
            .map_err(map_translation_error)?;
        (result.text, result.backend_timings)
    };

    // Apply cleanup unless disabled
    let (translated_text, removed, replaced) = if disable_cleanup {
        (translated_text, 0, 0)
    } else {
        let result = crate::cleanup::cleanup_output(&translated_text);
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

/// POST /translate — full translation endpoint with optional language detection,
/// chunking, and parallel translation.
#[post("/translate")]
pub async fn translate(
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
                // Single chunk, translate directly
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

                let results = future::join_all(futures).await;
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
    } else {
        // Source is explicitly specified, no detection.
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
                // Chunked translation
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

                let results = future::join_all(futures).await;
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

    // For API compatibility with LibreTranslate.
    if body.alternatives.is_some_and(|v| v > 0) {
        response["alternatives"] = q.shaped_empty_alternatives();
    }

    if source == "auto" {
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

        let backend_timings_json: Vec<serde_json::Value> = backend_timings
            .iter()
            .map(|t| match t {
                Some(bt) => serde_json::to_value(bt).unwrap_or(serde_json::Value::Null),
                None => serde_json::Value::Null,
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
        response["reports"] = serde_json::Value::Object(reports);
    }

    Ok(HttpResponse::Ok().json(response))
}
