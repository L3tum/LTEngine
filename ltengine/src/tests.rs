//! Integration tests with mock translation provider.

use crate::backend::{
    BackendError, ProviderConfig, ProviderManager, TranslateProvider, TranslationResult,
};
use crate::prompt::PromptBuilder;
use crate::{Args, QueryText, TranslateRequest, check_params, detect, translate};
use actix_web::{App, http::StatusCode, test, web};
use std::sync::{Arc, Mutex};

/// A simple mock provider that always returns a fixed translation.
struct MockProvider;

#[async_trait::async_trait]
impl TranslateProvider for MockProvider {
    async fn translate(&self, _system: &str, _user: &str) -> anyhow::Result<TranslationResult> {
        Ok(TranslationResult {
            text: "TranslatedText".to_string(),
            backend_timings: None,
        })
    }
    async fn ping(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

/// A mock provider that tracks call count and fails a fixed number of times.
struct RetryMockProvider {
    call_count: Arc<Mutex<u32>>,
    fail_count: u32,
}

impl RetryMockProvider {
    fn new(fail_count: u32) -> Self {
        RetryMockProvider {
            call_count: Arc::new(Mutex::new(0)),
            fail_count,
        }
    }
}

#[async_trait::async_trait]
impl TranslateProvider for RetryMockProvider {
    async fn translate(&self, _system: &str, _user: &str) -> anyhow::Result<TranslationResult> {
        let mut count = self.call_count.lock().unwrap();
        let current = *count;
        *count += 1;
        drop(count);

        if current < self.fail_count {
            // Simulate a transient retryable error (connection timeout, etc.)
            Err(BackendError::Retryable(format!("Simulated timeout (call {})", current)).into())
        } else {
            Ok(TranslationResult {
                text: "Finally worked".to_string(),
                backend_timings: None,
            })
        }
    }
    async fn ping(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn test_mock_provider_translate() {
    let provider = MockProvider;
    let result = provider
        .translate("You are an expert linguist.", "Hello, world!")
        .await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap().text, "TranslatedText");
}

#[tokio::test]
async fn test_prompt_builder() {
    let mut pb = PromptBuilder::new();
    pb.set_source_language("English");
    pb.set_target_language("Spanish");
    let prompt = pb.build(&"Hello!".to_string());
    assert!(prompt.system.contains("expert linguist"));
    assert!(prompt.user.contains("Spanish"));
    assert!(prompt.user.contains("Hello!"));
}

/// Test that ProviderManager retries correctly on transient (retryable) errors.
/// This uses RetryMockProvider which fails the first 2 calls, then succeeds.
#[tokio::test]
async fn test_retry_on_transient_failures() {
    // RetryMockProvider fails first 2 times with Retryable errors, then succeeds
    let provider = RetryMockProvider::new(2);
    let call_count = provider.call_count.clone(); // save for assertion
    let config = ProviderConfig {
        max_attempts: 3,
        base_delay_ms: 1, // 1ms to keep the test fast
        ..Default::default()
    };
    let provider_manager = ProviderManager::from_provider(Arc::new(provider), config);

    let result = provider_manager
        .translate("You are an expert linguist.", "Hello, world!")
        .await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap().text, "Finally worked");
    // Should have tried 3 times (2 failures + 1 success)
    assert_eq!(
        *call_count.lock().unwrap(),
        3,
        "Should have 3 attempts (2 failures + 1 success)"
    );
}

/// A mock provider that always returns a permanent (non-retryable) HTTP error.
struct PermanentErrorProvider;

#[async_trait::async_trait]
impl TranslateProvider for PermanentErrorProvider {
    async fn translate(&self, _system: &str, _user: &str) -> anyhow::Result<TranslationResult> {
        Err(BackendError::Http(401).into())
    }
    async fn ping(&self) -> anyhow::Result<()> {
        Err(BackendError::Http(401).into())
    }
}

/// Test that non-retryable errors (auth failure) are returned immediately without retry,
/// and the provider is dropped.
#[tokio::test]
async fn test_permanent_error_drops_provider() {
    let provider = PermanentErrorProvider;
    let config = ProviderConfig {
        max_attempts: 5, // even with many retries, permanent error should not retry
        base_delay_ms: 1000,
        ..Default::default()
    };
    let provider_manager = ProviderManager::from_provider(Arc::new(provider), config);

    let result = provider_manager.translate("system", "Hello").await;

    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("HTTP error 401"),
        "Error should mention 401 status"
    );

    // Verify the provider was dropped — a subsequent call should fail
    // (it will try to recreate the provider, which will fail since no backend is reachable)
    let result = provider_manager.translate("system", "Hello again").await;
    assert!(
        result.is_err(),
        "Second call should fail because provider was dropped and no backend is reachable"
    );
}

fn test_args() -> Args {
    Args {
        host: "127.0.0.1".to_string(),
        port: 5050,
        char_limit: 5000,
        batch_limit: 50,
        model: "test-model".to_string(),
        backend_host: "localhost:59999".to_string(),
        api_key: String::new(),
        backend_api_key: String::new(),
        retry_attempts: 1,
        retry_delay: 1,
        model_recheck_interval: 0,
        llm_detect: false,
        benchmark: false,
        dataset: None,
        disable_cleanup: false,
    }
}

#[actix_web::test]
async fn test_query_text_deserializes_string_and_array() {
    let scalar: TranslateRequest = serde_json::from_value(serde_json::json!({
        "q": "Hello",
        "source": "en",
        "target": "es"
    }))
    .unwrap();
    assert_eq!(scalar.q, Some(QueryText::Single("Hello".to_string())));

    let batch: TranslateRequest = serde_json::from_value(serde_json::json!({
        "q": ["Hello", "World"],
        "source": "en",
        "target": "es"
    }))
    .unwrap();
    assert_eq!(
        batch.q,
        Some(QueryText::Batch(vec![
            "Hello".to_string(),
            "World".to_string()
        ]))
    );
}

#[actix_web::test]
async fn test_query_text_validation_rejects_empty_blank_and_over_limit_requests() {
    let mut args = test_args();
    args.char_limit = 5;

    let empty_batch = TranslateRequest {
        q: Some(QueryText::Batch(Vec::new())),
        source: Some("en".to_string()),
        target: Some("es".to_string()),
        format: None,
        api_key: None,
        alternatives: None,
        enable_cleanup_reporting: None,
        enable_performance_reporting: None,
    };
    assert_eq!(
        check_params(
            &empty_batch,
            &args,
            &[
                ("source", &empty_batch.source),
                ("target", &empty_batch.target)
            ]
        )
        .unwrap_err()
        .status,
        400
    );

    let blank_item = TranslateRequest {
        q: Some(QueryText::Batch(vec![
            "Hello".to_string(),
            "  ".to_string(),
        ])),
        source: Some("en".to_string()),
        target: Some("es".to_string()),
        format: None,
        api_key: None,
        alternatives: None,
        enable_cleanup_reporting: None,
        enable_performance_reporting: None,
    };
    assert_eq!(
        check_params(
            &blank_item,
            &args,
            &[
                ("source", &blank_item.source),
                ("target", &blank_item.target)
            ]
        )
        .unwrap_err()
        .status,
        400
    );

    let total_over_limit = TranslateRequest {
        q: Some(QueryText::Batch(vec![
            "Hello".to_string(),
            "World".to_string(),
        ])),
        source: Some("en".to_string()),
        target: Some("es".to_string()),
        format: None,
        api_key: None,
        alternatives: None,
        enable_cleanup_reporting: None,
        enable_performance_reporting: None,
    };
    assert_eq!(
        check_params(
            &total_over_limit,
            &args,
            &[
                ("source", &total_over_limit.source),
                ("target", &total_over_limit.target)
            ]
        )
        .unwrap_err()
        .status,
        400
    );

    let item_over_limit = TranslateRequest {
        q: Some(QueryText::Batch(vec![
            "Hello".to_string(),
            "World!".to_string(),
        ])),
        source: Some("en".to_string()),
        target: Some("es".to_string()),
        format: None,
        api_key: None,
        alternatives: None,
        enable_cleanup_reporting: None,
        enable_performance_reporting: None,
    };
    assert_eq!(
        check_params(
            &item_over_limit,
            &args,
            &[
                ("source", &item_over_limit.source),
                ("target", &item_over_limit.target)
            ]
        )
        .unwrap_err()
        .status,
        400
    );

    args.batch_limit = 1;
    let batch_over_limit = TranslateRequest {
        q: Some(QueryText::Batch(vec!["Hi".to_string(), "Yo".to_string()])),
        source: Some("en".to_string()),
        target: Some("es".to_string()),
        format: None,
        api_key: None,
        alternatives: None,
        enable_cleanup_reporting: None,
        enable_performance_reporting: None,
    };
    let err = check_params(
        &batch_over_limit,
        &args,
        &[
            ("source", &batch_over_limit.source),
            ("target", &batch_over_limit.target),
        ],
    )
    .unwrap_err();
    assert_eq!(err.status, 400);
    assert!(err.error.contains("batch size"));
}

#[actix_web::test]
async fn test_translate_scalar_and_batch_response_shapes() {
    let args = Arc::new(test_args());
    let provider_manager = Arc::new(ProviderManager::from_provider(
        Arc::new(MockProvider),
        ProviderConfig::default(),
    ));

    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(args))
            .app_data(web::Data::new(provider_manager))
            .service(translate),
    )
    .await;

    let scalar_req = test::TestRequest::post()
        .uri("/translate")
        .insert_header(("Content-Type", "application/json"))
        .set_payload(r#"{"q":"Hello","source":"en","target":"es","alternatives":1}"#)
        .to_request();
    let scalar_resp: serde_json::Value = test::call_and_read_body_json(&app, scalar_req).await;
    assert_eq!(scalar_resp["translatedText"], "TranslatedText");
    assert!(
        scalar_resp["alternatives"]
            .as_array()
            .is_some_and(Vec::is_empty)
    );

    let batch_req = test::TestRequest::post()
        .uri("/translate")
        .insert_header(("Content-Type", "application/json"))
        .set_payload(r#"{"q":["Hello","World"],"source":"en","target":"es","alternatives":1}"#)
        .to_request();
    let batch_resp: serde_json::Value = test::call_and_read_body_json(&app, batch_req).await;
    assert_eq!(
        batch_resp["translatedText"],
        serde_json::json!(["TranslatedText", "TranslatedText"])
    );
    assert_eq!(batch_resp["alternatives"], serde_json::json!([[], []]));

    let form_req = test::TestRequest::post()
        .uri("/translate")
        .insert_header(("Content-Type", "application/x-www-form-urlencoded"))
        .set_payload("q=Hello&source=en&target=es")
        .to_request();
    let form_resp: serde_json::Value = test::call_and_read_body_json(&app, form_req).await;
    assert_eq!(form_resp["translatedText"], "TranslatedText");
}

#[actix_web::test]
async fn test_translate_auto_batch_detected_language_shape() {
    let args = Arc::new(test_args());
    let provider_manager = Arc::new(ProviderManager::from_provider(
        Arc::new(MockProvider),
        ProviderConfig::default(),
    ));

    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(args))
            .app_data(web::Data::new(provider_manager))
            .service(translate),
    )
    .await;

    let req = test::TestRequest::post()
        .uri("/translate")
        .insert_header(("Content-Type", "application/json"))
        .set_payload(
            r#"{"q":["This sentence is written in English with several common English words","Cette phrase est écrite en français avec plusieurs mots français courants"],"source":"auto","target":"es"}"#,
        )
        .to_request();
    let resp: serde_json::Value = test::call_and_read_body_json(&app, req).await;
    assert_eq!(
        resp["translatedText"],
        serde_json::json!(["TranslatedText", "TranslatedText"])
    );
    assert_eq!(resp["detectedLanguage"].as_array().map(Vec::len), Some(2));
    assert_eq!(resp["detectedLanguage"][0]["language"], "en");
    assert_eq!(resp["detectedLanguage"][1]["language"], "fr");
}

#[actix_web::test]
async fn test_detect_accepts_batch_q() {
    let args = Arc::new(test_args());
    let app = test::init_service(App::new().app_data(web::Data::new(args)).service(detect)).await;

    let req = test::TestRequest::post()
        .uri("/detect")
        .insert_header(("Content-Type", "application/json"))
        .set_payload(
            r#"{"q":["This sentence is written in English with several common English words","Cette phrase est écrite en français avec plusieurs mots français courants"]}"#,
        )
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body: serde_json::Value = test::read_body_json(resp).await;
    assert_eq!(body.as_array().map(Vec::len), Some(2));
    assert_eq!(body[0]["language"], "en");
    assert_eq!(body[1]["language"], "fr");
    assert!(body[0]["confidence"].is_number());
    assert!(body[1]["confidence"].is_number());
}

/// A mock provider that returns text with invisible Unicode characters (zero-width space + soft hyphen).
struct PollutedProvider;

#[async_trait::async_trait]
impl TranslateProvider for PollutedProvider {
    async fn translate(&self, _system: &str, _user: &str) -> anyhow::Result<TranslationResult> {
        // Returns "Hello\u{200B}wo\u{00AD}rld" (zero-width space + soft hyphen)
        Ok(TranslationResult {
            text: "Hello\u{200B}wo\u{00AD}rld".to_string(),
            backend_timings: None,
        })
    }
    async fn ping(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Test that enable_cleanup_reporting=true includes cleanup stats in the response.
/// Also verifies that the cleanup itself ran correctly (removed 1 char, replaced 1).
#[actix_web::test]
async fn test_cleanup_reporting_enabled_shows_stats() {
    let args = Arc::new(test_args());
    let provider_manager = Arc::new(ProviderManager::from_provider(
        Arc::new(PollutedProvider),
        ProviderConfig::default(),
    ));

    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(args))
            .app_data(web::Data::new(provider_manager))
            .service(translate),
    )
    .await;

    let req = test::TestRequest::post()
        .uri("/translate")
        .insert_header(("Content-Type", "application/json"))
        .set_payload(
            r#"{"q":"Hello world","source":"en","target":"es","enable_cleanup_reporting":true}"#,
        )
        .to_request();
    let resp: serde_json::Value = test::call_and_read_body_json(&app, req).await;

    // Cleanup removes the zero-width space and replaces the soft hyphen
    // Original: "Hello" + ZWSP + "wo" + soft-hyphen + "rld" -> "Hellowo-rld"
    assert_eq!(resp["translatedText"], "Hellowo-rld");
    assert!(resp["reports"].is_object());
    assert_eq!(resp["reports"]["cleanup"]["removed"], 1);
    assert_eq!(resp["reports"]["cleanup"]["replaced"], 1);
}

/// Test that enable_cleanup_reporting=false or omitting it results in no reports key.
#[actix_web::test]
async fn test_cleanup_reporting_disabled_no_reports() {
    let args = Arc::new(test_args());
    let provider_manager = Arc::new(ProviderManager::from_provider(
        Arc::new(PollutedProvider),
        ProviderConfig::default(),
    ));

    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(args))
            .app_data(web::Data::new(provider_manager))
            .service(translate),
    )
    .await;

    // With explicit false
    let req_false = test::TestRequest::post()
        .uri("/translate")
        .insert_header(("Content-Type", "application/json"))
        .set_payload(
            r#"{"q":"Hello world","source":"en","target":"es","enable_cleanup_reporting":false}"#,
        )
        .to_request();
    let resp_false: serde_json::Value = test::call_and_read_body_json(&app, req_false).await;
    assert_eq!(resp_false["translatedText"], "Hellowo-rld"); // cleanup still runs, just no stats
    assert!(resp_false.get("reports").is_none());

    // Without the field at all
    let req_omitted = test::TestRequest::post()
        .uri("/translate")
        .insert_header(("Content-Type", "application/json"))
        .set_payload(r#"{"q":"Hello world","source":"en","target":"es"}"#)
        .to_request();
    let resp_omitted: serde_json::Value = test::call_and_read_body_json(&app, req_omitted).await;
    assert_eq!(resp_omitted["translatedText"], "Hellowo-rld");
    assert!(resp_omitted.get("reports").is_none());
}
