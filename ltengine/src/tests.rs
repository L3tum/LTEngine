//! Integration tests with mock translation provider.

use crate::backend::{OpenAiProvider, ProviderConfig, ProviderManager, TranslateProvider};
use crate::prompt::PromptBuilder;
use crate::{Args, QueryText, TranslateRequest, check_params, detect, translate};
use actix_web::{App, http::StatusCode, test, web};
use std::sync::{Arc, Mutex};

/// A simple mock provider that always returns a fixed translation.
struct MockProvider;

#[async_trait::async_trait]
impl TranslateProvider for MockProvider {
    async fn translate(
        &self,
        _system: &str,
        _user: &str,
        _config: &ProviderConfig,
    ) -> anyhow::Result<String> {
        Ok("TranslatedText".to_string())
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

    fn call_count(&self) -> u32 {
        *self.call_count.lock().unwrap()
    }
}

#[async_trait::async_trait]
impl TranslateProvider for RetryMockProvider {
    async fn translate(
        &self,
        _system: &str,
        _user: &str,
        _config: &ProviderConfig,
    ) -> anyhow::Result<String> {
        let mut count = self.call_count.lock().unwrap();
        let current = *count;
        *count += 1;
        drop(count);

        if current < self.fail_count {
            // Simulate a transient error (e.g., timeout) — retryable
            Err(anyhow::anyhow!("Simulated timeout (call {})", current))
        } else {
            Ok("Finally worked".to_string())
        }
    }
    async fn ping(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn test_mock_provider_translate() {
    let provider = MockProvider;
    let config = ProviderConfig::default();
    let result = provider
        .translate("You are an expert linguist.", "Hello, world!", &config)
        .await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "TranslatedText");
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

/// Test the OpenAiProvider error handling (should fail gracefully without a real server).
#[tokio::test]
async fn test_openai_provider_connection_refused() {
    let provider = OpenAiProvider::new("localhost:59999", "test-model", None);
    let config = ProviderConfig::default();
    let result = provider.translate("system", "Hello", &config).await;
    assert!(result.is_err());
    // Just verify it fails — the exact error message depends on the network/OS
    let err_msg = result.unwrap_err().to_string();
    assert!(!err_msg.is_empty(), "Error should have a message");
}

/// Test that the RetryMockProvider correctly simulates transient failures:
/// it should fail the first N calls, then succeed on the (N+1)-th call.
#[tokio::test]
async fn test_retry_on_transient_failures() {
    // Mock provider fails first 2 times, then succeeds
    let provider = RetryMockProvider::new(2);
    let config = ProviderConfig::default();

    // First call: should fail
    let result = provider
        .translate("You are an expert linguist.", "Hello, world!", &config)
        .await;
    assert!(result.is_err());

    // Second call: should fail
    let result = provider
        .translate("You are an expert linguist.", "Hello, world!", &config)
        .await;
    assert!(result.is_err());

    // Third call: should succeed
    let result = provider
        .translate("You are an expert linguist.", "Hello, world!", &config)
        .await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "Finally worked");
    assert_eq!(
        provider.call_count(),
        3,
        "Should have 3 attempts (2 failures + 1 success)"
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
