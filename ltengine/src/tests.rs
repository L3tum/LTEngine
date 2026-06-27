//! Integration tests with mock translation provider.

use crate::backend::{TranslateProvider, OpenAiProvider, ProviderConfig};
use crate::prompt::PromptBuilder;
use std::sync::{Arc, Mutex};

/// A simple mock provider that always returns a fixed translation.
struct MockProvider;

#[async_trait::async_trait]
impl TranslateProvider for MockProvider {
    async fn translate(&self, _system: &str, _user: &str, _config: &ProviderConfig) -> anyhow::Result<String> {
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
    async fn translate(&self, _system: &str, _user: &str, _config: &ProviderConfig) -> anyhow::Result<String> {
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
    let result = provider
        .translate("system", "Hello", &config)
        .await;
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
    assert_eq!(provider.call_count(), 3, "Should have 3 attempts (2 failures + 1 success)");
}
