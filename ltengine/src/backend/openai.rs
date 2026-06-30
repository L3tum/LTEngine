//! OpenAI-compatible chat completions provider.

use anyhow::{Context, Result};
use reqwest::Client;
use serde::Deserialize;
use std::time::Duration;

use super::{BackendError, TranslateProvider};

/// Response structure for OpenAI-compatible chat completions API.
#[derive(Deserialize, Debug)]
struct CompletionResponse {
    choices: Vec<Choice>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    created: Option<u64>,
    #[serde(default)]
    model: Option<String>,
}

#[derive(Deserialize, Debug)]
struct Choice {
    message: ChatMessage,
    #[serde(default)]
    index: Option<u32>,
}

#[derive(Deserialize, Debug)]
struct ChatMessage {
    content: Option<String>,
    #[serde(default)]
    role: Option<String>,
}

/// A provider that sends requests to an OpenAI-compatible chat completions API.
pub struct OpenAiProvider {
    client: Client,
    base_url: String,
    model: String,
    api_key: Option<String>,
}

impl OpenAiProvider {
    /// Create a new provider that talks to `host` on port `port`, using `model`.
    ///
    /// The host should be e.g. `"localhost:11434"` or `"http://ollama:11434"` (HTTPS also supported).
    /// If the host doesn't include a scheme, `http://` is prepended.
    pub fn new(host: &str, model: &str, api_key: Option<&str>) -> Self {
        let base_url = if host.starts_with("http://") || host.starts_with("https://") {
            host.to_string()
        } else {
            format!("http://{}", host)
        };

        OpenAiProvider {
            client: Client::builder()
                .connect_timeout(Duration::from_secs(5))
                .timeout(Duration::from_secs(120))
                .build()
                .expect("Failed to build HTTP client"),
            base_url,
            model: model.to_string(),
            api_key: api_key.map(String::from),
        }
    }

    /// Send a single translation request to the backend. No retry — retry is handled by ProviderManager.
    /// Returns the translated text or a `BackendError` (retryable or not).
    pub async fn translate_with_config(&self, system: &str, user: &str) -> Result<String> {
        let url = format!("{}/v1/chat/completions", self.base_url);

        let mut request = self.client.post(&url).json(&serde_json::json!({
            "model": &self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        }));

        if let Some(key) = &self.api_key {
            request = request.header("Authorization", format!("Bearer {}", key));
        }

        match request.send().await {
            Ok(response) => match response.error_for_status() {
                Ok(response) => {
                    let body: CompletionResponse = response
                        .json()
                        .await
                        .with_context(|| "Failed to parse backend response as JSON")?;

                    body.choices
                        .first()
                        .and_then(|c| c.message.content.clone())
                        .ok_or_else(|| anyhow::anyhow!("Empty response from backend"))
                }
                Err(e) => {
                    if let Some(status) = e.status() {
                        let detail = e.to_string();
                        Err(BackendError::Http {
                            status_code: status.as_u16(),
                            detail,
                        }.into())
                    } else {
                        Err(e.into())
                    }
                }
            },
            Err(e) => {
                // Connection or timeout errors are retryable
                if e.is_timeout() || e.is_connect() {
                    Err(BackendError::Retryable(e.to_string()).into())
                } else {
                    Err(e.into())
                }
            }
        }
    }

    /// Quick connectivity check: hit the /health endpoint.
    /// All supported inference engines (Ollama, vLLM, llama.cpp server) expose this.
    pub async fn ping_backend(&self) -> Result<()> {
        let health_url = format!("{}/health", self.base_url);
        match self.client.get(&health_url).send().await {
            Ok(resp) => {
                if resp.status().is_success() {
                    return Ok(());
                }
            }
            Err(_) => {}
        }
        Err(anyhow::anyhow!(
            "Backend at {} is not reachable",
            self.base_url
        ))
    }

    /// Check if the backend is reachable and the model is available.
    ///
    /// Tries the standard OpenAI `/v1/models` first, then falls back to Ollama's `/api/tags`,
    /// and finally a generic `/health` check if model enumeration fails.
    /// Returns the available model names (if any).
    pub async fn check_model_availability(&self) -> Result<Vec<String>> {
        let mut available_models = Vec::new();

        // Try standard OpenAI /v1/models endpoint (vLLM, llama.cpp server, and any OpenAI-compatible backend)
        let models_url = format!("{}/v1/models", self.base_url);
        if let Ok(resp) = self.client.get(&models_url).send().await {
            if resp.status().is_success() {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    if let Some(data) = json["data"].as_array() {
                        for m in data {
                            if let Some(id) = m.get("id").and_then(|n| n.as_str()) {
                                available_models.push(id.to_string());
                            }
                        }
                    }
                    if !available_models.is_empty() {
                        return Ok(available_models);
                    }
                }
            }
        }

        // Fallback: try Ollama's /api/tags (older versions, or Ollama-specific)
        let ollama_url = format!("{}/api/tags", self.base_url);
        if let Ok(resp) = self.client.get(&ollama_url).send().await {
            if resp.status().is_success() {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    if let Some(models) = json["models"].as_array() {
                        for m in models {
                            if let Some(name) = m.get("name").and_then(|n| n.as_str()) {
                                available_models.push(name.to_string());
                            }
                        }
                    }
                    if !available_models.is_empty() {
                        return Ok(available_models);
                    }
                }
            }
        }

        // If model enumeration failed, fall back to /health to at least check reachability
        let health_url = format!("{}/health", self.base_url);
        if let Ok(resp) = self.client.get(&health_url).send().await {
            if resp.status().is_success() {
                // Rate-limit this warning: only log once per 10 seconds
                static_last_model_enum_warning_check(&format!("{}:{}", self.base_url, self.model));
                eprintln!(
                    "Warning: backend at {} is reachable but model enumeration failed, assuming model '{}' is available",
                    self.base_url, self.model
                );
                return Ok(Vec::new()); // backend reachable but we couldn't enumerate models
            }
        }

        Err(anyhow::anyhow!(
            "Backend at {} is not reachable",
            self.base_url
        ))
    }
}

/// Helper to rate-limit the model enumeration warning.
/// Logs at most once every 10 seconds per backend URL.
static LAST_MODEL_ENUM_WARN: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<String, std::time::Instant>>,
> = std::sync::OnceLock::new();

fn static_last_model_enum_warning_check(key: &str) {
    let map = LAST_MODEL_ENUM_WARN
        .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let now = std::time::Instant::now();
    let mut map = map.lock().unwrap();
    if let Some(&last) = map.get(key) {
        if now.duration_since(last) < std::time::Duration::from_secs(10) {
            // Too soon, skip
            return;
        }
    }
    map.insert(key.to_string(), now);
}

#[async_trait::async_trait]
impl TranslateProvider for OpenAiProvider {
    async fn ping(&self) -> Result<()> {
        self.ping_backend().await
    }

    async fn translate(&self, system: &str, user: &str) -> Result<String> {
        self.translate_with_config(system, user).await
    }
}
