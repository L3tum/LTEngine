//! Pluggable translation backends.
//!
//! Provides a `TranslateProvider` trait with an implementation that speaks the
//! OpenAI-compatible `/chat/completions` API (Ollama, vLLM, llama.cpp server, etc.).
//!
//! Supports configurable retry logic with exponential backoff and periodic
//! re-checking of model availability.

use anyhow::{Result, anyhow};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::sleep;

/// Result of checking model availability on the backend.
enum ModelAvailability {
    /// Backend is reachable and the requested model is available.
    Available,
    /// Backend is reachable but model enumeration failed (assume available).
    EnumerationFailed,
    /// Backend is reachable but model is not in the available list.
    ModelNotFound(Vec<String>),
    /// Backend is completely unreachable.
    Unreachable,
}

pub(crate) mod openai;
pub use openai::OpenAiProvider;

/// An error from the backend that may carry an HTTP status code and retryability hint.
#[derive(Debug)]
pub enum BackendError {
    /// Backend returned an HTTP error status (e.g., 401). Not retryable.
    Http { status_code: u16, detail: String },
    /// Model is not available on the backend (should return 404). Not retryable.
    ModelNotFound(String),
    /// Other error (parsing, empty response). Not retryable.
    Other(String),
    /// Network-level or timeout error that can be retried (connection refused, timeout, etc.).
    Retryable(String),
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendError::Http { status_code, .. } => {
                write!(f, "Backend HTTP error {}", status_code)
            }
            BackendError::ModelNotFound(msg) => write!(f, "{}", msg),
            BackendError::Other(msg) => write!(f, "{}", msg),
            BackendError::Retryable(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for BackendError {}

/// Configuration for provider creation and retry behavior.
#[derive(Clone)]
pub struct ProviderConfig {
    /// Maximum number of retry attempts for provider creation and translation requests.
    pub max_attempts: usize,
    /// Base delay in milliseconds for exponential backoff (e.g., 500ms, 1s, 2s).
    pub base_delay_ms: u64,
    /// Interval in seconds for periodic model availability rechecks.
    /// Set to 0 to disable periodic rechecking.
    pub recheck_interval_secs: u64,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        ProviderConfig {
            max_attempts: 5,
            base_delay_ms: 1000,
            recheck_interval_secs: 30,
        }
    }
}

/// A backend that can translate text via system + user prompts.
#[async_trait::async_trait]
pub trait TranslateProvider: Send + Sync {
    /// Translate the given system and user prompts, returning the model's response.
    async fn translate(&self, system: &str, user: &str, config: &ProviderConfig) -> Result<String>;

    /// Quick connectivity check without consuming a translation.
    /// Returns Ok(()) if the backend is reachable.
    async fn ping(&self) -> Result<()>;
}

/// A wrapper around a `TranslateProvider` that supports retry logic and periodic
/// model availability rechecking. If the provider becomes unreachable, subsequent
/// requests will attempt to recreate it (once the backend comes back online).
///
/// Uses a creation guard to prevent thundering herd problems when multiple requests
/// arrive simultaneously with no provider available.
#[derive(Clone)]
pub struct ProviderManager {
    provider: Arc<RwLock<Option<Arc<dyn TranslateProvider>>>>,
    creation_guard: Arc<tokio::sync::Mutex<()>>,
    host: String,
    model: String,
    api_key: Option<String>,
    config: ProviderConfig,
    shutdown_flag: Arc<AtomicBool>,
}

impl ProviderManager {
    /// Create a new `ProviderManager` that will manage the provider lifecycle.
    pub fn new(host: &str, model: &str, api_key: Option<&str>, config: ProviderConfig) -> Self {
        Self {
            provider: Arc::new(RwLock::new(None)),
            creation_guard: Arc::new(tokio::sync::Mutex::new(())),
            host: host.to_string(),
            model: model.to_string(),
            api_key: api_key.map(String::from),
            config,
            shutdown_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Create a manager around a supplied provider for tests.
    #[cfg(test)]
    pub fn from_provider(provider: Arc<dyn TranslateProvider>, config: ProviderConfig) -> Self {
        Self {
            provider: Arc::new(RwLock::new(Some(provider))),
            creation_guard: Arc::new(tokio::sync::Mutex::new(())),
            host: String::new(),
            model: String::new(),
            api_key: None,
            config,
            shutdown_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Check if the backend is reachable and the model is available.
    /// Returns a detailed status so the caller can decide what to do.
    async fn check_model_available(&self) -> ModelAvailability {
        let test_provider = OpenAiProvider::new(&self.host, &self.model, self.api_key.as_deref());
        let available_models = test_provider.check_model_availability().await;

        if let Ok(models) = available_models {
            if !models.is_empty() {
                let model_available = models
                    .iter()
                    .any(|m| m == &self.model || m.starts_with(&format!("{}:", self.model)));
                if model_available {
                    ModelAvailability::Available
                } else {
                    ModelAvailability::ModelNotFound(models)
                }
            } else {
                // Backend reachable but model enumeration failed - assume available
                ModelAvailability::EnumerationFailed
            }
        } else {
            ModelAvailability::Unreachable
        }
    }

    /// Start periodic rechecking of the model availability.
    /// This task runs in the background and will recreate the provider if the model
    /// is no longer available. It can be shut down by calling `shutdown`.
    pub fn start_rechecker(&self) {
        if self.config.recheck_interval_secs == 0 {
            return;
        }

        let shutdown_flag = self.shutdown_flag.clone();
        let manager = Arc::new(self.clone());

        tokio::spawn(async move {
            loop {
                sleep(Duration::from_secs(manager.config.recheck_interval_secs)).await;
                if shutdown_flag.load(Ordering::Relaxed) {
                    break;
                }

                if let Err(e) = manager.recheck_model_availability().await {
                    eprintln!("Periodic model recheck failed: {}", e);
                }
            }
        });
    }

    /// Stop the periodic rechecker.
    pub fn shutdown(&self) {
        self.shutdown_flag.store(true, Ordering::Relaxed);
    }

    /// Recheck model availability and recreate the provider if needed.
    /// If the backend is reachable (model available or enumeration failed), creates a new one with retry.
    pub async fn recheck_model_availability(&self) -> Result<()> {
        match self.check_model_available().await {
            ModelAvailability::Available | ModelAvailability::EnumerationFailed => {
                let new_provider = self.create_with_retry().await?;
                let mut provider_guard = self.provider.write().await;
                let old = provider_guard.replace(new_provider);
                if old.is_none() {
                    eprintln!(
                        "Provider recreated, model '{}' is now available",
                        self.model
                    );
                }
            }
            ModelAvailability::ModelNotFound(models) => {
                eprintln!(
                    "Warning: model '{}' not available, available models: {:?}",
                    self.model, models
                );
            }
            ModelAvailability::Unreachable => {
                // Backend not reachable — nothing to do
            }
        }
        Ok(())
    }

    /// Create the provider with retry and exponential backoff.
    /// Returns the provider once creation succeeds.
    async fn create_with_retry(&self) -> Result<Arc<dyn TranslateProvider>> {
        let mut attempt = 0;
        let max_attempts = self.config.max_attempts;

        loop {
            attempt += 1;

            // Create a provider (no model validation yet — we check after)
            let provider = OpenAiProvider::new(&self.host, &self.model, self.api_key.as_deref());

            match self.check_model_available().await {
                ModelAvailability::Available => {
                    return Ok(Arc::new(provider));
                }
                ModelAvailability::EnumerationFailed => {
                    eprintln!(
                        "Warning: backend is reachable but no models were enumerated, assuming model '{}' is available",
                        self.model
                    );
                    return Ok(Arc::new(provider));
                }
                ModelAvailability::ModelNotFound(models) => {
                    // Model not available — this is NOT retryable, return error
                    eprintln!(
                        "Model '{}' not available. Available models: {:?}",
                        self.model, models
                    );
                    return Err(anyhow!(BackendError::ModelNotFound(format!(
                        "Model '{}' not available. Available models: {:?}",
                        self.model, models
                    ))));
                }
                ModelAvailability::Unreachable => {
                    // Backend unreachable — retryable, fall through to delay loop
                }
            }

            if attempt < max_attempts {
                let delay =
                    Duration::from_millis(self.config.base_delay_ms * 2u64.pow(attempt as u32 - 1));
                eprintln!(
                    "Failed to create provider (attempt {}/{}), retrying in {:?}...",
                    attempt, max_attempts, delay
                );
                sleep(delay).await;
            } else {
                return Err(anyhow!(
                    "Failed to create provider after {} attempts. Backend may be unreachable or model '{}' not available.",
                    max_attempts,
                    self.model
                ));
            }
        }
    }

    /// Create the initial provider with retry. If the backend is unreachable, the
    /// provider will be recreated on the next request or via periodic rechecking.
    pub async fn initialize(&self) {
        // Try to create the provider, but don't fail immediately if the backend isn't ready
        if let Ok(provider) = self.create_with_retry().await {
            let mut provider_guard = self.provider.write().await;
            *provider_guard = Some(provider);
            eprintln!("Provider initialized successfully");
        } else {
            eprintln!(
                "Initial provider creation failed, will retry on first request or periodically"
            );
            // The provider will be None until it's successfully created
        }
    }

    /// Translate using the current provider. If the provider is None, attempts to
    /// create it once with retry. On failure, returns error.
    ///
    /// The provider's own `translate_with_config` handles retry with max attempts,
    /// so we don't add a wrapper retry loop. On transient (retryable) errors we
    /// drop the provider - the next request will trigger a new creation attempt.
    /// On permanent errors (auth failure, model not found) we also drop the provider
    /// so the rechecker can attempt recovery.
    pub async fn translate(&self, system: &str, user: &str) -> Result<String> {
        let provider = {
            let provider_guard = self.provider.read().await;
            provider_guard.clone()
        };

        if let Some(provider) = provider {
            match provider.translate(system, user, &self.config).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    // On any error, drop the provider so the next request will try to create
                    // a new one. For Retryable errors, the provider's own retry loop has
                    // already exhausted all retries, so the provider is dead and needs
                    // recreation. For permanent errors (401, 404), we also drop it.
                    eprintln!("Translation error, dropping provider: {}", e);
                    let mut provider_guard = self.provider.write().await;
                    *provider_guard = None;
                    Err(e)
                }
            }
        } else {
            // No provider, try to create one. Use creation_guard to prevent thundering herd.
            // Wait up to 2 seconds for another request to finish creating the provider.
            let guard_result =
                tokio::time::timeout(Duration::from_secs(2), self.creation_guard.lock()).await;

            let _guard = match guard_result {
                Ok(guard) => guard,
                Err(_) => {
                    // Timeout: another request is currently creating the provider.
                    // Wait a moment and check again.
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    let provider_guard = self.provider.read().await;
                    if let Some(provider) = provider_guard.clone() {
                        // Provider was created while we were waiting
                        return provider.translate(system, user, &self.config).await;
                    }
                    return Err(anyhow!(
                        "Could not create provider in time, backend may be unreachable"
                    ));
                }
            };

            // Double-check: another request might have created the provider while we waited
            {
                let provider_guard = self.provider.read().await;
                if let Some(provider) = provider_guard.clone() {
                    return provider.translate(system, user, &self.config).await;
                }
            }

            // Create the provider with retry
            eprintln!("No provider available, attempting to create...");
            let new_provider = self.create_with_retry().await?;
            {
                let mut provider_guard = self.provider.write().await;
                *provider_guard = Some(new_provider);
            }
            // Translate with the new provider (its own retry logic applies)
            {
                let provider_guard = self.provider.read().await;
                if let Some(provider) = provider_guard.clone() {
                    provider.translate(system, user, &self.config).await
                } else {
                    Err(anyhow!("Provider became unavailable after creation"))
                }
            }
        }
    }

    /// Ping the current provider. Returns Ok(()) if reachable, or error if not.
    /// On transient (retryable) errors, the provider stays alive - the next request
    /// might succeed. On permanent errors (401, 404), the provider is dropped.
    pub async fn ping(&self) -> Result<()> {
        let provider = {
            let provider_guard = self.provider.read().await;
            provider_guard.clone()
        };

        if let Some(provider) = provider {
            let result = provider.ping().await;
            // If ping fails with a non-retryable error (auth failure, etc.),
            // drop the provider so future requests trigger a recreation attempt.
            if let Err(e) = &result {
                let is_permanent = e
                    .downcast_ref::<BackendError>()
                    .map(|be| !matches!(be, BackendError::Retryable(_)))
                    .unwrap_or(false);

                if is_permanent {
                    eprintln!("Ping failed with permanent error, dropping provider");
                    let mut provider_guard = self.provider.write().await;
                    *provider_guard = None;
                }
            }
            result
        } else {
            Err(anyhow!("No provider available"))
        }
    }

    /// Get the current provider for health checks (cloned for use in async context).
    pub async fn get_provider(&self) -> Option<Arc<dyn TranslateProvider>> {
        let provider_guard = self.provider.read().await;
        provider_guard.clone()
    }
}
