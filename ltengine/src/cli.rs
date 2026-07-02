//! Command-line argument parsing.

use clap::{Parser, ValueEnum};

/// Output format for benchmark results.
#[derive(Parser, Debug, Clone, Default, ValueEnum)]
pub enum BenchmarkMode {
    /// Print benchmark results as plain text (default)
    #[default]
    Text,
    /// Generate a self-contained HTML report with charts and expandable details
    Html,
}

/// CLI arguments for LTEngine.
#[derive(Parser, Debug, Clone)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// Hostname to bind to (env var: LTE_HOST)
    #[arg(long, env = "LTE_HOST", default_value = "0.0.0.0")]
    pub host: String,

    /// Port to bind to (env var: LTE_PORT)
    #[arg(short, long, env = "LTE_PORT", default_value_t = 5050)]
    pub port: u16,

    /// Maximum total character length of all source text; rejects if exceeded (env var: LTE_CHAR_LIMIT)
    #[arg(long, env = "LTE_CHAR_LIMIT", default_value_t = 50000)]
    pub char_limit: usize,

    /// Maximum size of individual translation chunks (env var: LTE_CHUNK_SIZE)
    #[arg(long, env = "LTE_CHUNK_SIZE", default_value_t = 5000)]
    pub chunk_size: usize,

    /// Maximum number of concurrent translation requests per input text (env var: LTE_MAX_PARALLEL)
    #[arg(long, env = "LTE_MAX_PARALLEL", default_value_t = 2)]
    pub max_parallel: usize,

    /// Maximum number of strings accepted in JSON q arrays (env var: LTE_BATCH_LIMIT)
    #[arg(long, env = "LTE_BATCH_LIMIT", default_value_t = 50)]
    pub batch_limit: usize,

    /// Model to use (passed to the backend provider) (env var: LTE_MODEL)
    #[arg(short = 'm', long, env = "LTE_MODEL", default_value = "gemma3-4b")]
    pub model: String,

    /// Backend host (e.g., "localhost:11434" for Ollama, or full URL like "http://ollama:11434") (env var: LTE_BACKEND_HOST)
    #[arg(long, env = "LTE_BACKEND_HOST", default_value = "localhost:11434")]
    pub backend_host: String,

    /// Set an API key (optional, for authenticating translation requests to the API) (env var: LTE_API_KEY)
    #[arg(long, env = "LTE_API_KEY", default_value = "")]
    pub api_key: String,

    /// Set a backend API key (optional, for authenticating with the external backend) (env var: LTE_BACKEND_API_KEY)
    #[arg(
        long = "backend-api-key",
        env = "LTE_BACKEND_API_KEY",
        default_value = ""
    )]
    pub backend_api_key: String,

    /// Maximum number of retry attempts for provider creation and translation requests (env var: LTE_RETRY_ATTEMPTS)
    #[arg(long, env = "LTE_RETRY_ATTEMPTS", default_value_t = 5)]
    pub retry_attempts: usize,

    /// Base delay in milliseconds for exponential backoff (1000ms → 1s, 2s, 4s...) (env var: LTE_RETRY_DELAY)
    #[arg(long, env = "LTE_RETRY_DELAY", default_value_t = 1000)]
    pub retry_delay: u64,

    /// Interval in seconds for periodic model availability rechecks (0 to disable) (env var: LTE_MODEL_RECHECK_INTERVAL)
    #[arg(long, env = "LTE_MODEL_RECHECK_INTERVAL", default_value_t = 30)]
    pub model_recheck_interval: u64,

    /// Enable language detection via LLM (env var: LTE_LLM_DETECT)
    #[arg(long = "llm-detect", env = "LTE_LLM_DETECT", action)]
    pub llm_detect: bool,

    /// Run a benchmark against a dataset and exit (env var: LTE_BENCHMARK). Options: "text" or "html"
    #[arg(long, env = "LTE_BENCHMARK")]
    pub benchmark: Option<BenchmarkMode>,

    /// Path to the benchmark dataset JSON file (env var: LTE_DATASET)
    #[arg(long, env = "LTE_DATASET")]
    pub dataset: Option<String>,

    /// Disable Unicode cleanup of LLM output (cleanup is ON by default; use --disable-cleanup to turn it off) (env var: LTE_DISABLE_CLEANUP)
    #[arg(long, env = "LTE_DISABLE_CLEANUP", action)]
    pub disable_cleanup: bool,
}
