//! LTEngine — local, self-hosted AI translation API.
//!
//! This crate provides a LibreTranslate-compatible HTTP API powered by an
//! external inference backend via the OpenAI `/chat/completions` API.

mod app;
mod backend;
mod banner;
#[cfg(feature = "benchmark")]
mod benchmark;
mod chunking;
mod cleanup;
mod cli;
mod detect;
mod detection;
mod endpoints;
mod error_response;
mod languages;
mod postprocess;
mod prompt;
mod request;
mod static_handler;
mod translate;

use std::sync::Arc;

use clap::Parser;

use crate::app::{create_provider_manager, run_server};
use crate::cli::Args;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let args = Arc::new(Args::parse());

    let provider_manager = create_provider_manager(&args).await;

    // If benchmark mode, run and exit
    if let Some(ref mode) = args.benchmark {
        #[cfg(feature = "benchmark")]
        {
            crate::benchmark::run_benchmark(&args, mode, &provider_manager).await;
        }
        #[cfg(not(feature = "benchmark"))]
        {
            eprintln!("Benchmark feature not enabled. Rebuild with `--features benchmark`.");
        }
        return Ok(());
    }

    run_server(args, provider_manager).await
}

#[cfg(test)]
mod tests;
