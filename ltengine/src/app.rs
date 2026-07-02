//! Server setup and startup logic.

use std::sync::Arc;

use actix_web::{App, HttpServer};

use crate::backend::{ProviderConfig, ProviderManager};
use crate::banner::print_banner;
use crate::cli::{Args, BenchmarkMode};

/// Create and initialize the provider manager from CLI args.
pub async fn create_provider_manager(args: &Args) -> Arc<ProviderManager> {
    let backend_api_key = if args.backend_api_key.is_empty() {
        None
    } else {
        Some(args.backend_api_key.as_str())
    };
    let provider_config = ProviderConfig {
        max_attempts: args.retry_attempts,
        base_delay_ms: args.retry_delay,
        recheck_interval_secs: args.model_recheck_interval,
    };
    let provider_manager = Arc::new(ProviderManager::new(
        &args.backend_host,
        &args.model,
        backend_api_key,
        provider_config,
    ));

    // Initialize the provider with retry (non-blocking if backend isn't ready yet)
    provider_manager.initialize().await;

    // Start periodic rechecker if enabled
    if args.model_recheck_interval > 0 {
        provider_manager.start_rechecker();
    }

    provider_manager
}

/// Run benchmark and return immediately (exits process).
#[allow(dead_code)]
#[cfg(feature = "benchmark")]
pub async fn run_benchmark(
    args: &Args,
    mode: &BenchmarkMode,
    provider_manager: &Arc<ProviderManager>,
) {
    crate::benchmark::run_benchmark(args, mode, provider_manager).await;
}

/// Start the HTTP server with all routes registered.
pub async fn run_server(
    args: Arc<Args>,
    provider_manager: Arc<ProviderManager>,
) -> std::io::Result<()> {
    let host = args.host.clone();
    let port = args.port;

    print_banner();

    let backend_host = args.backend_host.clone();
    let model = args.model.clone();
    let retry_attempts = args.retry_attempts;
    let retry_delay = args.retry_delay;

    let args_data = Arc::clone(&args);
    let provider_manager_for_shutdown = provider_manager.clone();

    let server = HttpServer::new(move || {
        App::new()
            .app_data(actix_web::web::Data::new(provider_manager.clone()))
            .app_data(actix_web::web::Data::new(args_data.clone()))
            .service(crate::endpoints::get_languages)
            .service(crate::endpoints::get_frontend_settings)
            .service(crate::endpoints::health)
            .service(crate::endpoints::backend_health)
            .service(crate::translate::translate)
            .service(crate::endpoints::translate_file)
            .service(crate::detect::detect)
            .service(crate::endpoints::suggest)
            // Catch-all: any unmatched path goes to the static handler
            .default_service(actix_web::web::to(crate::static_handler::serve_static))
    })
    .bind((host.clone(), port))?
    .run();

    println!("Running on: http://{}:{}", host, port);
    println!("Using backend: {} (model: {})", backend_host, model);
    println!("Health endpoints: GET /health (service), GET /health/backend (backend status)");
    println!(
        "Provider will retry {} times with base delay {}ms for transient errors",
        retry_attempts, retry_delay
    );

    // Run server with graceful shutdown on SIGINT/SIGTERM
    tokio::select! {
        _ = server => {},
        _ = tokio::signal::ctrl_c() => {
            eprintln!("Shutting down...");
            provider_manager_for_shutdown.shutdown();
        }
    }

    Ok(())
}
