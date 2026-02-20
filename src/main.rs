mod api;
mod config;
mod model;
mod state;

use api::{health, not_found, status, tier_router, Tier};
use axum::{extract::DefaultBodyLimit, routing::get, Router};
use state::AppState;
use tower_http::cors::CorsLayer;
use tracing::info;

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "openrouter_api=info,tower_http=info".into()),
        )
        .init();

    let config = config::Config::from_env();
    let addr = format!("{}:{}", config.host, config.port);
    let state = AppState::new(config);

    state.full_refresh().await;
    state.spawn_scheduler();

    let app = Router::new()
        .nest("/free/v1", tier_router(Tier::Free))
        .nest("/stealth/v1", tier_router(Tier::Stealth))
        .route("/health", get(health))
        .route("/status", get(status))
        .fallback(not_found)
        .layer(CorsLayer::permissive())
        .layer(DefaultBodyLimit::max(10 * 1024 * 1024))
        .with_state(state);

    info!("Listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await.expect("failed to bind");
    axum::serve(listener, app).await.expect("server crashed");
}
