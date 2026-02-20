mod proxy;
pub mod responses;

use crate::model::Model;
use crate::state::SharedState;
use axum::{
    extract::{Path, Query, Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use proxy::Proxy;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy)]
pub enum Tier {
    Free,
    Stealth,
}

impl Tier {
    pub(crate) fn models(self, cache: &crate::state::ModelCache) -> std::sync::Arc<Vec<Model>> {
        match self {
            Self::Free => cache.free_models.clone(),
            Self::Stealth => cache.stealth_models.clone(),
        }
    }
}

#[derive(Deserialize, Default)]
pub(crate) struct ModelFilter {
    #[serde(default)]
    supports: Option<String>,
}

impl ModelFilter {
    pub(crate) fn matches(&self, model: &Model) -> bool {
        let Some(ref caps) = self.supports else {
            return true;
        };
        caps.split(',').all(|c| match c.trim() {
            "tools" => model.has_param("tools"),
            "tool_choice" => model.has_param("tool_choice"),
            "json_mode" => model.has_param("response_format"),
            "streaming" => model.has_param("stream"),
            "vision" => model.supports_vision(),
            _ => true,
        })
    }
}

macro_rules! tier_handlers {
    ($tier:expr, $list:ident, $get:ident, $fwd:ident, $resp:ident) => {
        async fn $list(
            State(s): State<SharedState>,
            Query(f): Query<ModelFilter>,
        ) -> impl IntoResponse {
            Proxy::list_models($tier, &s, &f).await
        }

        async fn $get(State(s): State<SharedState>, Path(id): Path<String>) -> Response {
            Proxy::get_model($tier, &s, &id).await
        }

        async fn $fwd(State(s): State<SharedState>, req: Request) -> Response {
            Proxy::forward($tier, &s, req).await
        }

        async fn $resp(State(s): State<SharedState>, req: Request) -> Response {
            Proxy::handle_responses($tier, &s, req).await
        }
    };
}

tier_handlers!(Tier::Free, list_free, get_free, fwd_free, resp_free);
tier_handlers!(Tier::Stealth, list_stealth, get_stealth, fwd_stealth, resp_stealth);

pub fn tier_router(tier: Tier) -> Router<SharedState> {
    match tier {
        Tier::Free => Router::new()
            .route("/models", get(list_free))
            .route("/models/*id", get(get_free))
            .route("/chat/completions", post(fwd_free))
            .route("/responses", post(resp_free)),
        Tier::Stealth => Router::new()
            .route("/models", get(list_stealth))
            .route("/models/*id", get(get_stealth))
            .route("/chat/completions", post(fwd_stealth))
            .route("/responses", post(resp_stealth)),
    }
}

pub async fn health() -> &'static str {
    "OK"
}

#[derive(Serialize)]
struct StatusBody {
    free_models: usize,
    stealth_models: usize,
    last_refreshed: String,
}

pub async fn status(State(s): State<SharedState>) -> impl IntoResponse {
    let c = s.cache.read().await;
    Json(StatusBody {
        free_models: c.free_models.len(),
        stealth_models: c.stealth_models.len(),
        last_refreshed: c.last_refreshed.to_rfc3339(),
    })
}

pub async fn not_found() -> Response {
    Proxy::error(
        StatusCode::NOT_FOUND,
        "Unknown API endpoint".into(),
        Some("unknown_url"),
    )
}
