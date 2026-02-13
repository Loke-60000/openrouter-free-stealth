use crate::model::{Model, OpenAIModelList};
use crate::state::SharedState;
use axum::{
    body::Body,
    extract::{Path, Query, Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use http_body_util::BodyExt;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

const UPSTREAM: &str = "https://openrouter.ai/api/v1";

const FORWARDED_HEADERS: &[&str] = &[
    "content-type",
    "accept",
    "accept-encoding",
    "authorization",
    "user-agent",
    "http-referer",
    "x-title",
];

#[derive(Clone, Copy)]
pub enum Tier {
    Free,
    Stealth,
}

impl Tier {
    fn models(self, cache: &crate::state::ModelCache) -> Arc<Vec<Model>> {
        match self {
            Self::Free => cache.free_models.clone(),
            Self::Stealth => cache.stealth_models.clone(),
        }
    }
}

#[derive(Deserialize, Default)]
pub struct ModelFilter {
    #[serde(default)]
    supports: Option<String>,
}

impl ModelFilter {
    fn matches(&self, model: &Model) -> bool {
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
    ($tier:expr, $list:ident, $get:ident, $fwd:ident) => {
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
    };
}

tier_handlers!(Tier::Free, list_free, get_free, fwd_free);
tier_handlers!(Tier::Stealth, list_stealth, get_stealth, fwd_stealth);

pub fn tier_router(tier: Tier) -> Router<SharedState> {
    match tier {
        Tier::Free => Router::new()
            .route("/models", get(list_free))
            .route("/models/*id", get(get_free))
            .fallback(fwd_free),
        Tier::Stealth => Router::new()
            .route("/models", get(list_stealth))
            .route("/models/*id", get(get_stealth))
            .fallback(fwd_stealth),
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

struct Proxy;

impl Proxy {
    async fn list_models(
        tier: Tier,
        state: &SharedState,
        filter: &ModelFilter,
    ) -> Json<OpenAIModelList> {
        let all = tier.models(&*state.cache.read().await);
        let filtered: Vec<_> = all.iter().filter(|m| filter.matches(m)).cloned().collect();
        Json(OpenAIModelList::from_models(&filtered))
    }

    async fn get_model(tier: Tier, state: &SharedState, raw_id: &str) -> Response {
        let models = tier.models(&*state.cache.read().await);
        let id = raw_id.trim_start_matches('/');
        match models.iter().find(|m| m.matches_display_id(id)) {
            Some(m) => Json(m.to_openai()).into_response(),
            None => Self::error(
                StatusCode::NOT_FOUND,
                format!("model \"{id}\" not found in this tier"),
            ),
        }
    }

    async fn forward(tier: Tier, state: &SharedState, req: Request) -> Response {
        let models = tier.models(&*state.cache.read().await);
        let (parts, body) = req.into_parts();

        let path = parts
            .uri
            .path_and_query()
            .map(|pq| pq.as_str())
            .unwrap_or(parts.uri.path());

        let url = format!("{UPSTREAM}{path}");

        let mut body_bytes = match body.collect().await {
            Ok(c) => c.to_bytes(),
            Err(e) => {
                return Self::error(
                    StatusCode::BAD_REQUEST,
                    format!("failed to read body: {e}"),
                )
            }
        };

        if let Some(mid) = Self::extract_model(&body_bytes) {
            match models.iter().find(|m| m.matches_display_id(&mid)) {
                Some(m) if m.id != mid => {
                    let mut json: serde_json::Value =
                        serde_json::from_slice(&body_bytes).unwrap();
                    json["model"] = serde_json::Value::String(m.id.clone());
                    body_bytes = axum::body::Bytes::from(json.to_string());
                }
                Some(_) => {}
                None => {
                    return Self::error(
                        StatusCode::NOT_FOUND,
                        format!("model \"{mid}\" not available in this tier"),
                    );
                }
            }
        }

        let mut upstream = state
            .client
            .request(parts.method, &url);

        for (name, value) in &parts.headers {
            if FORWARDED_HEADERS.contains(&name.as_str()) || name.as_str().starts_with("x-") {
                upstream = upstream.header(name, value);
            }
        }

        if !body_bytes.is_empty() {
            upstream = upstream.body(body_bytes);
        }

        match upstream.send().await {
            Ok(resp) => Self::stream(resp),
            Err(e) => Self::error(StatusCode::BAD_GATEWAY, format!("upstream error: {e}")),
        }
    }

    fn extract_model(body: &[u8]) -> Option<String> {
        if body.is_empty() {
            return None;
        }
        let json: serde_json::Value = serde_json::from_slice(body).ok()?;
        json.get("model")?.as_str().map(String::from)
    }

    fn stream(resp: reqwest::Response) -> Response {
        let status = resp.status();
        let headers = resp.headers().clone();
        let body = Body::from_stream(resp.bytes_stream());

        let mut builder = Response::builder().status(status.as_u16());
        for (name, value) in &headers {
            if !matches!(name.as_str(), "transfer-encoding" | "connection") {
                builder = builder.header(name, value);
            }
        }

        builder.body(body).unwrap_or_else(|_| {
            Self::error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to build response".into(),
            )
        })
    }

    fn error(status: StatusCode, message: String) -> Response {
        let body = serde_json::json!({
            "error": { "message": message, "type": "error", "code": status.as_u16() }
        });
        Response::builder()
            .status(status)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }
}
