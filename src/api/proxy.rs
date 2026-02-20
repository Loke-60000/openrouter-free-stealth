use super::responses;
use super::{ModelFilter, Tier};
use crate::model::OpenAIModelList;
use crate::state::SharedState;
use axum::{
    body::Body,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use http_body_util::BodyExt;

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

pub struct Proxy;

impl Proxy {
    pub async fn list_models(
        tier: Tier,
        state: &SharedState,
        filter: &ModelFilter,
    ) -> Json<OpenAIModelList> {
        let all = tier.models(&*state.cache.read().await);
        let filtered: Vec<_> = all.iter().filter(|m| filter.matches(m)).cloned().collect();
        Json(OpenAIModelList::from_models(&filtered))
    }

    pub async fn get_model(tier: Tier, state: &SharedState, raw_id: &str) -> Response {
        let models = tier.models(&*state.cache.read().await);
        let id = raw_id.trim_start_matches('/');
        match models.iter().find(|m| m.matches_display_id(id)) {
            Some(m) => Json(m.to_openai()).into_response(),
            None => Self::error(
                StatusCode::NOT_FOUND,
                format!("The model '{id}' does not exist"),
                Some("model_not_found"),
            ),
        }
    }

    pub async fn forward(tier: Tier, state: &SharedState, req: axum::extract::Request) -> Response {
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
                    None,
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
                        format!("The model '{mid}' does not exist"),
                        Some("model_not_found"),
                    );
                }
            }
        }

        let mut upstream = state.client.request(parts.method, &url);

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
            Err(e) => Self::error(StatusCode::BAD_GATEWAY, format!("upstream error: {e}"), None),
        }
    }

    pub async fn handle_responses(
        tier: Tier,
        state: &SharedState,
        req: axum::extract::Request,
    ) -> Response {
        let models = tier.models(&*state.cache.read().await);
        let (parts, body) = req.into_parts();

        let body_bytes = match body.collect().await {
            Ok(c) => c.to_bytes(),
            Err(e) => {
                return Self::error(
                    StatusCode::BAD_REQUEST,
                    format!("failed to read body: {e}"),
                    None,
                )
            }
        };

        let json_body: serde_json::Value = match serde_json::from_slice(&body_bytes) {
            Ok(v) => v,
            Err(e) => {
                return Self::error(
                    StatusCode::BAD_REQUEST,
                    format!("invalid JSON: {e}"),
                    None,
                )
            }
        };

        let model_str = json_body
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        if model_str.is_empty() {
            return Self::error(
                StatusCode::BAD_REQUEST,
                "missing required parameter: model".into(),
                Some("missing_parameter"),
            );
        }

        let resolved_model = match models.iter().find(|m| m.matches_display_id(&model_str)) {
            Some(m) => m,
            None => {
                return Self::error(
                    StatusCode::NOT_FOUND,
                    format!("The model '{}' does not exist", model_str),
                    Some("model_not_found"),
                );
            }
        };

        let api_key = parts
            .headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .unwrap_or("")
            .to_string();

        if api_key.is_empty() {
            return Self::error(
                StatusCode::UNAUTHORIZED,
                "Missing API key in Authorization header".into(),
                Some("missing_api_key"),
            );
        }

        responses::handle_responses(&state.client, &api_key, &resolved_model.id, json_body).await
    }

    fn extract_model(body: &[u8]) -> Option<String> {
        if body.is_empty() {
            return None;
        }
        let json: serde_json::Value = serde_json::from_slice(body).ok()?;
        json.get("model")?.as_str().map(String::from)
    }

    pub fn stream(resp: reqwest::Response) -> Response {
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
                None,
            )
        })
    }

    pub fn error(status: StatusCode, message: String, code: Option<&str>) -> Response {
        let error_type = match status.as_u16() {
            401 => "authentication_error",
            403 => "permission_error",
            429 => "rate_limit_error",
            400..=499 => "invalid_request_error",
            _ => "server_error",
        };
        let body = serde_json::json!({
            "error": {
                "message": message,
                "type": error_type,
                "param": null,
                "code": code,
            }
        });
        Response::builder()
            .status(status)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }
}
