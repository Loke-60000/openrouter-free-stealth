use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;
use tracing::{info, warn};

const API_BASE: &str = "https://openrouter.ai/api/v1";

const META_ROUTER_IDS: &[&str] = &[
    "openrouter/auto",
    "openrouter/free",
    "openrouter/bodybuilder",
    "switchpoint/router",
];

#[derive(Debug, Deserialize, Clone)]
struct ApiResponse {
    data: Vec<Model>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Model {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub created: i64,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub context_length: Option<u64>,
    #[serde(default)]
    pub pricing: Option<Pricing>,
    #[serde(default)]
    pub architecture: Option<Architecture>,
    #[serde(default)]
    pub top_provider: Option<TopProvider>,
    #[serde(default)]
    pub supported_parameters: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)]
pub struct Pricing {
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub completion: Option<String>,
    #[serde(default)]
    pub request: Option<String>,
    #[serde(default)]
    pub image: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)]
pub struct Architecture {
    #[serde(default)]
    pub modality: Option<String>,
    #[serde(default)]
    pub tokenizer: Option<String>,
    #[serde(default)]
    pub instruct_type: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)]
pub struct TopProvider {
    #[serde(default)]
    pub context_length: Option<u64>,
    #[serde(default)]
    pub max_completion_tokens: Option<u64>,
    #[serde(default)]
    pub is_moderated: Option<bool>,
}

impl Model {
    pub async fn fetch_all(client: &Client) -> anyhow::Result<Vec<Self>> {
        let resp = client
            .get(format!("{API_BASE}/models"))
            .timeout(Duration::from_secs(30))
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("OpenRouter returned {status}: {body}");
        }

        let data: ApiResponse = resp.json().await?;
        info!("Fetched {} models", data.data.len());
        Ok(data.data)
    }

    pub fn classify(all: &[Self]) -> (Vec<Self>, Vec<Self>) {
        let usable = |m: &&Self| !m.is_meta_router();
        let free: Vec<_> = all.iter().filter(|m| m.is_free()).filter(usable).cloned().collect();
        let stealth: Vec<_> = all.iter().filter(|m| m.is_stealth()).filter(usable).cloned().collect();
        info!("Classified {} free, {} stealth", free.len(), stealth.len());
        (free, stealth)
    }

    pub fn is_free(&self) -> bool {
        self.id.ends_with(":free")
            || self.pricing.as_ref().is_some_and(|p| {
                p.prompt.as_deref() == Some("0") && p.completion.as_deref() == Some("0")
            })
    }

    pub fn is_stealth(&self) -> bool {
        let has_keyword = |s: &str| {
            let l = s.to_lowercase();
            l.contains("cloaked") || l.contains("stealth")
        };
        self.description.as_deref().is_some_and(has_keyword)
            || has_keyword(&self.name)
            || self.id.starts_with("stealth/")
    }

    pub fn is_meta_router(&self) -> bool {
        META_ROUTER_IDS.contains(&self.id.as_str())
            || self.pricing.as_ref().is_some_and(|p| {
                p.prompt.as_deref() == Some("-1") || p.completion.as_deref() == Some("-1")
            })
    }

    pub fn has_param(&self, name: &str) -> bool {
        self.supported_parameters
            .as_ref()
            .is_some_and(|params| params.iter().any(|p| p == name))
    }

    pub fn supports_vision(&self) -> bool {
        self.architecture
            .as_ref()
            .and_then(|a| a.modality.as_deref())
            .is_some_and(|m| m.contains("image"))
    }

    pub fn capabilities(&self) -> Capabilities {
        Capabilities {
            tools: self.has_param("tools"),
            tool_choice: self.has_param("tool_choice"),
            parallel_tool_calls: self.has_param("parallel_tool_calls"),
            json_mode: self.has_param("response_format"),
            streaming: self.has_param("stream"),
            vision: self.supports_vision(),
        }
    }

    pub fn display_id(&self) -> String {
        let id = self.id.as_str();
        let id = id.strip_suffix(":free").unwrap_or(id);
        let id = id.split('/').last().unwrap_or(id);
        id.to_owned()
    }

    pub fn matches_display_id(&self, id: &str) -> bool {
        self.id == id || self.display_id() == id
    }

    pub fn to_openai(&self) -> OpenAIModel {
        OpenAIModel {
            id: self.display_id(),
            object: "model".into(),
            created: self.created,
            owned_by: self.provider().to_owned(),
            context_length: self.context_length,
            max_completion_tokens: self
                .top_provider
                .as_ref()
                .and_then(|t| t.max_completion_tokens),
            capabilities: self.capabilities(),
            pricing: self.pricing.as_ref().map(|p| OpenAIPricing {
                prompt: p.prompt.clone().unwrap_or_default(),
                completion: p.completion.clone().unwrap_or_default(),
            }),
        }
    }

    fn provider(&self) -> &str {
        self.id.split('/').next().unwrap_or("unknown")
    }

    pub async fn health_check_batch(
        client: &Client,
        api_key: &str,
        models: Vec<Self>,
        concurrency: usize,
    ) -> Vec<Self> {
        if models.is_empty() {
            return models;
        }
        info!("Health-checking {} models (concurrency={concurrency})", models.len());

        let sem = Arc::new(Semaphore::new(concurrency));
        let mut handles = Vec::with_capacity(models.len());

        for model in models {
            let permit = sem.clone().acquire_owned().await.unwrap();
            let client = client.clone();
            let key = api_key.to_owned();
            handles.push(tokio::spawn(async move {
                let ok = model.ping(&client, &key).await;
                drop(permit);
                ok.then_some(model)
            }));
        }

        let mut healthy = Vec::new();
        for h in handles {
            if let Ok(Some(m)) = h.await {
                healthy.push(m);
            }
        }
        info!("{} models passed health check", healthy.len());
        healthy
    }

    async fn ping(&self, client: &Client, api_key: &str) -> bool {
        let payload = serde_json::json!({
            "model": self.id,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1
        });

        match client
            .post(format!("{API_BASE}/chat/completions"))
            .bearer_auth(api_key)
            .json(&payload)
            .timeout(Duration::from_secs(30))
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => {
                info!("  + {}", self.id);
                true
            }
            Ok(r) if r.status() == reqwest::StatusCode::TOO_MANY_REQUESTS => {
                // 429 means the model exists but is rate-limited; treat as alive
                info!("  ~ {} (rate-limited, assumed alive)", self.id);
                true
            }
            Ok(r) => {
                let st = r.status();
                let body = r.text().await.unwrap_or_default();
                warn!("  - {} -> {st} {}", self.id, &body[..body.len().min(120)]);
                false
            }
            Err(e) => {
                warn!("  - {} -> {e}", self.id);
                false
            }
        }
    }
}

#[derive(Debug, Serialize, Clone)]
pub struct Capabilities {
    pub tools: bool,
    pub tool_choice: bool,
    pub parallel_tool_calls: bool,
    pub json_mode: bool,
    pub streaming: bool,
    pub vision: bool,
}

#[derive(Debug, Serialize, Clone)]
pub struct OpenAIModel {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u64>,
    pub capabilities: Capabilities,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pricing: Option<OpenAIPricing>,
}

#[derive(Debug, Serialize, Clone)]
pub struct OpenAIPricing {
    pub prompt: String,
    pub completion: String,
}

#[derive(Debug, Serialize)]
pub struct OpenAIModelList {
    pub object: String,
    pub data: Vec<OpenAIModel>,
}

impl OpenAIModelList {
    pub fn from_models(models: &[Model]) -> Self {
        Self {
            object: "list".into(),
            data: models.iter().map(Model::to_openai).collect(),
        }
    }
}
