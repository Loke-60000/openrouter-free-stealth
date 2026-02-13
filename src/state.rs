use crate::config::Config;
use crate::model::Model;
use chrono::{DateTime, Utc};
use reqwest::Client;
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

pub struct ModelCache {
    pub free_models: Arc<Vec<Model>>,
    pub stealth_models: Arc<Vec<Model>>,
    pub last_refreshed: DateTime<Utc>,
}

pub struct AppState {
    pub cache: RwLock<ModelCache>,
    pub client: Client,
    pub config: Config,
}

pub type SharedState = Arc<AppState>;

impl AppState {
    pub fn new(config: Config) -> SharedState {
        Arc::new(Self {
            cache: RwLock::new(ModelCache {
                free_models: Arc::new(Vec::new()),
                stealth_models: Arc::new(Vec::new()),
                last_refreshed: Utc::now(),
            }),
            client: Client::new(),
            config,
        })
    }

    /// Full refresh: classify all models and health-check everything.
    /// Used on first startup when the cache is empty.
    pub async fn full_refresh(self: &Arc<Self>) {
        info!("Full model refresh (startup)");

        let all = match Model::fetch_all(&self.client).await {
            Ok(m) => m,
            Err(e) => {
                error!("Failed to fetch models: {e}");
                return;
            }
        };

        let (mut free, mut stealth) = Model::classify(&all);

        if let Some(ref key) = self.config.health_check_key {
            let c = self.config.health_check_concurrency;
            free = Model::health_check_batch(&self.client, key, free, c).await;
            stealth = Model::health_check_batch(&self.client, key, stealth, c).await;
        } else {
            info!("No OPENROUTER_API_KEY set, skipping health checks");
        }

        let mut cache = self.cache.write().await;
        cache.free_models = Arc::new(free);
        cache.stealth_models = Arc::new(stealth);
        cache.last_refreshed = Utc::now();
        info!("Model cache updated");
    }

    /// Diff refresh: compare fresh model list against current cache.
    /// - Removed models are dropped.
    /// - Existing models are kept as-is (no re-ping).
    /// - New models are health-checked before being added.
    pub async fn diff_refresh(self: &Arc<Self>) {
        info!("Diff model refresh");

        let all = match Model::fetch_all(&self.client).await {
            Ok(m) => m,
            Err(e) => {
                error!("Failed to fetch models: {e}");
                return;
            }
        };

        let (fresh_free, fresh_stealth) = Model::classify(&all);

        let cache = self.cache.read().await;
        let old_free = cache.free_models.clone();
        let old_stealth = cache.stealth_models.clone();
        drop(cache);

        let new_free = self.diff_tier("free", &old_free, fresh_free).await;
        let new_stealth = self.diff_tier("stealth", &old_stealth, fresh_stealth).await;

        let mut cache = self.cache.write().await;
        cache.free_models = Arc::new(new_free);
        cache.stealth_models = Arc::new(new_stealth);
        cache.last_refreshed = Utc::now();
        info!("Model cache updated");
    }

    async fn diff_tier(
        self: &Arc<Self>,
        tier_name: &str,
        old: &[Model],
        fresh: Vec<Model>,
    ) -> Vec<Model> {
        let old_ids: HashSet<&str> = old.iter().map(|m| m.id.as_str()).collect();
        let fresh_ids: HashSet<&str> = fresh.iter().map(|m| m.id.as_str()).collect();

        // Models that disappeared upstream
        let removed: Vec<_> = old_ids.difference(&fresh_ids).collect();
        for id in &removed {
            warn!("[{tier_name}] Removed: {id}");
        }

        // Models that are new upstream
        let added_ids: HashSet<&&str> = fresh_ids.difference(&old_ids).collect();
        let added: Vec<Model> = fresh
            .iter()
            .filter(|m| added_ids.contains(&m.id.as_str()))
            .cloned()
            .collect();

        if !added.is_empty() {
            info!("[{tier_name}] {} new model(s) to check", added.len());
        }

        // Keep existing models that are still present upstream
        let mut result: Vec<Model> = old
            .iter()
            .filter(|m| fresh_ids.contains(m.id.as_str()))
            .cloned()
            .collect();

        // Health-check only new models
        if !added.is_empty() {
            if let Some(ref key) = self.config.health_check_key {
                let healthy =
                    Model::health_check_batch(&self.client, key, added, self.config.health_check_concurrency).await;
                result.extend(healthy);
            } else {
                result.extend(added);
            }
        }

        if !removed.is_empty() || !added_ids.is_empty() {
            info!(
                "[{tier_name}] {} kept, {} removed, {} added -> {} total",
                result.len() - added_ids.len().min(result.len()),
                removed.len(),
                added_ids.len(),
                result.len()
            );
        } else {
            info!("[{tier_name}] No changes ({} models)", result.len());
        }

        result
    }

    pub fn spawn_scheduler(self: &Arc<Self>) {
        let state = self.clone();
        let interval = self.config.refresh_interval_secs;
        tokio::spawn(async move {
            loop {
                info!(
                    "Next refresh in {}h {}m",
                    interval / 3600,
                    (interval % 3600) / 60
                );
                tokio::time::sleep(std::time::Duration::from_secs(interval)).await;

                state.diff_refresh().await;
            }
        });
    }
}
