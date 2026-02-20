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

        let (added_count, removed_count, total) = {
            let fresh_ids: HashSet<&str> = fresh.iter().map(|m| m.id.as_str()).collect();

            for id in old_ids.difference(&fresh_ids) {
                warn!("[{tier_name}] Removed upstream: {id}");
            }

            let added = fresh_ids.difference(&old_ids).count();
            if added > 0 {
                info!("[{tier_name}] {added} new model(s) from upstream");
            }

            let removed = old_ids.difference(&fresh_ids).count();
            (added, removed, fresh.len())
        };

        let result = if let Some(ref key) = self.config.health_check_key {
            info!("[{tier_name}] Health-checking {total} models");
            Model::health_check_batch(
                &self.client,
                key,
                fresh,
                self.config.health_check_concurrency,
            )
            .await
        } else {
            fresh
        };

        info!(
            "[{tier_name}] {}/{total} passed ({added_count} new, {removed_count} dropped upstream)",
            result.len()
        );

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
