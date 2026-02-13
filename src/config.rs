use std::env;

#[derive(Clone)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub health_check_key: Option<String>,
    pub health_check_concurrency: usize,
    pub refresh_interval_secs: u64,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            host: env::var("HOST").unwrap_or_else(|_| "0.0.0.0".into()),
            port: env::var("PORT")
                .unwrap_or_else(|_| "3000".into())
                .parse()
                .expect("PORT must be a valid number"),
            health_check_key: env::var("OPENROUTER_API_KEY").ok().filter(|k| !k.is_empty()),
            health_check_concurrency: env::var("HEALTH_CHECK_CONCURRENCY")
                .unwrap_or_else(|_| "5".into())
                .parse()
                .unwrap_or(5),
            refresh_interval_secs: env::var("REFRESH_INTERVAL_SECS")
                .unwrap_or_else(|_| "3600".into())
                .parse()
                .unwrap_or(3600),
        }
    }
}
