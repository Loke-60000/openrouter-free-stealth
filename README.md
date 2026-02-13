OpenAI-compatible proxy that splits OpenRouter models into **free** and **stealth** tiers.

## Why?

I use OpenWebUI to experiment with new models, but some have privacy restrictions or are "stealth/cloaked" (their real names are hidden and they're only available for testing). This proxy automatically fetches all free and stealth models from OpenRouter, health-checks them, and makes them available in my OpenWebUI instance so I never have to manually check which models work or are visible. It keeps my model list fresh and ready for testing, with no manual effort.

## Endpoints

| Endpoint                       | Description                 |
| ------------------------------ | --------------------------- |
| `/free/v1/models`              | List free models            |
| `/free/v1/chat/completions`    | Chat with free models       |
| `/stealth/v1/models`           | List stealth/cloaked models |
| `/stealth/v1/chat/completions` | Chat with stealth models    |
| `/health`                      | Health check                |
| `/status`                      | Cache stats                 |

Filter models: `/free/v1/models?supports=tools,vision`

## Run

```bash
# Docker
docker compose up -d

# Local
cargo run
```
