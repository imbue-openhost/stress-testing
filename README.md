# stress-testing

Load testing tools for OpenHost services.

`load_test.py` simulates multiple concurrent users typing code and requesting
completions. It ramps concurrency up in steps and reports latency percentiles
and throughput at each level, then rates each level as `YES` / `MARGINAL` / `NO`
against configurable p90 thresholds.

Four backends are supported:

| Backend     | Endpoint path           | Auth header                       | `--model` |
| ----------- | ----------------------- | --------------------------------- | --------- |
| `llama`     | `/infill`               | `Authorization: Bearer <token>`   | optional  |
| `ollama`    | `/api/generate`         | `Authorization: Bearer <token>`   | required  |
| `anthropic` | `/v1/messages`          | `x-api-key: <token>`              | required  |
| `openai`    | `/v1/chat/completions`  | `Authorization: Bearer <token>`   | required  |

## Install

Requires Python >= 3.10. The only runtime dependency is `aiohttp`.

```bash
uv venv
uv pip install -r pyproject.toml      # or: uv pip install -e .
```

This installs a `load-test` console script (equivalent to `python load_test.py`).

## Usage

### llama.cpp (`/infill`)

```bash
load-test \
  --endpoint https://code-completion.example.com \
  --token "$LLAMA_TOKEN" \
  --max-users 20 --requests-per-user 10
```

`--model` is optional for the llama backend.

### Ollama (`/api/generate`)

```bash
load-test \
  --backend ollama \
  --endpoint http://localhost:11434 \
  --model qwen2.5-coder:1.5b
```

Prompts are wrapped with `<|fim_prefix|> … <|fim_suffix|> … <|fim_middle|>`
sentinels for fill-in-the-middle completion.

### Anthropic (`/v1/messages`)

```bash
load-test \
  --backend anthropic \
  --endpoint https://api.anthropic.com \
  --token "$ANTHROPIC_API_KEY" \
  --model claude-3-5-sonnet-20241022
```

The token is sent as `x-api-key` and the `anthropic-version: 2023-06-01` header
is added automatically.

### OpenAI (`/v1/chat/completions`)

```bash
load-test \
  --backend openai \
  --endpoint https://api.openai.com \
  --token "$OPENAI_API_KEY" \
  --model gpt-4o-mini
```

## Options

| Flag                  | Default | Description                                                       |
| --------------------- | ------- | ----------------------------------------------------------------- |
| `--endpoint`          | —       | Server URL (required).                                            |
| `--token`             | `""`    | API token / bearer token.                                         |
| `--backend`           | `llama` | One of `llama`, `ollama`, `anthropic`, `openai`.                  |
| `--model`             | `""`    | Model name (required for ollama / anthropic / openai).           |
| `--max-users`         | `10`    | Max concurrent users to test.                                     |
| `--requests-per-user` | `5`     | Requests per user at each level.                                  |
| `--max-tokens`        | `64`    | Max tokens per completion.                                        |
| `--think-time`        | `2.0`   | Seconds between requests for a single user.                       |
| `--step`              | `2`     | Increment in concurrent users per level.                          |
| `--usable-p90`        | `3.0`   | p90 latency (s) below which a level is rated `YES`.              |
| `--marginal-p90`      | `5.0`   | p90 latency (s) below which a level is rated `MARGINAL`, else `NO`. |

## Metrics

Latency percentiles (`p50` / `p90` / `p99`) are computed with linear
interpolation between ranks (matching numpy's default method), so small samples
don't collapse the high percentiles onto `max`.

`throughput_rps` is reported as `successes / sum(latencies) * n` — i.e. an
estimate of aggregate requests/second assuming the per-request latencies were
served in parallel. Treat it as a rough indicator under concurrency, not an
exact wall-clock RPS.

## Development

```bash
uv pip install -e ".[dev]"
uv run pytest tests/ -v
```

Tests cover the pure functions (`build_request`, `parse_response`,
`LevelResult.summary`, and `percentile`). CI runs them on every push and pull
request (see `.github/workflows/test.yml`).
