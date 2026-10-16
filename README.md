# timestral-abel

A small HTTP service for Mistral AI's [Timestral](https://github.com/mistralai/timestral)
time-series forecasting models. Send history arrays (plus optional covariates), get back
probabilistic forecasts: a median and 21 quantiles per step.

- Super simple Python architecture
- Supports batching: series from one request, or from concurrent callers, share model calls.
- Bounded queue, request limits, per-user API keys and Prometheus metrics.
- Serves `mistralai/Timestral-209M` by default, pinned to a Hugging Face revision.

## Quick start (CPU)

Needs [uv](https://docs.astral.sh/uv/).

```sh
uv sync --python 3.12
cat > config.toml <<'EOF'
users_file = "users.toml"
[model]
id = "mistralai/Timestral-5M"
revision = ""
device = "cpu"
EOF
cat > users.toml <<'EOF'
[users.alice]
name = "Alice"
key = "replace-with-a-long-random-key"
EOF
uv run python -m timestral_abel      # downloads the model on first start
curl -s localhost:18320/healthz
```

Tests: `uv run python -m unittest discover -s tests`

## Docker (GPU)

```sh
cp config.example.toml config.toml
docker compose build
docker compose run --rm --no-deps --entrypoint python3 timestral -m timestral_abel.download
docker compose up -d
```

The Dockerfile expects a base image with Python 3.12 and CUDA-enabled PyTorch (see its
`FROM` line). Mount your users file via `TIMESTRAL_AUTH_DIR` (see `docker-compose.yml`).

## API

All endpoints except `/healthz` and `/metrics` need a key, sent as
`Authorization: Bearer <key>` or `x-api-key: <key>`.

| Endpoint | |
| --- | --- |
| `GET /healthz` | Readiness (503 while loading) |
| `GET /v1/info` | Model, revision, device, quantile levels, limits |
| `POST /v1/forecast` | Forecasts for one or many series |
| `GET /metrics` | Prometheus metrics (`timestral_*`) |

```sh
curl -s localhost:18320/v1/forecast -H "Authorization: Bearer $KEY" -d '{
  "prediction_length": 4,
  "quantiles": [0.1, 0.9],
  "series": [
    {"id": "a", "context": [1, 2, 3, 4, 5, 6, 7, 8]},
    {"id": "b", "context": [5, null, 4, 6, 5, 4, 6, 5],
     "known_covariates": [[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]]}
  ]
}'
```

- `context`: history, oldest first; `null` marks a missing value (up to 32768 steps).
- `known_covariates`: rows known over the context and the horizon (context + horizon values).
  `past_covariates`: rows known over the context only.
- `quantiles`: optional subset of the 21 levels; `median` is always returned.
- `symmetric_averaging` (default `true`) and `clip_nonnegative` (default `true`) are passed to
  the model.

Each forecast comes back with its `id`, `median` and `quantiles`. Errors are
`{"error": "..."}` with a 4xx/5xx status; a full queue returns 503 with `Retry-After`.
The full contract is in [`docs/SPEC.md`](docs/SPEC.md).

## Configuration

See [`config.example.toml`](config.example.toml); every setting is optional. The users file
holds one `[users.<id>]` table per caller with a `name` and a `key`; it is re-read when it
changes.

## Licence

This repository is MIT-licensed (see [`LICENSE`](LICENSE)). The Timestral inference code is
Apache-2.0. The Timestral model weights are © 2026 Mistral AI, released under a
[Modified MIT license](https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE);
keep that licence and attribution with any copy of the weights.
