"""Configuration from config.toml (or $TIMESTRAL_ABEL_CONFIG); every key is optional."""

import dataclasses
import logging
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# Timestral's trained context length; longer contexts would be cut by the model anyway.
MODEL_MAX_CONTEXT = 32768


@dataclass(frozen=True)
class Model:
    """Which checkpoint to load and where."""

    # Hugging Face repo id, or a local directory holding config.json + model.safetensors.
    id: str = "mistralai/Timestral-209M"
    # Commit of the Hub repo; ignored for a local directory.
    revision: str = "9874d252ab0aae16f7f7c999f98f98d6ad7d8365"
    # Hugging Face cache directory (the ./models mount in the container).
    cache_dir: str = "models"
    # Never touch the network; the revision must already be in cache_dir.
    local_files_only: bool = False
    device: str = "cuda"


@dataclass(frozen=True)
class Batching:
    """How series are packed into model calls on the single GPU worker."""

    # Series per model call.
    max_batch_series: int = 16
    # Cap on series x rows x (context + horizon) steps per call, after padding,
    # so memory stays bounded whatever the shapes. One series always fits.
    max_batch_cells: int = 1_200_000
    # How long the worker waits for more series before starting a part-full batch.
    batch_wait_ms: int = 5
    # Series waiting for the worker; a request that would exceed it gets 503.
    max_queue_series: int = 2048
    # Values (context, covariates) held by queued series, 4 bytes each: bounds the
    # queue's host memory whatever the shapes (64M = 256 MB). A request that would
    # exceed it gets 503. Must be >= max_body_bytes / 2 so any valid request fits.
    max_queue_values: int = 64 * 1024 * 1024
    # A request still waiting after this many seconds gets 503.
    request_timeout: int = 900


@dataclass(frozen=True)
class Limits:
    max_body_bytes: int = 64 * 1024 * 1024
    max_series_per_request: int = 1024
    max_context: int = MODEL_MAX_CONTEXT
    max_prediction_length: int = 2880
    # Known plus past covariate rows per series.
    max_covariates: int = 32
    # Values in a response: series x prediction_length x (1 + quantiles asked
    # for), about 10 bytes each as JSON. 8M is about 80 MB of response.
    max_output_values: int = 8_000_000
    # Forecast requests reading and parsing their body, or encoding their
    # response, at once. A 64 MiB body takes ~300 MB while it is parsed; others wait.
    max_parallel_parses: int = 2
    # Open connections; more are closed on arrival. Each has its own thread.
    max_connections: int = 64
    # Seconds a socket read or write may stall before the connection is closed.
    socket_timeout: int = 60


@dataclass(frozen=True)
class Warmup:
    """One forecast of this shape runs before the service reports ready, so the
    CUDA kernels are loaded and the allocator already holds the memory a
    typical batch needs. series = 0 skips it."""

    series: int = 16
    context: int = 5376
    prediction_length: int = 96
    known_covariates: int = 4
    past_covariates: int = 1


@dataclass(frozen=True)
class Config:
    listen_host: str = "0.0.0.0"
    listen_port: int = 18320
    # Users file with per-user API keys; re-read whenever it changes.
    users_file: str = "auth/users.toml"
    model: Model = field(default_factory=Model)
    batching: Batching = field(default_factory=Batching)
    limits: Limits = field(default_factory=Limits)
    warmup: Warmup = field(default_factory=Warmup)


SECTIONS = {"model": Model, "batching": Batching, "limits": Limits, "warmup": Warmup}


def _build(cls, data: dict, section: str):
    """Instantiate a config dataclass from a TOML table, coercing to the default's type."""
    known = {f.name: f for f in dataclasses.fields(cls) if f.name not in SECTIONS}
    values = {}
    for key, value in data.items():
        if key not in known:
            log.warning("config: ignoring unknown key %s%s", section, key)
            continue
        kind = type(known[key].default)
        if kind is bool and not isinstance(value, bool):
            raise ValueError(f"config: {section}{key} must be true or false")
        values[key] = kind(value)
    return cls(**values)


def validate(cfg: Config) -> Config:
    b, lim = cfg.batching, cfg.limits
    positive = {
        "batching.max_batch_series": b.max_batch_series, "batching.max_batch_cells": b.max_batch_cells,
        "batching.max_queue_series": b.max_queue_series, "batching.request_timeout": b.request_timeout,
        "limits.max_body_bytes": lim.max_body_bytes, "limits.max_series_per_request": lim.max_series_per_request,
        "limits.max_context": lim.max_context, "limits.max_prediction_length": lim.max_prediction_length,
        "batching.max_queue_values": b.max_queue_values, "limits.max_output_values": lim.max_output_values,
        "limits.max_parallel_parses": lim.max_parallel_parses, "limits.max_connections": lim.max_connections,
        "limits.socket_timeout": lim.socket_timeout,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"config: {name} must be positive")
    if b.batch_wait_ms < 0 or lim.max_covariates < 0:
        raise ValueError("config: batching.batch_wait_ms and limits.max_covariates must not be negative")
    if lim.max_context > MODEL_MAX_CONTEXT:
        raise ValueError(f"config: limits.max_context cannot exceed the model's {MODEL_MAX_CONTEXT}")
    if b.max_queue_series < lim.max_series_per_request:
        raise ValueError("config: batching.max_queue_series must be >= limits.max_series_per_request")
    # Every JSON value takes at least 2 bytes ("0,"), so a body of max_body_bytes
    # holds at most half as many values; such a request must fit an empty queue.
    if b.max_queue_values < lim.max_body_bytes // 2:
        raise ValueError("config: batching.max_queue_values must be >= limits.max_body_bytes / 2")
    return cfg


def load(path: str | os.PathLike | None = None) -> Config:
    path = Path(path or os.environ.get("TIMESTRAL_ABEL_CONFIG") or "config.toml")
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        log.info("config: %s not found, using defaults", path)
        data = {}
    sections = {name: _build(cls, data.pop(name, {}), f"{name}.") for name, cls in SECTIONS.items()}
    return validate(dataclasses.replace(_build(Config, data, ""), **sections))
