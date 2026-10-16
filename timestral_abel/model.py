"""CUDA start-up, model loading, warm-up and the facts /v1/info reports."""

import importlib.metadata
import json
import logging
import re
import time

import numpy as np

from .config import Config, Model, Warmup

log = logging.getLogger(__name__)

# Exit status when the CUDA context cannot be created. This can fail transiently
# with "out of memory" when GPU memory is tight, and a fresh process a few seconds
# later succeeds; Docker's restart policy provides that fresh process.
EXIT_CUDA_INIT = 75

# Required with any use of the weights; see README.md "Licence".
ATTRIBUTION = ("Timestral model weights: 2026 - Mistral AI, Modified MIT license. "
               "timestral inference code: Apache-2.0.")


class CudaInitError(RuntimeError):
    pass


def init_cuda(device: str) -> None:
    """Create the CUDA context before anything else, so a failure is quick and obvious."""
    import torch

    if not device.startswith("cuda"):
        return
    if not torch.cuda.is_available():
        raise CudaInitError("torch.cuda.is_available() is False (is the GPU passed to the container?)")
    try:
        torch.zeros(1, device=device)
        torch.cuda.synchronize(device)
    except Exception as exc:
        raise CudaInitError(str(exc).strip().splitlines()[0] if str(exc).strip() else repr(exc)) from exc


def load(cfg: Model):
    """The forecaster for the configured checkpoint (downloaded to cache_dir if needed)."""
    from timestral import TimestralForecaster

    from .download import fetch

    start = time.monotonic()
    forecaster = TimestralForecaster.from_pretrained(fetch(cfg), device=cfg.device)
    log.info("loaded %s (revision %s) on %s in %.1fs", cfg.id, cfg.revision[:12] or "-", cfg.device,
             time.monotonic() - start)
    return forecaster


def warm_up(forecaster, cfg: Warmup, rng_seed: int = 0) -> float:
    """One synthetic forecast of the configured shape; returns its seconds."""
    if cfg.series <= 0:
        return 0.0
    rng = np.random.default_rng(rng_seed)
    t, h = cfg.context, cfg.prediction_length
    context = rng.random((cfg.series, 1, t), dtype=np.float32)
    known = rng.random((cfg.series, cfg.known_covariates, t + h), dtype=np.float32) if cfg.known_covariates else None
    past = rng.random((cfg.series, cfg.past_covariates, t), dtype=np.float32) if cfg.past_covariates else None
    if not forecaster.supports_covariates:
        known = past = None
    elif not forecaster.supports_known_covariates:
        known = None
    start = time.monotonic()
    forecaster.forecast(context, h, known_covariates=known, past_covariates=past).quantiles.cpu()
    return time.monotonic() - start


def gpu_memory(device: str):
    """A callable returning (allocated, reserved) bytes; zeros off the GPU."""
    if not device.startswith("cuda"):
        return lambda: (0, 0)
    import torch

    return lambda: (torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device))


def timestral_commit() -> str:
    """The commit the installed timestral package was built from, if pip recorded it."""
    try:
        direct = json.loads(importlib.metadata.distribution("timestral").read_text("direct_url.json") or "{}")
    except (importlib.metadata.PackageNotFoundError, ValueError):
        return "unknown"
    commit = direct.get("vcs_info", {}).get("commit_id")
    if commit:
        return commit
    match = re.search(r"\b[0-9a-f]{40}\b", direct.get("url", ""))
    return match.group(0) if match else "unknown"


def info(cfg: Config, forecaster) -> dict:
    """The static part of GET /v1/info."""
    import torch

    model = forecaster.model
    device = str(forecaster.device)
    if device.startswith("cuda"):
        device = f"{device} ({torch.cuda.get_device_name(forecaster.device)})"
    lim = cfg.limits
    return {
        "model": cfg.model.id,
        "revision": cfg.model.revision,
        "timestral_commit": timestral_commit(),
        "torch": torch.__version__,
        "device": device,
        "parameters": sum(p.numel() for p in model.parameters()),
        "patch_size": forecaster.patch_size,
        "supports_past_covariates": forecaster.supports_covariates,
        "supports_known_covariates": forecaster.supports_known_covariates,
        "quantile_levels": list(forecaster.quantile_levels),
        "limits": {
            "max_body_bytes": lim.max_body_bytes,
            "max_series_per_request": lim.max_series_per_request,
            "max_context": min(lim.max_context, forecaster.max_context),
            "max_prediction_length": lim.max_prediction_length,
            "max_covariates": lim.max_covariates,
            "max_batch_series": cfg.batching.max_batch_series,
            "max_batch_cells": cfg.batching.max_batch_cells,
            "max_queue_series": cfg.batching.max_queue_series,
            "max_queue_values": cfg.batching.max_queue_values,
            "max_output_values": lim.max_output_values,
        },
        "license": ATTRIBUTION,
        "license_url": f"https://huggingface.co/{cfg.model.id}/blob/main/LICENSE"
        if cfg.model.id.count("/") == 1 and not cfg.model.id.startswith((".", "/")) else None,
    }
