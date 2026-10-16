"""Shared test helpers: a tiny random Timestral, a fake forecaster and a live server."""

import dataclasses
import json
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from timestral import TimestralConfig, TimestralForecaster, TimestralModel
from timestral_abel import auth
from timestral_abel.batching import Batcher
from timestral_abel.config import Batching, Config, Limits
from timestral_abel.metrics import Metrics
from timestral_abel.server import ForecastServer, Service

LEVELS = (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
          0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)

KEY_A = "a" * 32
KEY_B = "b" * 32
UPSTREAM = "u" * 32
USERS_TOML = f"""
upstream_key = "{UPSTREAM}"
[users.alice]
name = "Alice"
key = "{KEY_A}"
[users.bob]
key = "{KEY_B}"
"""


def tiny_forecaster(seed: int = 0) -> TimestralForecaster:
    """A randomly initialised Timestral small enough for CPU tests (no download).

    The layer schedule supports known covariates (a time op followed by more time
    ops and a variate op), so every code path of the real model runs.
    """
    torch.manual_seed(seed)
    cfg = TimestralConfig(patch_size=8, d_model=32, num_heads=2, mlp_hidden_dim=64,
                          block_schedule=("time", "time", "time_variate", "time", "time_variate"),
                          quantile_knots=LEVELS, block_size=4, context_length=512)
    model = TimestralModel(cfg).eval()
    model.transformer.rope.rebuild_inv_freq()
    return TimestralForecaster(model)


class FakeForecaster:
    """Quantiles = last observed value + level, per series; records every call.

    A context holding 666.0 makes the call raise ValueError (like timestral on a
    degenerate input); `gate` (an Event) blocks every call until set.
    """

    quantile_levels = LEVELS
    supports_covariates = True
    supports_known_covariates = True

    def __init__(self, gate: threading.Event | None = None):
        self.gate = gate
        self.calls = []  # (batch size, context length, known rows, past rows, horizon)

    def forecast(self, context, prediction_length, known_covariates=None, past_covariates=None,
                 symmetric_averaging=True, clip_nonnegative=True):
        if self.gate is not None:
            self.gate.wait(10)
        self.calls.append((context.shape[0], context.shape[-1],
                           0 if known_covariates is None else known_covariates.shape[1],
                           0 if past_covariates is None else past_covariates.shape[1], prediction_length))
        if np.any(context == 666.0):
            raise ValueError("forecast produced NaN values; the input is degenerate")
        last = np.array([row[np.isfinite(row)][-1] for row in context[:, 0]], dtype=np.float32)
        q = last[:, None, None, None] + np.array(LEVELS, dtype=np.float32)
        q = np.broadcast_to(q, (context.shape[0], 1, prediction_length, len(LEVELS))).copy()
        return SimpleNamespace(quantiles=torch.from_numpy(q), quantile_levels=LEVELS)


def series(context, known=None, past=None, sid=None) -> dict:
    """One series as JSON-ready lists, NaN as null."""
    def lst(a):
        return [None if not np.isfinite(v) else float(v) for v in np.asarray(a, dtype=np.float64)]
    out = {"context": lst(context)}
    if sid is not None:
        out["id"] = sid
    if known is not None:
        out["known_covariates"] = [lst(r) for r in known]
    if past is not None:
        out["past_covariates"] = [lst(r) for r in past]
    return out


class LiveServer:
    """The HTTP server on a free port with a users file in a temporary directory."""

    def __init__(self, forecaster=None, ready=True, batching: Batching | None = None,
                 limits: Limits | None = None):
        self._dir = tempfile.TemporaryDirectory()
        self.users_path = Path(self._dir.name) / "users.toml"
        self.users_path.write_text(USERS_TOML)
        self.cfg = dataclasses.replace(Config(), listen_host="127.0.0.1", listen_port=0,
                                       users_file=str(self.users_path),
                                       batching=batching or Batching(batch_wait_ms=1),
                                       limits=limits or Limits())
        self.metrics = Metrics("test/model", "rev123", "cpu")
        self.service = Service(self.cfg, auth.Directory(self.users_path), self.metrics)
        self.forecaster = forecaster or FakeForecaster()
        self.batcher = None
        if ready:
            self.make_ready()
        self.server = ForecastServer(self.service)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def make_ready(self):
        self.batcher = Batcher(self.forecaster, self.cfg.batching, self.metrics)
        self.batcher.start()
        self.service.set_ready(self.forecaster, self.batcher,
                               {"model": "test/model", "quantile_levels": list(LEVELS)})

    def request(self, path, body=None, key=KEY_A, header="bearer", method=None, raw=None):
        headers = {}
        if key is not None:
            headers = {"Authorization": f"Bearer {key}"} if header == "bearer" else {"x-api-key": key}
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status, resp.headers, resp.read()
        except urllib.error.HTTPError as err:
            return err.code, err.headers, err.read()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        if self.batcher is not None:
            self.batcher.stop(5)
        self._dir.cleanup()
