"""Counters and gauges, rendered as Prometheus text for GET /metrics.

The names are a stable contract for dashboards and scrapers (see docs/SPEC.md).
"""

import threading
import time

REQUEST_STATUSES = ("ok", "client_error", "server_error", "rejected")


def _label(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _num(value: float) -> str:
    return repr(float(value)) if isinstance(value, float) else str(value)


class Metrics:
    def __init__(self, model: str, revision: str, device: str, start_time: float | None = None):
        self.model, self.revision, self.device = model, revision, device
        self.start_time = start_time if start_time is not None else time.time()
        self._lock = threading.Lock()
        self.ready = False
        self.requests = dict.fromkeys(REQUEST_STATUSES, 0)
        self.series_total = 0
        self.batches_total = 0
        self.forecast_seconds = 0.0
        self.duration_sum = 0.0
        self.duration_count = 0
        self.last_request_at = 0.0
        self.user_series: dict[str, int] = {}
        # Set by the service once the model is loaded.
        self.queue_depth = lambda: 0
        self.gpu_memory = lambda: (0, 0)

    def record_request(self, status: str, seconds: float) -> None:
        """One POST /v1/forecast, whatever its outcome."""
        with self._lock:
            self.requests[status] += 1
            self.duration_sum += seconds
            self.duration_count += 1
            if status == "ok":
                self.last_request_at = time.time()

    def record_batch(self, seconds: float, users: list[str | None] = ()) -> None:
        """One model call (a GPU batch) and the user of each series in it.

        Series are counted here, where the model runs for them, so series_total
        matches forecast_seconds_total even when a client leaves before its
        response is delivered or a request fails part-way.
        """
        with self._lock:
            self.batches_total += 1
            self.forecast_seconds += seconds
            self.series_total += len(users)
            for user in users:
                if user is not None:
                    self.user_series[user] = self.user_series.get(user, 0) + 1

    def render(self) -> str:
        allocated, reserved = self.gpu_memory()
        depth = self.queue_depth()
        with self._lock:
            out = []

            def metric(name, kind, help_, samples):
                out.append(f"# HELP {name} {help_}")
                out.append(f"# TYPE {name} {kind}")
                for labels, value in samples:
                    lbl = ",".join(f'{k}="{_label(v)}"' for k, v in labels.items())
                    out.append(f"{name}{{{lbl}}} {_num(value)}" if lbl else f"{name} {_num(value)}")

            metric("timestral_up", "gauge", "The service is running.", [({}, 1)])
            metric("timestral_model_info", "gauge", "Loaded model, Hub revision and device.",
                   [({"model": self.model, "revision": self.revision, "device": self.device}, 1)])
            metric("timestral_ready", "gauge", "1 once the model is loaded and warmed up.",
                   [({}, int(self.ready))])
            metric("timestral_requests_total", "counter", "POST /v1/forecast requests by outcome.",
                   [({"status": s}, n) for s, n in self.requests.items()])
            metric("timestral_series_total", "counter", "Series forecast (the model ran for them), any request outcome.",
                   [({}, self.series_total)])
            metric("timestral_batches_total", "counter", "Model calls (GPU batches).",
                   [({}, self.batches_total)])
            metric("timestral_forecast_seconds_total", "counter", "Seconds spent in model calls.",
                   [({}, self.forecast_seconds)])
            out.append("# HELP timestral_request_duration_seconds POST /v1/forecast wall time.")
            out.append("# TYPE timestral_request_duration_seconds summary")
            out.append(f"timestral_request_duration_seconds_sum {_num(self.duration_sum)}")
            out.append(f"timestral_request_duration_seconds_count {self.duration_count}")
            metric("timestral_queue_depth", "gauge", "Series waiting for the GPU worker.", [({}, depth)])
            metric("timestral_gpu_memory_allocated_bytes", "gauge", "torch.cuda.memory_allocated.",
                   [({}, allocated)])
            metric("timestral_gpu_memory_reserved_bytes", "gauge",
                   "torch.cuda.memory_reserved (held by the caching allocator).", [({}, reserved)])
            metric("timestral_last_request_timestamp_seconds", "gauge",
                   "Unix time of the last successful forecast request (0 = none yet).",
                   [({}, self.last_request_at)])
            metric("timestral_user_series_total", "counter", "Series forecast per API user.",
                   [({"user": u}, n) for u, n in sorted(self.user_series.items())])
            metric("process_start_time_seconds", "gauge", "Unix time the process started.",
                   [({}, self.start_time)])
        return "\n".join(out) + "\n"
