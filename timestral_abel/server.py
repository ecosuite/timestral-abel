"""HTTP server: /healthz, /metrics, /v1/info and /v1/forecast.

Memory is bounded per stage, so the service can share a host with other work:
connections are capped (`max_connections`) and time out when they stall
(`socket_timeout`); at most `max_parallel_parses` requests read and parse a
body, or encode and send a response, at once; the queue in between is bounded
in series and values (batching.py); a response is capped at `max_output_values`.
"""

import json
import logging
import math
import select
import socket
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from . import auth, request
from .batching import Batcher, QueueFull, ShuttingDown
from .config import Config
from .metrics import Metrics

log = logging.getLogger(__name__)

# Paths polled by health checks and metrics scrapers; logged only when they fail.
QUIET = {"/healthz", "/metrics"}
# How long a request waits for a parse slot before it gets 503.
PARSE_SLOT_WAIT = 30
# A body left unread after an error is drained for at most this long.
DRAIN_SECONDS = 30
# How often a request waiting for the GPU worker checks that its client is still there.
CLIENT_POLL = 1.0


class Service:
    """Everything the handlers need. `batcher` and `info` are set once the model is warm."""

    def __init__(self, cfg: Config, users: auth.Directory, metrics: Metrics):
        self.cfg = cfg
        self.users = users
        self.metrics = metrics
        self.batcher: Batcher | None = None
        self.failed = False  # the GPU worker gave up; the process is about to exit
        self.info: dict = {}
        self.levels: tuple[float, ...] = ()
        self.known_supported = True
        self.covariates_supported = True
        self.io_slots = threading.BoundedSemaphore(cfg.limits.max_parallel_parses)

    def set_ready(self, forecaster, batcher: Batcher, info: dict) -> None:
        self.levels = tuple(forecaster.quantile_levels)
        self.known_supported = forecaster.supports_known_covariates
        self.covariates_supported = forecaster.supports_covariates
        self.info = info
        self.metrics.queue_depth = lambda: batcher.queue_depth
        self.batcher = batcher
        self.metrics.ready = True

    def set_failed(self) -> None:
        self.failed = True
        self.metrics.ready = False

    @property
    def ready(self) -> bool:
        return self.batcher is not None and not self.failed

    @property
    def state(self) -> str:
        return "failed" if self.failed else "ok" if self.batcher is not None else "loading"


class ForecastServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 64  # listen backlog

    def __init__(self, service: Service):
        self.service = service
        self.max_connections = service.cfg.limits.max_connections
        self._active = 0
        self._active_cond = threading.Condition()
        self._refused_logged = 0.0
        super().__init__((service.cfg.listen_host, service.cfg.listen_port), Handler)

    def process_request(self, request, client_address):
        """Start a handler thread, unless max_connections are already open."""
        with self._active_cond:
            full = self._active >= self.max_connections
            if not full:
                self._active += 1
        if not full:
            super().process_request(request, client_address)
            return
        if time.monotonic() - self._refused_logged > 60:
            self._refused_logged = time.monotonic()
            log.warning("%d connections open; refusing new ones", self.max_connections)
        try:  # a best-effort answer that cannot block the accept loop
            request.setblocking(False)
            request.send(b"HTTP/1.0 503 Service Unavailable\r\nRetry-After: 5\r\n"
                         b"Content-Length: 0\r\nConnection: close\r\n\r\n")
        except OSError:
            pass
        self.shutdown_request(request)

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self._active_cond:
                self._active -= 1
                self._active_cond.notify_all()

    def wait_idle(self, timeout: float) -> bool:
        """Wait until no handler is running (used at shutdown); False on timeout."""
        deadline = time.monotonic() + timeout
        with self._active_cond:
            while self._active:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._active_cond.wait(left)
        return True


class HttpError(Exception):
    def __init__(self, status: int, message: str, outcome: str, headers: dict | None = None):
        super().__init__(message)
        self.status, self.outcome, self.headers = status, outcome, headers or {}


class ClientGone(Exception):
    """The client closed its connection while its forecasts were pending."""


class Handler(BaseHTTPRequestHandler):
    server: ForecastServer

    def setup(self) -> None:
        # Every socket read and write (request line, headers, body, response)
        # times out, so a stalled or idle client cannot hold its thread forever.
        self.timeout = self.server.service.cfg.limits.socket_timeout
        super().setup()

    def version_string(self) -> str:
        return "timestral-abel"

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        svc = self.server.service
        try:
            if path == "/healthz":
                body = {"status": svc.state, "ready": svc.ready,
                        "model": svc.cfg.model.id, "device": svc.cfg.model.device}
                self._send_json(200 if svc.ready else 503, body)
            elif path == "/metrics":
                self._send(200, "text/plain; version=0.0.4; charset=utf-8", svc.metrics.render().encode())
            elif path == "/v1/info":
                self._authenticate()
                self._check_ready()
                self._send_json(200, {**svc.info, "ready": True,
                                      "queue_depth": svc.batcher.queue_depth})
            elif path == "/v1/forecast":
                self._send_json(405, {"error": "use POST"}, {"Allow": "POST"})
            else:
                self._send_json(404, {"error": "not found"})
        except HttpError as exc:
            self._send_json(exc.status, {"error": str(exc)}, exc.headers)
        except Exception:
            log.exception("error handling GET %s", path)
            self._send_json(500, {"error": "internal error"})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path != "/v1/forecast":
            self._drain(1024 * 1024)
            known = path in ("/v1/info", "/healthz", "/metrics")
            self._send_json(405 if known else 404, {"error": "use GET" if known else "not found"},
                            {"Allow": "GET"} if known else None)
            return
        self._body_read = False
        self._forecast()

    def _forecast(self) -> None:
        svc = self.server.service
        start = time.monotonic()
        user, n_series, outcome = None, 0, "server_error"
        pending = None
        try:
            user = self._authenticate()
            self._check_ready()
            # Reading and parsing a large body costs several times its size in
            # memory, so only a few requests do it at once.
            if not svc.io_slots.acquire(timeout=PARSE_SLOT_WAIT):
                raise HttpError(503, "busy parsing other requests; retry shortly", "rejected",
                                {"Retry-After": "5"})
            try:
                body = self._read_body(svc.cfg.limits.max_body_bytes)
                try:
                    req = request.parse(body, svc.cfg.limits, svc.levels, svc.known_supported,
                                        svc.covariates_supported)
                except request.BadRequest as exc:
                    raise HttpError(400, str(exc), "client_error") from None
                del body
            finally:
                svc.io_slots.release()
            n_series = len(req.series)
            try:
                pending = svc.batcher.submit(req.series, user.id)
            except QueueFull:
                raise HttpError(503, "the forecast queue is full; retry later", "rejected",
                                {"Retry-After": str(self._retry_after())}) from None
            except ShuttingDown:
                raise HttpError(503, "the service is restarting; retry shortly", "rejected",
                                {"Retry-After": "30"}) from None
            self._wait(pending, svc.cfg.batching.request_timeout)
            if pending.error is not None:
                status, message = pending.error
                if status == 503:
                    raise HttpError(503, message, "rejected", {"Retry-After": "30"})
                raise HttpError(status, message, "client_error" if status < 500 else "server_error")
            done = time.monotonic()
            started = pending.started_at or done
            timing = {
                "queue_seconds": round(started - pending.submitted_at, 4),
                "compute_seconds": round(pending.compute_seconds, 4),
                "total_seconds": round(done - start, 4),
                "batches": len(pending.batch_ids),
            }
            # The forecasts are done: from here on a lost connection is the client's doing.
            outcome = "client_error"
            with svc.io_slots:  # the response is several times the size of its numbers
                chunks = request.encode({"model": svc.cfg.model.id, "revision": svc.cfg.model.revision},
                                        req, pending.results, svc.levels, {"timing": timing})
                self._send(200, "application/json", chunks)
            del chunks
            outcome = "ok"
            log.info("forecast user=%s series=%d batches=%d queue=%.3fs compute=%.3fs total=%.3fs status=200",
                     user.id, n_series, timing["batches"], timing["queue_seconds"],
                     timing["compute_seconds"], time.monotonic() - start)
        except HttpError as exc:
            outcome = exc.outcome
            if not self._body_read:
                # Refused before reading the body (bad key, still loading, too
                # large, busy): read it anyway, or closing the socket on unread
                # data resets the connection and clients that send the whole
                # body before reading (urllib, requests) never see this answer.
                # A body over 4 x the limit is not read; the connection just closes.
                limit = svc.cfg.limits.max_body_bytes
                self._drain(4 * limit if exc.status == 413 else limit)
            self._reply_error(exc)
            log.info("forecast user=%s series=%d status=%d error=%s",
                     user.id if user else "-", n_series, exc.status, str(exc)[:200])
        except ClientGone:
            outcome = "client_error"
            log.info("forecast user=%s series=%d: client went away while waiting; its queued series were dropped",
                     user.id if user else "-", n_series)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            outcome = "client_error"
            self.close_connection = True
            log.info("forecast user=%s series=%d: client went away", user.id if user else "-", n_series)
        except Exception:
            log.exception("error handling POST /v1/forecast")
            self._reply_error(HttpError(500, "internal error", "server_error"))
        finally:
            if pending is not None and not pending.done.is_set():
                svc.batcher.cancel(pending)
            svc.metrics.record_request(outcome, time.monotonic() - start)

    def _wait(self, pending, timeout: float) -> None:
        """Wait for a request's forecasts, dropping them if its client disconnects."""
        svc = self.server.service
        deadline = time.monotonic() + timeout
        while not pending.done.wait(min(CLIENT_POLL, max(0.0, deadline - time.monotonic()))):
            if self._client_gone():
                svc.batcher.cancel(pending)
                raise ClientGone()
            if time.monotonic() >= deadline:
                svc.batcher.cancel(pending)
                raise HttpError(503, "timed out waiting for the GPU worker", "server_error",
                                {"Retry-After": "30"})

    def _client_gone(self) -> bool:
        """True when the peer has closed its side (readable with nothing to read)."""
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            return bool(readable) and self.connection.recv(1, socket.MSG_PEEK) == b""
        except (OSError, ValueError):
            return True

    # --- helpers ----------------------------------------------------------------

    def _check_ready(self) -> None:
        svc = self.server.service
        if svc.failed:
            raise HttpError(503, "the service is restarting; retry shortly", "rejected", {"Retry-After": "30"})
        if not svc.ready:
            raise HttpError(503, "model is loading", "rejected", {"Retry-After": "30"})

    def _authenticate(self) -> auth.User:
        user = self.server.service.users.authenticate(auth.request_key(self.headers))
        if user is None:
            raise HttpError(401, "missing or invalid API key (Authorization: Bearer <key> or x-api-key)",
                            "client_error", {"WWW-Authenticate": "Bearer"})
        return user

    def _read_body(self, limit: int) -> bytes:
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            raise HttpError(411, "send a Content-Length (chunked bodies are not supported)", "client_error")
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            raise HttpError(411, "Content-Length required", "client_error") from None
        if length < 0:
            raise HttpError(400, "bad Content-Length", "client_error")
        if length > limit:
            self.close_connection = True  # the body is not read
            raise HttpError(413, f"body larger than {limit} bytes", "client_error")
        self._body_read = True  # whatever happens next, the body is not drained again
        try:
            body = self.rfile.read(length)
        except TimeoutError:
            self.close_connection = True
            raise HttpError(408, "timed out reading the body", "client_error") from None
        if len(body) != length:
            raise HttpError(400, "body shorter than Content-Length", "client_error")
        return body

    def _drain(self, limit: int) -> None:
        """Read and drop a body of up to `limit` bytes so the reply is not lost to a
        reset. Stops after DRAIN_SECONDS, so a trickling client cannot hold the thread."""
        self._body_read = True
        self.close_connection = True
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if not 0 < length <= limit:
            return
        deadline = time.monotonic() + DRAIN_SECONDS
        try:
            while length > 0 and time.monotonic() < deadline:
                chunk = self.rfile.read1(min(length, 1 << 20))
                if not chunk:
                    break
                length -= len(chunk)
        except OSError:  # includes a socket timeout
            pass

    def _reply_error(self, exc: HttpError) -> None:
        try:
            self._send_json(exc.status, {"error": str(exc)}, exc.headers)
        except OSError:  # the client is gone; nothing to tell it
            self.close_connection = True

    def _retry_after(self) -> int:
        """Seconds until the queue has likely drained: queued batches x mean batch time."""
        svc = self.server.service
        m = svc.metrics
        per_batch = m.forecast_seconds / m.batches_total if m.batches_total else 1.0
        batches = svc.batcher.queue_depth / svc.cfg.batching.max_batch_series
        return max(1, min(120, math.ceil(batches * per_batch)))

    def _send_json(self, status: int, data, headers: dict | None = None) -> None:
        self._send(status, "application/json", json.dumps(data, separators=(",", ":")).encode(), headers)

    def _send(self, status: int, ctype: str, body: bytes | list[bytes], headers: dict | None = None) -> None:
        chunks = [body] if isinstance(body, bytes) else body
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(sum(map(len, chunks))))
        self.send_header("Cache-Control", "no-store")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            for chunk in chunks:
                self.wfile.write(chunk)

    def log_request(self, code="-", size="-") -> None:
        path = urlsplit(self.path).path
        status = int(code) if isinstance(code, (int, HTTPStatus)) else code
        # /v1/forecast logs its own line; polled paths only when they fail.
        if path == "/v1/forecast" or (path in QUIET and status == 200):
            return
        log.info("%s %s %s", self.command, path, status)

    def log_message(self, format, *args) -> None:
        log.info(format, *args)
