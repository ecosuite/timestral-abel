import http.client
import json
import re
import socket
import threading
import time
import unittest
from unittest import mock

import numpy as np

from timestral_abel.config import Batching, Limits
from helpers import KEY_A, KEY_B, LEVELS, UPSTREAM, FakeForecaster, LiveServer, series, tiny_forecaster

METRIC_NAMES = [
    "timestral_up", "timestral_model_info", "timestral_ready", "timestral_requests_total",
    "timestral_series_total", "timestral_batches_total", "timestral_forecast_seconds_total",
    "timestral_request_duration_seconds_sum", "timestral_request_duration_seconds_count",
    "timestral_queue_depth", "timestral_gpu_memory_allocated_bytes", "timestral_gpu_memory_reserved_bytes",
    "timestral_last_request_timestamp_seconds", "timestral_user_series_total", "process_start_time_seconds",
]


def metric_values(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            name, value = line.rsplit(" ", 1)
            out[name] = float(value)
    return out


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.live = LiveServer(tiny_forecaster(), limits=Limits(max_series_per_request=8,
                                                               max_body_bytes=200_000))

    @classmethod
    def tearDownClass(cls):
        cls.live.close()

    def forecast(self, body, **kw):
        status, headers, raw = self.live.request("/v1/forecast", body, **kw)
        return status, headers, json.loads(raw)

    def test_healthz_and_info(self):
        status, _, raw = self.live.request("/healthz", key=None)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw), {"status": "ok", "ready": True, "model": "mistralai/Timestral-209M",
                                           "device": "cuda"})
        status, _, raw = self.live.request("/v1/info")
        self.assertEqual(status, 200)
        info = json.loads(raw)
        self.assertTrue(info["ready"])
        self.assertEqual(info["queue_depth"], 0)
        self.assertEqual(self.live.request("/v1/info", key=None)[0], 401)

    def test_auth(self):
        body = {"prediction_length": 2, "series": [{"context": [1, 2, 3]}]}
        self.assertEqual(self.forecast(body)[0], 200)
        self.assertEqual(self.forecast(body, key=KEY_B, header="x-api-key")[0], 200)
        status, headers, out = self.forecast(body, key=UPSTREAM)
        self.assertEqual(status, 401)
        self.assertEqual(headers["WWW-Authenticate"], "Bearer")
        self.assertNotIn(UPSTREAM, json.dumps(out))
        self.assertEqual(self.forecast(body, key=None)[0], 401)
        self.assertEqual(self.forecast(body, key="z" * 32)[0], 401)

    def test_batched_forecast(self):
        rng = np.random.default_rng(1)
        h = 10
        items = [series(rng.random(40) * 3, sid="a"),
                 series(np.r_[np.nan, rng.random(24)], known=rng.random((2, 25 + h)), sid="b"),
                 series(rng.random(33), known=rng.random((2, 33 + h)), past=rng.random((1, 33)))]
        status, _, out = self.forecast({"prediction_length": h, "series": items, "quantiles": [0.1, 0.9]})
        self.assertEqual(status, 200, out)
        self.assertEqual(out["model"], "mistralai/Timestral-209M")
        self.assertEqual(out["quantile_levels"], [0.1, 0.9])
        self.assertEqual([f["id"] for f in out["forecasts"]], ["a", "b", 2])
        for f in out["forecasts"]:
            self.assertEqual(len(f["median"]), h)
            self.assertEqual(set(f["quantiles"]), {"0.1", "0.9"})
            self.assertTrue(all(lo <= m <= hi for lo, m, hi in
                                zip(f["quantiles"]["0.1"], f["median"], f["quantiles"]["0.9"])))
        self.assertEqual(set(out["timing"]), {"queue_seconds", "compute_seconds", "total_seconds", "batches"})
        self.assertEqual(out["timing"]["batches"], 3)  # three layouts

        # The same series one per request give the same numbers.
        for item, f in zip(items, out["forecasts"]):
            _, _, single = self.forecast({"prediction_length": h, "series": [item]})
            np.testing.assert_allclose(single["forecasts"][0]["median"], f["median"], rtol=1e-5, atol=1e-5)
            self.assertEqual(len(single["forecasts"][0]["quantiles"]), len(LEVELS))

    def test_client_errors(self):
        cases = [
            ({"prediction_length": 2}, 400, "series"),
            ({"prediction_length": 2, "series": [{"context": [1, "x"]}]}, 400, "series[0].context"),
            ({"prediction_length": 2, "series": [{"context": [1]}] * 9}, 400, "at most 8"),
        ]
        for body, code, needle in cases:
            with self.subTest(body=str(body)[:60]):
                status, _, out = self.forecast(body)
                self.assertEqual(status, code)
                self.assertIn(needle, out["error"])
        status, _, raw = self.live.request("/v1/forecast", raw=b"x" * 200_001)
        self.assertEqual(status, 413)
        # urllib sends the whole body before reading the answer: the server reads
        # (and drops) it, so the 413 arrives instead of a reset connection.
        status, _, raw = self.live.request("/v1/forecast", raw=b"x" * 790_000)
        self.assertEqual(status, 413)
        self.assertIn("larger than", json.loads(raw)["error"])
        self.assertEqual(self.live.request("/v1/forecast")[0], 405)  # GET
        self.assertEqual(self.live.request("/nope", key=None)[0], 404)
        self.assertEqual(self.live.request("/v1/info", body={}, key=None)[0], 405)

    def test_metrics(self):
        self.forecast({"prediction_length": 3, "series": [{"context": [1, 2]}, {"context": [3, 4, 5]}]})
        status, headers, raw = self.live.request("/metrics", key=None)
        self.assertEqual(status, 200)
        self.assertTrue(headers["Content-Type"].startswith("text/plain"))
        text = raw.decode()
        for name in METRIC_NAMES:
            self.assertRegex(text, rf"(?m)^{name}[{{ ]")
        for status_ in ("ok", "client_error", "server_error", "rejected"):
            self.assertIn(f'timestral_requests_total{{status="{status_}"}}', text)
        self.assertIn('timestral_model_info{model="test/model",revision="rev123",device="cpu"} 1', text)
        values = metric_values(text)
        self.assertEqual(values["timestral_ready"], 1)
        self.assertGreaterEqual(values["timestral_series_total"], 2)
        self.assertGreaterEqual(values['timestral_user_series_total{user="alice"}'], 2)
        self.assertGreater(values["timestral_forecast_seconds_total"], 0)
        self.assertAlmostEqual(values["timestral_last_request_timestamp_seconds"], time.time(), delta=60)
        self.assertNotRegex(text, re.escape(KEY_A))


class LoadingTest(unittest.TestCase):
    def test_not_ready(self):
        live = LiveServer(ready=False)
        try:
            status, _, raw = live.request("/healthz", key=None)
            self.assertEqual(status, 503)
            self.assertEqual(json.loads(raw)["status"], "loading")
            status, headers, _ = live.request("/v1/forecast", {"prediction_length": 1,
                                                                "series": [{"context": [1]}]})
            self.assertEqual(status, 503)
            self.assertEqual(headers["Retry-After"], "30")
            self.assertIn("timestral_ready 0", live.request("/metrics", key=None)[2].decode())
            self.assertIn('timestral_requests_total{status="rejected"} 1',
                          live.request("/metrics", key=None)[2].decode())
        finally:
            live.close()


class QueueFullTest(unittest.TestCase):
    def test_rejects_with_retry_after(self):
        gate = threading.Event()
        live = LiveServer(FakeForecaster(gate), batching=Batching(max_queue_series=3, max_batch_series=1,
                                                                  batch_wait_ms=0),
                          limits=Limits(max_series_per_request=3))
        body = {"prediction_length": 1, "series": [{"context": [1]}] * 3}
        results = []
        try:
            first = threading.Thread(target=lambda: results.append(live.request("/v1/forecast", body)))
            first.start()
            deadline = time.time() + 5
            while live.batcher.queue_depth < 2 and time.time() < deadline:
                time.sleep(0.01)
            status, headers, raw = live.request("/v1/forecast", body)
            self.assertEqual(status, 503)
            self.assertIn("queue is full", json.loads(raw)["error"])
            self.assertGreaterEqual(int(headers["Retry-After"]), 1)
            gate.set()
            first.join(10)
            self.assertEqual(results[0][0], 200)
            text = live.request("/metrics", key=None)[2].decode()
            self.assertIn('timestral_requests_total{status="rejected"} 1', text)
            self.assertIn('timestral_requests_total{status="ok"} 1', text)
        finally:
            gate.set()
            live.close()

    def test_failed_series_is_422(self):
        live = LiveServer()
        try:
            with self.assertLogs("timestral_abel.batching", "WARNING"):
                status, _, raw = live.request("/v1/forecast", {"prediction_length": 1,
                                                               "series": [{"context": [1]}, {"context": [666]}]})
            self.assertEqual(status, 422)
            self.assertIn("series[1]", json.loads(raw)["error"])
        finally:
            live.close()


class RobustnessTest(unittest.TestCase):
    """Connections, slots, departed clients and shutdown."""

    def metrics(self, live):
        return metric_values(live.request("/metrics", key=None)[2].decode())

    def test_stalled_connection_times_out(self):
        live = LiveServer(limits=Limits(socket_timeout=1))
        try:
            with socket.create_connection(live.server.server_address, timeout=5) as sock:
                sock.sendall(b"POST /v1/forecast HTTP/1.1\r\n")  # and nothing more
                start = time.monotonic()
                self.assertEqual(sock.recv(1024), b"")  # the server closed it
                self.assertLess(time.monotonic() - start, 4)
            # An authenticated body that stalls part-way gets 408.
            with socket.create_connection(live.server.server_address, timeout=5) as sock:
                sock.sendall(f"POST /v1/forecast HTTP/1.1\r\nAuthorization: Bearer {KEY_A}\r\n"
                             "Content-Length: 100\r\n\r\n{".encode())
                self.assertIn(b" 408 ", sock.recv(1024))
        finally:
            live.close()

    def test_connection_cap(self):
        live = LiveServer(limits=Limits(max_connections=2, socket_timeout=5))
        try:
            idle = [socket.create_connection(live.server.server_address, timeout=5) for _ in range(2)]
            time.sleep(0.2)
            with self.assertLogs("timestral_abel.server", "WARNING"):
                with socket.create_connection(live.server.server_address, timeout=5) as sock:
                    self.assertTrue(sock.recv(1024).startswith(b"HTTP/1.0 503"))
            for sock in idle:
                sock.close()
            deadline = time.time() + 5
            while live.server._active and time.time() < deadline:
                time.sleep(0.01)
            self.assertEqual(live.request("/healthz", key=None)[0], 200)
        finally:
            live.close()

    def test_client_that_leaves_has_its_series_dropped(self):
        gate = threading.Event()
        live = LiveServer(FakeForecaster(gate), batching=Batching(max_batch_series=1, batch_wait_ms=0))
        try:
            body = json.dumps({"prediction_length": 1, "series": [{"context": [1]}] * 5})
            conn = http.client.HTTPConnection("127.0.0.1", live.server.server_address[1], timeout=0.5)
            conn.request("POST", "/v1/forecast", body, {"Authorization": f"Bearer {KEY_A}"})
            with self.assertRaises(TimeoutError):
                conn.getresponse()
            self.assertGreaterEqual(live.batcher.queue_depth, 4)
            conn.close()  # the client gives up
            deadline = time.time() + 5
            while live.batcher.queue_depth and time.time() < deadline:
                time.sleep(0.05)
            self.assertEqual(live.batcher.queue_depth, 0)
            gate.set()
            deadline = time.time() + 5
            while (self.metrics(live)['timestral_requests_total{status="client_error"}'] < 1
                   or self.metrics(live)["timestral_series_total"] < 1) and time.time() < deadline:
                time.sleep(0.05)
            values = self.metrics(live)
            self.assertEqual(values['timestral_requests_total{status="client_error"}'], 1)
            self.assertEqual(values["timestral_series_total"], 1)  # the one the model ran for
            self.assertEqual(values['timestral_user_series_total{user="alice"}'], 1)
            self.assertEqual(values['timestral_requests_total{status="server_error"}'], 0)
        finally:
            gate.set()
            live.close()

    def test_parse_slots_bound_concurrent_parsing(self):
        live = LiveServer(limits=Limits(max_parallel_parses=1))
        try:
            live.service.io_slots.acquire()  # someone else is parsing
            with mock.patch("timestral_abel.server.PARSE_SLOT_WAIT", 0.2):
                status, headers, raw = live.request("/v1/forecast", {"prediction_length": 1,
                                                                     "series": [{"context": [1]}]})
            self.assertEqual(status, 503)
            self.assertEqual(headers["Retry-After"], "5")
            live.service.io_slots.release()
            self.assertEqual(live.request("/v1/forecast", {"prediction_length": 1,
                                                           "series": [{"context": [1]}]})[0], 200)
        finally:
            live.close()

    def test_client_errors_not_server_errors(self):
        live = LiveServer(limits=Limits(max_output_values=1000, max_series_per_request=100))
        try:
            status, _, raw = live.request("/v1/forecast", raw=b'{"series":' + b"[" * 100_000)
            self.assertEqual(status, 400)
            self.assertIn("nested too deeply", json.loads(raw)["error"])
            status, _, raw = live.request("/v1/forecast", {"prediction_length": 100,
                                                           "series": [{"context": [1]}]})
            self.assertEqual(status, 400)
            self.assertIn("response would hold", json.loads(raw)["error"])
            self.assertEqual(self.metrics(live)['timestral_requests_total{status="server_error"}'], 0)
        finally:
            live.close()

    def test_shutdown_answers_queued_requests(self):
        gate = threading.Event()
        live = LiveServer(FakeForecaster(gate), batching=Batching(max_batch_series=1, batch_wait_ms=0))
        results = []
        try:
            body = {"prediction_length": 1, "series": [{"context": [1]}] * 3}
            client = threading.Thread(target=lambda: results.append(live.request("/v1/forecast", body)))
            client.start()
            deadline = time.time() + 5
            while live.batcher.queue_depth < 2 and time.time() < deadline:
                time.sleep(0.01)
            stopper = threading.Thread(target=live.batcher.stop, args=(10,))
            stopper.start()
            client.join(5)
            status, headers, raw = results[0]
            self.assertEqual(status, 503)
            self.assertEqual(headers["Retry-After"], "30")
            self.assertIn("restarting", json.loads(raw)["error"])
            gate.set()
            stopper.join(10)
            status, _, raw = live.request("/v1/forecast", body)  # the batcher takes no more
            self.assertEqual(status, 503)
            self.assertTrue(live.server.wait_idle(5))
        finally:
            gate.set()
            live.close()

    def test_failed_service_is_unhealthy(self):
        live = LiveServer()
        try:
            live.service.set_failed()
            status, _, raw = live.request("/healthz", key=None)
            self.assertEqual(status, 503)
            self.assertEqual(json.loads(raw)["status"], "failed")
            self.assertEqual(self.metrics(live)["timestral_ready"], 0)
            status, _, raw = live.request("/v1/forecast", {"prediction_length": 1, "series": [{"context": [1]}]})
            self.assertEqual(status, 503)
            self.assertIn("restarting", json.loads(raw)["error"])
        finally:
            live.close()


if __name__ == "__main__":
    unittest.main()
