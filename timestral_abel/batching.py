"""Dynamic batching onto one worker thread that owns the model.

Requests put their series on one queue. The worker takes the oldest series,
then the queued series with the same layout (covariate counts, horizon,
flags) closest to it in context length, up to `max_batch_series` series and
`max_batch_cells` padded cells, and runs them as one model call. Series from
different requests share a call when their layouts match. A part-full batch
waits up to `batch_wait_ms` for company first.

Contexts in a batch are padded on the left with NaN to the longest one. The
model never attends to leading all-missing patches and anchors positions at
the first observed patch, so padding changes a forecast only by float
rounding (tests/test_batching.py checks this).

The queue is bounded twice: by series (`max_queue_series`) and by the values
they hold (`max_queue_values`), so its host memory stays bounded whatever the
shapes. When model calls keep failing with CUDA errors the context is probably
broken for good; after FATAL_FAILURES in a row the worker reports it through
`on_fatal` so the process can exit and Docker restart it with a fresh context.
"""

import itertools
import logging
import threading
import time

import numpy as np

from .config import Batching
from .metrics import Metrics
from .request import Layout, Series

log = logging.getLogger(__name__)


# Consecutive model calls failing with a CUDA error before the worker gives up.
FATAL_FAILURES = 5

SHUTTING_DOWN = (503, "the service is restarting; retry shortly")


class QueueFull(Exception):
    """The request would push the queue past max_queue_series or max_queue_values."""


class ShuttingDown(Exception):
    """The batcher has stopped and takes no more series."""


class Pending:
    """One request's series on their way through the worker."""

    def __init__(self, series: list[Series], user: str | None = None):
        self.series = series
        self.user = user  # counted in timestral_user_series_total
        self.results: list[np.ndarray | None] = [None] * len(series)
        self.remaining = len(series)
        self.error: tuple[int, str] | None = None  # (HTTP status, message)
        self.done = threading.Event()
        self.submitted_at = time.monotonic()
        self.started_at: float | None = None
        self.batch_ids: set[int] = set()
        self.compute_seconds = 0.0  # model time of every batch this request took part in

    def _fail(self, status: int, message: str) -> None:
        if self.error is None:
            self.error = (status, message)
        self.done.set()


class Job:
    __slots__ = ("pending", "slot", "series", "values")

    def __init__(self, pending: Pending, slot: int, series: Series):
        self.pending, self.slot, self.series = pending, slot, series
        self.values = series.values


def pad_batch(series: list[Series]) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Stack same-layout series, left-padding shorter contexts (and their covariates) with NaN.

    Returns context `(B, 1, T)`, known covariates `(B, C_known, T + H)` or None,
    and past covariates `(B, C_past, T)` or None, T being the longest context.
    """
    layout = series[0].layout
    b, t, h = len(series), max(s.length for s in series), layout.prediction_length
    context = np.full((b, 1, t), np.nan, dtype=np.float32)
    known = np.full((b, layout.n_known, t + h), np.nan, dtype=np.float32) if layout.n_known else None
    past = np.full((b, layout.n_past, t), np.nan, dtype=np.float32) if layout.n_past else None
    for i, s in enumerate(series):
        pad = t - s.length
        context[i, 0, pad:] = s.context
        if known is not None:
            known[i, :, pad:] = s.known
        if past is not None:
            past[i, :, pad:] = s.past
    return context, known, past


def batch_cells(layout: Layout, n_series: int, longest: int) -> int:
    return n_series * layout.rows * (longest + layout.prediction_length)


class Batcher:
    def __init__(self, forecaster, cfg: Batching, metrics: Metrics | None = None, on_fatal=None):
        self.forecaster = forecaster
        self.cfg = cfg
        self.metrics = metrics
        self.on_fatal = on_fatal  # called once, from the worker, with a reason
        self._queue: list[Job] = []
        self._queued_values = 0
        self._cond = threading.Condition()
        self._stop = False
        self._device_failures = 0  # consecutive model calls that failed with a CUDA error
        self._fatal = False
        self._batch_ids = itertools.count(1)
        self._thread = threading.Thread(target=self._run, name="gpu-worker", daemon=True)

    # --- request side ---------------------------------------------------------

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 30) -> None:
        """Take no more series, fail the queued ones with 503 and let the current call finish."""
        with self._cond:
            self._stop = True
            jobs, self._queue, self._queued_values = self._queue, [], 0
            self._cond.notify_all()
        for job in jobs:
            job.pending._fail(*SHUTTING_DOWN)
        if self._thread.is_alive():
            self._thread.join(timeout)

    @property
    def queue_depth(self) -> int:
        with self._cond:
            return len(self._queue)

    def submit(self, series: list[Series], user: str | None = None) -> Pending:
        """Queue a request's series; raises QueueFull when there is no room for all of them."""
        pending = Pending(series, user)
        # Shortest first within a request, so its long series end up together and
        # short ones are not padded to their length (results go back by slot).
        order = sorted(range(len(series)), key=lambda i: series[i].length)
        jobs = [Job(pending, i, series[i]) for i in order]
        values = sum(job.values for job in jobs)
        with self._cond:
            if self._stop:
                raise ShuttingDown()
            if (len(self._queue) + len(jobs) > self.cfg.max_queue_series
                    or self._queued_values + values > self.cfg.max_queue_values):
                raise QueueFull()
            self._queue.extend(jobs)
            self._queued_values += values
            self._cond.notify_all()
        return pending

    def cancel(self, pending: Pending) -> None:
        """Drop a request's queued series (it timed out, failed or its client left)."""
        with self._cond:
            self._set_queue([job for job in self._queue if job.pending is not pending])

    def _set_queue(self, jobs: list[Job]) -> None:  # with self._cond held
        self._queue = jobs
        self._queued_values = sum(job.values for job in jobs)

    # --- worker side ----------------------------------------------------------

    def _take(self) -> list[Job]:
        """The next batch: the oldest series plus same-layout companions, within the caps.

        Companions are the queued series closest to the oldest one in length
        (oldest first among equals), so concurrent requests with mixed lengths
        are not each padded to their longest series. The oldest series always
        goes, so nothing waits for more than its turn.
        """
        first = self._queue[0]
        layout, target = first.series.layout, first.series.length
        candidates = sorted((job for job in self._queue[1:] if job.series.layout == layout),
                            key=lambda job: abs(job.series.length - target))
        chosen, longest = [first], target
        for job in candidates:
            if len(chosen) == self.cfg.max_batch_series:
                break
            new_longest = max(longest, job.series.length)
            if batch_cells(layout, len(chosen) + 1, new_longest) > self.cfg.max_batch_cells:
                continue  # a shorter one further on may still fit
            chosen.append(job)
            longest = new_longest
        taken = set(map(id, chosen))
        self._set_queue([job for job in self._queue if id(job) not in taken])
        return chosen

    def _run(self) -> None:
        wait = self.cfg.batch_wait_ms / 1000
        while True:
            with self._cond:
                while not self._queue and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                # Give concurrent callers a few ms to add series, unless a full batch is waiting.
                deadline = time.monotonic() + wait
                while len(self._queue) < self.cfg.max_batch_series and not self._stop:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        break
                    self._cond.wait(left)
                if self._stop or not self._queue:
                    continue
                jobs = self._take()
            try:
                self._process(jobs)
            except Exception:  # never let the worker die
                log.exception("batch failed")
                for job in jobs:
                    self.cancel(job.pending)
                    job.pending._fail(500, "internal error during forecast")

    def _process(self, jobs: list[Job]) -> None:
        live = [job for job in jobs if job.pending.error is None]
        if not live:
            return
        try:
            self._call(live)
        except Exception as exc:
            if len(live) == 1:
                self._fail_one(live[0], exc)
                return
            # One bad series (or a transient out-of-memory) must not fail the
            # other callers in the batch: retry each series on its own.
            log.warning("batch of %d failed (%s); retrying series one by one", len(live), _short(exc))
            for job in live:
                if job.pending.error is None:
                    try:
                        self._call([job])
                    except Exception as exc_one:
                        self._fail_one(job, exc_one)

    def _call(self, jobs: list[Job]) -> None:
        """One model call, counting consecutive CUDA failures (see FATAL_FAILURES)."""
        try:
            self._forecast(jobs)
        except Exception as exc:
            if _device_fault(exc):
                self._device_failures += 1
                if self._device_failures >= FATAL_FAILURES and not self._fatal:
                    self._fatal = True
                    reason = f"{self._device_failures} model calls in a row failed: {_short(exc)}"
                    log.error("GPU worker: %s; the CUDA context looks unusable", reason)
                    if self.on_fatal is not None:
                        self.on_fatal(reason)
            raise
        self._device_failures = 0

    def _fail_one(self, job: Job, exc: Exception) -> None:
        self.cancel(job.pending)  # the request has failed; its other series need not run
        where = f"series[{job.series.index}]"
        if isinstance(exc, ValueError):  # timestral's verdict on the input itself
            job.pending._fail(422, f"{where}: {_short(exc)}")
        else:
            log.error("forecast of %s failed: %s", where, _short(exc), exc_info=exc)
            job.pending._fail(500, f"{where}: forecast failed on the server")

    def _forecast(self, jobs: list[Job]) -> None:
        layout = jobs[0].series.layout
        context, known, past = pad_batch([job.series for job in jobs])
        batch_id = next(self._batch_ids)
        start = time.monotonic()
        for job in jobs:
            if job.pending.started_at is None:
                job.pending.started_at = start
        forecast = self.forecaster.forecast(
            context, layout.prediction_length, known_covariates=known, past_covariates=past,
            symmetric_averaging=layout.symmetric_averaging, clip_nonnegative=layout.clip_nonnegative)
        quantiles = forecast.quantiles[:, 0].float().cpu().numpy()  # (B, H, Q); waits for the GPU
        seconds = time.monotonic() - start
        if self.metrics is not None:
            self.metrics.record_batch(seconds, [job.pending.user for job in jobs])
        for job, q in zip(jobs, quantiles):
            pending = job.pending
            if pending.error is not None:
                continue
            pending.results[job.slot] = q
            if batch_id not in pending.batch_ids:
                pending.batch_ids.add(batch_id)
                pending.compute_seconds += seconds
            pending.remaining -= 1
            if pending.remaining == 0:
                pending.done.set()


def _device_fault(exc: Exception) -> bool:
    """A CUDA error that may have broken the context, as opposed to a bad input
    (ValueError) or an allocation the caching allocator could not satisfy
    (torch.OutOfMemoryError, which leaves the context usable)."""
    if isinstance(exc, ValueError) or type(exc).__name__ == "OutOfMemoryError":
        return False
    return type(exc).__name__ == "AcceleratorError" or "cuda" in str(exc).lower()


def _short(exc: Exception) -> str:
    text = str(exc).strip().splitlines()
    return text[0] if text else type(exc).__name__
