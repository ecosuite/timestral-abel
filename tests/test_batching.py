import threading
import time
import unittest

import numpy as np

from timestral_abel import batching
from timestral_abel.batching import FATAL_FAILURES, Batcher, QueueFull, ShuttingDown
from timestral_abel.config import Batching
from timestral_abel.request import Layout, Series
from helpers import FakeForecaster, tiny_forecaster

H = 12


def make(index, length, n_known=0, n_past=0, h=H, sym=True, clip=True, seed=None, value=None):
    rng = np.random.default_rng(index if seed is None else seed)
    context = (rng.random(length, dtype=np.float32) * 5 + np.sin(np.arange(length) / 6)).astype(np.float32)
    if value is not None:
        context[-1] = value
    context[rng.random(length) < 0.05] = np.nan  # some missing values
    context[-1] = context[-1] if np.isfinite(context[-1]) else 1.0
    known = rng.random((n_known, length + h), dtype=np.float32) if n_known else None
    past = rng.random((n_past, length), dtype=np.float32) if n_past else None
    return Series(index, index, Layout(n_known, n_past, h, sym, clip), context, known, past)


def run(forecaster, series_lists, **cfg):
    """Submit each list as one request before the worker starts; return the results."""
    batcher = Batcher(forecaster, Batching(**{"batch_wait_ms": 0, **cfg}))
    pendings = [batcher.submit(s) for s in series_lists]
    batcher.start()
    try:
        for p in pendings:
            assert p.done.wait(60), "timed out"
    finally:
        batcher.stop(5)
    return pendings


class PadTest(unittest.TestCase):
    def test_left_pads_mixed_lengths(self):
        a, b = make(0, 5, n_known=1, n_past=2), make(1, 3, n_known=1, n_past=2)
        context, known, past = batching.pad_batch([a, b])
        self.assertEqual(context.shape, (2, 1, 5))
        self.assertEqual(known.shape, (2, 1, 5 + H))
        self.assertEqual(past.shape, (2, 2, 5))
        np.testing.assert_array_equal(context[0, 0], a.context)
        self.assertTrue(np.isnan(context[1, 0, :2]).all())
        np.testing.assert_array_equal(context[1, 0, 2:], b.context)
        self.assertTrue(np.isnan(known[1, :, :2]).all())
        np.testing.assert_array_equal(known[1, :, 2:], b.known)
        self.assertTrue(np.isnan(past[1, :, :2]).all())
        np.testing.assert_array_equal(past[1, :, 2:], b.past)

    def test_no_covariates(self):
        context, known, past = batching.pad_batch([make(0, 4)])
        self.assertEqual(context.shape, (1, 1, 4))
        self.assertIsNone(known)
        self.assertIsNone(past)


class GroupingTest(unittest.TestCase):
    def test_groups_by_layout(self):
        fake = FakeForecaster()
        series = [make(0, 40), make(1, 40, n_known=2), make(2, 30), make(3, 40, n_known=2, n_past=1),
                  make(4, 20, n_known=2), make(5, 40, h=7), make(6, 40, sym=False), make(7, 25)]
        (p,) = run(fake, [series])
        self.assertIsNone(p.error)
        # (batch, padded length, known, past, horizon): one call per distinct layout.
        self.assertCountEqual(fake.calls, [(3, 40, 0, 0, H), (2, 40, 2, 0, H), (1, 40, 2, 1, H),
                                           (1, 40, 0, 0, 7), (1, 40, 0, 0, H)])
        for s, q in zip(series, p.results):
            self.assertEqual(q.shape, (s.layout.prediction_length, 21))
            last = s.context[np.isfinite(s.context)][-1]
            self.assertAlmostEqual(float(q[0, 10]), float(last) + 0.5, places=5)  # the right series
        self.assertEqual(len(p.batch_ids), 5)

    def test_series_cap(self):
        fake = FakeForecaster()
        run(fake, [[make(i, 10) for i in range(7)]], max_batch_series=3)
        self.assertEqual([c[0] for c in fake.calls], [3, 3, 1])

    def test_cell_cap(self):
        fake = FakeForecaster()
        # rows 1 x (100 + 12) = 112 cells per series; 250 cells fit two.
        run(fake, [[make(i, 100) for i in range(5)]], max_batch_cells=250)
        self.assertEqual([c[0] for c in fake.calls], [2, 2, 1])
        # A single series larger than the cap still runs, alone.
        fake = FakeForecaster()
        run(fake, [[make(0, 100), make(1, 100)]], max_batch_cells=50)
        self.assertEqual([c[0] for c in fake.calls], [1, 1])

    def test_requests_share_batches(self):
        fake = FakeForecaster()
        p1, p2 = run(fake, [[make(0, 30), make(1, 30)], [make(2, 30)]])
        self.assertEqual(fake.calls, [(3, 30, 0, 0, H)])
        self.assertEqual(p1.batch_ids, p2.batch_ids)

    def test_long_series_padding_is_bounded(self):
        fake = FakeForecaster()
        run(fake, [[make(0, 500), make(1, 10), make(2, 12)]], max_batch_cells=3 * (12 + H) + 1)
        # Shortest first: the two short ones share a call, the long one goes alone.
        self.assertEqual(sorted(fake.calls), [(1, 500, 0, 0, H), (2, 12, 0, 0, H)])

    def test_concurrent_requests_batch_by_length(self):
        fake = FakeForecaster()
        # Two requests of mixed lengths: the short series of both share a call
        # instead of each request's call being padded to its 200-step series.
        p1, p2 = run(fake, [[make(0, 200), make(1, 10), make(2, 200), make(3, 10)],
                            [make(4, 10), make(5, 200), make(6, 10), make(7, 200)]], max_batch_series=4)
        self.assertEqual(fake.calls, [(4, 10, 0, 0, H), (4, 200, 0, 0, H)])
        self.assertIsNone(p1.error)
        self.assertIsNone(p2.error)
        for p in (p1, p2):
            for s, q in zip(p.series, p.results):
                last = s.context[np.isfinite(s.context)][-1]
                self.assertAlmostEqual(float(q[0, 10]), float(last) + 0.5, places=5)

    def test_cell_cap_skips_to_shorter_series(self):
        fake = FakeForecaster()
        # Oldest is 50 steps; the 60-step series would break the cap, the 40-step one fits.
        run(fake, [[make(0, 50)], [make(1, 60)], [make(2, 40)]], max_batch_cells=2 * (50 + H) + 1)
        self.assertEqual(fake.calls, [(2, 50, 0, 0, H), (1, 60, 0, 0, H)])

    def test_bad_series_fails_only_its_request(self):
        fake = FakeForecaster()
        with self.assertLogs("timestral_abel.batching", "WARNING"):  # "retrying series one by one"
            good, bad = run(fake, [[make(0, 20), make(1, 20)], [make(2, 20, value=666.0), make(3, 20)]])
        self.assertIsNone(good.error)
        self.assertEqual(good.remaining, 0)
        self.assertEqual(bad.error[0], 422)
        self.assertIn("series[2]", bad.error[1])  # Series.index, its place in the request

    def test_queue_full_and_cancel(self):
        gate = threading.Event()
        batcher = Batcher(FakeForecaster(gate), Batching(max_queue_series=5, max_batch_series=2,
                                                         batch_wait_ms=0))
        batcher.start()
        try:
            first = batcher.submit([make(i, 10) for i in range(4)])
            with self.assertRaises(QueueFull):
                batcher.submit([make(i, 10) for i in range(4)])
            self.assertGreaterEqual(batcher.queue_depth, 2)
            batcher.cancel(first)
            self.assertEqual(batcher.queue_depth, 0)
            second = batcher.submit([make(i, 10) for i in range(5)])
            gate.set()
            self.assertTrue(second.done.wait(10))
            self.assertIsNone(second.error)
        finally:
            gate.set()
            batcher.stop(5)


class BoundsAndShutdownTest(unittest.TestCase):
    def test_queue_bounded_by_values(self):
        gate = threading.Event()
        # Each 100-step series with 2 known rows holds 100 + 2 x (100 + H) values.
        per_series = 100 + 2 * (100 + H)
        batcher = Batcher(FakeForecaster(gate), Batching(max_queue_values=3 * per_series, batch_wait_ms=0))
        try:
            first = batcher.submit([make(i, 100, n_known=2) for i in range(2)])
            with self.assertRaises(QueueFull):
                batcher.submit([make(i, 100, n_known=2) for i in range(2)])
            batcher.submit([make(9, 100, n_known=2)])  # exactly at the bound
            batcher.cancel(first)  # cancelling gives the room back
            batcher.submit([make(i, 100, n_known=2) for i in range(2)])
        finally:
            gate.set()
            batcher.stop(5)

    def test_stop_fails_queued_requests_with_503(self):
        gate = threading.Event()
        batcher = Batcher(FakeForecaster(gate), Batching(max_batch_series=1, batch_wait_ms=0))
        batcher.start()
        running = batcher.submit([make(0, 10)])
        deadline = time.time() + 5
        while batcher.queue_depth and time.time() < deadline:  # the worker has taken it
            time.sleep(0.01)
        queued = batcher.submit([make(1, 10), make(2, 10)])
        stopper = threading.Thread(target=batcher.stop, args=(10,))
        stopper.start()
        self.assertTrue(queued.done.wait(5))
        self.assertEqual(queued.error[0], 503)
        gate.set()  # the call in progress still completes
        stopper.join(10)
        self.assertTrue(running.done.is_set())
        self.assertIsNone(running.error)
        with self.assertRaises(ShuttingDown):
            batcher.submit([make(3, 10)])

    def test_repeated_cuda_errors_are_fatal(self):
        class Broken(FakeForecaster):
            def forecast(self, *args, **kw):
                raise RuntimeError("CUDA error: unspecified launch failure")
        fatal = []
        batcher = Batcher(Broken(), Batching(batch_wait_ms=0), on_fatal=fatal.append)
        with self.assertLogs("timestral_abel.batching", "ERROR"):
            pendings = [batcher.submit([make(i, 10)]) for i in range(FATAL_FAILURES + 2)]
            batcher.start()
            try:
                for p in pendings:
                    self.assertTrue(p.done.wait(10))
            finally:
                batcher.stop(5)
        self.assertEqual(len(fatal), 1)  # reported once
        self.assertIn("unspecified launch failure", fatal[0])
        self.assertTrue(all(p.error[0] == 500 for p in pendings))

    def test_input_errors_and_ooms_are_not_fatal(self):
        class OutOfMemoryError(RuntimeError):
            pass

        class Flaky(FakeForecaster):
            def forecast(self, *args, **kw):
                raise OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB")
        fatal = []
        batcher = Batcher(Flaky(), Batching(batch_wait_ms=0), on_fatal=fatal.append)
        with self.assertLogs("timestral_abel.batching", "ERROR"):
            pendings = [batcher.submit([make(i, 10)]) for i in range(FATAL_FAILURES + 2)]
            batcher.start()
            try:
                for p in pendings:
                    self.assertTrue(p.done.wait(10))
            finally:
                batcher.stop(5)
        self.assertEqual(fatal, [])
        fake = FakeForecaster()
        batcher = Batcher(fake, Batching(batch_wait_ms=0), on_fatal=fatal.append)
        with self.assertLogs("timestral_abel.batching", "WARNING"):  # "retrying series one by one"
            pendings = [batcher.submit([make(i, 10, value=666.0)]) for i in range(FATAL_FAILURES + 2)]
            batcher.start()
            try:
                for p in pendings:
                    self.assertTrue(p.done.wait(10))
            finally:
                batcher.stop(5)
        self.assertEqual(fatal, [])
        self.assertTrue(all(p.error[0] == 422 for p in pendings))


class InvarianceTest(unittest.TestCase):
    """Real (tiny, random) Timestral: results must not depend on how series are batched."""

    @classmethod
    def setUpClass(cls):
        cls.fc = tiny_forecaster()

    def check(self, series, atol=1e-4):
        together = run(self.fc, [series])[0].results
        alone = [p.results[0] for p in run(self.fc, [[s] for s in series], max_batch_series=1)]
        for cap in (2, 3, 16):
            with self.subTest(max_batch_series=cap):
                capped = run(self.fc, [series], max_batch_series=cap)[0].results
                for a, b in zip(capped, alone):
                    np.testing.assert_allclose(a, b, rtol=1e-4, atol=atol)
        for a, b in zip(together, alone):
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=atol)
        # And the direct, unpadded library call agrees with the service's answer.
        for s, a in zip(series, alone):
            direct = self.fc.forecast(s.context[None, None], s.layout.prediction_length,
                                      known_covariates=None if s.known is None else s.known[None],
                                      past_covariates=None if s.past is None else s.past[None],
                                      symmetric_averaging=s.layout.symmetric_averaging,
                                      clip_nonnegative=s.layout.clip_nonnegative)
            np.testing.assert_allclose(a, direct.quantiles[0, 0].numpy(), rtol=1e-4, atol=atol)

    def test_mixed_lengths_no_covariates(self):
        self.check([make(i, n) for i, n in enumerate([64, 37, 100, 9, 64, 81])])

    def test_mixed_lengths_with_covariates(self):
        self.check([make(i, n, n_known=2, n_past=1) for i, n in enumerate([50, 77, 64, 13, 90])])

    def test_multi_block_horizon(self):
        # Block size 4 patches x 8 steps = 32: a 70-step horizon takes three blocks.
        self.check([make(i, n, n_known=1, h=70) for i, n in enumerate([40, 96, 61])])

    def test_flags_off(self):
        self.check([make(i, n, sym=False, clip=False) for i, n in enumerate([30, 55, 80])])

    def test_concurrent_requests(self):
        series = [make(i, n, n_known=1) for i, n in enumerate([30, 45, 60, 75])]
        alone = [p.results[0] for p in run(self.fc, [[s] for s in series], max_batch_series=1)]
        batcher = Batcher(self.fc, Batching(batch_wait_ms=20))
        batcher.start()
        try:
            results = [None] * 4

            def call(i):
                p = batcher.submit([series[i]])
                p.done.wait(60)
                results[i] = p.results[0]

            threads = [threading.Thread(target=call, args=(i,)) for i in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(60)
        finally:
            batcher.stop(5)
        for a, b in zip(results, alone):
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
