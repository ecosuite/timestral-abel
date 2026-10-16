"""The real Timestral-5M on CPU, if it is already in the Hugging Face cache.

Skipped otherwise (the tests never download). To enable it:
    uv run python -c "from huggingface_hub import snapshot_download as d; d('mistralai/Timestral-5M')"
"""

import unittest

import numpy as np

from timestral_abel.batching import Batcher
from timestral_abel.config import Batching, Model
from timestral_abel.request import Layout, Series

try:
    from timestral_abel import model as service_model

    FORECASTER = service_model.load(Model(id="mistralai/Timestral-5M", revision="", cache_dir=None,
                                          local_files_only=True, device="cpu"))
except Exception as exc:  # not cached (or no huggingface_hub cache at all)
    FORECASTER, SKIP = None, f"Timestral-5M not in the local Hugging Face cache ({type(exc).__name__})"
else:
    SKIP = ""


def daily(index: int, length: int, h: int, n_known: int) -> Series:
    """A noisy daily cycle (96 steps per day) with a matching known covariate."""
    rng = np.random.default_rng(index)
    t = np.arange(length + h)
    shape = np.clip(np.sin((t % 96) / 96 * 2 * np.pi - np.pi / 2) + 0.2, 0, None).astype(np.float32)
    context = np.clip(shape[:length] * (3 + index) + rng.normal(0, 0.1, length), 0, None).astype(np.float32)
    known = np.stack([shape] * n_known) if n_known else None
    return Series(index, index, Layout(n_known, 0, h, True, True), context, known, None)


@unittest.skipIf(FORECASTER is None, SKIP)
class RealModelTest(unittest.TestCase):
    def test_batched_equals_direct(self):
        h = 96
        series = [daily(i, n, h, 1) for i, n in enumerate([960, 1440, 700, 1920])]
        batcher = Batcher(FORECASTER, Batching(batch_wait_ms=0))
        pending = batcher.submit(series)
        batcher.start()
        try:
            self.assertTrue(pending.done.wait(300))
        finally:
            batcher.stop(5)
        self.assertIsNone(pending.error)
        for s, q in zip(series, pending.results):
            direct = FORECASTER.forecast(s.context[None, None], h, known_covariates=s.known[None])
            np.testing.assert_allclose(q, direct.quantiles[0, 0].numpy(), rtol=1e-4, atol=1e-4)
            median = q[:, 10]
            # Sanity: a plausible forecast of the daily shape, non-negative (clipped).
            self.assertGreaterEqual(float(median.min()), 0)
            self.assertLess(float(np.abs(median - s.known[0, -h:] * (3 + s.index)).mean()), 1.0)


if __name__ == "__main__":
    unittest.main()
