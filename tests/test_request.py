import json
import math
import unittest

import numpy as np

from timestral_abel import request
from timestral_abel.config import Limits
from helpers import LEVELS

LIMITS = Limits(max_series_per_request=4, max_context=100, max_prediction_length=10, max_covariates=2)


def parse(body, **kw):
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    return request.parse(raw, LIMITS, LEVELS, **kw)


class ParseTest(unittest.TestCase):
    def test_valid_request(self):
        req = parse({"prediction_length": 3, "quantiles": [0.9, 0.1, 0.5],
                     "series": [{"id": "a", "context": [1, None, 2.5],
                                 "known_covariates": [[0, 1, 0, 1, 0, 1]], "past_covariates": [[1, 2, 3]]},
                                {"context": [4.0, 5.0]}]})
        self.assertEqual(req.prediction_length, 3)
        self.assertEqual(req.quantiles, (0.1, 0.5, 0.9))  # model order
        a, b = req.series
        self.assertEqual(a.id, "a")
        self.assertEqual(b.id, 1)  # defaults to the index
        self.assertEqual(a.context.dtype, np.float32)
        self.assertTrue(np.isnan(a.context[1]))
        self.assertEqual(a.known.shape, (1, 6))
        self.assertEqual(a.past.shape, (1, 3))
        self.assertEqual(a.layout, request.Layout(1, 1, 3, True, True))
        self.assertEqual(b.layout, request.Layout(0, 0, 3, True, True))
        self.assertEqual(a.layout.rows, 3)

    def test_non_finite_tokens_are_missing(self):
        req = parse(b'{"prediction_length": 1, "series": [{"context": [NaN, Infinity, -Infinity, 1]}]}')
        self.assertEqual(int(np.isnan(req.series[0].context).sum()), 3)

    def test_flags(self):
        req = parse({"prediction_length": 1, "symmetric_averaging": False, "clip_nonnegative": False,
                     "series": [{"context": [1]}]})
        self.assertEqual(req.series[0].layout, request.Layout(0, 0, 1, False, False))

    def test_errors(self):
        ok = {"context": [1.0, 2.0]}
        cases = {
            b"not json": "not valid JSON",
            b"[1]": "JSON object",
            json.dumps({"prediction_length": 1, "series": [ok], "extra": 1}): "unknown field",
            json.dumps({"series": [ok]}): "prediction_length",
            json.dumps({"prediction_length": 0, "series": [ok]}): "prediction_length",
            json.dumps({"prediction_length": 11, "series": [ok]}): "prediction_length",
            json.dumps({"prediction_length": True, "series": [ok]}): "prediction_length",
            json.dumps({"prediction_length": 1.5, "series": [ok]}): "prediction_length",
            json.dumps({"prediction_length": 1, "series": []}): "non-empty",
            json.dumps({"prediction_length": 1, "series": [ok] * 5}): "at most 4 series",
            json.dumps({"prediction_length": 1, "series": [1]}): "series[0] must be an object",
            json.dumps({"prediction_length": 1, "series": [{"context": [1], "known": []}]}): "unknown field",
            json.dumps({"prediction_length": 1, "series": [{}]}): "series[0].context must be a list",
            json.dumps({"prediction_length": 1, "series": [{"context": []}]}): "1 to 100 values",
            json.dumps({"prediction_length": 1, "series": [{"context": [1] * 101}]}): "1 to 100 values",
            json.dumps({"prediction_length": 1, "series": [{"context": [None, None]}]}): "no observed",
            json.dumps({"prediction_length": 1, "series": [{"context": ["1"]}]}): "only numbers",
            json.dumps({"prediction_length": 1, "series": [{"context": [True]}]}): "only numbers",
            json.dumps({"prediction_length": 1, "series": [{"context": [[1]]}]}): "only numbers",
            json.dumps({"prediction_length": 1, "series": [{"context": [1e39]}]}): "float32 range",
            json.dumps({"prediction_length": 1, "series": [{"context": [10 ** 400]}]}): "float32 range",
            json.dumps({"prediction_length": 1, "series": [{"id": [1], "context": [1]}]}): ".id",
            json.dumps({"prediction_length": 2, "series": [{"context": [1, 2],
                        "known_covariates": [[1, 2, 3]]}]}): "series[0].known_covariates[0] must have 4 values",
            json.dumps({"prediction_length": 2, "series": [{"context": [1, 2],
                        "past_covariates": [[1, 2, 3]]}]}): "series[0].past_covariates[0] must have 2 values",
            json.dumps({"prediction_length": 1, "series": [{"context": [1],
                        "past_covariates": [1]}]}): "list of rows",
            json.dumps({"prediction_length": 1, "series": [{"context": [1],
                        "past_covariates": [[1], [1], [1]]}]}): "at most 2 covariate rows",
            json.dumps({"prediction_length": 1, "quantiles": [0.33], "series": [ok]}): "not one of",
            json.dumps({"prediction_length": 1, "quantiles": [], "series": [ok]}): "quantiles",
            json.dumps({"prediction_length": 1, "symmetric_averaging": 1, "series": [ok]}): "true or false",
        }
        for body, needle in cases.items():
            with self.subTest(body=body[:80]):
                with self.assertRaises(request.BadRequest) as ctx:
                    parse(body.encode() if isinstance(body, str) else body)
                self.assertIn(needle, str(ctx.exception))

    def test_deep_nesting_is_a_client_error(self):
        with self.assertRaisesRegex(request.BadRequest, "nested too deeply"):
            parse(b'{"series":' + b"[" * 100_000)

    def test_response_size_cap(self):
        limits = Limits(max_series_per_request=10, max_prediction_length=100, max_output_values=2000)
        body = lambda n, q=None: json.dumps({"prediction_length": 100, "quantiles": q,
                                             "series": [{"context": [1]}] * n}).encode()
        # 2 series x 100 steps x (1 + 9 quantiles) fits; with all 21 quantiles it does not.
        request.parse(body(2, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]), limits, LEVELS)
        with self.assertRaisesRegex(request.BadRequest, "response would hold 4400 values"):
            request.parse(body(2), limits, LEVELS)

    def test_capabilities(self):
        body = {"prediction_length": 1, "series": [{"context": [1], "known_covariates": [[1, 2]]}]}
        with self.assertRaisesRegex(request.BadRequest, "known_covariates"):
            parse(body, known_supported=False)
        with self.assertRaisesRegex(request.BadRequest, "covariates"):
            parse(body, covariates_supported=False)


class EncodeTest(unittest.TestCase):
    def test_nan_becomes_null_and_json_is_valid(self):
        req = parse({"prediction_length": 2, "quantiles": [0.1, 0.5],
                     "series": [{"id": "x", "context": [1]}, {"context": [2]}]})
        q = np.tile(np.arange(len(LEVELS), dtype=np.float32), (2, 1))
        q[1, 10] = np.nan
        q[0, 2] = np.inf
        other = np.full((2, len(LEVELS)), 1 / 3, dtype=np.float32)
        out = json.loads(b"".join(request.encode({"model": "m", "revision": "r"}, req, [q, other], LEVELS,
                                                 {"timing": {"total_seconds": 1.5}})))
        self.assertEqual(list(out), ["model", "revision", "quantile_levels", "forecasts", "timing"])
        self.assertEqual(out["quantile_levels"], [0.1, 0.5])
        first, second = out["forecasts"]
        self.assertEqual(first["id"], "x")
        self.assertEqual(second["id"], 1)
        self.assertEqual(first["median"], [10, None])
        self.assertEqual(first["quantiles"], {"0.1": [None, 2], "0.5": [10, None]})
        self.assertTrue(math.isclose(second["median"][0], 1 / 3, rel_tol=1e-6))
        self.assertEqual(out["timing"], {"total_seconds": 1.5})

    def test_empty_trailer(self):
        req = parse({"prediction_length": 1, "series": [{"context": [1]}]})
        out = json.loads(b"".join(request.encode({"model": "m"}, req, [np.zeros((1, len(LEVELS)))], LEVELS, {})))
        self.assertEqual(len(out["forecasts"][0]["quantiles"]), len(LEVELS))


if __name__ == "__main__":
    unittest.main()
