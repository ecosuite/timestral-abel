"""Parse and validate a POST /v1/forecast body into arrays, and encode the response.

Each series is univariate: one target row `context` of T steps, plus optional
known-future covariate rows (T + H steps each) and past-only covariate rows (T
steps each). null, NaN and +-Infinity all mean "missing". Different series in
one request may have different context lengths and covariate counts.
"""

import json
from dataclasses import dataclass

import numpy as np

from .config import Limits

TOP_KEYS = {"prediction_length", "series", "quantiles", "symmetric_averaging", "clip_nonnegative"}
SERIES_KEYS = {"id", "context", "known_covariates", "past_covariates"}
NUMBER_TYPES = {float, int, type(None)}
F32_MAX = float(np.finfo(np.float32).max)


class BadRequest(ValueError):
    """A client error; the message is returned to the caller as is."""


@dataclass(frozen=True)
class Layout:
    """Series with the same layout can share one model call."""

    n_known: int
    n_past: int
    prediction_length: int
    symmetric_averaging: bool
    clip_nonnegative: bool

    @property
    def rows(self) -> int:
        return 1 + self.n_known + self.n_past


@dataclass
class Series:
    index: int  # position in the request
    id: object
    layout: Layout
    context: np.ndarray  # (T,) float32, NaN = missing
    known: np.ndarray | None  # (n_known, T + H) float32
    past: np.ndarray | None  # (n_past, T) float32

    @property
    def length(self) -> int:
        return self.context.shape[0]

    @property
    def values(self) -> int:
        """Numbers held for this series (what it costs in the queue)."""
        return sum(a.size for a in (self.context, self.known, self.past) if a is not None)


@dataclass
class ForecastRequest:
    prediction_length: int
    series: list[Series]
    quantiles: tuple[float, ...]  # levels to return, in the model's order


def _vector(values, where: str, length: int | None = None) -> np.ndarray:
    """A JSON list of numbers/nulls as float32 with NaN for missing values."""
    if not isinstance(values, list):
        raise BadRequest(f"{where} must be a list of numbers or null")
    if length is not None and len(values) != length:
        raise BadRequest(f"{where} must have {length} values, got {len(values)}")
    # A type check in C over the whole list; bool, str and nested lists are refused.
    if not set(map(type, values)) <= NUMBER_TYPES:
        raise BadRequest(f"{where} must hold only numbers or null")
    try:
        wide = np.array(values, dtype=np.float64)
    except OverflowError:  # an integer beyond float64
        raise BadRequest(f"{where} holds a value outside the float32 range") from None
    finite = np.isfinite(wide)
    if np.any(np.abs(wide[finite]) > F32_MAX):
        raise BadRequest(f"{where} holds a value outside the float32 range")
    return np.where(finite, wide, np.nan).astype(np.float32)


def _rows(values, where: str, length: int) -> np.ndarray | None:
    if values is None:
        return None
    if not isinstance(values, list) or not all(isinstance(row, list) for row in values):
        raise BadRequest(f"{where} must be a list of rows (lists of numbers or null)")
    if not values:
        return None
    return np.stack([_vector(row, f"{where}[{i}]", length) for i, row in enumerate(values)])


def _flag(body: dict, name: str) -> bool:
    value = body.get(name, True)
    if not isinstance(value, bool):
        raise BadRequest(f"{name} must be true or false")
    return value


def parse(body: bytes, limits: Limits, levels: tuple[float, ...], known_supported: bool = True,
          covariates_supported: bool = True) -> ForecastRequest:
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError) as exc:
        raise BadRequest(f"body is not valid JSON: {exc}") from None
    except RecursionError:
        raise BadRequest("body is not valid JSON: nested too deeply") from None
    if not isinstance(data, dict):
        raise BadRequest("body must be a JSON object")
    unknown = set(data) - TOP_KEYS
    if unknown:
        raise BadRequest(f"unknown field(s): {', '.join(sorted(unknown))}")

    h = data.get("prediction_length")
    if not isinstance(h, int) or isinstance(h, bool) or not 1 <= h <= limits.max_prediction_length:
        raise BadRequest(f"prediction_length must be an integer from 1 to {limits.max_prediction_length}")
    sym, clip = _flag(data, "symmetric_averaging"), _flag(data, "clip_nonnegative")

    quantiles = levels
    if data.get("quantiles") is not None:
        wanted = data["quantiles"]
        if not isinstance(wanted, list) or not wanted:
            raise BadRequest("quantiles must be a non-empty list of quantile levels")
        picked = set()
        for q in wanted:
            match = [lv for lv in levels if isinstance(q, (int, float)) and not isinstance(q, bool)
                     and abs(lv - q) < 1e-9]
            if not match:
                raise BadRequest(f"quantile {q!r} is not one of the model's levels {list(levels)}")
            picked.add(match[0])
        quantiles = tuple(lv for lv in levels if lv in picked)

    raw = data.get("series")
    if not isinstance(raw, list) or not raw:
        raise BadRequest("series must be a non-empty list")
    if len(raw) > limits.max_series_per_request:
        raise BadRequest(f"at most {limits.max_series_per_request} series per request, got {len(raw)}")
    # The response holds the median plus each quantile asked for, per step.
    output = len(raw) * h * (1 + len(quantiles))
    if output > limits.max_output_values:
        raise BadRequest(f"the response would hold {output} values (series x prediction_length x "
                         f"(1 + quantiles)); at most {limits.max_output_values}: ask for fewer "
                         "quantiles or split the request")

    series = []
    for i, item in enumerate(raw):
        where = f"series[{i}]"
        if not isinstance(item, dict):
            raise BadRequest(f"{where} must be an object")
        unknown = set(item) - SERIES_KEYS
        if unknown:
            raise BadRequest(f"{where}: unknown field(s): {', '.join(sorted(unknown))}")
        sid = item.get("id", i)
        if not isinstance(sid, (str, int)) or isinstance(sid, bool):
            raise BadRequest(f"{where}.id must be a string or an integer")
        context = _vector(item.get("context"), f"{where}.context")
        t = context.shape[0]
        if not 1 <= t <= limits.max_context:
            raise BadRequest(f"{where}.context must have 1 to {limits.max_context} values, got {t}")
        if not np.isfinite(context).any():
            raise BadRequest(f"{where}.context has no observed (finite) values")
        known = _rows(item.get("known_covariates"), f"{where}.known_covariates", t + h)
        past = _rows(item.get("past_covariates"), f"{where}.past_covariates", t)
        n_known = 0 if known is None else known.shape[0]
        n_past = 0 if past is None else past.shape[0]
        if n_known + n_past > limits.max_covariates:
            raise BadRequest(f"{where}: at most {limits.max_covariates} covariate rows, got {n_known + n_past}")
        if n_known and not known_supported:
            raise BadRequest("this model cannot use known_covariates")
        if (n_known or n_past) and not covariates_supported:
            raise BadRequest("this model cannot use covariates")
        series.append(Series(i, sid, Layout(n_known, n_past, h, sym, clip), context, known, past))
    return ForecastRequest(h, series, quantiles)


# --- response -----------------------------------------------------------------

def _numbers(values: np.ndarray) -> str:
    """A JSON array body: 7 significant digits (float32 precision), null for non-finite."""
    text = np.char.mod("%.7g", values.astype(np.float64))
    text = np.where(np.isfinite(values), text, "null")
    return ",".join(text.tolist())


def encode(header: dict, req: ForecastRequest, results: list[np.ndarray], levels: tuple[float, ...],
           trailer: dict) -> list[bytes]:
    """The response JSON as consecutive chunks: `header` keys, quantile_levels,
    forecasts, then `trailer` keys. `results[i]` is the `(H, Q)` quantile array
    of series i, Q = `levels`.

    Built by hand around json.dumps so a large batch of floats is formatted by
    numpy instead of one Python float at a time, and kept as one chunk per
    series so the response is never also held as a single joined copy.
    """
    pick = [levels.index(q) for q in req.quantiles]
    median = levels.index(0.5)
    head = json.dumps({**header, "quantile_levels": list(req.quantiles)}, separators=(",", ":"))
    chunks = [(head[:-1] + ',"forecasts":[').encode()]
    for i, (s, q) in enumerate(zip(req.series, results)):
        quantiles = ",".join(f'"{levels[j]}":[{_numbers(q[:, j])}]' for j in pick)
        chunks.append(f'{"," if i else ""}{{"id":{json.dumps(s.id)},"median":[{_numbers(q[:, median])}],'
                      f'"quantiles":{{{quantiles}}}}}'.encode())
    tail = "," + json.dumps(trailer, separators=(",", ":"))[1:] if trailer else "}"
    chunks.append(("]" + tail).encode())
    return chunks
