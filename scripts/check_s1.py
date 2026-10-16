"""End-to-end check of a running service against reference scores.

Sends the day-ahead cases of an inputs file (.npz with context, t1_known,
t2_known, t2_past, actual, daylight, night and cap; not included in this
repository) to /v1/forecast in requests of many series, several in flight so
dynamic batching is exercised, rebuilds the medians and prints the nMAE of each
test with the throughput.

REFERENCE holds the scores of the same cases run directly through the timestral
library (Timestral-209M on a GPU, symmetric averaging on). A correct service
lands within about 0.05 points of them.

Usage:
    python3 scripts/check_s1.py http://localhost:18320 KEY_FILE [--inputs S1_inputs.npz]
        [--user ID] [--chunk 48] [--concurrency 2] [--limit N] [--tests T1,T2] [--no-sym]

KEY_FILE holds one API key (the first non-empty, non-# line), or is a
users.toml when --user ID is given. The key is never printed.
Needs Python 3.11+ and numpy (the service image has both).
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REFERENCE = {"T1": 12.47, "T2": 22.77}
H = 96


def read_key(path: str, user: str | None) -> str:
    """The key in a one-line key file, or users.<user>.key of a users.toml."""
    text = Path(path).read_text()
    if user:
        import tomllib

        try:
            return tomllib.loads(text)["users"][user]["key"]
        except (tomllib.TOMLDecodeError, KeyError, TypeError):
            raise SystemExit(f"{path}: no users.{user}.key") from None
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    raise SystemExit(f"{path}: no key found")


def dumps(a: np.ndarray) -> str:
    """A float32 array as a JSON list, exact (float64 repr of each float32), NaN as null."""
    return json.dumps(a.astype(np.float64).tolist()).replace("NaN", "null")


def body(z, test: str, idx: range, sym: bool) -> bytes:
    parts = []
    for i in idx:
        s = f'{{"id":{i},"context":{dumps(z["context"][i, 0])}'
        if test == "T1":
            s += ',"known_covariates":[' + ",".join(dumps(r) for r in z["t1_known"][i]) + "]"
        else:
            s += ',"known_covariates":[' + ",".join(dumps(r) for r in z["t2_known"][i]) + "]"
            s += ',"past_covariates":[' + ",".join(dumps(r) for r in z["t2_past"][i]) + "]"
        parts.append(s + "}")
    return (f'{{"prediction_length":{H},"quantiles":[0.5],"symmetric_averaging":{json.dumps(sym)},'
            f'"series":[' + ",".join(parts) + "]}").encode()


def post(url: str, key: str, data: bytes) -> dict:
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(20):
        try:
            with urllib.request.urlopen(req, timeout=1800) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as err:
            detail = err.read().decode(errors="replace")[:300]
            if err.code == 503 and attempt < 19:  # queue full / loading: honour Retry-After
                time.sleep(float(err.headers.get("Retry-After") or 5))
                continue
            raise SystemExit(f"HTTP {err.code}: {detail}") from None
    raise SystemExit("gave up after repeated 503s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url", help="service base URL, e.g. http://localhost:18320")
    ap.add_argument("key_file", help="file holding an API key, or a users.toml with --user")
    ap.add_argument("--user", help="take users.<USER>.key from KEY_FILE (a users.toml)")
    ap.add_argument("--inputs", default=str(Path.home() / "timestral-bench" / "S1_inputs.npz"))
    ap.add_argument("--chunk", type=int, default=48, help="series per request (default %(default)s)")
    ap.add_argument("--concurrency", type=int, default=2, help="requests in flight (default %(default)s)")
    ap.add_argument("--limit", type=int, default=0, help="first N cases only (no reference comparison)")
    ap.add_argument("--tests", default="T1,T2")
    ap.add_argument("--no-sym", action="store_true", help="symmetric_averaging off (no reference)")
    a = ap.parse_args()

    base = a.url.rstrip("/")
    key = read_key(a.key_file, a.user)
    info_req = urllib.request.Request(base + "/v1/info", headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(info_req, timeout=30) as resp:
        info = json.load(resp)
    print(f"service: {info['model']} @ {info['revision'][:12]} on {info['device']}, "
          f"timestral {info['timestral_commit'][:12]}, {info['parameters']:,} parameters", flush=True)

    z = dict(np.load(a.inputs))
    n = min(a.limit or len(z["context"]), len(z["context"]))
    actual, night, cap = z["actual"][:n], z["night"][:n], float(z["cap"])
    ok = z["daylight"][:n] & np.isfinite(actual)
    sym = not a.no_sym
    failed = False

    for test in a.tests.split(","):
        chunks = [range(i, min(i + a.chunk, n)) for i in range(0, n, a.chunk)]
        median = np.full((n, H), np.nan)
        compute = 0.0
        batches = 0
        start = time.time()

        def run(idx):
            return idx, post(base + "/v1/forecast", key, body(z, test, idx, sym))

        with ThreadPoolExecutor(a.concurrency) as pool:
            for idx, out in pool.map(run, chunks):
                for f in out["forecasts"]:
                    median[f["id"]] = np.array(f["median"], dtype=np.float64)
                compute += out["timing"]["compute_seconds"]
                batches += out["timing"]["batches"]
        secs = time.time() - start
        pred = np.where(night, 0.0, np.clip(median, 0, cap))
        nmae = 100 * np.abs(pred - actual)[ok].sum() / actual[ok].sum()
        line = (f"{test}: nMAE {nmae:.2f}%  {n} series in {secs:.1f}s = {secs / n:.3f}s/series "
                f"({n / secs:.2f} series/s), {len(chunks)} requests, {batches} request-batches, "
                f"model time per request summed {compute:.1f}s")
        if n == len(z["context"]) and sym:
            diff = nmae - REFERENCE[test]
            verdict = "OK" if abs(diff) <= 0.05 else "MISMATCH"
            failed |= verdict != "OK"
            line += f"  | reference {REFERENCE[test]:.2f}% (diff {diff:+.3f}) {verdict}"
        print(line, flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
