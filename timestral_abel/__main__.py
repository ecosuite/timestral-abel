"""`python -m timestral_abel`: CUDA first, then the HTTP server, the model and a warm-up.

Start-up order:
1. Create the CUDA context. When GPU memory is tight this can fail transiently
   ("out of memory") while a fresh process seconds later works, so on failure
   exit with status 75 and let Docker restart the container.
2. Start the HTTP server: /healthz answers 503 "loading" until step 4 is done.
3. Load the checkpoint (downloading the pinned revision on first start).
4. Run one warm-up forecast so the allocator holds the memory a batch needs.

On SIGTERM it stops accepting connections, answers queued requests with 503,
lets the current model call finish and gives open requests a few seconds to
send their responses. If model calls keep failing with CUDA errors after
start-up, it does the same and exits with EXIT_GPU_FAULT so Docker restarts it.
"""

import logging
import signal
import sys
import threading
import time

from . import auth, config, model
from .batching import Batcher
from .metrics import Metrics
from .server import ForecastServer, Service

log = logging.getLogger("timestral_abel")

# The GPU worker gave up (see batching.FATAL_FAILURES).
EXIT_GPU_FAULT = 70
# Shutdown budget, inside compose's stop_grace_period (30 s): the current model
# call, then responses still being written.
STOP_WORKER_SECONDS = 20
STOP_REQUESTS_SECONDS = 8


def main() -> None:
    start_time = time.time()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    cfg = config.load()

    try:
        model.init_cuda(cfg.model.device)
    except model.CudaInitError as exc:
        log.error("CUDA init failed on %s: %s; exiting with %d so the container restarts",
                  cfg.model.device, exc, model.EXIT_CUDA_INIT)
        sys.exit(model.EXIT_CUDA_INIT)

    stop = threading.Event()  # set by the GPU worker when it gives up
    # Docker stop sends SIGTERM. The handler only appends to a list: it runs on
    # the main thread between bytecodes, and Event.set() there would deadlock
    # if a second signal arrived while the first was inside set() (its lock is
    # not reentrant). The main loop below polls the list.
    signals: list[int] = []
    signal.signal(signal.SIGTERM, lambda signum, _: signals.append(signum))
    signal.signal(signal.SIGINT, lambda signum, _: signals.append(signum))

    metrics = Metrics(cfg.model.id, cfg.model.revision, cfg.model.device, start_time)
    metrics.gpu_memory = model.gpu_memory(cfg.model.device)
    users = auth.Directory(cfg.users_file)
    users.refresh()
    service = Service(cfg, users, metrics)
    server = ForecastServer(service)
    threading.Thread(target=server.serve_forever, name="http", daemon=True).start()
    log.info("listening on %s:%d (loading the model)", cfg.listen_host, cfg.listen_port)

    try:
        forecaster = model.load(cfg.model)
        seconds = model.warm_up(forecaster, cfg.warmup)
    except Exception:
        log.exception("model load or warm-up failed; exiting so the container restarts")
        server.shutdown()
        sys.exit(1)
    allocated, reserved = metrics.gpu_memory()
    log.info("warm-up forecast took %.2fs; GPU memory allocated %.2f GiB, reserved %.2f GiB",
             seconds, allocated / 2**30, reserved / 2**30)

    exit_code = 0

    def on_fatal(reason: str) -> None:
        nonlocal exit_code
        exit_code = EXIT_GPU_FAULT
        service.set_failed()  # /healthz 503, timestral_ready 0, new requests 503
        log.error("exiting with %d so the container restarts with a fresh CUDA context", EXIT_GPU_FAULT)
        stop.set()

    batcher = Batcher(forecaster, cfg.batching, metrics, on_fatal=on_fatal)
    batcher.start()
    service.set_ready(forecaster, batcher, model.info(cfg, forecaster))
    log.info("ready: %s on %s", cfg.model.id, cfg.model.device)

    while not signals and not stop.wait(0.5):
        pass
    log.info("shutting down")
    server.shutdown()  # no new connections
    server.server_close()
    batcher.stop(STOP_WORKER_SECONDS)  # queued requests get 503; the current call finishes
    if not server.wait_idle(STOP_REQUESTS_SECONDS):
        log.warning("exiting with requests still open")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
