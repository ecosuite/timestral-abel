"""`python -m timestral_abel.download`: fetch the configured checkpoint into cache_dir.

The service downloads it on first start anyway; running this once beforehand
keeps that out of the start-up (and the healthcheck's start period). Needs no GPU.

The LICENSE and README of the model repository are fetched with the weights:
the Modified MIT license asks that its notice travels with every copy.
"""

import logging
import sys
from pathlib import Path

from . import config

FILES = ["config.json", "model.safetensors", "LICENSE", "README.md"]


def fetch(cfg: config.Model) -> Path:
    """A local directory with the checkpoint: `cfg.id` itself, or its Hub snapshot."""
    local = Path(cfg.id).expanduser()
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(cfg.id, revision=cfg.revision or None, cache_dir=cfg.cache_dir,
                                  local_files_only=cfg.local_files_only, allow_patterns=FILES))


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
    cfg = config.load().model
    print(f"{cfg.id} @ {cfg.revision or 'main'} is in {fetch(cfg)}")


if __name__ == "__main__":
    main()
