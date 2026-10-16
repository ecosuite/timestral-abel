"""API keys from a users file, read-only, re-read when it changes.

    upstream_key = "..."        # optional; never accepted as a key here

    [users.alice]               # stable id: series are counted against it
    name = "Alice"
    key  = "..."

The optional top-level upstream_key lets the file be shared with a service that
holds its own credential there; that credential is never a forecasting key.
"""

import hashlib
import logging
import os
import re
import threading
import tomllib
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
MIN_KEY_LENGTH = 16


@dataclass(frozen=True)
class User:
    id: str
    name: str


@dataclass(frozen=True)
class Users:
    users: tuple[User, ...]
    by_hash: dict  # sha256(key) -> User


def _digest(key: str) -> bytes:
    return hashlib.sha256(key.encode()).digest()


def parse(data: dict) -> Users:
    """Validate a parsed users file. Raises ValueError with a readable message.

    The upstream key is another service's credential and is never a
    forecasting key.
    """
    upstream_key = data.get("upstream_key")
    upstream = _digest(upstream_key) if isinstance(upstream_key, str) and upstream_key else None
    table = data.get("users", {})
    if not isinstance(table, dict):
        raise ValueError("[users] must be a table of [users.<id>] entries")
    users, by_hash = [], {}
    for uid, entry in table.items():
        if not ID_RE.match(uid):
            raise ValueError(f"user id {uid!r}: use letters, digits, - and _ (max 40)")
        if not isinstance(entry, dict):
            raise ValueError(f"users.{uid} must be a table")
        name, key = entry.get("name", uid), entry.get("key")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"users.{uid}.name must be a non-empty string")
        if not isinstance(key, str) or len(key) < MIN_KEY_LENGTH:
            raise ValueError(f"users.{uid}.key must be a string of at least {MIN_KEY_LENGTH} characters")
        digest = _digest(key)
        if digest == upstream:
            # Rejecting the whole file would lock everyone out; the one entry is
            # left out instead.
            log.warning("users.%s has the same key as upstream_key; not accepted here", uid)
            continue
        if digest in by_hash:
            raise ValueError(f"users.{uid} has the same key as users.{by_hash[digest].id}")
        user = User(uid, name.strip())
        users.append(user)
        by_hash[digest] = user
    return Users(tuple(users), by_hash)


def load(path: str | os.PathLike) -> Users:
    with open(path, "rb") as f:
        return parse(tomllib.load(f))


class Directory:
    """The users file, re-read when its size, mtime or inode changes.

    A file that fails to parse is logged and ignored: the previous good version
    stays in effect, so a typo while editing never locks everyone out. A file
    that is gone or cannot be read refuses every key until it is back, so
    removing it (or the mount) cuts off access at once.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._signature = ()  # never a real signature, so the first refresh reads (or logs)
        self.current: Users | None = None

    def refresh(self) -> Users | None:
        try:
            st = os.stat(self.path)
            signature = (st.st_ino, st.st_size, st.st_mtime_ns)
        except FileNotFoundError:
            signature = None
        except OSError as exc:  # e.g. the mounted directory is not readable by our uid
            signature = ("error", str(exc))
        with self._lock:
            if signature == self._signature:
                return self.current
            self._signature = signature
            if signature is None or signature[0] == "error":
                return self._refuse_all("not found" if signature is None else signature[1])
            try:
                users = load(self.path)
            except OSError as exc:  # removed or made unreadable between stat and open
                return self._refuse_all(str(exc))
            except (ValueError, tomllib.TOMLDecodeError) as exc:
                log.error("users file %s: %s (keeping the previous version)", self.path, exc)
                return self.current
            self.current = users
            log.info("users file loaded: %d user(s)", len(users.users))
            return users

    def _refuse_all(self, reason: str) -> None:
        log.error("users file %s unreadable (%s); every API request is refused", self.path, reason)
        self.current = None
        return None

    def authenticate(self, key: str | None) -> User | None:
        users = self.refresh()
        if users is None or not key:
            return None
        return users.by_hash.get(_digest(key))


def request_key(headers) -> str | None:
    """The caller's key from `Authorization: Bearer <key>` or `x-api-key`."""
    auth = headers.get("Authorization", "")
    scheme, _, token = auth.partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        return token.strip()
    key = headers.get("x-api-key", "").strip()
    return key or None
