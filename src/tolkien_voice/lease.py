from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import errno
import fcntl
import json
import os
from pathlib import Path
import re
from typing import Iterator, Literal


OWNER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@dataclass(frozen=True)
class LeaseOwner:
    kind: Literal["voice", "text"]
    name: str
    pid: int | None = None
    acquired_at: str | None = None

    def __post_init__(self) -> None:
        if not OWNER_NAME_PATTERN.fullmatch(self.name):
            raise ValueError("lease owner name contains unsafe characters")

    def activated(self) -> LeaseOwner:
        return replace(
            self,
            pid=os.getpid(),
            acquired_at=datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        )


class LeaseBusyError(RuntimeError):
    def __init__(self, owner: LeaseOwner | None) -> None:
        super().__init__("Spark large-model slot is busy")
        self.owner = owner


class LeaseTimeoutError(TimeoutError):
    def __init__(self, owner: LeaseOwner | None) -> None:
        super().__init__("timed out waiting for Spark large-model slot")
        self.owner = owner


class SparkModelLease:
    def __init__(
        self,
        lock_path: str | Path,
        owner_path: str | Path,
        *,
        poll_seconds: float = 0.5,
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        self.lock_path = Path(lock_path)
        self.owner_path = Path(owner_path)
        self.poll_seconds = poll_seconds
        self._fd: int | None = None
        self.owner: LeaseOwner | None = None

    def try_acquire(self, owner: LeaseOwner) -> bool:
        if self._fd is not None:
            raise RuntimeError("lease instance already holds the lock")
        self.lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            os.close(fd)
            if error.errno in (errno.EACCES, errno.EAGAIN):
                return False
            raise
        active_owner = owner.activated()
        self._fd = fd
        self.owner = active_owner
        try:
            self._write_owner(active_owner)
        except BaseException:
            self.release()
            raise
        return True

    async def acquire(self, owner: LeaseOwner, timeout_seconds: float) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        while True:
            if self.try_acquire(owner):
                return
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise LeaseTimeoutError(self.current_owner())
            await asyncio.sleep(min(self.poll_seconds, remaining))

    def current_owner(self) -> LeaseOwner | None:
        if self._fd is not None:
            return self.owner
        self.lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                return self._read_owner()
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
                return None
        finally:
            os.close(fd)

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            current = self._read_owner()
            if current == self.owner:
                self.owner_path.unlink(missing_ok=True)
        finally:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None
            self.owner = None

    @contextmanager
    def held(self, owner: LeaseOwner) -> Iterator[SparkModelLease]:
        if not self.try_acquire(owner):
            raise LeaseBusyError(self.current_owner())
        try:
            yield self
        finally:
            self.release()

    def _write_owner(self, owner: LeaseOwner) -> None:
        self.owner_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = self.owner_path.with_name(f".{self.owner_path.name}.{os.getpid()}.tmp")
        descriptor = os.open(temporary, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
        try:
            payload = json.dumps(asdict(owner), separators=(",", ":")).encode()
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, self.owner_path)

    def _read_owner(self) -> LeaseOwner | None:
        try:
            raw = json.loads(self.owner_path.read_text(encoding="utf-8"))
            return LeaseOwner(
                kind=raw["kind"],
                name=raw["name"],
                pid=int(raw["pid"]),
                acquired_at=str(raw["acquired_at"]),
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

