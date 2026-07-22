from __future__ import annotations

import asyncio
from dataclasses import dataclass
import re
from typing import Sequence


HF_TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{20,}")


def redact_secrets(value: str) -> str:
    return HF_TOKEN_PATTERN.sub("[REDACTED_HF_TOKEN]", value)


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str


class CommandTimeoutError(TimeoutError):
    """Raised after terminating a command that exceeded its deadline."""


class CommandRunner:
    async def run(self, argv: Sequence[str], *, timeout: float) -> CommandResult:
        if not argv:
            raise ValueError("command argv cannot be empty")
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except TimeoutError as error:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                process.kill()
                await process.wait()
            raise CommandTimeoutError(f"command timed out after {timeout:g} seconds") from error
        return CommandResult(
            exit_code=process.returncode,
            stdout=redact_secrets(stdout_bytes.decode(errors="replace")),
            stderr=redact_secrets(stderr_bytes.decode(errors="replace")),
        )

