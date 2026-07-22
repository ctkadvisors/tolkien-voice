from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from tolkien_voice.commands import CommandResult


class FakeCommandRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], float]] = []
        self._results: dict[tuple[str, ...], list[CommandResult]] = defaultdict(list)

    def add_result(self, prefix: Sequence[str], result: CommandResult) -> None:
        self._results[tuple(prefix)].append(result)

    async def run(self, argv: Sequence[str], *, timeout: float) -> CommandResult:
        command = tuple(argv)
        self.calls.append((command, timeout))
        matches = [prefix for prefix in self._results if command[: len(prefix)] == prefix]
        if matches:
            prefix = max(matches, key=len)
            return self._results[prefix].pop(0)
        return CommandResult(exit_code=0, stdout="", stderr="")


@dataclass
class FakeResponse:
    status_code: int


class FakeHttpClient:
    def __init__(
        self,
        statuses: list[int | Exception],
        *,
        default: int | Exception = 404,
    ) -> None:
        self.statuses = list(statuses)
        self.default = default
        self.calls: list[str] = []

    async def get(self, url: str, *, timeout: float) -> FakeResponse:
        self.calls.append(url)
        value = self.statuses.pop(0) if self.statuses else self.default
        if isinstance(value, Exception):
            raise value
        return FakeResponse(value)
