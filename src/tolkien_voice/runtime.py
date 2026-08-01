from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import stat
from typing import Any, Protocol

import httpx

from tolkien_voice.commands import CommandResult, CommandRunner, redact_secrets
from tolkien_voice.settings import RuntimeSettings


PERSONAPLEX_BASE_REVISION = "fdaf4090a61cb315c138a1faee287ffd6c716309"


class AsyncHttpClient(Protocol):
    async def get(self, url: str, *, timeout: float) -> Any: ...


class RuntimeCredentialError(RuntimeError):
    """Raised when the gated-model credential is absent or unsafe."""


class RuntimeStartError(RuntimeError):
    def __init__(self, message: str, *, diagnostics: str = "") -> None:
        super().__init__(message)
        self.diagnostics = redact_secrets(diagnostics)


class _DefaultHttpClient:
    async def get(self, url: str, *, timeout: float) -> httpx.Response:
        async with httpx.AsyncClient() as client:
            return await client.get(url, timeout=timeout)


class PersonaPlexRuntime:
    def __init__(
        self,
        settings: RuntimeSettings,
        *,
        runner: CommandRunner | None = None,
        http: AsyncHttpClient | None = None,
        poll_seconds: float = 1.0,
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        self.settings = settings
        self.runner = runner or CommandRunner()
        self.http = http or _DefaultHttpClient()
        self.poll_seconds = poll_seconds
        self._started = False

    @property
    def cache_path(self) -> Path:
        return self.settings.data_root / "cache" / "huggingface"

    @property
    def token_path(self) -> Path:
        return self.settings.data_root / "secrets" / "hf_token"

    @property
    def log_path(self) -> Path:
        return self.settings.data_root / "logs"

    async def recover(self) -> None:
        result = await self._inspect()
        if result.exit_code == 0:
            await self._stop_command()
        self._started = False

    async def start(self, character_id: str | None = None) -> None:
        self._prepare_directories()
        self._validate_token()
        # A leftover container (exited after a prior session, or orphaned by a
        # crash) squats the fixed --name and makes docker run fail -- remove
        # it unconditionally before starting; failure (name absent) is fine.
        await self._remove_command(force=True)
        result = await self.runner.run(self._docker_run_command(character_id), timeout=60)
        if result.exit_code != 0:
            raise RuntimeStartError(
                "PersonaPlex container failed to launch",
                diagnostics=f"{result.stdout}\n{result.stderr}".strip(),
            )
        self._started = True
        try:
            await self.wait_ready()
        except BaseException as error:
            diagnostics = await self.logs()
            await self.stop()
            if isinstance(error, asyncio.CancelledError):
                raise
            raise RuntimeStartError(
                "PersonaPlex did not become ready before the startup deadline",
                diagnostics=diagnostics,
            ) from error

    async def wait_ready(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.settings.runtime_start_timeout_seconds
        last_error: BaseException | None = None
        while True:
            try:
                response = await self.http.get(
                    f"{self.settings.personaplex_url}/api/health", timeout=2
                )
                if response.status_code == 200:
                    return
            except (httpx.HTTPError, OSError) as error:
                last_error = error
            container = await self._inspect()
            state = container.stdout.strip().lower()
            if container.exit_code != 0 or state in {"dead", "exited", "removing"}:
                raise RuntimeError(
                    f"PersonaPlex container exited before readiness (state={state or 'absent'})"
                )
            remaining = deadline - loop.time()
            if remaining <= 0:
                detail = f": {last_error}" if last_error else ""
                raise TimeoutError(f"PersonaPlex readiness timed out{detail}")
            await asyncio.sleep(min(self.poll_seconds, remaining))

    async def logs(self) -> str:
        result = await self.runner.run(
            [
                "docker",
                "logs",
                "--tail",
                "200",
                self.settings.personaplex_container,
            ],
            timeout=15,
        )
        return redact_secrets("\n".join(part for part in (result.stdout, result.stderr) if part))

    async def stop(self) -> None:
        if not self._started:
            return
        await self._stop_command()
        self._started = False

    def _prepare_directories(self) -> None:
        directories = {
            self.settings.data_root: 0o700,
            self.settings.data_root / "cache": 0o750,
            self.cache_path: 0o750,
            self.settings.data_root / "secrets": 0o700,
            self.log_path: 0o700,
        }
        for path, mode in directories.items():
            self._assert_under_data_root(path)
            path.mkdir(parents=True, exist_ok=True, mode=mode)
            path.chmod(mode)

    def _validate_token(self) -> None:
        self._assert_under_data_root(self.token_path)
        if not self.token_path.is_file() or self.token_path.stat().st_size == 0:
            raise RuntimeCredentialError(f"missing Hugging Face token file: {self.token_path}")
        mode = stat.S_IMODE(self.token_path.stat().st_mode)
        if mode & 0o077:
            raise RuntimeCredentialError("Hugging Face token must have no group/other access")

    def _validate_adapter(self, character_id: str | None = None) -> tuple[Path, str] | None:
        selected = self.settings.adapter_bank.get(character_id or "")
        if selected is not None:
            adapter = selected.path
            metadata_path = selected.metadata_path
        else:
            adapter = self.settings.adapter_path
            metadata_path = self.settings.adapter_metadata_path
        if adapter is None and metadata_path is None:
            return None
        if adapter is None or metadata_path is None:
            raise ValueError("adapter path and metadata path must be configured together")
        self._assert_under_data_root(adapter)
        self._assert_under_data_root(metadata_path)
        adapter = adapter.expanduser().resolve()
        metadata_path = metadata_path.expanduser().resolve()
        if not adapter.is_file() or adapter.suffix != ".safetensors":
            raise ValueError(f"adapter must be a safetensors file: {adapter}")
        if adapter.stat().st_size == 0:
            raise ValueError(f"adapter is empty: {adapter}")
        if not metadata_path.is_file():
            raise ValueError(f"adapter metadata is missing: {metadata_path}")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"adapter metadata is invalid: {metadata_path}") from error
        if metadata.get("base_revision") != PERSONAPLEX_BASE_REVISION:
            raise ValueError("adapter base revision does not match the pinned PersonaPlex base")
        model_cache = (
            self.cache_path / "hub" / "models--nvidia--personaplex-7b-v1"
        )
        base_ref = model_cache / "refs" / "main"
        try:
            cached_revision = base_ref.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise ValueError("pinned PersonaPlex base is not present in the runtime cache") from error
        if cached_revision != PERSONAPLEX_BASE_REVISION:
            raise ValueError("runtime cache does not point at the adapter's pinned base revision")
        snapshot = model_cache / "snapshots" / PERSONAPLEX_BASE_REVISION
        if not (snapshot / "model.safetensors").is_file():
            raise ValueError("pinned PersonaPlex model checkpoint is missing from the cache")
        digest = hashlib.sha256()
        with adapter.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_sha256 = digest.hexdigest()
        if metadata.get("adapter_sha256") != actual_sha256:
            raise ValueError("adapter SHA-256 does not match its run metadata")
        return adapter, actual_sha256

    def _assert_under_data_root(self, path: Path) -> None:
        root = self.settings.data_root.expanduser().resolve()
        candidate = path.expanduser().resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"runtime path escapes data root: {candidate}")

    def _docker_run_command(self, character_id: str | None = None) -> list[str]:
        self._assert_under_data_root(self.cache_path)
        self._assert_under_data_root(self.token_path)
        command = [
            "docker",
            "run",
            "--detach",
            "--name",
            self.settings.personaplex_container,
            "--gpus",
            "all",
            "--network",
            "host",
            "--ipc",
            "host",
            "--env",
            "HF_HOME=/cache/huggingface",
            "--env",
            "NO_TORCH_COMPILE=1",
            "--mount",
            f"type=bind,src={self.cache_path},dst=/cache/huggingface",
            "--mount",
            f"type=bind,src={self.token_path},dst=/run/secrets/hf_token,readonly",
        ]
        adapter = self._validate_adapter(character_id)
        if adapter is not None:
            adapter_path, adapter_sha256 = adapter
            command.extend(
                [
                    "--mount",
                    f"type=bind,src={adapter_path},dst=/adapter/lora.safetensors,readonly",
                    "--env",
                    "TOLKIEN_VOICE_LORA_WEIGHT=/adapter/lora.safetensors",
                    "--env",
                    f"TOLKIEN_VOICE_LORA_SHA256={adapter_sha256}",
                    "--env",
                    f"TOLKIEN_VOICE_BASE_REVISION={PERSONAPLEX_BASE_REVISION}",
                    "--env",
                    "HF_HUB_OFFLINE=1",
                ]
            )
        command.append(self.settings.personaplex_image)
        return command

    async def _inspect(self) -> CommandResult:
        return await self.runner.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{.State.Status}}",
                self.settings.personaplex_container,
            ],
            timeout=10,
        )

    async def _stop_command(self) -> None:
        await self.runner.run(
            [
                "docker",
                "stop",
                "--time",
                "20",
                self.settings.personaplex_container,
            ],
            timeout=30,
        )
        # docker stop leaves the exited container behind, and its fixed name
        # would block the next docker run -- remove it as part of stopping.
        # (Not --rm on docker run: startup-failure diagnostics read
        # `docker logs` from the exited container before stop() runs, and
        # --rm would delete it out from under that.)
        await self._remove_command(force=False)

    async def _remove_command(self, *, force: bool) -> None:
        command = ["docker", "rm"]
        if force:
            command.append("--force")
        command.append(self.settings.personaplex_container)
        await self.runner.run(command, timeout=30)
