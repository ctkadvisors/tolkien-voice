from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.fakes import FakeCommandRunner, FakeHttpClient
from tolkien_voice.commands import CommandResult, redact_secrets
from tolkien_voice.runtime import PersonaPlexRuntime, RuntimeStartError
from tolkien_voice.settings import RuntimeSettings


def settings_for(tmp_path: Path, *, timeout: int = 1) -> RuntimeSettings:
    return RuntimeSettings.model_validate(
        {
            "controller_host": "127.0.0.1",
            "controller_port": 8788,
            "personaplex_url": "http://127.0.0.1:8998",
            "personaplex_container": "tolkien-voice-personaplex",
            "personaplex_image": "tolkien-voice/personaplex:3428dfd",
            "hf_repo": "nvidia/personaplex-7b-v1",
            "disconnect_idle_seconds": 300,
            "lease_wait_seconds": 900,
            "runtime_start_timeout_seconds": timeout,
            "llama_swap_url": "http://127.0.0.1:9090",
            "lock_path": tmp_path / "run/big-model.lock",
            "owner_path": tmp_path / "run/big-model-owner.json",
            "data_root": tmp_path / "data",
        }
    )


def install_fake_token(settings: RuntimeSettings) -> Path:
    secret_dir = settings.data_root / "secrets"
    secret_dir.mkdir(parents=True, mode=0o700)
    token = secret_dir / "hf_token"
    token.write_text("hf_test_token_not_logged_12345\n", encoding="utf-8")
    token.chmod(0o600)
    return token


@pytest.mark.asyncio
async def test_start_uses_exact_private_docker_contract(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    install_fake_token(settings)
    runner = FakeCommandRunner()
    runtime = PersonaPlexRuntime(settings, runner=runner, http=FakeHttpClient([200]))

    await runtime.start()

    command = runner.calls[0][0]
    assert command == (
        "docker",
        "run",
        "--detach",
        "--name",
        "tolkien-voice-personaplex",
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
        f"type=bind,src={settings.data_root}/cache/huggingface,dst=/cache/huggingface",
        "--mount",
        f"type=bind,src={settings.data_root}/secrets/hf_token,dst=/run/secrets/hf_token,readonly",
        "tolkien-voice/personaplex:3428dfd",
    )
    joined = " ".join(command).lower()
    assert "audio" not in joined
    assert "transcript" not in joined


@pytest.mark.asyncio
async def test_start_creates_only_runtime_directories(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    install_fake_token(settings)
    runtime = PersonaPlexRuntime(
        settings, runner=FakeCommandRunner(), http=FakeHttpClient([200])
    )

    await runtime.start()

    assert {path.name for path in settings.data_root.iterdir()} == {"cache", "secrets", "logs"}
    assert (settings.data_root / "cache/huggingface").is_dir()


@pytest.mark.asyncio
async def test_recover_stops_only_preexisting_managed_container(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    runner = FakeCommandRunner()
    runner.add_result(
        ["docker", "inspect"], CommandResult(0, "running\n", "")
    )
    runtime = PersonaPlexRuntime(settings, runner=runner, http=FakeHttpClient([]))

    await runtime.recover()

    commands = [call[0] for call in runner.calls]
    assert commands == [
        (
            "docker",
            "inspect",
            "--format",
            "{{.State.Status}}",
            "tolkien-voice-personaplex",
        ),
        ("docker", "stop", "--time", "20", "tolkien-voice-personaplex"),
    ]
    assert all("unrelated-container" not in command for command in commands)


@pytest.mark.asyncio
async def test_recover_is_noop_when_managed_container_is_absent(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    runner = FakeCommandRunner()
    runner.add_result(["docker", "inspect"], CommandResult(1, "", "not found"))
    runtime = PersonaPlexRuntime(settings, runner=runner, http=FakeHttpClient([]))

    await runtime.recover()

    assert len(runner.calls) == 1


@pytest.mark.asyncio
async def test_startup_fails_fast_when_container_exits(tmp_path: Path) -> None:
    settings = settings_for(tmp_path, timeout=30)
    install_fake_token(settings)
    runner = FakeCommandRunner()
    runner.add_result(["docker", "inspect"], CommandResult(0, "exited\n", ""))
    runner.add_result(["docker", "logs"], CommandResult(0, "container crashed\n", ""))
    runtime = PersonaPlexRuntime(
        settings,
        runner=runner,
        http=FakeHttpClient([], default=ConnectionError("not ready")),
        poll_seconds=10,
    )

    with pytest.raises(RuntimeStartError) as caught:
        await asyncio.wait_for(runtime.start(), timeout=2)

    assert "container crashed" in caught.value.diagnostics
    assert "state=exited" in str(caught.value.__cause__)
    assert ("docker", "stop", "--time", "20", "tolkien-voice-personaplex") in [
        call[0] for call in runner.calls
    ]


@pytest.mark.asyncio
async def test_startup_failure_redacts_logs_and_stops_container(tmp_path: Path) -> None:
    settings = settings_for(tmp_path, timeout=1)
    install_fake_token(settings)
    runner = FakeCommandRunner()
    runner.add_result(
        ["docker", "logs"],
        CommandResult(0, "download failed for hf_abcdefghijklmnopqrstuvwxyz123456", ""),
    )
    runtime = PersonaPlexRuntime(
        settings,
        runner=runner,
        http=FakeHttpClient([], default=ConnectionError("not ready")),
        poll_seconds=0.001,
    )

    with pytest.raises(RuntimeStartError) as caught:
        await asyncio.wait_for(runtime.start(), timeout=2)

    assert "hf_abcdefghijklmnopqrstuvwxyz123456" not in caught.value.diagnostics
    assert "[REDACTED_HF_TOKEN]" in caught.value.diagnostics
    assert ("docker", "stop", "--time", "20", "tolkien-voice-personaplex") in [
        call[0] for call in runner.calls
    ]


@pytest.mark.asyncio
async def test_stop_is_idempotent(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    install_fake_token(settings)
    runner = FakeCommandRunner()
    runtime = PersonaPlexRuntime(settings, runner=runner, http=FakeHttpClient([200]))
    await runtime.start()

    await runtime.stop()
    await runtime.stop()

    stops = [call for call in runner.calls if call[0][:2] == ("docker", "stop")]
    assert len(stops) == 1


def test_secret_redaction() -> None:
    assert redact_secrets("prefix hf_abcdefghijklmnopqrstuvwxyz123456 suffix") == (
        "prefix [REDACTED_HF_TOKEN] suffix"
    )
