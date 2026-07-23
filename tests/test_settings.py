from pathlib import Path

import pytest

from tolkien_voice.settings import RuntimeSettings, load_runtime_settings


CONFIG = Path(__file__).parents[1] / "configs" / "runtime.yaml"


@pytest.fixture
def runtime_settings() -> RuntimeSettings:
    return load_runtime_settings(CONFIG)


def test_runtime_defaults_are_spark_safe(runtime_settings: RuntimeSettings) -> None:
    assert runtime_settings.controller_host == "127.0.0.1"
    assert runtime_settings.controller_port == 8788
    assert runtime_settings.personaplex_url == "http://127.0.0.1:8998"
    assert runtime_settings.disconnect_idle_seconds == 300
    assert runtime_settings.lease_wait_seconds == 900


def test_environment_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOLKIEN_VOICE_DISCONNECT_IDLE_SECONDS", "420")

    settings = load_runtime_settings(CONFIG)

    assert settings.disconnect_idle_seconds == 420


def test_persona_adapter_bank_is_loaded_without_global_adapter(tmp_path: Path) -> None:
    values = {
        "controller_host": "127.0.0.1",
        "controller_port": 8788,
        "personaplex_url": "http://127.0.0.1:8998",
        "personaplex_container": "tolkien-voice-personaplex",
        "personaplex_image": "tolkien-voice/personaplex:3428dfd",
        "hf_repo": "nvidia/personaplex-7b-v1",
        "disconnect_idle_seconds": 300,
        "lease_wait_seconds": 900,
        "runtime_start_timeout_seconds": 900,
        "llama_swap_url": "http://127.0.0.1:9090",
        "lock_path": tmp_path / "big-model.lock",
        "owner_path": tmp_path / "big-model-owner.json",
        "data_root": tmp_path / "data",
        "adapter_bank": {
            "aragorn": {
                "path": tmp_path / "data/aragorn/lora.safetensors",
                "metadata_path": tmp_path / "data/aragorn/metadata.json",
            }
        },
    }

    settings = RuntimeSettings.model_validate(values)

    assert settings.adapter_bank["aragorn"].path.name == "lora.safetensors"


def test_global_adapter_and_persona_bank_are_mutually_exclusive(tmp_path: Path) -> None:
    values = {
        "controller_host": "127.0.0.1",
        "controller_port": 8788,
        "personaplex_url": "http://127.0.0.1:8998",
        "personaplex_container": "tolkien-voice-personaplex",
        "personaplex_image": "tolkien-voice/personaplex:3428dfd",
        "hf_repo": "nvidia/personaplex-7b-v1",
        "disconnect_idle_seconds": 300,
        "lease_wait_seconds": 900,
        "runtime_start_timeout_seconds": 900,
        "llama_swap_url": "http://127.0.0.1:9090",
        "lock_path": tmp_path / "big-model.lock",
        "owner_path": tmp_path / "big-model-owner.json",
        "data_root": tmp_path / "data",
        "adapter_path": tmp_path / "data/global/lora.safetensors",
        "adapter_metadata_path": tmp_path / "data/global/metadata.json",
        "adapter_bank": {
            "aragorn": {
                "path": tmp_path / "data/aragorn/lora.safetensors",
                "metadata_path": tmp_path / "data/aragorn/metadata.json",
            }
        },
    }

    with pytest.raises(ValueError, match="mutually exclusive"):
        RuntimeSettings.model_validate(values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("controller_host", "0.0.0.0"),
        ("personaplex_url", "http://100.98.79.30:8998"),
        ("disconnect_idle_seconds", 29),
        ("hf_repo", "someone/other-model"),
    ],
)
def test_rejects_unsafe_runtime_values(field: str, value: object) -> None:
    values = {
        "controller_host": "127.0.0.1",
        "controller_port": 8788,
        "personaplex_url": "http://127.0.0.1:8998",
        "personaplex_container": "tolkien-voice-personaplex",
        "personaplex_image": "tolkien-voice/personaplex:3428dfd",
        "hf_repo": "nvidia/personaplex-7b-v1",
        "disconnect_idle_seconds": 300,
        "lease_wait_seconds": 900,
        "runtime_start_timeout_seconds": 900,
        "llama_swap_url": "http://127.0.0.1:9090",
        "lock_path": "/run/user/1000/tolkien-voice/big-model.lock",
        "owner_path": "/run/user/1000/tolkien-voice/big-model-owner.json",
        "data_root": "/home/chrisknuteson/tolkien-voice-data",
    }
    values[field] = value

    with pytest.raises(ValueError):
        RuntimeSettings.model_validate(values)
