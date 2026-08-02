from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import yaml


ENV_PREFIX = "TOLKIEN_VOICE_"
ALLOWED_HF_REPOS = {
    "nvidia/personaplex-7b-v1",
    # Same architecture, GRPO-RL-tuned by Kyutai for turn-taking/interruption
    # (arXiv 2606.11167) -- drop-in for the serving stack. Stage 0 of the
    # language program swaps production to this base.
    "kyutai/personaplex-rl-seamless",
}
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class AdapterArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    metadata_path: Path


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    controller_host: str
    controller_port: int = Field(ge=1, le=65535)
    personaplex_url: str
    personaplex_container: str
    personaplex_image: str
    hf_repo: str
    disconnect_idle_seconds: int = Field(ge=30)
    lease_wait_seconds: int = Field(gt=0)
    runtime_start_timeout_seconds: int = Field(gt=0)
    llama_swap_url: str
    lock_path: Path
    owner_path: Path
    data_root: Path
    adapter_path: Path | None = None
    adapter_metadata_path: Path | None = None
    adapter_bank: dict[str, AdapterArtifact] = Field(default_factory=dict)

    @model_validator(mode="after")
    def adapter_configuration_is_unambiguous(self) -> RuntimeSettings:
        has_global = self.adapter_path is not None or self.adapter_metadata_path is not None
        if (self.adapter_path is None) != (self.adapter_metadata_path is None):
            raise ValueError("adapter path and metadata path must be configured together")
        if has_global and self.adapter_bank:
            raise ValueError("global adapter and persona adapter bank are mutually exclusive")
        for character_id in self.adapter_bank:
            if not character_id or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
                for character in character_id
            ):
                raise ValueError(f"invalid adapter-bank character ID: {character_id!r}")
        return self

    @field_validator("controller_host")
    @classmethod
    def controller_must_be_loopback(cls, value: str) -> str:
        if value not in LOOPBACK_HOSTS:
            raise ValueError("controller_host must be loopback-only")
        return value

    @field_validator("personaplex_url", "llama_swap_url")
    @classmethod
    def service_url_must_be_loopback(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme != "http" or parsed.hostname not in LOOPBACK_HOSTS:
            raise ValueError("service URL must use HTTP on a loopback host")
        return value.rstrip("/")

    @field_validator("hf_repo")
    @classmethod
    def model_repository_must_be_allowlisted(cls, value: str) -> str:
        if value not in ALLOWED_HF_REPOS:
            raise ValueError("hf_repo is not allowlisted")
        return value


def _environment_overrides() -> dict[str, str]:
    overrides: dict[str, str] = {}
    for field_name in RuntimeSettings.model_fields:
        env_name = f"{ENV_PREFIX}{field_name.upper()}"
        if env_name in os.environ:
            overrides[field_name] = os.environ[env_name]
    return overrides


def load_runtime_settings(config_path: str | Path) -> RuntimeSettings:
    path = Path(config_path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        raw: Any = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"runtime config must be a mapping: {path}")
    values = {**raw, **_environment_overrides()}
    return RuntimeSettings.model_validate(values)
