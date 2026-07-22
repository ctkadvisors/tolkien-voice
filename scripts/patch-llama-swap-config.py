#!/usr/bin/env python3
"""Safely add the shared Spark big-model lease to llama-swap."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any, Iterable
from urllib.error import URLError
from urllib.request import Request, urlopen

from ruamel.yaml import YAML


BIG_MODELS = {"coder", "coder-fp8", "qwen36", "gemma", "tolkien"}
ALL_MODELS = BIG_MODELS | {"embed"}
LIVE_CONFIG = Path("/home/chrisknuteson/inference/llama-swap/config.yaml")
LLAMA_SWAP_BINARY = Path("/home/chrisknuteson/inference/llama-swap/llama-swap")
LLAMA_SWAP_URL = "http://127.0.0.1:9090"
LEASE_WRAPPER = Path(
    "/home/chrisknuteson/tolkien-voice/scripts/run-with-big-model-lease.sh"
)

ORIGINAL_COMMANDS = {
    "coder": (
        "bash -lc 'exec ~/inference/companions/run-supervised.sh "
        "vllm_coder_gb10_marlin ~/inference/companions/launch-coder-gb10-marlin.sh'"
    ),
    "coder-fp8": (
        "bash -lc 'exec ~/inference/companions/run-supervised.sh "
        "vllm_node ~/inference/companions/relaunch-base.sh'"
    ),
    "qwen36": (
        "bash -lc 'exec ~/inference/companions/run-supervised.sh "
        "vllm_qwen36 ~/inference/companions/launch-qwen36.sh'"
    ),
    "gemma": (
        "bash -lc 'exec ~/inference/companions/run-supervised.sh "
        "vllm_gemma ~/inference/companions/launch-gemma.sh'"
    ),
    "tolkien": "bash -lc 'exec ~/inference/companions/launch-tolkien.sh'",
}

PATCHED_COMMANDS = {
    "coder": (
        f"{LEASE_WRAPPER} coder "
        "/home/chrisknuteson/inference/companions/run-supervised.sh "
        "vllm_coder_gb10_marlin "
        "/home/chrisknuteson/inference/companions/launch-coder-gb10-marlin.sh"
    ),
    "coder-fp8": (
        f"{LEASE_WRAPPER} coder-fp8 "
        "/home/chrisknuteson/inference/companions/run-supervised.sh "
        "vllm_node /home/chrisknuteson/inference/companions/relaunch-base.sh"
    ),
    "qwen36": (
        f"{LEASE_WRAPPER} qwen36 "
        "/home/chrisknuteson/inference/companions/run-supervised.sh "
        "vllm_qwen36 /home/chrisknuteson/inference/companions/launch-qwen36.sh"
    ),
    "gemma": (
        f"{LEASE_WRAPPER} gemma "
        "/home/chrisknuteson/inference/companions/run-supervised.sh "
        "vllm_gemma /home/chrisknuteson/inference/companions/launch-gemma.sh"
    ),
    "tolkien": (
        f"{LEASE_WRAPPER} tolkien "
        "/home/chrisknuteson/inference/companions/launch-tolkien.sh"
    ),
}

EXPECTED_MODELS: dict[str, dict[str, Any]] = {
    "coder": {
        "proxy": "http://127.0.0.1:8000",
        "checkEndpoint": "/health",
        "healthCheckTimeout": 900,
        "cmdStop": "docker stop vllm_coder_gb10_marlin",
        "useModelName": "Qwen3-Coder-Next-GB10",
        "aliases": [
            "Qwen3-Coder-Next-GB10",
            "coder-gb10",
            "Qwen3-Coder-Next-NVFP4",
            "coder-nvfp4",
        ],
    },
    "coder-fp8": {
        "proxy": "http://127.0.0.1:8000",
        "checkEndpoint": "/health",
        "healthCheckTimeout": 900,
        "cmdStop": "docker stop vllm_node",
        "useModelName": "Qwen/Qwen3-Coder-Next-FP8",
        "aliases": ["Qwen/Qwen3-Coder-Next-FP8"],
    },
    "qwen36": {
        "proxy": "http://127.0.0.1:8005",
        "checkEndpoint": "/health",
        "healthCheckTimeout": 600,
        "cmdStop": "docker stop vllm_qwen36",
        "useModelName": "Qwen3.6-35B-A3B-NVFP4",
        "aliases": ["Qwen3.6-35B-A3B-NVFP4"],
    },
    "gemma": {
        "proxy": "http://127.0.0.1:8203",
        "checkEndpoint": "/health",
        "healthCheckTimeout": 300,
        "cmdStop": "docker stop vllm_gemma",
        "useModelName": "google/gemma-4-12B-it-qat-w4a16-ct",
        "aliases": ["google/gemma-4-12B-it-qat-w4a16-ct"],
    },
    "embed": {
        "proxy": "http://127.0.0.1:8201",
        "checkEndpoint": "/health",
        "healthCheckTimeout": 300,
        "cmdStop": "docker stop vllm_embed",
        "useModelName": "Qwen/Qwen3-Embedding-0.6B",
        "aliases": ["Qwen/Qwen3-Embedding-0.6B"],
    },
    "tolkien": {
        "proxy": "http://127.0.0.1:8206",
        "checkEndpoint": "/health",
        "healthCheckTimeout": 600,
        "useModelName": "tolkien",
        "aliases": ["tolkien-27b"],
    },
}


class ConfigDriftError(RuntimeError):
    """The live config is not the reviewed Spark config."""


class ActiveBigModelError(RuntimeError):
    """A large model owns the Spark memory slot."""


def _yaml() -> YAML:
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096
    return yaml


def _clean_command(value: Any) -> str:
    return str(value).strip()


def _validate_config(config: Any) -> None:
    if not isinstance(config, dict) or not isinstance(config.get("models"), dict):
        raise ConfigDriftError("config has no models mapping")
    if set(config["models"]) != ALL_MODELS:
        raise ConfigDriftError(f"routes changed: expected {sorted(ALL_MODELS)}")

    groups = config.get("groups", {})
    expected_groups = {
        "always": {
            "swap": False,
            "exclusive": False,
            "persistent": True,
            "members": ["embed"],
        },
        "big": {
            "swap": True,
            "exclusive": False,
            "members": ["coder", "coder-fp8", "qwen36", "gemma", "tolkien"],
        },
    }
    if dict(groups) != expected_groups:
        raise ConfigDriftError("llama-swap group membership or behavior changed")

    for route_id, expected in EXPECTED_MODELS.items():
        model = config["models"][route_id]
        for key, expected_value in expected.items():
            if model.get(key) != expected_value:
                raise ConfigDriftError(f"{route_id}: unexpected {key}")
        allowed_keys = set(expected) | {"cmd", "ttl"}
        if set(model) - allowed_keys:
            raise ConfigDriftError(f"{route_id}: unexpected fields {sorted(set(model) - allowed_keys)}")

        command = _clean_command(model.get("cmd", ""))
        if route_id == "embed":
            expected_embed = (
                "bash -lc 'exec ~/inference/companions/run-supervised.sh "
                "vllm_embed ~/inference/companions/launch-embedding.sh'"
            )
            if command != expected_embed or "ttl" in model:
                raise ConfigDriftError("embed: command or persistence changed")
        elif command not in {ORIGINAL_COMMANDS[route_id], PATCHED_COMMANDS[route_id]}:
            raise ConfigDriftError(f"{route_id}: unknown command")
        elif "ttl" in model and model["ttl"] != 60:
            raise ConfigDriftError(f"{route_id}: unexpected ttl")


def patch_config_text(source: str) -> str:
    yaml = _yaml()
    config = yaml.load(source)
    _validate_config(config)
    already_patched = all(
        _clean_command(config["models"][route_id]["cmd"]) == PATCHED_COMMANDS[route_id]
        and config["models"][route_id].get("ttl") == 60
        for route_id in BIG_MODELS
    )
    if already_patched:
        return source

    for route_id in BIG_MODELS:
        model = config["models"][route_id]
        model["cmd"] = PATCHED_COMMANDS[route_id]
        model["ttl"] = 60

    output = StringIO()
    yaml.dump(config, output)
    rendered = output.getvalue()
    _validate_config(_yaml().load(rendered))
    return rendered


def assert_big_models_idle(running_models: Iterable[str]) -> None:
    active = BIG_MODELS.intersection(running_models)
    if active:
        raise ActiveBigModelError(
            f"active big model(s): {', '.join(sorted(active))}; wait for the current workload"
        )


def make_backup_path(config_path: Path, now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return config_path.with_name(f"{config_path.name}.bak.tolkien-voice-{timestamp}")


def validate_restore_path(config_path: Path, backup_path: Path) -> Path:
    selected = backup_path.expanduser().absolute()
    resolved_config = config_path.expanduser().resolve()
    resolved_backup = selected.resolve()
    prefix = f"{resolved_config.name}.bak.tolkien-voice-"
    if (
        resolved_backup.parent != resolved_config.parent
        or re.fullmatch(rf"{re.escape(prefix)}\d{{8}}T\d{{6}}Z", resolved_backup.name) is None
    ):
        raise ValueError("backup must be a Tolkien Voice backup beside the live config")
    return selected


def _get_json(url: str, timeout: float = 5.0) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _health_ok(url: str, timeout: float = 1.0) -> bool:
    request = Request(url, headers={"Accept": "text/plain"})
    with urlopen(request, timeout=timeout) as response:
        return response.status == 200


def _running_models(base_url: str) -> list[str]:
    payload = _get_json(f"{base_url}/running")
    return [entry["model"] for entry in payload.get("running", []) if "model" in entry]


def _free_loopback_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _validate_with_binary(config_path: Path, binary: Path) -> None:
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"llama-swap binary is not executable: {binary}")
    port = _free_loopback_port()
    process = subprocess.Popen(
        [str(binary), "-config", str(config_path), "-listen", f"127.0.0.1:{port}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stderr = process.stderr.read() if process.stderr else ""
                raise RuntimeError(f"temporary llama-swap validation failed: {stderr.strip()}")
            try:
                if _health_ok(f"http://127.0.0.1:{port}/health"):
                    return
            except (URLError, TimeoutError):
                time.sleep(0.25)
        raise RuntimeError("temporary llama-swap validation timed out")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _verify_routes(base_url: str) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            payload = _get_json(f"{base_url}/v1/models")
            available = {item["id"] for item in payload.get("data", []) if "id" in item}
            if available == ALL_MODELS:
                return
        except (URLError, TimeoutError, json.JSONDecodeError):
            pass
        time.sleep(0.5)
    raise RuntimeError(f"llama-swap did not expose the expected routes: {sorted(ALL_MODELS)}")


def _write_atomic(config_path: Path, content: str, *, validate_binary: bool) -> None:
    mode = config_path.stat().st_mode
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=config_path.parent, prefix=".tolkien-voice-", delete=False
    ) as temporary:
        temporary.write(content)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.chmod(temporary_path, mode)
        if validate_binary:
            _validate_with_binary(temporary_path, LLAMA_SWAP_BINARY)
        os.replace(temporary_path, config_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def check(config_path: Path, base_url: str) -> str:
    assert_big_models_idle(_running_models(base_url))
    return patch_config_text(config_path.read_text())


def apply(config_path: Path, base_url: str) -> Path:
    rendered = check(config_path, base_url)
    backup = make_backup_path(config_path)
    if backup.exists():
        raise RuntimeError(f"refusing to overwrite backup: {backup}")
    shutil.copy2(config_path, backup)
    try:
        _write_atomic(config_path, rendered, validate_binary=True)
        _verify_routes(base_url)
        assert_big_models_idle(_running_models(base_url))
    except Exception:
        shutil.copy2(backup, config_path)
        raise
    return backup


def restore(config_path: Path, backup_path: Path, base_url: str) -> None:
    selected = validate_restore_path(config_path, backup_path)
    if not selected.is_file():
        raise FileNotFoundError(selected)
    assert_big_models_idle(_running_models(base_url))
    source = selected.read_text()
    patch_config_text(source)  # drift-check the backup before replacing anything
    _write_atomic(config_path, source, validate_binary=True)
    _verify_routes(base_url)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--check", action="store_true", help="validate and preview safely")
    operation.add_argument("--apply", action="store_true", help="back up and patch live config")
    operation.add_argument("--restore", type=Path, metavar="BACKUP", help="restore a validated backup")
    parser.add_argument("--config", type=Path, default=LIVE_CONFIG)
    parser.add_argument("--llama-swap-url", default=LLAMA_SWAP_URL)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.check:
            rendered = check(args.config, args.llama_swap_url)
            state = "already patched" if rendered == args.config.read_text() else "ready to patch"
            print(f"llama-swap config is safe: {state}")
        elif args.apply:
            backup = apply(args.config, args.llama_swap_url)
            print(f"patched llama-swap config; backup: {backup}")
        else:
            restore(args.config, args.restore, args.llama_swap_url)
            print(f"restored llama-swap config from: {args.restore}")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
