from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests/fixtures/llama-swap-config.yaml"
SCRIPT = ROOT / "scripts/patch-llama-swap-config.py"
BIG_MODELS = {"coder", "coder-fp8", "qwen36", "gemma", "tolkien"}


def load_patcher():
    spec = importlib.util.spec_from_file_location("patch_llama_swap_config", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load patcher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_text() -> str:
    return FIXTURE.read_text()


def parsed(text: str) -> dict:
    return yaml.safe_load(text)


def test_patch_preserves_routes_and_wraps_only_big_models() -> None:
    patcher = load_patcher()
    before = parsed(source_text())
    rendered = patcher.patch_config_text(source_text())
    after = parsed(rendered)

    assert after["models"]["embed"] == before["models"]["embed"]
    assert after["groups"]["always"] == before["groups"]["always"]
    assert after["hooks"] == before["hooks"]
    for route_id, original in before["models"].items():
        updated = after["models"][route_id]
        for key in ("proxy", "checkEndpoint", "healthCheckTimeout", "useModelName", "aliases"):
            assert updated.get(key) == original.get(key)
        if route_id in BIG_MODELS:
            assert updated["ttl"] == 60
            wrapper = (
                "/home/chrisknuteson/tolkien-voice/scripts/"
                f"run-with-big-model-lease.sh {route_id}"
            )
            assert updated["cmd"].count(wrapper) == 1
        else:
            assert updated["cmd"] == original["cmd"]

    assert rendered.startswith("# llama-swap")
    assert "# NVFP4 80B: ~5min weight load + compile" in rendered


def test_patch_is_idempotent() -> None:
    patcher = load_patcher()
    once = patcher.patch_config_text(source_text())
    assert patcher.patch_config_text(once) == once


@pytest.mark.parametrize("route_id", sorted(BIG_MODELS))
def test_patch_refuses_unknown_big_model_command(route_id: str) -> None:
    patcher = load_patcher()
    config = parsed(source_text())
    config["models"][route_id]["cmd"] = "launch-something-unexpected"
    with pytest.raises(patcher.ConfigDriftError, match=route_id):
        patcher.patch_config_text(yaml.safe_dump(config, sort_keys=False))


def test_patch_refuses_missing_route_or_changed_groups() -> None:
    patcher = load_patcher()
    missing = parsed(source_text())
    del missing["models"]["gemma"]
    with pytest.raises(patcher.ConfigDriftError, match="routes"):
        patcher.patch_config_text(yaml.safe_dump(missing, sort_keys=False))

    regrouped = parsed(source_text())
    regrouped["groups"]["big"]["members"].remove("tolkien")
    with pytest.raises(patcher.ConfigDriftError, match="group"):
        patcher.patch_config_text(yaml.safe_dump(regrouped, sort_keys=False))


def test_running_big_model_is_rejected() -> None:
    patcher = load_patcher()
    patcher.assert_big_models_idle(["embed"])
    with pytest.raises(patcher.ActiveBigModelError, match="coder"):
        patcher.assert_big_models_idle(["embed", "coder"])


def test_backup_and_restore_paths_are_constrained() -> None:
    patcher = load_patcher()
    live = Path("/home/chrisknuteson/inference/llama-swap/config.yaml")
    now = datetime(2026, 7, 22, 4, 15, 0, tzinfo=UTC)
    backup = patcher.make_backup_path(live, now)
    assert backup == Path(
        "/home/chrisknuteson/inference/llama-swap/"
        "config.yaml.bak.tolkien-voice-20260722T041500Z"
    )
    assert patcher.validate_restore_path(live, backup) == backup
    with pytest.raises(ValueError, match="backup"):
        patcher.validate_restore_path(live, Path("/tmp/config.yaml"))


def test_cli_supports_exactly_one_operation() -> None:
    patcher = load_patcher()
    parser = patcher.build_parser()
    assert parser.parse_args(["--check"]).check
    assert parser.parse_args(["--apply"]).apply
    assert parser.parse_args(["--restore", "/tmp/backup"]).restore == Path("/tmp/backup")
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--check", "--apply"])


def test_health_probe_accepts_plain_text_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    patcher = load_patcher()

    class PlainTextResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(patcher, "urlopen", lambda *_args, **_kwargs: PlainTextResponse())
    assert patcher._health_ok("http://127.0.0.1:19090/health") is True


def test_apply_is_a_noop_when_config_is_already_patched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    patcher = load_patcher()
    live = tmp_path / "config.yaml"
    live.write_text(patcher.patch_config_text(source_text()))
    monkeypatch.setattr(patcher, "_running_models", lambda _url: ["embed"])
    monkeypatch.setattr(patcher, "_verify_routes", lambda _url: None)
    monkeypatch.setattr(
        patcher,
        "_write_atomic",
        lambda *_args, **_kwargs: pytest.fail("already-patched config must not be rewritten"),
    )

    assert patcher.apply(live, "http://127.0.0.1:9090") is None
    assert list(tmp_path.glob("*.bak.tolkien-voice-*")) == []
