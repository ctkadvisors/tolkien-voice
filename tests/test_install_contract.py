from pathlib import Path


ROOT = Path(__file__).parents[1]


def script(name: str) -> str:
    return (ROOT / "scripts" / name).read_text()


def test_installer_is_ordered_and_non_destructive() -> None:
    source = script("install-spark.sh")
    execution = source[source.rindex("\npreflight_host\n") :]
    ordered_markers = [
        "preflight_host",
        "preflight_secret",
        "preflight_model_slot",
        "prepare_data_directories",
        "build_assets",
        '"$patcher" --check',
        '"$patcher" --apply',
        "install_user_service",
        "configure-tailscale-serve.sh",
        "smoke-test.sh",
    ]
    positions = [execution.index(marker) for marker in ordered_markers]
    assert positions == sorted(positions)

    for forbidden in (
        "sudo ",
        "/unload",
        "docker system prune",
        "tailscale serve reset",
        "rm -rf",
        "docker stop vllm_embed",
    ):
        assert forbidden not in source

    assert "100 * 1024 * 1024" in source
    assert "vendor/personaplex-revision.txt" in source
    assert "export PATH=/home/chrisknuteson/.local/bin:$PATH" in source
    assert "node --version" in source
    assert "v22." in source
    assert "NATM" not in source


def test_rollback_targets_only_tolkien_voice_resources() -> None:
    source = script("rollback-spark.sh")
    assert "tolkien-voice.service" in source
    assert "tolkien-voice-personaplex" in source
    assert "patch-llama-swap-config.py" in source
    assert "--restore" in source
    assert 'configure-tailscale-serve.sh" --remove' in source
    for forbidden in (
        "sudo ",
        "tailscale serve reset",
        "docker system prune",
        "rm -rf",
        "vllm_embed",
        "vllm_node",
        "vllm_gemma",
    ):
        assert forbidden not in source
