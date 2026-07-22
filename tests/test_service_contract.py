from pathlib import Path


ROOT = Path(__file__).parents[1]
UNIT = ROOT / "systemd/tolkien-voice.service"


def test_user_service_is_private_and_self_cleaning() -> None:
    source = UNIT.read_text()
    assert "WorkingDirectory=/home/chrisknuteson/tolkien-voice" in source
    assert "ExecStart=/home/chrisknuteson/tolkien-voice/.venv/bin/tolkien-voice" in source
    assert "ExecStop=/home/chrisknuteson/tolkien-voice/scripts/stop-runtime.sh" in source
    assert "RuntimeDirectory=tolkien-voice" in source
    assert "RuntimeDirectoryMode=0700" in source
    assert "Restart=on-failure" in source
    assert "RestartSec=5" in source
    assert "UMask=0077" in source
    assert "NoNewPrivileges=true" in source
    assert "TOLKIEN_VOICE_CONFIG=/home/chrisknuteson/tolkien-voice/configs/runtime.yaml" in source
    assert "0.0.0.0" not in source
    assert "HF_TOKEN" not in source
    assert "EnvironmentFile" not in source
    assert "User=root" not in source


def test_stop_script_only_stops_the_managed_container() -> None:
    source = (ROOT / "scripts/stop-runtime.sh").read_text()
    assert "tolkien-voice-personaplex" in source
    assert "docker inspect" in source
    assert "docker stop --time 20" in source
    for forbidden in ("docker stop $(", "docker ps -q", "docker kill", "rm -rf"):
        assert forbidden not in source


def test_tailscale_script_is_private_and_scoped() -> None:
    source = (ROOT / "scripts/configure-tailscale-serve.sh").read_text()
    assert "spark-1250.tail71b73b.ts.net." in source
    assert "BackendState" in source
    assert "controller=http://127.0.0.1:8788" in source
    assert '"$controller/api/health"' in source
    assert "tailscale serve get-config" in source
    assert "tailscale serve --bg --yes 8788" in source
    assert "tailscale serve --https=443 off" in source
    assert "tailscale serve status --json" in source
    assert "http://127.0.0.1:8788" in source
    assert "funnel" not in source.lower()
    assert "tailscale serve reset" not in source


def test_smoke_test_covers_idle_privacy_and_cleanup() -> None:
    source = (ROOT / "scripts/smoke-test.sh").read_text()
    for required in (
        "controller=http://127.0.0.1:8788",
        "private_url=https://spark-1250.tail71b73b.ts.net",
        "sauron",
        "NATM3.pt",
        "tolkien-voice-personaplex",
        "/v1/embeddings",
        "vllm_embed",
        "0.0.0.0:8788",
        "0.0.0.0:8998",
        "/api/session",
        "DELETE",
        "eval-runs",
    ):
        assert required in source
    assert "SKIP: PersonaPlex weights" in source
    assert "docker stop vllm_embed" not in source
    assert source.index("/v1/embeddings") < source.index(
        "docker inspect --format '{{.State.Status}}' vllm_embed"
    )
