from fastapi.testclient import TestClient

from tolkien_voice.main import create_app, runtime_config_path


def test_health_is_loopback_service_health() -> None:
    client = TestClient(create_app())

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "tolkien-voice"}


def test_runtime_config_path_honors_service_environment(monkeypatch, tmp_path) -> None:
    selected = tmp_path / "runtime.yaml"
    monkeypatch.setenv("TOLKIEN_VOICE_CONFIG", str(selected))

    assert runtime_config_path() == selected.resolve()
