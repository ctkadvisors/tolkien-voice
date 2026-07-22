from fastapi.testclient import TestClient

from tolkien_voice.main import create_app


def test_health_is_loopback_service_health() -> None:
    client = TestClient(create_app())

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "tolkien-voice"}
