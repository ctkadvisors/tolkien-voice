from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient

from tests.test_session import FakeLease, FakeRuntime
from tolkien_voice.api import build_upstream_websocket_url
from tolkien_voice.characters import CharacterRegistry
from tolkien_voice.main import create_app
from tolkien_voice.session import SessionController


CARDS = Path(__file__).parents[1] / "configs" / "characters"


def make_client(*, web_dist: Path | None = None) -> TestClient:
    registry = CharacterRegistry.from_directory(CARDS)
    controller = SessionController(
        registry=registry,
        lease=FakeLease(),
        runtime=FakeRuntime(),
        lease_wait_seconds=1,
        disconnect_idle_seconds=300,
    )
    return TestClient(
        create_app(
            session_controller=controller,
            character_registry=registry,
            web_dist=web_dist,
        )
    )


def test_characters_endpoint_exposes_roster_without_prompts() -> None:
    with make_client() as client:
        response = client.get("/api/characters")

    assert response.status_code == 200
    characters = response.json()["characters"]
    assert len(characters) == 9
    sauron = next(card for card in characters if card["id"] == "sauron")
    assert sauron["allowed_voices"] == ["NATM3.pt"]
    assert "role" not in sauron
    assert "text_prompt" not in sauron


def test_start_returns_202_and_only_public_session_fields() -> None:
    with make_client() as client:
        response = client.post(
            "/api/session",
            json={"character_id": "gandalf", "voice_id": "NATM1.pt"},
        )

    assert response.status_code == 202
    payload = response.json()
    assert payload["state"] == "waiting_for_lease"
    assert payload["character_id"] == "gandalf"
    assert "text_prompt" not in payload
    assert "role" not in payload


def test_unknown_character_and_disallowed_voice_are_rejected() -> None:
    with make_client() as client:
        unknown = client.post(
            "/api/session", json={"character_id": "saruman", "voice_id": "NATM1.pt"}
        )
        disallowed = client.post(
            "/api/session", json={"character_id": "sauron", "voice_id": "NATF0.pt"}
        )

    assert unknown.status_code == 404
    assert disallowed.status_code == 422


def test_second_session_returns_conflict_and_delete_returns_idle() -> None:
    with make_client() as client:
        first = client.post(
            "/api/session", json={"character_id": "gandalf", "voice_id": "NATM1.pt"}
        )
        second = client.post(
            "/api/session", json={"character_id": "sauron", "voice_id": "NATM3.pt"}
        )
        ended = client.delete(f"/api/session/{first.json()['session_id']}")
        status = client.get("/api/session")

    assert second.status_code == 409
    assert second.json()["character_id"] == "gandalf"
    assert ended.status_code == 204
    assert status.json()["state"] == "idle"


def test_upstream_url_encodes_server_side_prompt_and_voice() -> None:
    url = build_upstream_websocket_url(
        "http://127.0.0.1:8998",
        text_prompt="<system> Speak as Gandalf & listen. <system>",
        voice_prompt="NATM1.pt",
    )

    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    assert parsed.scheme == "ws"
    assert parsed.netloc == "127.0.0.1:8998"
    assert parsed.path == "/api/chat"
    assert query == {
        "text_prompt": ["<system> Speak as Gandalf & listen. <system>"],
        "voice_prompt": ["NATM1.pt"],
    }


def test_production_bundle_is_served_without_shadowing_api(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    (tmp_path / "index.html").write_text("<h1>Tolkien Voice</h1>", encoding="utf-8")
    (assets / "app.js").write_text("console.log('voice')", encoding="utf-8")

    with make_client(web_dist=tmp_path) as client:
        page = client.get("/")
        asset = client.get("/assets/app.js")
        health = client.get("/api/health")

    assert page.status_code == 200
    assert "Tolkien Voice" in page.text
    assert asset.status_code == 200
    assert health.json()["status"] == "ok"
