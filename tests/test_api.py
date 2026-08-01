from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient
import pytest

from tests.test_session import FakeLease, FakeRuntime
from tolkien_voice.api import (
    _browser_to_upstream,
    _upstream_to_browser,
    build_upstream_websocket_url,
)
from tolkien_voice.barge_in import BARGE_IN_CONTROL_BYTE, BargeInDetector, BargeInGate
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


class _RecordingReader:
    def __init__(self) -> None:
        self.next_pcm: list[float] = []

    def append_bytes(self, _payload: bytes) -> None:
        return None

    def read_pcm(self) -> list[float]:
        return self.next_pcm


class _FakeWebSocket:
    def __init__(self, messages: list[dict[str, object]] | None = None) -> None:
        self._messages = list(messages or [])
        self.sent: list[bytes] = []
        self.closed: tuple[int, str] | None = None

    async def receive(self) -> dict[str, object]:
        if not self._messages:
            return {"type": "websocket.disconnect"}
        return self._messages.pop(0)

    async def send_bytes(self, payload: bytes) -> None:
        self.sent.append(payload)

    async def close(self, code: int, reason: str) -> None:
        self.closed = (code, reason)


class _FakeUpstream:
    def __init__(self) -> None:
        self.sent: list[bytes] = []

    async def send(self, payload: bytes) -> None:
        self.sent.append(payload)


def _one_frame_gate() -> tuple[BargeInGate, _RecordingReader]:
    reader = _RecordingReader()
    gate = BargeInGate(
        reader=reader,
        detector=BargeInDetector(
            activation_threshold=0.02,
            activation_sustained_frames=1,
            deactivation_threshold=0.015,
            deactivation_sustained_frames=1,
        ),
        sample_rate=24_000,
    )
    reader.next_pcm = [0.03] * round(24_000 * 0.04)
    return gate, reader


@pytest.mark.asyncio
async def test_browser_to_upstream_sends_control_message_on_barge_in() -> None:
    gate, _ = _one_frame_gate()
    websocket = _FakeWebSocket(
        [{"type": "websocket.receive", "bytes": b"\x01opus-frame"}]
    )
    upstream = _FakeUpstream()

    await _browser_to_upstream(websocket, upstream, gate)

    assert upstream.sent == [b"\x01opus-frame"]
    assert websocket.sent == [bytes([0x03, BARGE_IN_CONTROL_BYTE])]


@pytest.mark.asyncio
async def test_browser_to_upstream_relays_normally_without_barge_in() -> None:
    quiet_reader = _RecordingReader()
    quiet_reader.next_pcm = [0.0] * round(24_000 * 0.04)
    gate = BargeInGate(
        reader=quiet_reader,
        detector=BargeInDetector(
            activation_threshold=0.02,
            activation_sustained_frames=3,
            deactivation_threshold=0.015,
            deactivation_sustained_frames=12,
        ),
        sample_rate=24_000,
    )
    websocket = _FakeWebSocket(
        [{"type": "websocket.receive", "bytes": b"\x01opus-frame"}]
    )
    upstream = _FakeUpstream()

    await _browser_to_upstream(websocket, upstream, gate)

    assert upstream.sent == [b"\x01opus-frame"]
    assert websocket.sent == []


@pytest.mark.asyncio
async def test_upstream_to_browser_withholds_audio_but_forwards_text_while_muting() -> None:
    gate, _ = _one_frame_gate()
    gate.observe_user_audio(b"opus")
    assert gate.muting is True

    async def upstream_frames():
        yield b"\x01assistant-audio"
        yield b"\x02assistant-text"

    websocket = _FakeWebSocket()

    await _upstream_to_browser(websocket, upstream_frames(), gate)

    assert websocket.sent == [b"\x02assistant-text"]


@pytest.mark.asyncio
async def test_upstream_to_browser_forwards_audio_when_not_muting() -> None:
    quiet_reader = _RecordingReader()
    gate = BargeInGate(
        reader=quiet_reader,
        detector=BargeInDetector(
            activation_threshold=0.02,
            activation_sustained_frames=3,
            deactivation_threshold=0.015,
            deactivation_sustained_frames=12,
        ),
        sample_rate=24_000,
    )
    assert gate.muting is False

    async def upstream_frames():
        yield b"\x01assistant-audio"

    websocket = _FakeWebSocket()

    await _upstream_to_browser(websocket, upstream_frames(), gate)

    assert websocket.sent == [b"\x01assistant-audio"]


@pytest.mark.asyncio
async def test_browser_to_upstream_tolerates_undecodable_audio_and_keeps_relaying() -> None:
    class _RaisingReader:
        def append_bytes(self, _payload: bytes) -> None:
            raise ValueError("corrupt opus payload")

        def read_pcm(self) -> list[float]:
            return []

    gate = BargeInGate(
        reader=_RaisingReader(),
        detector=BargeInDetector(
            activation_threshold=0.02,
            activation_sustained_frames=3,
            deactivation_threshold=0.015,
            deactivation_sustained_frames=12,
        ),
        sample_rate=24_000,
    )
    websocket = _FakeWebSocket(
        [{"type": "websocket.receive", "bytes": b"\x01broken-opus"}]
    )
    upstream = _FakeUpstream()

    await _browser_to_upstream(websocket, upstream, gate)

    assert upstream.sent == [b"\x01broken-opus"]
    assert websocket.sent == []
