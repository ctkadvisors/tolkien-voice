from __future__ import annotations

import asyncio
from dataclasses import asdict
from urllib.parse import urlencode, urlparse, urlunparse

from fastapi import APIRouter, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
import websockets

from tolkien_voice.characters import (
    CharacterRegistry,
    UnknownCharacterError,
    VoiceNotAllowedError,
)
from tolkien_voice.session import SessionConflictError, SessionController


MAX_FRAME_BYTES = 1024 * 1024


class StartSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    character_id: str
    voice_id: str | None = None


def build_upstream_websocket_url(
    personaplex_url: str,
    *,
    text_prompt: str,
    voice_prompt: str,
) -> str:
    parsed = urlparse(personaplex_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    query = urlencode({"text_prompt": text_prompt, "voice_prompt": voice_prompt})
    return urlunparse((scheme, parsed.netloc, "/api/chat", "", query, ""))


def create_api_router(
    controller: SessionController,
    registry: CharacterRegistry,
    *,
    personaplex_url: str,
) -> APIRouter:
    router = APIRouter()

    @router.get("/api/characters")
    async def characters() -> dict[str, object]:
        return {
            "characters": [
                {
                    "id": card.id,
                    "display_name": card.display_name,
                    "summary": card.summary,
                    "default_voice": card.default_voice,
                    "allowed_voices": card.allowed_voices,
                }
                for card in registry.cards()
            ]
        }

    @router.get("/api/session")
    async def session_status() -> dict[str, object]:
        return _public_session(controller)

    @router.post("/api/session", status_code=202)
    async def start_session(request: StartSessionRequest) -> JSONResponse:
        try:
            await controller.start(request.character_id, request.voice_id)
        except UnknownCharacterError as error:
            raise HTTPException(status_code=404, detail="unknown character") from error
        except VoiceNotAllowedError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except SessionConflictError as error:
            payload = asdict(error.session)
            payload["state"] = error.session.state.value
            return JSONResponse(status_code=409, content=payload)
        return JSONResponse(status_code=202, content=_public_session(controller))

    @router.delete("/api/session/{session_id}", status_code=204)
    async def end_session(session_id: str) -> Response:
        try:
            await controller.end(session_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="unknown session") from error
        return Response(status_code=204)

    @router.websocket("/api/session/{session_id}/chat")
    async def chat(websocket: WebSocket, session_id: str) -> None:
        try:
            character = controller.resolved_for_websocket(session_id)
        except KeyError:
            await websocket.close(code=4404)
            return
        except RuntimeError:
            await websocket.close(code=4409)
            return
        await websocket.accept()
        upstream_url = build_upstream_websocket_url(
            personaplex_url,
            text_prompt=character.text_prompt,
            voice_prompt=character.voice_prompt,
        )
        marked_connected = False
        try:
            async with websockets.connect(
                upstream_url,
                compression=None,
                max_size=MAX_FRAME_BYTES,
                open_timeout=30,
            ) as upstream:
                first = await upstream.recv()
                if not isinstance(first, bytes) or not first or first[0] != 0:
                    await websocket.close(code=1011, reason="invalid PersonaPlex handshake")
                    return
                await websocket.send_bytes(first)
                await controller.mark_connected(session_id)
                marked_connected = True
                browser_to_upstream = asyncio.create_task(
                    _browser_to_upstream(websocket, upstream)
                )
                upstream_to_browser = asyncio.create_task(
                    _upstream_to_browser(websocket, upstream)
                )
                done, pending = await asyncio.wait(
                    {browser_to_upstream, upstream_to_browser},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for task in done:
                    task.result()
        except WebSocketDisconnect:
            pass
        except Exception:
            try:
                await websocket.close(code=1011, reason="PersonaPlex connection failed")
            except RuntimeError:
                pass
        finally:
            if marked_connected:
                try:
                    await controller.mark_disconnected(session_id)
                except KeyError:
                    pass

    return router


async def _browser_to_upstream(websocket: WebSocket, upstream: object) -> None:
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
        payload = message.get("bytes")
        if payload is None:
            await websocket.close(code=1003, reason="binary frames required")
            return
        if len(payload) > MAX_FRAME_BYTES:
            await websocket.close(code=1009, reason="frame too large")
            return
        await upstream.send(payload)  # type: ignore[attr-defined]


async def _upstream_to_browser(websocket: WebSocket, upstream: object) -> None:
    async for payload in upstream:  # type: ignore[attr-defined]
        if not isinstance(payload, bytes):
            await websocket.close(code=1011, reason="invalid upstream frame")
            return
        await websocket.send_bytes(payload)


def _public_session(controller: SessionController) -> dict[str, object]:
    view = controller.status()
    payload = asdict(view)
    payload["state"] = view.state.value
    return {key: value for key, value in payload.items() if value is not None}
