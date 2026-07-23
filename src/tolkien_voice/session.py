from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol
from uuid import uuid4

from tolkien_voice.characters import CharacterRegistry, ResolvedCharacter
from tolkien_voice.lease import LeaseOwner, LeaseTimeoutError


class SessionState(StrEnum):
    IDLE = "idle"
    WAITING_FOR_LEASE = "waiting_for_lease"
    LOADING = "loading"
    READY = "ready"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    STOPPING = "stopping"
    FAILED = "failed"


class Lease(Protocol):
    async def acquire(self, owner: LeaseOwner, timeout_seconds: float) -> None: ...

    def current_owner(self) -> LeaseOwner | None: ...

    def release(self) -> None: ...


class Runtime(Protocol):
    async def recover(self) -> None: ...

    async def start(self, character_id: str) -> None: ...

    async def stop(self) -> None: ...


@dataclass(frozen=True)
class SessionView:
    state: SessionState
    session_id: str | None = None
    character_id: str | None = None
    character_name: str | None = None
    voice_id: str | None = None
    blocking_owner: dict[str, str] | None = None
    error: str | None = None


@dataclass
class _ActiveSession:
    session_id: str
    character: ResolvedCharacter
    state: SessionState
    activation_task: asyncio.Task[None] | None = None
    cleanup_task: asyncio.Task[None] | None = None
    lease_acquired: bool = False


class SessionConflictError(RuntimeError):
    def __init__(self, session: SessionView) -> None:
        super().__init__("a Tolkien Voice session is already active")
        self.session = session


class SessionController:
    def __init__(
        self,
        *,
        registry: CharacterRegistry,
        lease: Lease,
        runtime: Runtime,
        lease_wait_seconds: float,
        disconnect_idle_seconds: float,
    ) -> None:
        self.registry = registry
        self.lease = lease
        self.runtime = runtime
        self.lease_wait_seconds = lease_wait_seconds
        self.disconnect_idle_seconds = disconnect_idle_seconds
        self._active: _ActiveSession | None = None
        self._lock = asyncio.Lock()
        self._last_failure: SessionView | None = None
        self.transition_history: list[SessionState] = [SessionState.IDLE]

    async def initialize(self) -> None:
        await self.runtime.recover()
        self.lease.current_owner()

    def status(self) -> SessionView:
        if self._active is None:
            if self._last_failure is not None:
                return SessionView(
                    state=SessionState.IDLE,
                    blocking_owner=self._last_failure.blocking_owner,
                    error=self._last_failure.error,
                )
            return SessionView(state=SessionState.IDLE)
        return self._view(self._active)

    async def start(self, character_id: str, voice_id: str | None = None) -> SessionView:
        character = self.registry.resolve(character_id, voice_id)
        async with self._lock:
            if self._active is not None:
                raise SessionConflictError(self._view(self._active))
            self._last_failure = None
            session = _ActiveSession(
                session_id=str(uuid4()),
                character=character,
                state=SessionState.WAITING_FOR_LEASE,
            )
            self._active = session
            self._record_transition(SessionState.WAITING_FOR_LEASE)
            session.activation_task = asyncio.create_task(self._activate(session))
            return self._view(session)

    async def end(self, session_id: str) -> None:
        async with self._lock:
            session = self._require_active(session_id)
            previous_state = session.state
            self._set_state(session, SessionState.STOPPING)
            activation_task = session.activation_task
            cleanup_task = session.cleanup_task
            session.cleanup_task = None
        current = asyncio.current_task()
        for task in (cleanup_task, activation_task):
            if task is not None and task is not current and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if previous_state not in {SessionState.WAITING_FOR_LEASE, SessionState.FAILED}:
            await self.runtime.stop()
        if session.lease_acquired:
            self.lease.release()
            session.lease_acquired = False
        async with self._lock:
            if self._active is session:
                self._active = None
                self._record_transition(SessionState.IDLE)

    async def mark_connected(self, session_id: str) -> None:
        async with self._lock:
            session = self._require_active(session_id)
            if session.state not in {SessionState.READY, SessionState.DISCONNECTED}:
                raise RuntimeError(f"session cannot connect from {session.state}")
            if session.cleanup_task is not None:
                session.cleanup_task.cancel()
                session.cleanup_task = None
            self._set_state(session, SessionState.CONNECTED)

    async def mark_disconnected(self, session_id: str) -> None:
        async with self._lock:
            session = self._require_active(session_id)
            if session.state != SessionState.CONNECTED:
                return
            self._set_state(session, SessionState.DISCONNECTED)
            session.cleanup_task = asyncio.create_task(self._expire_disconnected(session))

    def resolved_for_websocket(self, session_id: str) -> ResolvedCharacter:
        session = self._require_active(session_id)
        if session.state not in {SessionState.READY, SessionState.DISCONNECTED}:
            raise RuntimeError(f"session is not ready: {session.state}")
        return session.character

    async def _activate(self, session: _ActiveSession) -> None:
        try:
            await self.lease.acquire(
                LeaseOwner(kind="voice", name="tolkien-voice"),
                timeout_seconds=self.lease_wait_seconds,
            )
            session.lease_acquired = True
            async with self._lock:
                if self._active is not session:
                    return
                self._set_state(session, SessionState.LOADING)
            await self.runtime.start(session.character.id)
            async with self._lock:
                if self._active is session:
                    self._set_state(session, SessionState.READY)
        except asyncio.CancelledError:
            raise
        except LeaseTimeoutError as error:
            await self._fail(session, "large-model lease timed out", error.owner)
        except Exception:
            if session.lease_acquired:
                await self.runtime.stop()
            await self._fail(session, "PersonaPlex failed to start", None)

    async def _fail(
        self,
        session: _ActiveSession,
        message: str,
        owner: LeaseOwner | None,
    ) -> None:
        if session.lease_acquired:
            self.lease.release()
            session.lease_acquired = False
        blocking_owner = None if owner is None else {"kind": owner.kind, "name": owner.name}
        async with self._lock:
            if self._active is not session:
                return
            self._set_state(session, SessionState.FAILED)
            self._last_failure = SessionView(
                state=SessionState.FAILED,
                session_id=session.session_id,
                character_id=session.character.id,
                character_name=session.character.display_name,
                voice_id=session.character.voice_prompt,
                blocking_owner=blocking_owner,
                error=message,
            )
            self._active = None
            self._record_transition(SessionState.IDLE)

    async def _expire_disconnected(self, session: _ActiveSession) -> None:
        try:
            await asyncio.sleep(self.disconnect_idle_seconds)
            async with self._lock:
                if self._active is not session or session.state != SessionState.DISCONNECTED:
                    return
            await self.end(session.session_id)
        except asyncio.CancelledError:
            return

    def _require_active(self, session_id: str) -> _ActiveSession:
        if self._active is None or self._active.session_id != session_id:
            raise KeyError(session_id)
        return self._active

    def _set_state(self, session: _ActiveSession, state: SessionState) -> None:
        session.state = state
        self._record_transition(state)

    def _record_transition(self, state: SessionState) -> None:
        self.transition_history.append(state)

    @staticmethod
    def _view(session: _ActiveSession) -> SessionView:
        return SessionView(
            state=session.state,
            session_id=session.session_id,
            character_id=session.character.id,
            character_name=session.character.display_name,
            voice_id=session.character.voice_prompt,
        )
