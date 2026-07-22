from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tolkien_voice.characters import CharacterRegistry
from tolkien_voice.lease import LeaseOwner, LeaseTimeoutError
from tolkien_voice.session import SessionConflictError, SessionController, SessionState


CARDS = Path(__file__).parents[1] / "configs" / "characters"


class FakeLease:
    def __init__(self) -> None:
        self.acquired = asyncio.Event()
        self.allow_acquire = asyncio.Event()
        self.allow_acquire.set()
        self.released = 0
        self.owner: LeaseOwner | None = None
        self.timeout_owner: LeaseOwner | None = None

    async def acquire(self, owner: LeaseOwner, timeout_seconds: float) -> None:
        if self.timeout_owner is not None:
            raise LeaseTimeoutError(self.timeout_owner)
        await asyncio.wait_for(self.allow_acquire.wait(), timeout=timeout_seconds)
        self.owner = owner.activated()
        self.acquired.set()

    def current_owner(self) -> LeaseOwner | None:
        return self.timeout_owner or self.owner

    def release(self) -> None:
        self.owner = None
        self.released += 1


class FakeRuntime:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.recovered = 0
        self.allow_start = asyncio.Event()
        self.allow_start.set()
        self.start_error: Exception | None = None

    async def recover(self) -> None:
        self.recovered += 1

    async def start(self) -> None:
        self.started += 1
        await self.allow_start.wait()
        if self.start_error is not None:
            raise self.start_error

    async def stop(self) -> None:
        self.stopped += 1


async def wait_for_state(controller: SessionController, expected: SessionState) -> None:
    for _ in range(100):
        if controller.status().state == expected:
            return
        await asyncio.sleep(0.001)
    raise AssertionError(f"session did not reach {expected}; got {controller.status().state}")


def make_controller(
    *, lease: FakeLease | None = None, runtime: FakeRuntime | None = None, idle: float = 0.02
) -> SessionController:
    return SessionController(
        registry=CharacterRegistry.from_directory(CARDS),
        lease=lease or FakeLease(),
        runtime=runtime or FakeRuntime(),
        lease_wait_seconds=1,
        disconnect_idle_seconds=idle,
    )


@pytest.mark.asyncio
async def test_start_returns_immediately_then_reaches_ready() -> None:
    controller = make_controller()

    created = await controller.start("gandalf", "NATM1.pt")

    assert created.state == SessionState.WAITING_FOR_LEASE
    assert created.session_id is not None
    await wait_for_state(controller, SessionState.READY)
    assert controller.status().character_id == "gandalf"


@pytest.mark.asyncio
async def test_second_start_is_rejected_with_active_character_and_state() -> None:
    lease = FakeLease()
    lease.allow_acquire.clear()
    controller = make_controller(lease=lease)
    await controller.start("sauron", "NATM3.pt")

    with pytest.raises(SessionConflictError) as caught:
        await controller.start("gandalf", "NATM1.pt")

    assert caught.value.session.character_id == "sauron"
    assert caught.value.session.state == SessionState.WAITING_FOR_LEASE
    await controller.end(controller.status().session_id or "")


@pytest.mark.asyncio
async def test_controller_waits_for_lease_without_touching_runtime() -> None:
    lease = FakeLease()
    lease.allow_acquire.clear()
    runtime = FakeRuntime()
    controller = make_controller(lease=lease, runtime=runtime)

    await controller.start("gandalf", "NATM1.pt")
    await asyncio.sleep(0)

    assert controller.status().state == SessionState.WAITING_FOR_LEASE
    assert runtime.started == 0
    await controller.end(controller.status().session_id or "")


@pytest.mark.asyncio
async def test_lease_timeout_records_failed_then_returns_idle() -> None:
    lease = FakeLease()
    lease.timeout_owner = LeaseOwner(kind="text", name="coder").activated()
    controller = make_controller(lease=lease)

    await controller.start("gandalf", "NATM1.pt")
    await wait_for_state(controller, SessionState.IDLE)

    status = controller.status()
    assert SessionState.FAILED in controller.transition_history
    assert status.blocking_owner == {"kind": "text", "name": "coder"}


@pytest.mark.asyncio
async def test_runtime_failure_releases_lease() -> None:
    lease = FakeLease()
    runtime = FakeRuntime()
    runtime.start_error = RuntimeError("boom")
    controller = make_controller(lease=lease, runtime=runtime)

    await controller.start("gandalf", "NATM1.pt")
    await wait_for_state(controller, SessionState.IDLE)

    assert lease.released == 1
    assert SessionState.FAILED in controller.transition_history


@pytest.mark.asyncio
async def test_explicit_end_stops_runtime_and_releases_lease() -> None:
    lease = FakeLease()
    runtime = FakeRuntime()
    controller = make_controller(lease=lease, runtime=runtime)
    started = await controller.start("gandalf", "NATM1.pt")
    await wait_for_state(controller, SessionState.READY)

    await controller.end(started.session_id or "")

    assert controller.status().state == SessionState.IDLE
    assert runtime.stopped == 1
    assert lease.released == 1


@pytest.mark.asyncio
async def test_disconnect_expires_after_idle_timeout() -> None:
    runtime = FakeRuntime()
    controller = make_controller(runtime=runtime, idle=0.01)
    started = await controller.start("gandalf", "NATM1.pt")
    await wait_for_state(controller, SessionState.READY)
    await controller.mark_connected(started.session_id or "")

    await controller.mark_disconnected(started.session_id or "")
    await wait_for_state(controller, SessionState.IDLE)

    assert runtime.stopped == 1


@pytest.mark.asyncio
async def test_reconnect_cancels_idle_cleanup() -> None:
    controller = make_controller(idle=0.03)
    started = await controller.start("gandalf", "NATM1.pt")
    await wait_for_state(controller, SessionState.READY)
    await controller.mark_connected(started.session_id or "")
    await controller.mark_disconnected(started.session_id or "")

    await asyncio.sleep(0.005)
    await controller.mark_connected(started.session_id or "")
    await asyncio.sleep(0.04)

    assert controller.status().state == SessionState.CONNECTED
    await controller.end(started.session_id or "")


@pytest.mark.asyncio
async def test_unknown_session_cannot_end_active_session() -> None:
    controller = make_controller()
    started = await controller.start("gandalf", "NATM1.pt")

    with pytest.raises(KeyError):
        await controller.end("00000000-0000-0000-0000-000000000000")

    assert controller.status().session_id == started.session_id
    await controller.end(started.session_id or "")

