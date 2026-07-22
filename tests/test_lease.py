import asyncio
import json
from pathlib import Path

import pytest

from tolkien_voice.lease import LeaseOwner, LeaseTimeoutError, SparkModelLease


def make_lease(tmp_path: Path, *, poll_seconds: float = 0.01) -> SparkModelLease:
    return SparkModelLease(
        lock_path=tmp_path / "big-model.lock",
        owner_path=tmp_path / "big-model-owner.json",
        poll_seconds=poll_seconds,
    )


def test_exclusive_acquisition_exposes_current_owner(tmp_path: Path) -> None:
    first = make_lease(tmp_path)
    second = make_lease(tmp_path)
    owner = LeaseOwner(kind="voice", name="tolkien-voice")

    assert first.try_acquire(owner) is True
    assert second.try_acquire(LeaseOwner(kind="text", name="coder")) is False
    assert second.current_owner() == first.owner

    first.release()


def test_release_is_idempotent_and_allows_next_holder(tmp_path: Path) -> None:
    first = make_lease(tmp_path)
    second = make_lease(tmp_path)
    assert first.try_acquire(LeaseOwner(kind="voice", name="tolkien-voice"))

    first.release()
    first.release()

    assert second.try_acquire(LeaseOwner(kind="text", name="coder"))
    second.release()


def test_context_manager_releases_lease(tmp_path: Path) -> None:
    first = make_lease(tmp_path)
    second = make_lease(tmp_path)

    with first.held(LeaseOwner(kind="voice", name="tolkien-voice")):
        assert not second.try_acquire(LeaseOwner(kind="text", name="coder"))

    assert second.try_acquire(LeaseOwner(kind="text", name="coder"))
    second.release()


def test_stale_metadata_is_ignored_when_kernel_lock_is_free(tmp_path: Path) -> None:
    owner_path = tmp_path / "big-model-owner.json"
    owner_path.write_text(
        json.dumps(
            {
                "kind": "voice",
                "name": "stale-session",
                "pid": 999999,
                "acquired_at": "2026-07-21T20:00:00Z",
                "secret": "hf_abcdefghijklmnopqrstuvwxyz",
            }
        ),
        encoding="utf-8",
    )

    assert make_lease(tmp_path).current_owner() is None


@pytest.mark.asyncio
async def test_async_acquisition_times_out_with_public_owner(tmp_path: Path) -> None:
    first = make_lease(tmp_path)
    second = make_lease(tmp_path)
    first.try_acquire(LeaseOwner(kind="voice", name="tolkien-voice"))

    with pytest.raises(LeaseTimeoutError) as caught:
        await second.acquire(
            LeaseOwner(kind="text", name="coder"), timeout_seconds=0.03
        )

    assert caught.value.owner is not None
    assert caught.value.owner.kind == "voice"
    assert caught.value.owner.name == "tolkien-voice"
    first.release()


@pytest.mark.asyncio
async def test_async_acquisition_waits_until_release(tmp_path: Path) -> None:
    first = make_lease(tmp_path)
    second = make_lease(tmp_path)
    first.try_acquire(LeaseOwner(kind="text", name="coder"))

    async def release_first() -> None:
        await asyncio.sleep(0.02)
        first.release()

    release_task = asyncio.create_task(release_first())
    await second.acquire(LeaseOwner(kind="voice", name="tolkien-voice"), 0.2)
    await release_task

    assert second.owner is not None
    assert second.owner.kind == "voice"
    second.release()


def test_owner_json_has_only_public_fields(tmp_path: Path) -> None:
    lease = make_lease(tmp_path)
    lease.try_acquire(LeaseOwner(kind="voice", name="tolkien-voice"))

    written = json.loads((tmp_path / "big-model-owner.json").read_text(encoding="utf-8"))

    assert set(written) == {"kind", "name", "pid", "acquired_at"}
    assert "hf_" not in json.dumps(written)
    lease.release()

