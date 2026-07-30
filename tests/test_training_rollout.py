import json
from pathlib import Path

import httpx
import pytest

import tolkien_voice.training.rollout as rollout_module
from tolkien_voice.training.judge import JudgeClient
from tolkien_voice.training.rollout import Rollout, RolloutGroup, collect_rollout_group


def _rollout(
    *,
    yield_delay_reward: float,
    quality_reward: float,
    raw_audio_path: Path = Path("/tmp/rollout.opus"),
    barge_in_seconds: float = 1.1,
    first_output_received_seconds: float | None = 0.2,
) -> Rollout:
    return Rollout(
        transcript="text",
        yield_delay_reward=yield_delay_reward,
        quality_reward=quality_reward,
        delay_seconds=0.5,
        raw_audio_path=raw_audio_path,
        barge_in_seconds=barge_in_seconds,
        first_output_received_seconds=first_output_received_seconds,
    )


def test_single_rollout_group_is_degenerate() -> None:
    group = RolloutGroup(
        probe_id="aragorn-lore-rangers",
        character="aragorn",
        rollouts=[_rollout(yield_delay_reward=-0.5, quality_reward=0.7)],
    )

    assert group.is_degenerate is True


def test_group_is_degenerate_when_both_rewards_are_constant() -> None:
    group = RolloutGroup(
        probe_id="aragorn-lore-rangers",
        character="aragorn",
        rollouts=[
            _rollout(yield_delay_reward=-0.5, quality_reward=0.7),
            _rollout(yield_delay_reward=-0.5, quality_reward=0.7),
        ],
    )

    assert group.is_degenerate is True


def test_group_is_not_degenerate_when_yield_reward_varies() -> None:
    group = RolloutGroup(
        probe_id="aragorn-lore-rangers",
        character="aragorn",
        rollouts=[
            _rollout(yield_delay_reward=-0.5, quality_reward=0.7),
            _rollout(yield_delay_reward=-1.2, quality_reward=0.7),
        ],
    )

    assert group.is_degenerate is False


def test_group_is_not_degenerate_when_quality_reward_varies() -> None:
    group = RolloutGroup(
        probe_id="aragorn-lore-rangers",
        character="aragorn",
        rollouts=[
            _rollout(yield_delay_reward=-0.5, quality_reward=0.7),
            _rollout(yield_delay_reward=-0.5, quality_reward=0.2),
        ],
    )

    assert group.is_degenerate is False


@pytest.mark.asyncio
async def test_collect_rollout_group_scores_post_barge_text_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[float] = []
    raw_audio_paths: list[Path] = []

    async def fake_probe_native_interruption(
        url: str,
        opening_wav_path: Path,
        interruption_wav_path: Path,
        *,
        max_wait_seconds: float,
        response_seconds: float,
        raw_audio_path: Path,
    ) -> dict:
        calls.append(response_seconds)
        raw_audio_paths.append(raw_audio_path)
        index = len(calls)
        return {
            "output_frames": [(0.0, 0.08), (0.04, 0.01), (0.08, 0.01), (0.12, 0.01), (0.16, 0.01)],
            "barge_in_ms": 100,
            "max_wait_seconds": max_wait_seconds,
            "first_output_received_seconds": 0.0,
            "interruption_response_ms": 40,
            "interruption_text": f"pre-barge speech. post-barge reply {index}",
            "post_barge_text": f"post-barge reply {index}",
        }

    monkeypatch.setattr(rollout_module, "probe_native_interruption", fake_probe_native_interruption)

    scored_user_messages: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        scored_user_messages.append(body["messages"][1]["content"])
        content = json.dumps(
            {
                "persona_score": 0.1 * len(scored_user_messages),
                "lore_score": 0.5,
                "identified_character": "aragorn",
                "rationale": "ok",
            }
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    judge = JudgeClient(base_url="http://teacher.test/v1", transport=httpx.MockTransport(handler))
    opening_wav = tmp_path / "opening.wav"
    interruption_wav = tmp_path / "interruption.wav"
    opening_wav.touch()
    interruption_wav.touch()

    group = await collect_rollout_group(
        "ws://probe",
        opening_wav,
        interruption_wav,
        probe_id="aragorn-lore-rangers",
        character="aragorn",
        probe_prompt="What did your years leading the Rangers of the North teach you?",
        expected_traits=["grounded", "resolute", "first-person"],
        judge=judge,
        raw_audio_directory=tmp_path,
        group_size=3,
        seed=44,
    )

    assert len(calls) == 3
    assert len(group.rollouts) == 3
    assert group.probe_id == "aragorn-lore-rangers"
    assert group.character == "aragorn"
    assert all("post-barge reply" in message for message in scored_user_messages)
    assert all("pre-barge speech" not in message for message in scored_user_messages)
    assert group.rollouts[0].transcript == "post-barge reply 1"
    assert group.rollouts[0].delay_seconds == 0.04
    assert len(set(raw_audio_paths)) == 3
    assert group.rollouts[0].raw_audio_path == raw_audio_paths[0]
    assert group.rollouts[0].barge_in_seconds == 0.1
    assert group.rollouts[0].first_output_received_seconds == 0.0


@pytest.mark.asyncio
async def test_collect_rollout_group_propagates_mid_group_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[float] = []

    async def failing_probe_native_interruption(
        url: str,
        opening_wav_path: Path,
        interruption_wav_path: Path,
        *,
        max_wait_seconds: float,
        response_seconds: float,
        raw_audio_path: Path,
    ) -> dict:
        calls.append(max_wait_seconds)
        if len(calls) == 2:
            raise RuntimeError("probe failed")
        return {
            "output_frames": [(0.0, 0.08), (0.04, 0.01), (0.08, 0.01), (0.12, 0.01), (0.16, 0.01)],
            "barge_in_ms": 100,
            "max_wait_seconds": max_wait_seconds,
            "first_output_received_seconds": 0.0,
            "interruption_response_ms": 40,
            "interruption_text": "pre-barge speech. post-barge reply",
            "post_barge_text": "post-barge reply",
        }

    monkeypatch.setattr(rollout_module, "probe_native_interruption", failing_probe_native_interruption)

    def handler(request: httpx.Request) -> httpx.Response:
        content = json.dumps(
            {
                "persona_score": 0.5,
                "lore_score": 0.5,
                "identified_character": "aragorn",
                "rationale": "ok",
            }
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    judge = JudgeClient(base_url="http://teacher.test/v1", transport=httpx.MockTransport(handler))
    opening_wav = tmp_path / "opening.wav"
    interruption_wav = tmp_path / "interruption.wav"
    opening_wav.touch()
    interruption_wav.touch()

    with pytest.raises(RuntimeError, match="probe failed"):
        await collect_rollout_group(
            "ws://probe",
            opening_wav,
            interruption_wav,
            probe_id="aragorn-lore-rangers",
            character="aragorn",
            probe_prompt="What did your years leading the Rangers of the North teach you?",
            expected_traits=["grounded", "resolute", "first-person"],
            judge=judge,
            raw_audio_directory=tmp_path,
            group_size=3,
            seed=44,
        )

    assert len(calls) == 2


@pytest.mark.asyncio
async def test_collect_rollout_group_rejects_group_size_below_one() -> None:
    with pytest.raises(ValueError, match="group_size"):
        await collect_rollout_group(
            "ws://probe",
            Path("/tmp/opening.wav"),
            Path("/tmp/interruption.wav"),
            probe_id="x",
            character="aragorn",
            probe_prompt="p",
            expected_traits=[],
            judge=JudgeClient(base_url="http://teacher.test/v1"),
            raw_audio_directory=Path("/tmp"),
            group_size=0,
            seed=1,
        )
