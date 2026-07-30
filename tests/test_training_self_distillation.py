from array import array
from pathlib import Path
import wave

from tolkien_voice.training.rollout import Rollout, RolloutGroup
from tolkien_voice.training.self_distillation import (
    leading_silence_seconds,
    select_best_rollout,
)


def _rollout(*, yield_delay_reward: float, quality_reward: float, index: int = 0) -> Rollout:
    return Rollout(
        transcript=f"reply {index}",
        yield_delay_reward=yield_delay_reward,
        quality_reward=quality_reward,
        delay_seconds=0.5,
        raw_audio_path=Path(f"/tmp/rollout-{index}.opus"),
        barge_in_seconds=1.1,
        first_output_received_seconds=0.2,
    )


def test_select_best_rollout_excludes_slow_yields_and_ranks_by_quality() -> None:
    group = RolloutGroup(
        probe_id="aragorn-lore-rangers",
        character="aragorn",
        rollouts=[
            _rollout(yield_delay_reward=-0.05, quality_reward=0.1, index=0),
            _rollout(yield_delay_reward=-0.9, quality_reward=0.9, index=1),
            _rollout(yield_delay_reward=-2.5, quality_reward=1.0, index=2),
        ],
    )

    best = select_best_rollout(group, yield_delay_reward_gate=-1.0)

    assert best is not None
    assert best.transcript == "reply 1"


def test_select_best_rollout_returns_none_when_nothing_clears_the_gate() -> None:
    group = RolloutGroup(
        probe_id="aragorn-lore-rangers",
        character="aragorn",
        rollouts=[
            _rollout(yield_delay_reward=-2.0, quality_reward=0.9, index=0),
            _rollout(yield_delay_reward=-3.0, quality_reward=0.5, index=1),
        ],
    )

    assert select_best_rollout(group, yield_delay_reward_gate=-1.0) is None


def test_select_best_rollout_gate_is_inclusive() -> None:
    group = RolloutGroup(
        probe_id="aragorn-lore-rangers",
        character="aragorn",
        rollouts=[_rollout(yield_delay_reward=-1.0, quality_reward=0.4, index=0)],
    )

    best = select_best_rollout(group, yield_delay_reward_gate=-1.0)

    assert best is not None
    assert best.transcript == "reply 0"


def _write_mono_wav(path: Path, *, silence_seconds: float, speech_seconds: float) -> None:
    sample_rate = 24_000
    samples = array("h")
    samples.extend([0] * round(silence_seconds * sample_rate))
    samples.extend([5000] * round(speech_seconds * sample_rate))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())


def test_leading_silence_seconds_finds_first_audible_frame(tmp_path: Path) -> None:
    wav_path = tmp_path / "interruption.wav"
    _write_mono_wav(wav_path, silence_seconds=0.3, speech_seconds=0.5)

    result = leading_silence_seconds(wav_path)

    assert 0.28 <= result <= 0.32


def test_leading_silence_seconds_is_zero_when_speech_starts_immediately(tmp_path: Path) -> None:
    wav_path = tmp_path / "interruption.wav"
    _write_mono_wav(wav_path, silence_seconds=0.0, speech_seconds=0.5)

    assert leading_silence_seconds(wav_path) == 0.0
