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


import json

import sphn

from tolkien_voice.training.manifest import validate_rendered_examples
from tolkien_voice.training.schema import Split
from tolkien_voice.training.self_distillation import build_self_distillation_example


def _write_opus_fixture(path: Path, *, seconds: float, amplitude: float) -> None:
    import numpy as np

    sample_rate = 24_000
    frame_samples = round(sample_rate * 0.04)
    writer = sphn.OpusStreamWriter(sample_rate)
    total_samples = round(seconds * sample_rate)
    audio_bytes = bytearray()
    for offset in range(0, total_samples, frame_samples):
        length = min(frame_samples, total_samples - offset)
        frame = np.full(frame_samples, amplitude, dtype=np.float32)
        frame[length:] = 0.0
        writer.append_pcm(frame)
        payload = writer.read_bytes()
        if payload:
            audio_bytes.extend(payload)
    path.write_bytes(bytes(audio_bytes))


def test_build_self_distillation_example_produces_a_valid_rendered_example(
    tmp_path: Path,
) -> None:
    opening_wav = tmp_path / "opening.wav"
    interruption_wav = tmp_path / "interruption.wav"
    _write_mono_wav(opening_wav, silence_seconds=0.0, speech_seconds=1.0)
    _write_mono_wav(interruption_wav, silence_seconds=0.2, speech_seconds=0.6)

    raw_audio_path = tmp_path / "rollout-0.opus"
    _write_opus_fixture(raw_audio_path, seconds=1.0, amplitude=0.2)

    rollout = Rollout(
        transcript="Go on, then.",
        yield_delay_reward=-0.3,
        quality_reward=0.8,
        delay_seconds=0.3,
        raw_audio_path=raw_audio_path,
        barge_in_seconds=1.0,
        first_output_received_seconds=0.1,
    )
    output_directory = tmp_path / "corpus"

    example = build_self_distillation_example(
        rollout,
        scenario_id="aragorn-lore-rangers",
        character="aragorn",
        opening_wav_path=opening_wav,
        interruption_wav_path=interruption_wav,
        system_prompt="<system>You are Aragorn.<system>",
        split=Split.TRAIN,
        output_directory=output_directory,
    )

    validate_rendered_examples([example])
    assert example.split == Split.TRAIN
    assert example.voice_license == "self-distillation-v6-continuation"
    assert example.wav_path.is_file()
    assert example.alignment_path.is_file()
    payload = json.loads(example.alignment_path.read_text())
    assert payload["system_prompt"] == "<system>You are Aragorn.<system>"
    assert len(payload["alignments"]) >= 1


def test_build_self_distillation_example_raises_on_channel_clip(
    tmp_path: Path,
) -> None:
    """Tests that build_self_distillation_example raises ValueError when mixed
    samples exceed MAX_SAFE_SAMPLE, matching the behavior in render.py."""
    opening_wav = tmp_path / "opening.wav"
    interruption_wav = tmp_path / "interruption.wav"

    # Write high-amplitude WAV files that will exceed MAX_SAFE_SAMPLE when mixed.
    # MAX_SAFE_SAMPLE = int(32_767 * 0.99) = 32438
    sample_rate = 24_000

    # opening: 1.0 second of amplitude 20000
    opening_samples = array("h")
    opening_samples.extend([20000] * round(1.0 * sample_rate))
    with wave.open(str(opening_wav), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(opening_samples.tobytes())

    # interruption: 0.2 seconds silence + 0.6 seconds at amplitude 15000
    # This overlaps with the opening on the right channel and will exceed the limit.
    interruption_samples = array("h")
    interruption_samples.extend([0] * round(0.2 * sample_rate))
    interruption_samples.extend([15000] * round(0.6 * sample_rate))
    with wave.open(str(interruption_wav), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(interruption_samples.tobytes())

    raw_audio_path = tmp_path / "rollout-0.opus"
    _write_opus_fixture(raw_audio_path, seconds=1.0, amplitude=0.05)

    rollout = Rollout(
        transcript="Go on, then.",
        yield_delay_reward=-0.3,
        quality_reward=0.8,
        delay_seconds=0.3,
        raw_audio_path=raw_audio_path,
        barge_in_seconds=0.3,  # Position interruption to overlap with opening
        first_output_received_seconds=0.1,
    )
    output_directory = tmp_path / "corpus"

    try:
        build_self_distillation_example(
            rollout,
            scenario_id="aragorn-lore-rangers",
            character="aragorn",
            opening_wav_path=opening_wav,
            interruption_wav_path=interruption_wav,
            system_prompt="<system>You are Aragorn.<system>",
            split=Split.TRAIN,
            output_directory=output_directory,
        )
        # If we get here, the function did not raise as expected
        assert False, "Expected ValueError for clipped audio"
    except ValueError as e:
        # Verify the error message contains "clipped" as specified
        assert "clipped" in str(e), f"Error message should contain 'clipped': {e}"
        assert "aragorn-lore-rangers" in str(e), f"Error message should contain scenario_id: {e}"
