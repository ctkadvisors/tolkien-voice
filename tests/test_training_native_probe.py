from array import array
from pathlib import Path
import wave

from tolkien_voice.training.native_probe import (
    ActivityDetector,
    load_user_channel,
    measure_interruption_response,
    summarize_audio_packets,
)


def test_probe_loads_right_user_channel_and_finds_speech_end(tmp_path: Path) -> None:
    path = tmp_path / "duplex.wav"
    frames = array("h")
    for index in range(2400):
        frames.extend((2000, 1000 if 400 <= index < 1600 else 0))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(frames.tobytes())

    sample_rate, samples, speech_end_seconds = load_user_channel(path)

    assert sample_rate == 24_000
    assert len(samples) == 2400
    assert 0.066 <= speech_end_seconds <= 0.067
    assert samples[0] == 0
    assert samples[500] > 0


def test_packet_summary_counts_only_playout_sized_gaps() -> None:
    summary = summarize_audio_packets([0.10, 0.14, 0.18, 0.36, 0.40], gap_threshold=0.12)

    assert summary["audio_messages"] == 5
    assert summary["underrun_count"] == 1
    assert summary["max_packet_gap_ms"] == 180


def test_interruption_response_requires_sustained_agent_silence() -> None:
    frames = [
        (1.00, 0.08),
        (1.04, 0.07),
        (1.08, 0.06),
        (1.12, 0.01),
        (1.16, 0.06),
        (1.20, 0.01),
        (1.24, 0.008),
        (1.28, 0.006),
        (1.32, 0.005),
        (1.36, 0.004),
    ]

    response_ms = measure_interruption_response(
        frames,
        barge_in_seconds=1.10,
        silence_threshold=0.015,
        sustained_frames=4,
    )

    assert response_ms == 100


def test_interruption_response_is_unmeasured_when_agent_never_yields() -> None:
    frames = [(1.0 + index * 0.04, 0.08) for index in range(10)]

    response_ms = measure_interruption_response(
        frames,
        barge_in_seconds=1.10,
        silence_threshold=0.015,
        sustained_frames=4,
    )

    assert response_ms is None


def test_activity_detector_requires_consecutive_active_frames() -> None:
    detector = ActivityDetector(threshold=0.03, sustained_frames=3)

    assert detector.append(0.04) is False
    assert detector.append(0.01) is False
    assert detector.append(0.04) is False
    assert detector.append(0.05) is False
    assert detector.append(0.06) is True
