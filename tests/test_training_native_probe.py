from array import array
from pathlib import Path
import wave

from tolkien_voice.training.native_probe import load_user_channel, summarize_audio_packets


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
