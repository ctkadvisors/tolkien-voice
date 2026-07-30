from array import array
import json
from pathlib import Path
import struct
import sys
from types import ModuleType
import wave

import pytest

from tolkien_voice.training.manifest import validate_rendered_examples
from tolkien_voice.training.rollout import Rollout, RolloutGroup
from tolkien_voice.training.schema import Split
from tolkien_voice.training.self_distillation import (
    build_self_distillation_example,
    leading_silence_seconds,
    select_best_rollout,
)


def _rollout(
    *,
    yield_delay_reward: float,
    quality_reward: float,
    index: int = 0,
    first_output_received_seconds: float | None = 0.2,
    last_output_received_seconds: float | None = 0.2,
    post_barge_text_frames: list[tuple[float, str]] | None = None,
) -> Rollout:
    return Rollout(
        transcript=f"reply {index}",
        yield_delay_reward=yield_delay_reward,
        quality_reward=quality_reward,
        delay_seconds=0.5,
        raw_audio_path=Path(f"/tmp/rollout-{index}.opus"),
        barge_in_seconds=1.1,
        first_output_received_seconds=first_output_received_seconds,
        last_output_received_seconds=last_output_received_seconds,
        post_barge_text_frames=(
            post_barge_text_frames
            if post_barge_text_frames is not None
            else [(first_output_received_seconds or 0.0, f"reply {index}")]
        ),
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


# --- Fake sphn/numpy modules -------------------------------------------------
#
# tolkien_voice.training.self_distillation._decode_opus_pcm imports sphn and
# numpy locally (not at module level) specifically so tests can inject fakes
# via sys.modules and avoid a hard dependency on the `evaluation` extra (see
# tests/test_training_native_probe.py for the same pattern applied to
# probe_native_interruption). These fakes don't do real Opus encode/decode --
# they just round-trip raw float32 PCM through struct-packed bytes -- but the
# FakeOpusStreamReader still models the real sphn async-decode contract that
# _decode_opus_pcm's close()+poll loop depends on: a read before close()
# returns empty, and only after close() does polling read_pcm() yield the
# real decoded data followed by None. That way, if _decode_opus_pcm ever
# regressed back to a single naive read_pcm() call, this test would catch it.


class FakeArray:
    def __init__(self, values) -> None:
        self._values = list(values)

    @property
    def size(self) -> int:
        return len(self._values)

    def reshape(self, _shape: int) -> "FakeArray":
        return self

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return FakeArray(self._values[item])
        return self._values[item]

    def __setitem__(self, item, value) -> None:
        if isinstance(item, slice):
            for index in range(*item.indices(len(self._values))):
                self._values[index] = value
        else:
            self._values[item] = value

    def __mul__(self, scalar: float) -> "FakeArray":
        return FakeArray([value * scalar for value in self._values])

    def astype(self, _dtype: object) -> "FakeArray":
        return FakeArray([int(round(value)) for value in self._values])

    def tolist(self) -> list:
        return list(self._values)


def _make_fake_numpy() -> ModuleType:
    fake_numpy = ModuleType("numpy")

    def asarray(values: object, dtype: object | None = None) -> FakeArray:
        del dtype
        if isinstance(values, FakeArray):
            return FakeArray(values._values)
        return FakeArray(list(values))

    def full(shape: int, value: float, dtype: object | None = None) -> FakeArray:
        del dtype
        return FakeArray([value] * shape)

    def zeros(shape: int, dtype: object | None = None) -> FakeArray:
        del dtype
        return FakeArray([0.0] * shape)

    def clip(values: FakeArray, low: float, high: float) -> FakeArray:
        return FakeArray([max(low, min(high, value)) for value in values])

    def concatenate(arrays: list[FakeArray]) -> FakeArray:
        combined: list[float] = []
        for item in arrays:
            combined.extend(item)
        return FakeArray(combined)

    fake_numpy.asarray = asarray
    fake_numpy.full = full
    fake_numpy.zeros = zeros
    fake_numpy.clip = clip
    fake_numpy.concatenate = concatenate
    fake_numpy.float32 = "float32"
    fake_numpy.int16 = "int16"
    return fake_numpy


def _make_fake_sphn() -> ModuleType:
    fake_sphn = ModuleType("sphn")

    class FakeOpusStreamWriter:
        def __init__(self, sample_rate: int) -> None:
            self.sample_rate = sample_rate
            self._pending: list[float] = []

        def append_pcm(self, frame: object) -> None:
            self._pending.extend(frame)

        def read_bytes(self) -> bytes:
            if not self._pending:
                return b""
            payload = struct.pack(f"{len(self._pending)}f", *self._pending)
            self._pending = []
            return payload

    class FakeOpusStreamReader:
        def __init__(self, sample_rate: int) -> None:
            self.sample_rate = sample_rate
            self._buffer = bytearray()
            self._closed = False
            self._served = False

        def append_bytes(self, payload: bytes) -> None:
            self._buffer.extend(payload)

        def close(self) -> None:
            self._closed = True

        def read_pcm(self) -> FakeArray | None:
            # A read before close() races the (simulated) background worker
            # and must return empty -- exactly like real sphn.
            if not self._closed:
                return FakeArray([])
            if not self._served:
                self._served = True
                count = len(self._buffer) // 4
                values = (
                    struct.unpack(f"{count}f", bytes(self._buffer[: count * 4]))
                    if count
                    else ()
                )
                return FakeArray(values)
            return None

    fake_sphn.OpusStreamWriter = FakeOpusStreamWriter
    fake_sphn.OpusStreamReader = FakeOpusStreamReader
    return fake_sphn


def _install_fake_audio_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "numpy", _make_fake_numpy())
    monkeypatch.setitem(sys.modules, "sphn", _make_fake_sphn())


def _write_opus_fixture(path: Path, *, seconds: float, amplitude: float) -> None:
    # sphn/numpy are imported locally here (not at module level) so this
    # helper only ever reaches whatever fake modules the calling test has
    # already installed into sys.modules via _install_fake_audio_modules.
    import numpy as np
    import sphn

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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_audio_modules(monkeypatch)

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
        # The fixture decodes to exactly 1.0s of audio (24_000 samples at
        # 24kHz), so a wall-clock span of exactly 1.0s is fully consistent
        # with the decoded duration and the timing-consistency guard in
        # build_self_distillation_example should not fire.
        last_output_received_seconds=1.1,
        post_barge_text_frames=[(0.1, "Go on, then.")],
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tests that build_self_distillation_example raises ValueError when mixed
    samples exceed MAX_SAFE_SAMPLE, matching the behavior in render.py."""
    _install_fake_audio_modules(monkeypatch)

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
        # Consistent with the 1.0s-decoded fixture so the timing-consistency
        # guard doesn't preempt the clip check this test targets.
        last_output_received_seconds=1.1,
        post_barge_text_frames=[(0.1, "Go on, then.")],
    )
    output_directory = tmp_path / "corpus"

    with pytest.raises(ValueError, match="clipped") as excinfo:
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
    assert "aragorn-lore-rangers" in str(excinfo.value)


def test_build_self_distillation_example_raises_on_assistant_timing_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wall-clock span between first_output_received_seconds and
    last_output_received_seconds is measured on a different clock basis
    than the decoded assistant audio duration might suggest (see
    probe_native_interruption's docstring/comments on wall-clock drift).
    build_self_distillation_example must refuse to silently place the
    assistant audio on a reconstructed timeline when those two
    measurements disagree by more than the tolerated jitter budget.
    """
    _install_fake_audio_modules(monkeypatch)

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
        # The fixture decodes to exactly 1.0s of audio, but this claims a
        # 1.7s wall-clock span -- a 0.7s drift, far outside the ~150ms
        # jitter budget -- so the guard must raise.
        last_output_received_seconds=1.8,
        post_barge_text_frames=[(0.1, "Go on, then.")],
    )
    output_directory = tmp_path / "corpus"

    with pytest.raises(ValueError, match="timing is inconsistent") as excinfo:
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
    assert "aragorn-lore-rangers" in str(excinfo.value)
    assert not output_directory.exists() or not any(output_directory.iterdir())


def test_build_self_distillation_example_aligns_words_to_real_chunk_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Word alignments must come from each post_barge_text_frames chunk's own
    real timestamp, not from spreading the whole transcript uniformly across
    the entire captured-audio span (first_output_received_seconds to
    assistant_end_sec), which spans the model's pre-barge continuation plus
    a trailing silence window well beyond where these words were actually
    spoken. This is a regression test for that bug.
    """
    _install_fake_audio_modules(monkeypatch)

    opening_wav = tmp_path / "opening.wav"
    interruption_wav = tmp_path / "interruption.wav"
    _write_mono_wav(opening_wav, silence_seconds=0.0, speech_seconds=1.0)
    _write_mono_wav(interruption_wav, silence_seconds=0.2, speech_seconds=0.6)

    raw_audio_path = tmp_path / "rollout-0.opus"
    # 15s of captured assistant audio -- typical of a session that includes
    # a pre-barge continuation, the post-barge yield, and a trailing
    # response_seconds silence window.
    _write_opus_fixture(raw_audio_path, seconds=15.0, amplitude=0.1)

    rollout = Rollout(
        transcript="First chunk words here. Second chunk words here.",
        yield_delay_reward=-0.3,
        quality_reward=0.8,
        delay_seconds=0.3,
        raw_audio_path=raw_audio_path,
        barge_in_seconds=1.0,
        first_output_received_seconds=0.5,
        # Consistent with the 15.0s-decoded fixture (0.5 + 15.0 = 15.5) so
        # the timing-consistency guard doesn't fire.
        last_output_received_seconds=15.5,
        post_barge_text_frames=[
            (0.5, "First chunk words here. "),
            (3.0, "Second chunk words here."),
        ],
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
    payload = json.loads(example.alignment_path.read_text())
    words_by_text = {word: (start, end) for word, (start, end), _ in payload["alignments"]}

    first_start, first_end = words_by_text["First"]
    second_start, second_end = words_by_text["Second"]

    # Each chunk's words must land near that chunk's own real timestamp --
    # not proportionally spread across the full ~15.5s captured-audio span,
    # which is what the uniform-spacing bug produced.
    assert 0.5 <= first_start < 3.0
    assert 0.5 <= first_end < 3.0
    assert 3.0 <= second_start < 15.5
    assert 3.0 <= second_end < 15.5
