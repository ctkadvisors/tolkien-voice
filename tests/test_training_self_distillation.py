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
from tolkien_voice.training.self_distillation import _rescale_to_placed_span


def test_rescale_to_placed_span_maps_endpoints_and_interpolates() -> None:
    kwargs = dict(
        first_output_received_seconds=2.0,
        wall_clock_span=10.0,
        assistant_start_sec=1.0,
        decoded_duration=5.0,
    )

    assert _rescale_to_placed_span(2.0, **kwargs) == pytest.approx(1.0)
    assert _rescale_to_placed_span(12.0, **kwargs) == pytest.approx(6.0)
    assert _rescale_to_placed_span(7.0, **kwargs) == pytest.approx(3.5)


def test_rescale_to_placed_span_returns_start_when_span_is_zero() -> None:
    result = _rescale_to_placed_span(
        5.0,
        first_output_received_seconds=5.0,
        wall_clock_span=0.0,
        assistant_start_sec=2.0,
        decoded_duration=3.0,
    )

    assert result == 2.0


def _assert_close(
    actual: tuple[float, float], expected: tuple[float, float], *, abs_tol: float = 1e-6
) -> None:
    # Deliberately not pytest.approx: several tests in this file install a
    # fake `numpy` module into sys.modules (see _install_fake_audio_modules)
    # so build_self_distillation_example's local `import numpy` picks it up
    # without requiring the real `evaluation` extra. pytest.approx's
    # equality check also probes sys.modules["numpy"] (to special-case numpy
    # arrays) and crashes on the fake module's missing attributes, so plain
    # abs-difference comparisons are used here instead.
    actual_start, actual_end = actual
    expected_start, expected_end = expected
    assert abs(actual_start - expected_start) <= abs_tol, (actual, expected)
    assert abs(actual_end - expected_end) <= abs_tol, (actual, expected)


def _rollout(
    *,
    yield_delay_reward: float,
    quality_reward: float,
    index: int = 0,
    first_output_received_seconds: float | None = 0.2,
    last_output_received_seconds: float | None = 0.2,
    post_barge_text_frames: list[tuple[float, str]] | None = None,
    output_frames: list[tuple[float, float]] | None = None,
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
        output_frames=(
            output_frames
            if output_frames is not None
            else [(first_output_received_seconds or 0.0, 0.08)]
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
        # Real audible output for the full decoded span, so the real-speech-end
        # bound (fixed for the last-chunk case) lands at the same place the
        # old assistant_end_sec bound did for this fixture.
        output_frames=[(0.1, 0.2), (1.1, 0.2)],
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
        output_frames=[(0.1, 0.2), (1.1, 0.2)],
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


def test_build_self_distillation_example_tolerates_sub_realtime_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact shape of drift observed live on Spark (2026-07-30): audio
    genuinely decodes to less playback duration than the wall-clock time it
    took to arrive, because PersonaPlex generated it slower than real-time
    on that hardware (~1.4x observed) -- not because of corpus-corrupting
    timing drift. The old absolute-tolerance guard rejected every live
    example with this shape; this test proves the new guard does not.
    """
    _install_fake_audio_modules(monkeypatch)

    opening_wav = tmp_path / "opening.wav"
    interruption_wav = tmp_path / "interruption.wav"
    _write_mono_wav(opening_wav, silence_seconds=0.0, speech_seconds=1.0)
    _write_mono_wav(interruption_wav, silence_seconds=0.2, speech_seconds=0.6)

    raw_audio_path = tmp_path / "rollout-0.opus"
    # Decodes to 10.0s -- deliberately much less than the 14.0s wall-clock
    # span below, reproducing the live ~1.4x sub-realtime ratio observed on
    # Spark (18.66-18.68s wall-clock for 13.36s decoded in two live runs).
    _write_opus_fixture(raw_audio_path, seconds=10.0, amplitude=0.1)

    rollout = Rollout(
        transcript="Go on, then.",
        yield_delay_reward=-0.3,
        quality_reward=0.8,
        delay_seconds=0.3,
        raw_audio_path=raw_audio_path,
        barge_in_seconds=1.0,
        first_output_received_seconds=0.5,
        last_output_received_seconds=14.5,
        post_barge_text_frames=[(3.0, "Go on, then.")],
        output_frames=[(0.5, 0.1), (5.0, 0.1), (9.0, 0.1)],
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

    assert example.wav_path.is_file()


def test_build_self_distillation_example_places_interruption_relative_to_real_yield(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The interruption's audible onset in the reconstructed WAV must land
    delay_seconds before the assistant's real (rescaled) speech end -- not
    at a raw barge_in_seconds, which can place the interruption after the
    assistant's compressed audio has already gone quiet under sub-realtime
    generation, teaching the model to yield before an interruption it
    hasn't heard yet. See the design spec's "Correction" section.
    """
    _install_fake_audio_modules(monkeypatch)

    opening_wav = tmp_path / "opening.wav"
    interruption_wav = tmp_path / "interruption.wav"
    # Short opening (0.1s) so it can never overlap the interruption, which
    # this scenario places around t=5.5s.
    _write_mono_wav(opening_wav, silence_seconds=0.0, speech_seconds=0.1)
    # Silence prefix is an exact multiple of FRAME_SECONDS (0.04s * 2 =
    # 0.08s) so leading_silence_seconds returns exactly 0.08, avoiding
    # frame-quantization ambiguity in this test's expected value.
    _write_mono_wav(interruption_wav, silence_seconds=0.08, speech_seconds=0.6)

    raw_audio_path = tmp_path / "rollout-0.opus"
    # Same 5/7 compression ratio as
    # test_build_self_distillation_example_tolerates_sub_realtime_generation
    # (decoded 10.0s over a 14.0s wall-clock span).
    _write_opus_fixture(raw_audio_path, seconds=10.0, amplitude=0.1)

    rollout = Rollout(
        transcript="Go on, then.",
        yield_delay_reward=-1.0,
        quality_reward=0.8,
        delay_seconds=1.0,
        raw_audio_path=raw_audio_path,
        barge_in_seconds=6.0,
        first_output_received_seconds=0.5,
        last_output_received_seconds=14.5,
        post_barge_text_frames=[(9.5, "Go on, then.")],
        # Last active frame at raw t=9.0 -- rescales to
        # 0.5 + (9.0 - 0.5) * (10.0 / 14.0) = 46/7 (~6.571s).
        output_frames=[(0.5, 0.1), (5.0, 0.1), (9.0, 0.1)],
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

    # real_speech_end_sec (rescaled) = 46/7. The interruption's own silence
    # prefix (onset_offset) exactly cancels out of the audible-onset
    # position: interruption_start_sec = (46/7 - delay_seconds) -
    # onset_offset, and the loud content starts onset_offset into the
    # clip, so audible onset = 46/7 - delay_seconds regardless of
    # onset_offset's value.
    expected_audible_onset_sec = 46 / 7 - 1.0  # ~5.571428571
    assistant_end_sec = 0.5 + 10.0  # assistant_start_sec + decoded_duration

    with wave.open(str(example.wav_path), "rb") as source:
        sample_rate = source.getframerate()
        interleaved = array("h")
        interleaved.frombytes(source.readframes(source.getnframes()))
    right_channel = interleaved[1::2]  # user/interruption channel

    def _right_channel_rms(start_sec: float, duration_sec: float) -> float:
        start = round(start_sec * sample_rate)
        end = round((start_sec + duration_sec) * sample_rate)
        window = right_channel[max(0, start) : max(0, end)]
        if not window:
            return 0.0
        return (sum(sample * sample for sample in window) / len(window)) ** 0.5

    # Silent just before the expected onset -- the interruption has not
    # started yet, and the short opening ended long ago.
    assert _right_channel_rms(expected_audible_onset_sec - 0.5, 0.3) < 100
    # Audible at and after the expected onset.
    assert _right_channel_rms(expected_audible_onset_sec + 0.05, 0.3) > 1000
    # The interruption's onset lands comfortably before the assistant's
    # placed audio ends -- the concrete failure mode this correction fixes
    # (the pre-correction code could place it after this point).
    assert expected_audible_onset_sec < assistant_end_sec


def test_build_self_distillation_example_raises_on_empty_decoded_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_audio_modules(monkeypatch)

    opening_wav = tmp_path / "opening.wav"
    interruption_wav = tmp_path / "interruption.wav"
    _write_mono_wav(opening_wav, silence_seconds=0.0, speech_seconds=1.0)
    _write_mono_wav(interruption_wav, silence_seconds=0.2, speech_seconds=0.6)

    raw_audio_path = tmp_path / "rollout-0.opus"
    raw_audio_path.write_bytes(b"")  # empty Opus stream decodes to zero samples

    rollout = Rollout(
        transcript="",
        yield_delay_reward=-0.3,
        quality_reward=0.8,
        delay_seconds=0.3,
        raw_audio_path=raw_audio_path,
        barge_in_seconds=1.0,
        first_output_received_seconds=0.5,
        last_output_received_seconds=0.5,
        post_barge_text_frames=[],
        output_frames=[],
    )
    output_directory = tmp_path / "corpus"

    with pytest.raises(ValueError, match="decoded to no samples"):
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


def test_build_self_distillation_example_raises_on_out_of_order_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_audio_modules(monkeypatch)

    opening_wav = tmp_path / "opening.wav"
    interruption_wav = tmp_path / "interruption.wav"
    _write_mono_wav(opening_wav, silence_seconds=0.0, speech_seconds=1.0)
    _write_mono_wav(interruption_wav, silence_seconds=0.2, speech_seconds=0.6)

    raw_audio_path = tmp_path / "rollout-0.opus"
    _write_opus_fixture(raw_audio_path, seconds=1.0, amplitude=0.1)

    rollout = Rollout(
        transcript="Go on, then.",
        yield_delay_reward=-0.3,
        quality_reward=0.8,
        delay_seconds=0.3,
        raw_audio_path=raw_audio_path,
        barge_in_seconds=1.0,
        first_output_received_seconds=5.0,
        last_output_received_seconds=1.0,  # before first_output -- impossible
        post_barge_text_frames=[(3.0, "Go on, then.")],
        output_frames=[(5.0, 0.1)],
    )
    output_directory = tmp_path / "corpus"

    with pytest.raises(ValueError, match="out of order"):
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


def test_build_self_distillation_example_aligns_words_to_real_chunk_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Word alignments must come from each post_barge_text_frames chunk's own
    real timestamp, not from spreading the whole transcript uniformly across
    the entire captured-audio span (first_output_received_seconds to
    assistant_end_sec), which spans the model's pre-barge continuation plus
    a trailing silence window well beyond where these words were actually
    spoken. This is a regression test for that bug -- specifically the
    interior-chunk case, where a chunk is bounded by the *next* chunk's real
    timestamp (already correct before this test was tightened).

    Bounds are exact (not a wide window): with output_frames active across
    the whole decoded span, the interior chunk's [0.5, 3.0) window and
    _word_alignments' own uniform-spacing-within-a-chunk formula fully
    determine each word's start/end, so this test can assert precise values
    instead of a multi-second range that both the buggy and fixed
    implementations would satisfy.
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
        # wall_clock_span (22.5) is 1.5x decoded_duration (15.0) -- the same
        # direction and a similar magnitude to the ~1.4x sub-realtime ratio
        # observed live on Spark (2026-07-30), so this test exercises the
        # rescale path instead of degenerating to an identity mapping.
        last_output_received_seconds=23.0,
        post_barge_text_frames=[
            (0.5, "First chunk words here. "),
            (3.0, "Second chunk words here."),
        ],
        # Real audible output for the entire decoded span, matching the new
        # last_output_received_seconds so the last active frame rescales to
        # exactly assistant_end_sec -- this test is about the interior-chunk
        # bound, not the last-chunk fix (see the dedicated test below).
        output_frames=[(0.5, 0.1), (23.0, 0.1)],
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
    # Indexed by position, not by a word-text-keyed dict: "chunk", "words",
    # and "here" each appear once per chunk, so a text-keyed lookup would
    # silently collapse the two chunks' distinct timings onto one entry.
    words = [(word, tuple(span)) for word, span, _ in payload["alignments"]]
    assert [word for word, _ in words] == [
        "First", "chunk", "words", "here", "Second", "chunk", "words", "here",
    ]

    # Rescale factor: decoded_duration (15.0) / wall_clock_span (22.5) = 2/3.
    # Interior chunk: "First chunk words here." spaced uniformly across
    # [rescale(0.5), rescale(3.0)) = [0.5, 0.5 + 5/3) -- 4 words.
    _assert_close(words[0][1], (0.5, 0.5 + 5 / 12), abs_tol=1e-4)
    _assert_close(words[1][1], (0.5 + 5 / 12, 0.5 + 10 / 12), abs_tol=1e-4)
    _assert_close(words[2][1], (0.5 + 10 / 12, 0.5 + 15 / 12), abs_tol=1e-4)
    _assert_close(words[3][1], (0.5 + 15 / 12, 0.5 + 5 / 3), abs_tol=1e-4)

    # Last chunk: "Second chunk words here." spaced uniformly across
    # [rescale(3.0), rescale(23.0)) = [0.5 + 5/3, 15.5) -- 4 words.
    _assert_close(words[4][1], (0.5 + 5 / 3, 5.5), abs_tol=1e-4)


def test_build_self_distillation_example_bounds_last_chunk_by_real_speech_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The *last* post_barge_text_frames chunk must be bounded by the real
    end of audible assistant speech (derived from rollout.output_frames),
    not by assistant_end_sec -- the end of the entire captured assistant
    audio, which includes a trailing response_seconds window (12.0s by
    default) of mostly silence after the assistant actually stops talking.

    This is the concrete case the whole-branch review identified as still
    broken by commit b90ee96 (which only fixed *interior* chunk bounds):
    a single/last chunk arrives at t=3.0s, but the assistant's real speech
    (per output_frames) ends at t=4.0s, while the full captured audio spans
    all the way to t=15.5s (0.5s pre-roll + 15.0s decoded audio). The old
    `end_sec=assistant_end_sec` behavior would spread this chunk's words
    across [3.0, 15.5) -- ending near 15.5, seconds after the assistant
    stopped speaking. The fix must instead bound it to [3.0, 4.0).
    """
    _install_fake_audio_modules(monkeypatch)

    opening_wav = tmp_path / "opening.wav"
    interruption_wav = tmp_path / "interruption.wav"
    _write_mono_wav(opening_wav, silence_seconds=0.0, speech_seconds=1.0)
    _write_mono_wav(interruption_wav, silence_seconds=0.2, speech_seconds=0.6)

    raw_audio_path = tmp_path / "rollout-0.opus"
    # 15s of captured assistant audio -- the assistant is only actually
    # talking for the first ~3.5s of it (per output_frames below); the rest
    # is the trailing response_seconds silence window.
    _write_opus_fixture(raw_audio_path, seconds=15.0, amplitude=0.1)

    rollout = Rollout(
        transcript="Second chunk words here.",
        yield_delay_reward=-0.3,
        quality_reward=0.8,
        delay_seconds=0.3,
        raw_audio_path=raw_audio_path,
        barge_in_seconds=1.0,
        first_output_received_seconds=0.5,
        # Same 1.5x wall_clock_span:decoded_duration ratio as the interior-
        # chunk test above, so this test also exercises the rescale path.
        last_output_received_seconds=23.0,
        post_barge_text_frames=[(3.0, "Second chunk words here.")],
        # Real audible output (rms above the 0.02 activity threshold) stops
        # at raw wall-clock t=4.0; everything after that is quiet, all the
        # way out to the new 23.0s end of the captured audio.
        output_frames=[
            (0.5, 0.1),
            (1.0, 0.1),
            (2.0, 0.1),
            (3.0, 0.1),
            (4.0, 0.1),
            (4.5, 0.01),
            (10.0, 0.01),
            (23.0, 0.01),
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

    # Rescale factor: decoded_duration (15.0) / wall_clock_span (22.5) = 2/3.
    # real_speech_end_sec = rescale(4.0) = 0.5 + 7/3 (the raw 4.0s last-active
    # frame, rescaled); chunk start = rescale(3.0) = 0.5 + 5/3. Only chunk,
    # bounded by [0.5 + 5/3, 0.5 + 7/3) -- 4 words, duration 2/3, 1/6 per word.
    _assert_close(words_by_text["Second"], (0.5 + 5 / 3, 0.5 + 5 / 3 + 1 / 6), abs_tol=1e-4)
    _assert_close(
        words_by_text["chunk"], (0.5 + 5 / 3 + 1 / 6, 0.5 + 5 / 3 + 2 / 6), abs_tol=1e-4
    )
    _assert_close(
        words_by_text["words"], (0.5 + 5 / 3 + 2 / 6, 0.5 + 5 / 3 + 3 / 6), abs_tol=1e-4
    )
    last_word_start, last_word_end = words_by_text["here"]
    assert abs(last_word_start - (0.5 + 5 / 3 + 3 / 6)) <= 1e-4
    assert abs(last_word_end - (0.5 + 7 / 3)) <= 1e-4
    # The discriminating assertion: the pre-A1 behavior (bounding by
    # assistant_end_sec=15.5, unrescaled) or the pre-this-fix behavior
    # (bounding by the raw, un-rescaled real_speech_end=4.0) would both put
    # this word's end far from 0.5 + 7/3 (~2.83s). This tight bound close to
    # the true rescaled value, and far from either prior bug's output, is
    # what makes this test catch a regression of either fix.
    assert last_word_end < 5.0
