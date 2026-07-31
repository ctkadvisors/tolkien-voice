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
    assert abs(actual_start - expected_start) <= abs_tol
    assert abs(actual_end - expected_end) <= abs_tol


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
    # 2.0s decoded -- long enough that barge-in (at raw t=1.0, well within
    # the first 40ms-frame-quantized second) plus a 0.3s extraction has
    # real content to extract.
    _write_opus_fixture(raw_audio_path, seconds=2.0, amplitude=0.2)

    rollout = Rollout(
        transcript="Go on, then.",
        yield_delay_reward=-0.3,
        quality_reward=0.8,
        delay_seconds=0.3,
        raw_audio_path=raw_audio_path,
        barge_in_seconds=1.0,
        first_output_received_seconds=0.1,
        last_output_received_seconds=2.1,
        post_barge_text_frames=[(1.0, "Go on, then.")],
        # 25 frames (0.0 to 0.96s in 0.04s steps) before barge_in=1.0.
        output_frames=[(round(i * 0.04, 6), 0.1) for i in range(50)],
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


def test_build_self_distillation_example_interruption_never_overlaps_opening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The concrete case the whole-branch review found broken in the prior
    (rescale-based) design: a long opening plus a plausible delay_seconds
    could pull the interruption's audible onset earlier than the opening's
    own end, overlapping two different user utterances on one channel with
    no error raised. The new design's max(opening_end_sec, barge_in_seconds)
    placement must never do this, regardless of delay_seconds.
    """
    _install_fake_audio_modules(monkeypatch)

    opening_wav = tmp_path / "opening.wav"
    interruption_wav = tmp_path / "interruption.wav"
    # 4.5s opening, matching the whole-branch review's own reproduction.
    _write_mono_wav(opening_wav, silence_seconds=0.0, speech_seconds=4.5)
    _write_mono_wav(interruption_wav, silence_seconds=0.08, speech_seconds=0.6)

    raw_audio_path = tmp_path / "rollout-0.opus"
    _write_opus_fixture(raw_audio_path, seconds=6.0, amplitude=0.1)

    rollout = Rollout(
        transcript="Go on, then.",
        yield_delay_reward=-0.3,
        quality_reward=0.8,
        # A large delay_seconds is exactly what pulled the interruption
        # earlier than the opening's end under the old rescale-based
        # formula -- the new formula must not be sensitive to this at all.
        delay_seconds=3.0,
        raw_audio_path=raw_audio_path,
        barge_in_seconds=4.9,  # shortly after the 4.5s opening ends
        first_output_received_seconds=0.1,
        last_output_received_seconds=6.1,
        post_barge_text_frames=[(4.9, "Go on, then.")],
        output_frames=[(round(i * 0.04, 6), 0.1) for i in range(150)],
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

    with wave.open(str(example.wav_path), "rb") as source:
        sample_rate = source.getframerate()
        interleaved = array("h")
        interleaved.frombytes(source.readframes(source.getnframes()))
    right_channel = interleaved[1::2]

    opening_end_sample = round(4.5 * sample_rate)
    # Every sample up to and including the opening's own end must be
    # exactly the opening's own content (amplitude 5000, from
    # _write_mono_wav) -- nothing from the interruption clip has bled in
    # before the opening finished.
    assert all(sample == 5000 for sample in right_channel[: opening_end_sample - 1])


def test_build_self_distillation_example_raises_when_decoded_audio_does_not_reach_barge_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_audio_modules(monkeypatch)

    opening_wav = tmp_path / "opening.wav"
    interruption_wav = tmp_path / "interruption.wav"
    _write_mono_wav(opening_wav, silence_seconds=0.0, speech_seconds=1.0)
    _write_mono_wav(interruption_wav, silence_seconds=0.2, speech_seconds=0.6)

    raw_audio_path = tmp_path / "rollout-0.opus"
    # Only 0.2s of decoded audio (4800 samples at 24kHz), but barge-in is
    # claimed at 5.0s with 10 output_frames entries all before it -- 10
    # frames * 960 samples/frame = 9600, which exceeds the 4800-sample
    # decoded buffer, so extraction_start lands past its end.
    _write_opus_fixture(raw_audio_path, seconds=0.2, amplitude=0.1)

    rollout = Rollout(
        transcript="Go on, then.",
        yield_delay_reward=-0.3,
        quality_reward=0.8,
        delay_seconds=0.3,
        raw_audio_path=raw_audio_path,
        barge_in_seconds=5.0,
        first_output_received_seconds=0.1,
        last_output_received_seconds=0.3,
        post_barge_text_frames=[],
        output_frames=[(round(i * 0.04, 6), 0.1) for i in range(10)],
    )
    output_directory = tmp_path / "corpus"

    with pytest.raises(ValueError, match="does not extend to barge-in"):
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


def test_build_self_distillation_example_places_assistant_segment_at_interruption_onset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The assistant's extracted post-barge segment must start exactly at
    the interruption's own audible onset -- physically: this is what the
    model said starting when the user interrupted.
    """
    _install_fake_audio_modules(monkeypatch)

    opening_wav = tmp_path / "opening.wav"
    interruption_wav = tmp_path / "interruption.wav"
    _write_mono_wav(opening_wav, silence_seconds=0.0, speech_seconds=1.0)
    # Silence prefix is an exact multiple of FRAME_SECONDS (0.04s * 2 =
    # 0.08s) so leading_silence_seconds returns exactly 0.08.
    _write_mono_wav(interruption_wav, silence_seconds=0.08, speech_seconds=0.6)

    raw_audio_path = tmp_path / "rollout-0.opus"
    _write_opus_fixture(raw_audio_path, seconds=2.0, amplitude=0.2)

    rollout = Rollout(
        transcript="Go on, then.",
        yield_delay_reward=-0.3,
        quality_reward=0.8,
        delay_seconds=0.5,
        raw_audio_path=raw_audio_path,
        barge_in_seconds=1.5,
        first_output_received_seconds=0.1,
        last_output_received_seconds=2.1,
        post_barge_text_frames=[(1.5, "Go on, then.")],
        output_frames=[(round(i * 0.04, 6), 0.1) for i in range(50)],
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

    # interruption_start_sec = max(1.0, 1.5) - 0.08 = 1.42
    # audible onset = interruption_start_sec + onset_offset = 1.5
    # assistant_start_sec = 1.5 (shares the interruption's own onset)
    expected_onset_sec = 1.5

    with wave.open(str(example.wav_path), "rb") as source:
        sample_rate = source.getframerate()
        interleaved = array("h")
        interleaved.frombytes(source.readframes(source.getnframes()))
    left_channel = interleaved[0::2]  # assistant channel

    def _left_channel_rms(start_sec: float, duration_sec: float) -> float:
        start = round(start_sec * sample_rate)
        end = round((start_sec + duration_sec) * sample_rate)
        window = left_channel[max(0, start) : max(0, end)]
        if not window:
            return 0.0
        return (sum(sample * sample for sample in window) / len(window)) ** 0.5

    assert _left_channel_rms(expected_onset_sec - 0.3, 0.2) < 100
    assert _left_channel_rms(expected_onset_sec + 0.05, 0.2) > 1000


def test_build_self_distillation_example_clamps_interruption_start_when_onset_offset_exceeds_onset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for the missing max(0.0, ...) floor: a short opening
    (0.5s) combined with an interruption clip whose leading silence (2.0s)
    exceeds interruption_onset_sec (max(0.5, 0.2) = 0.5) used to drive
    interruption_start_sec negative (0.5 - 2.0 = -1.5), which _mix's inner
    loop would then silently wrap into negative array indices instead of
    raising. With the max(0.0, ...) clamp in place, this must complete
    without error and produce a valid example, with the interruption's
    audible content landing at or after opening_end_sec.
    """
    _install_fake_audio_modules(monkeypatch)

    opening_wav = tmp_path / "opening.wav"
    interruption_wav = tmp_path / "interruption.wav"
    _write_mono_wav(opening_wav, silence_seconds=0.0, speech_seconds=0.5)
    # 2.0s leading silence -- exceeds interruption_onset_sec (0.5) so
    # onset_offset > interruption_onset_sec, the exact trigger condition.
    _write_mono_wav(interruption_wav, silence_seconds=2.0, speech_seconds=0.3)

    raw_audio_path = tmp_path / "rollout-0.opus"
    _write_opus_fixture(raw_audio_path, seconds=2.0, amplitude=0.2)

    rollout = Rollout(
        transcript="Go on, then.",
        yield_delay_reward=-0.3,
        quality_reward=0.8,
        delay_seconds=0.3,
        raw_audio_path=raw_audio_path,
        # Smaller than opening_end_sec (0.5), so max(opening_end_sec,
        # barge_in_seconds) picks the opening's own end -- 0.5 -- matching
        # the reviewer's demonstrated reproduction.
        barge_in_seconds=0.2,
        first_output_received_seconds=0.1,
        last_output_received_seconds=2.1,
        post_barge_text_frames=[(0.2, "Go on, then.")],
        output_frames=[(round(i * 0.04, 6), 0.1) for i in range(50)],
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

    with wave.open(str(example.wav_path), "rb") as source:
        sample_rate = source.getframerate()
        interleaved = array("h")
        interleaved.frombytes(source.readframes(source.getnframes()))
    right_channel = interleaved[1::2]  # user channel (opening + interruption)

    def _right_channel_rms(start_sec: float, duration_sec: float) -> float:
        start = round(start_sec * sample_rate)
        end = round((start_sec + duration_sec) * sample_rate)
        window = right_channel[max(0, start) : max(0, end)]
        if not window:
            return 0.0
        return (sum(sample * sample for sample in window) / len(window)) ** 0.5

    opening_end_sec = 0.5
    # No interruption content should be audible before the opening ends --
    # if the pre-fix bug had wrapped negative indices to the buffer's end,
    # this would instead show spurious energy at the wrong place, or the
    # call would have raised/crashed before ever reaching this assertion.
    assert _right_channel_rms(0.0, opening_end_sec) > 1000
    # The interruption's own audible content lands at its natural onset
    # (interruption_start_sec + onset_offset = 0.0 + 2.0 = 2.0), which is
    # at or after opening_end_sec -- never overlapping/pre-empting it.
    assert _right_channel_rms(2.05, 0.2) > 1000


def test_build_self_distillation_example_raises_clear_error_when_delay_seconds_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test: rollout.delay_seconds is typed float | None, but the
    extraction-end calculation used to multiply it by SAMPLE_RATE
    unconditionally, raising an unrelated TypeError deep inside the
    function when it was None. Must now raise a clear, specific ValueError
    instead.
    """
    _install_fake_audio_modules(monkeypatch)

    opening_wav = tmp_path / "opening.wav"
    interruption_wav = tmp_path / "interruption.wav"
    _write_mono_wav(opening_wav, silence_seconds=0.0, speech_seconds=1.0)
    _write_mono_wav(interruption_wav, silence_seconds=0.2, speech_seconds=0.6)

    raw_audio_path = tmp_path / "rollout-0.opus"
    _write_opus_fixture(raw_audio_path, seconds=2.0, amplitude=0.2)

    rollout = Rollout(
        transcript="Go on, then.",
        yield_delay_reward=-0.3,
        quality_reward=0.8,
        delay_seconds=None,
        raw_audio_path=raw_audio_path,
        barge_in_seconds=1.0,
        first_output_received_seconds=0.1,
        last_output_received_seconds=2.1,
        post_barge_text_frames=[(1.0, "Go on, then.")],
        output_frames=[(round(i * 0.04, 6), 0.1) for i in range(50)],
    )
    output_directory = tmp_path / "corpus"

    with pytest.raises(ValueError, match="delay_seconds is None"):
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


def test_build_self_distillation_example_aligns_words_by_direct_offset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Alignment timestamps are a direct offset translation from
    barge_in_seconds onto the assistant segment's placed position -- no
    compression factor, since the extracted segment plays at its own
    natural (unscaled) pace.
    """
    _install_fake_audio_modules(monkeypatch)

    opening_wav = tmp_path / "opening.wav"
    interruption_wav = tmp_path / "interruption.wav"
    _write_mono_wav(opening_wav, silence_seconds=0.0, speech_seconds=1.0)
    _write_mono_wav(interruption_wav, silence_seconds=0.08, speech_seconds=0.6)

    raw_audio_path = tmp_path / "rollout-0.opus"
    _write_opus_fixture(raw_audio_path, seconds=3.0, amplitude=0.2)

    rollout = Rollout(
        transcript="First words. Second words.",
        yield_delay_reward=-0.3,
        quality_reward=0.8,
        delay_seconds=1.0,
        raw_audio_path=raw_audio_path,
        barge_in_seconds=1.5,
        first_output_received_seconds=0.1,
        last_output_received_seconds=3.1,
        post_barge_text_frames=[
            (1.5, "First words. "),
            (1.9, "Second words."),
        ],
        output_frames=[(round(i * 0.04, 6), 0.1) for i in range(75)],
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
    words = [(word, tuple(span)) for word, span, _ in payload["alignments"]]
    # _word_alignments (unchanged by this plan) strips trailing punctuation
    # via word.strip(string.punctuation), so "words." tokenizes to "words".
    assert [word for word, _ in words] == ["First", "words", "Second", "words"]

    # assistant_start_sec = max(1.0, 1.5) = 1.5 (shares interruption onset).
    # Chunk 1 offset from barge_in (1.5): 1.5 - 1.5 = 0.0 -> placed at 1.5.
    # Chunk 2 offset from barge_in: 1.9 - 1.5 = 0.4 -> placed at 1.9.
    # Interior chunk 1 spans [1.5, 1.9) -- 2 words, 0.2s each.
    _assert_close(words[0][1], (1.5, 1.7))
    _assert_close(words[1][1], (1.7, 1.9))
    # Last chunk spans [1.9, assistant_start_sec + delay_seconds) = [1.9, 2.5)
    # -- 2 words, 0.3s each.
    _assert_close(words[2][1], (1.9, 2.2))
    _assert_close(words[3][1], (2.2, 2.5))


def test_build_self_distillation_example_translates_alignments_by_content_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reproduces PersonaPlex's sub-realtime generation (measured ~1.4x on
    real hardware: 1 wall-clock second yields less than 1 second of actual
    audio content) and proves alignment timestamps are translated by
    content position (frame-counting), not by a raw wall-clock delta.

    output_frames here are spaced 0.056s apart in wall-clock time -- 1.4x
    FRAME_SECONDS (0.04s) -- so every FRAME_SECONDS of decoded content
    takes 1.4x as long, in wall-clock terms, to arrive as it would under
    1:1 realtime generation.

    barge_in_seconds=1.0 lands after the 18th output_frame (17*0.056=0.952
    < 1.0 <= 18*0.056=1.008), so barge_in_frame_index=18, matching the
    extraction_start computed a few lines earlier in the same function.
    delay_seconds=0.5 extracts exactly 0.5s of post-barge content, so the
    placed assistant segment spans content-seconds [0.0, 0.5) relative to
    barge-in -- i.e. wall-clock seconds [assistant_start_sec,
    segment_end_sec) = [1.0, 1.5).

    The single post_barge_text_frames chunk is timestamped at wall-clock
    1.65s -- 0.65s after barge_in_seconds, comfortably past
    barge_in_seconds + delay_seconds (1.5s) in wall-clock terms, which is
    exactly the shape of chunk the old wall-clock-delta code mis-translated
    (see docstring below for the pre-fix arithmetic). Translated by content
    position, though, this chunk lands at real content-second 0.48 past
    barge-in (12 frames * 0.04s, since 30 output_frames arrive before
    1.65s and 30 - 18 = 12) -- i.e. wall-clock 1.48s -- safely inside the
    placed segment, with no clamping required.
    """
    _install_fake_audio_modules(monkeypatch)

    opening_wav = tmp_path / "opening.wav"
    interruption_wav = tmp_path / "interruption.wav"
    _write_mono_wav(opening_wav, silence_seconds=0.0, speech_seconds=1.0)
    # Silence prefix is an exact multiple of FRAME_SECONDS so
    # leading_silence_seconds returns exactly 0.08.
    _write_mono_wav(interruption_wav, silence_seconds=0.08, speech_seconds=0.6)

    raw_audio_path = tmp_path / "rollout-0.opus"
    _write_opus_fixture(raw_audio_path, seconds=3.0, amplitude=0.2)

    # Pre-fix (wall-clock-delta) arithmetic, for contrast -- NOT exercised
    # by the fixed code, just documenting what it would have produced:
    #   assistant_start_sec + (chunk_start - barge_in_seconds)
    #   = 1.0 + (1.65 - 1.0) = 1.65
    # -- which is past segment_end_sec (1.5) *and* past the WAV's own total
    # duration (1.52, set by the interruption clip's own placement below),
    # exactly the out-of-bounds failure mode the whole-branch review found.
    old_buggy_translated_position = 1.0 + (1.65 - 1.0)

    rollout = Rollout(
        transcript="Actually never mind.",
        yield_delay_reward=-0.3,
        quality_reward=0.8,
        delay_seconds=0.5,
        raw_audio_path=raw_audio_path,
        barge_in_seconds=1.0,
        first_output_received_seconds=0.1,
        last_output_received_seconds=3.1,
        post_barge_text_frames=[(1.65, "Actually never mind.")],
        # Spaced 0.056s apart (1.4x FRAME_SECONDS) -- sub-realtime content
        # generation. 200 frames covers wall-clock up to 200*0.056=11.2s.
        output_frames=[(round(i * 0.056, 6), 0.1) for i in range(200)],
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

    # assistant_start_sec = interruption_start_sec + onset_offset
    #                      = max(0.0, 1.0 - 0.08) + 0.08 = 1.0
    # segment_end_sec = assistant_start_sec + delay_seconds = 1.5
    assistant_start_sec = 1.0
    segment_end_sec = assistant_start_sec + 0.5
    assert old_buggy_translated_position > segment_end_sec
    assert old_buggy_translated_position > example.duration_sec

    payload = json.loads(example.alignment_path.read_text())
    words = [(word, tuple(span)) for word, span, _ in payload["alignments"]]
    assert [word for word, _ in words] == ["Actually", "never", "mind"]

    # Every word's timestamps must land inside the placed assistant
    # segment -- proving the content-position translation (not the old
    # wall-clock delta) is what actually landed here.
    for _, (word_start, word_end) in words:
        assert assistant_start_sec <= word_start <= segment_end_sec
        assert assistant_start_sec <= word_end <= segment_end_sec
        assert word_end <= example.duration_sec

    # And the concrete expected start: content delta = (30 - 18) frames
    # * 0.04s = 0.48s past barge-in -> wall-clock 1.48s, well inside
    # [1.0, 1.5) and nowhere near the buggy 1.65s the old code would have
    # produced. (30 output_frames arrive before wall-clock 1.65s: frames
    # 0..29, since 29*0.056=1.624 < 1.65 <= 30*0.056=1.68.)
    assert abs(words[0][1][0] - 1.48) <= 1e-6
