# Self-Distillation Post-Barge-Only Audio Placement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `build_self_distillation_example`'s whole-session
wall-clock-rescale placement (from the prior "realtime timing
recalibration" plan's Tasks 1-2, currently in this branch but not merged)
with a simpler, more robust design: extract only the assistant's
post-barge audio by content position (frame count, not wall-clock time),
and place it and the interruption directly, with no compression/rescale
math anywhere. See `docs/superpowers/specs/2026-07-30-self-distillation-post-barge-audio-design.md`
for the full rationale.

**Architecture:** One function (`build_self_distillation_example`)
rewritten; two helper functions removed (`_rescale_to_placed_span`,
`_last_active_frame_seconds`'s use here); `_word_alignments_from_frames`
kept but fed differently (offset-translated timestamps instead of
rescaled ones). No other module changes.

**Tech Stack:** Python 3.12, pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-30-self-distillation-post-barge-audio-design.md`
  -- read this first for the full rationale, including the live diagnostic
  data that motivated this redesign.
- This branch (`fix/self-distillation-realtime-timing-recal`) currently has
  the prior (rescale-based) design at commit `7f0bff5`. This plan replaces
  that design's placement logic; it does not build on top of it.
- No wall-clock-to-placed-span rescaling remains anywhere in
  `build_self_distillation_example` after this plan -- every placement is
  either a direct value (`opening_end_sec`, `barge_in_seconds`,
  `onset_offset`) or a simple offset translation.
- `_decode_opus_pcm` and `leading_silence_seconds` are unchanged.
- `frame_samples = round(SAMPLE_RATE * FRAME_SECONDS)` is the same
  chunking convention `native_probe.py` already uses to build
  `output_frames` -- reuse it, don't redefine it with a different value.

---

### Task 1: Rewrite `build_self_distillation_example` with post-barge-only extraction

**Files:**
- Modify: `src/tolkien_voice/training/self_distillation.py`
- Test: `tests/test_training_self_distillation.py`

**Interfaces:**
- Consumes: `Rollout.barge_in_seconds`, `Rollout.delay_seconds`,
  `Rollout.output_frames`, `Rollout.post_barge_text_frames` (all existing
  fields, unchanged).
- Produces: `build_self_distillation_example`'s public signature and
  return type are unchanged -- `scripts/build-self-distillation-corpus.py`
  needs no update.
- Removes: `_rescale_to_placed_span` (delete entirely -- no longer called
  anywhere). `_last_active_frame_seconds` (delete entirely -- the
  extraction-end clamp in the new design replaces its purpose, and no
  other caller uses it).

- [ ] **Step 1: Write the failing tests for the new placement model**

Replace the entire contents of `tests/test_training_self_distillation.py`
from its start through the `_write_opus_fixture` helper (i.e., keep
everything from `_install_fake_audio_modules`/`_write_opus_fixture`
onward unchanged for now) with:

```python
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
```

(This keeps `_assert_close` -- still needed by the alignment test below --
and drops the now-irrelevant `_rescale_to_placed_span` import/tests
entirely, since that function is being deleted in Step 3.)

Then, after the existing `_rollout`, `test_select_best_rollout_*`,
`_write_mono_wav`, `test_leading_silence_seconds_*` sections (all
unrelated to this change -- leave them exactly as they are), **delete**
every test from `_install_fake_audio_modules`'s section onward that
constructs a `Rollout` and calls `build_self_distillation_example`,
EXCEPT keep `_install_fake_audio_modules`, `_write_opus_fixture`, and
`FakeArray`/`_make_fake_numpy`/`_make_fake_sphn` (the fake-module
infrastructure -- still needed). Specifically delete:
`test_build_self_distillation_example_produces_a_valid_rendered_example`,
`test_build_self_distillation_example_raises_on_channel_clip`,
`test_build_self_distillation_example_tolerates_sub_realtime_generation`,
`test_build_self_distillation_example_raises_on_empty_decoded_audio`,
`test_build_self_distillation_example_raises_on_out_of_order_timestamps`,
`test_build_self_distillation_example_aligns_words_to_real_chunk_timestamps`,
`test_build_self_distillation_example_bounds_last_chunk_by_real_speech_end`,
`test_build_self_distillation_example_places_interruption_relative_to_real_yield`
-- all of these test the removed rescale-based design's specific
behavior. (`test_build_self_distillation_example_raises_on_channel_clip`
in particular tested opening-vs-interruption overlap, which the new
design's `max(opening_end_sec, barge_in_seconds)` placement makes
structurally impossible -- there is nothing left for that scenario to
test.)

Add these new tests after `_write_opus_fixture`:

```python
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
    assert [word for word, _ in words] == ["First", "words.", "Second", "words."]

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training_self_distillation.py -v`
Expected: the new tests FAIL (the production code still implements the old
rescale-based design); tests unrelated to `build_self_distillation_example`
(`select_best_rollout`, `leading_silence_seconds`) still PASS unchanged.

- [ ] **Step 3: Replace `build_self_distillation_example` and remove the two obsolete helpers**

In `src/tolkien_voice/training/self_distillation.py`:

Delete `_rescale_to_placed_span` entirely (the whole function, lines
139-170 in the current file).

Delete `_last_active_frame_seconds` entirely (the whole function, lines
117-136 in the current file).

Replace `build_self_distillation_example` (its full body, from `def
build_self_distillation_example(` through the final `return
RenderedExample(...)`) with:

```python
def build_self_distillation_example(
    rollout: Rollout,
    *,
    scenario_id: str,
    character: str,
    opening_wav_path: Path,
    interruption_wav_path: Path,
    system_prompt: str,
    split: Split,
    output_directory: Path,
) -> RenderedExample:
    """Reconstructs a training example from only the post-barge portion of
    the model's real session: opening_wav_path at t=0 on the user (right)
    channel; interruption_wav_path positioned so its audible onset lands
    at max(opening_end_sec, rollout.barge_in_seconds) -- never before the
    opening finishes, by construction; the model's real post-barge audio
    (extracted from rollout.raw_audio_path by content position -- counting
    decoded output_frames chunks before barge-in, not converting wall-clock
    time -- for exactly rollout.delay_seconds) placed starting at that same
    onset point on the assistant (left) channel.

    Deliberately does not attempt to reconstruct the assistant's pre-barge
    turn (backchannel activity, near-silent listening) at all -- the
    corpus's semantic content (rollout.transcript) has always been
    post-barge-only, and reconstructing the full session's audio requires
    correctly modeling genuine full-duplex overlap between the assistant's
    pre-barge activity and the user's opening, which is not needed for
    this corpus's actual purpose (teaching correct post-interruption yield
    behavior). See docs/superpowers/specs/2026-07-30-self-distillation-post-barge-audio-design.md.
    """
    del character  # not needed for construction; kept for caller symmetry with scenario metadata
    output_directory.mkdir(parents=True, exist_ok=True, mode=0o700)

    opening_samples = _mono_pcm16(opening_wav_path)
    interruption_samples = _mono_pcm16(interruption_wav_path)
    assistant_samples = _decode_opus_pcm(rollout.raw_audio_path)

    frame_samples = round(SAMPLE_RATE * FRAME_SECONDS)
    barge_in_frame_index = sum(
        1 for received_seconds, _ in rollout.output_frames
        if received_seconds < rollout.barge_in_seconds
    )
    extraction_start = barge_in_frame_index * frame_samples
    if extraction_start >= len(assistant_samples):
        raise ValueError(
            f"decoded assistant audio for {scenario_id} does not extend to "
            f"barge-in (decoded {len(assistant_samples)} samples, barge-in "
            f"at sample {extraction_start})"
        )
    extraction_end = min(
        len(assistant_samples), extraction_start + round(rollout.delay_seconds * SAMPLE_RATE)
    )
    post_barge_samples = assistant_samples[extraction_start:extraction_end]

    onset_offset = leading_silence_seconds(interruption_wav_path)
    opening_end_sec = len(opening_samples) / SAMPLE_RATE
    interruption_onset_sec = max(opening_end_sec, rollout.barge_in_seconds)
    interruption_start_sec = interruption_onset_sec - onset_offset
    assistant_start_sec = interruption_onset_sec

    interruption_end_sec = interruption_start_sec + len(interruption_samples) / SAMPLE_RATE
    assistant_end_sec = assistant_start_sec + len(post_barge_samples) / SAMPLE_RATE
    total_seconds = max(opening_end_sec, interruption_end_sec, assistant_end_sec)
    total_frames = math.ceil(total_seconds * SAMPLE_RATE)

    left = array("i", [0] * total_frames)   # assistant
    right = array("i", [0] * total_frames)  # user

    def _mix(channel: array, samples: array, start_sec: float) -> None:
        start_frame = round(start_sec * SAMPLE_RATE)
        for offset, sample in enumerate(samples):
            target = start_frame + offset
            if target >= len(channel):
                break
            mixed = channel[target] + sample
            if abs(mixed) >= MAX_SAFE_SAMPLE:
                raise ValueError(f"mixed audio is clipped in {scenario_id}")
            channel[target] = mixed

    _mix(right, opening_samples, 0.0)
    _mix(right, interruption_samples, interruption_start_sec)
    _mix(left, post_barge_samples, assistant_start_sec)

    interleaved = array("h")
    for left_sample, right_sample in zip(left, right, strict=True):
        interleaved.extend((left_sample, right_sample))

    example_id = f"{scenario_id}-self-distill"
    wav_path = output_directory / f"{example_id}.wav"
    temporary_wav = wav_path.with_suffix(".wav.tmp")
    with wave.open(str(temporary_wav), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(interleaved.tobytes())
    os.replace(temporary_wav, wav_path)
    wav_path.chmod(0o600)

    segment_end_sec = assistant_start_sec + len(post_barge_samples) / SAMPLE_RATE
    translated_frames = [
        (assistant_start_sec + (chunk_start - rollout.barge_in_seconds), text)
        for chunk_start, text in rollout.post_barge_text_frames
    ]
    alignments = _word_alignments_from_frames(
        translated_frames, real_speech_end_sec=segment_end_sec
    )
    alignment_path = output_directory / f"{example_id}.json"
    temporary_alignment = alignment_path.with_suffix(".json.tmp")
    temporary_alignment.write_text(
        json.dumps(
            {
                "alignments": alignments,
                "system_prompt": system_prompt,
            },
            indent=2,
        )
        + "\n"
    )
    temporary_alignment.chmod(0o600)
    os.replace(temporary_alignment, alignment_path)

    blueprint_hash = hashlib.sha256(
        f"{scenario_id}:{rollout.transcript}".encode("utf-8")
    ).hexdigest()

    return RenderedExample(
        id=example_id,
        blueprint_hash=blueprint_hash,
        wav_path=wav_path,
        alignment_path=alignment_path,
        duration_sec=total_frames / SAMPLE_RATE,
        split=split,
        voice_license=VOICE_LICENSE,
    )
```

Note `_word_alignments_from_frames` itself (the per-chunk interior/last-chunk
bounding function) is unchanged -- only what's passed into it changes
(`translated_frames`, an offset translation, instead of rescaled
timestamps; `segment_end_sec`, the extracted segment's own natural end,
instead of a separately-derived "real speech end" value).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_training_self_distillation.py -v`
Expected: PASS, all tests in this file.

- [ ] **Step 5: Run the full suite and lint**

Run: `uv run pytest -q`
Expected: PASS, no regressions in any other file. (No other file imports
`_rescale_to_placed_span` or `_last_active_frame_seconds`, so their
removal should not ripple anywhere else -- confirm this with
`grep -rn "_rescale_to_placed_span\|_last_active_frame_seconds" src/ tests/ scripts/`
returning no results outside the diff you just made.)

Run: `uv run ruff check --exclude vendor .`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/tolkien_voice/training/self_distillation.py tests/test_training_self_distillation.py
git commit -m "fix: extract post-barge audio by content position instead of reconstructing the whole session"
```

---

## After this task: live re-verification (not automated, requires Spark)

Same as the prior spec's verification bar: run
`scripts/build-self-distillation-corpus.py` against a small scenario
subset via a live PersonaPlex instance (`scripts/_personaplex_ctl.py
start`/`stop`, with the documented warmup probe run first). The build
should succeed, and a manual listen/inspection of a resulting WAV should
confirm the interruption never overlaps the opening and the assistant's
segment starts audibly at the interruption's own onset -- this is the
first design in this whole effort where that can be verified by simple
inspection (the assistant channel should be silent until exactly where
the interruption becomes audible), rather than needing careful frame-level
arithmetic to reason about.
