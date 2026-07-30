# Self-Distillation Realtime Timing Recalibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `build_self_distillation_example`'s timing guard, which currently rejects essentially every live-session corpus example because it wrongly treats PersonaPlex's sub-realtime generation speed (observed ~1.4x on Spark) as corpus-corrupting timing drift. Replace the absolute-tolerance drift check with a much weaker sanity check, and rescale alignment timestamps onto the audio's actual placed span so word boundaries never fall outside the audio that's actually there.

**Architecture:** One file's internal logic changes (`self_distillation.py`): a new pure `_rescale_to_placed_span` helper, a narrowed guard (two genuine invariants instead of one flawed tolerance comparison), and rewiring `build_self_distillation_example` to rescale wall-clock timestamps before computing alignment chunk boundaries. No other module changes.

**Tech Stack:** Python 3.12, pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-30-self-distillation-realtime-timing-recalibration-design.md` -- read this first for the full rationale (the live-diagnostic evidence, why sub-realtime generation isn't corpus corruption, and the two confirmed design decisions).
- Audio placement itself (`assistant_end_sec = assistant_start_sec + decoded_duration`) is **already correct** and must not change -- this plan only changes the guard and the alignment-timestamp rescaling.
- The new guard raises on exactly two conditions: `decoded_duration <= 0` and `wall_clock_span < 0` (when computable). No tolerance/magnitude-based drift check remains.
- Rescaling happens at the call site in `build_self_distillation_example`, not inside `_word_alignments_from_frames` or `_last_active_frame_seconds` -- both stay pure over whatever timestamps they're handed.
- `_rescale_to_placed_span` returns `assistant_start_sec` unconditionally when `wall_clock_span <= 0` (no interpolation possible, not a division error).

---

### Task 1: Recalibrate the timing guard and alignment rescaling

**Files:**
- Modify: `src/tolkien_voice/training/self_distillation.py`
- Test: `tests/test_training_self_distillation.py`

**Interfaces:**
- Consumes: `Rollout.first_output_received_seconds`, `Rollout.last_output_received_seconds`, `Rollout.output_frames`, `Rollout.post_barge_text_frames` (all unchanged from the merged branch).
- Produces: `_rescale_to_placed_span(...)` (new, pure function); `build_self_distillation_example`'s behavior changes (narrower guard, rescaled alignments) but its signature is unchanged -- no caller (`scripts/build-self-distillation-corpus.py`) needs updating.

- [ ] **Step 1: Write the failing tests for the new guard**

In `tests/test_training_self_distillation.py`, find and replace the existing
`test_build_self_distillation_example_raises_on_assistant_timing_drift` test
(it tests the guard being removed) with three new tests. First, locate the
existing `_rollout` helper and any shared fixture-building helpers already
in the file (`_write_mono_wav`, `_write_opus_fixture`, `_install_fake_audio_modules`)
and reuse them as-is.

Replace the old drift-raising test with:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training_self_distillation.py -k "tolerates_sub_realtime or raises_on_empty_decoded or raises_on_out_of_order" -v`
Expected: `test_build_self_distillation_example_tolerates_sub_realtime_generation`
FAILs with the old guard's `ValueError: assistant audio timing is
inconsistent...` (drift 4.0s > 0.15s tolerance). The other two are new
error messages the old code doesn't produce yet, so they FAIL with
`AttributeError`/mismatched-message or simply because the old guard raises
a different message first.

- [ ] **Step 3: Write the failing test for `_rescale_to_placed_span`**

Add near the top of the test file (after existing imports), a new test
function:

```python
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
```

Note: `pytest.approx` is used here directly (not via the file's
`_assert_close` helper, if one exists from a prior fix round) since this
test operates on plain Python floats, not through the fake-numpy module
injection that made `pytest.approx` unsafe elsewhere in this file (per the
A1 fix's own note: "`pytest.approx` was avoided [elsewhere] because it
probes `sys.modules["numpy"]`... and this file injects a fake numpy
module" -- that concern only applies to tests using
`_install_fake_audio_modules`; these two rescale tests don't call it, so
real `pytest.approx`/real numpy resolution is fine here).

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_training_self_distillation.py -k "rescale_to_placed_span" -v`
Expected: FAIL with `ImportError`/`AttributeError` -- `_rescale_to_placed_span` doesn't exist yet.

- [ ] **Step 5: Implement `_rescale_to_placed_span`**

In `src/tolkien_voice/training/self_distillation.py`, add this function
after `_last_active_frame_seconds` and before `_word_alignments_from_frames`:

```python
def _rescale_to_placed_span(
    wall_clock_seconds: float,
    *,
    first_output_received_seconds: float,
    wall_clock_span: float,
    assistant_start_sec: float,
    decoded_duration: float,
) -> float:
    """Maps a wall-clock timestamp within
    [first_output_received_seconds, first_output_received_seconds + wall_clock_span]
    onto the placed audio's actual span
    [assistant_start_sec, assistant_start_sec + decoded_duration].

    PersonaPlex can generate audio slower than real-time (observed ~1.4x on
    Spark, 2026-07-30 live diagnostics), so the wall-clock time between the
    first and last received packet routinely exceeds the audio's own
    decoded playback duration. The audio itself is placed starting at
    assistant_start_sec and playing continuously for decoded_duration (see
    build_self_distillation_example) -- unrelated wall-clock timestamps
    (word-chunk arrival times, last-active-frame time) need to be
    reprojected onto that placed span, not used as-is, or they can land
    after the audio has already ended in the reconstructed WAV.

    wall_clock_span may be 0 (a session with exactly one received packet)
    -- in that degenerate case there is nothing to interpolate across, so
    the result is assistant_start_sec unconditionally rather than dividing
    by zero.
    """
    if wall_clock_span <= 0:
        return assistant_start_sec
    fraction = (wall_clock_seconds - first_output_received_seconds) / wall_clock_span
    return assistant_start_sec + fraction * decoded_duration
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_training_self_distillation.py -k "rescale_to_placed_span" -v`
Expected: PASS, both tests.

- [ ] **Step 7: Replace the guard and wire in rescaling**

In `src/tolkien_voice/training/self_distillation.py`, in
`build_self_distillation_example`, replace this entire block (currently
spanning from `assistant_samples = _decode_opus_pcm(...)` through the end
of the old guard's `if drift > 0.15:` raise):

```python
    assistant_samples = _decode_opus_pcm(rollout.raw_audio_path)

    if (
        rollout.first_output_received_seconds is not None
        and rollout.last_output_received_seconds is not None
    ):
        # first_output_received_seconds and last_output_received_seconds are
        # both wall-clock receive timestamps from probe_native_interruption,
        # but they are captured at very different points in that function's
        # sender loop (which sleeps FRAME_SECONDS after every send, so
        # wall-clock time drifts ahead of nominal stream time as the session
        # runs). If PersonaPlex's server-side model free-runs on its own
        # clock rather than staying in lockstep with received frames, that
        # drift is harmless; if it runs in lockstep, the drift is real and
        # would misplace assistant audio relative to the interruption on the
        # reconstructed timeline -- corrupting exactly the yield-timing
        # signal this corpus exists to teach. Nobody can currently verify
        # which case holds (PersonaPlex's server source isn't available), so
        # rather than guess-correct the timing, cross-check the wall-clock
        # span against the independently-measured decoded-audio duration and
        # fail loudly here, before any partial artifact is written, rather
        # than silently emitting a corpus example with corrupted timing.
        wall_clock_span = rollout.last_output_received_seconds - rollout.first_output_received_seconds
        decoded_duration = len(assistant_samples) / SAMPLE_RATE
        drift = abs(wall_clock_span - decoded_duration)
        if drift > 0.15:
            raise ValueError(
                f"assistant audio timing is inconsistent for {scenario_id}: "
                f"wall-clock span {wall_clock_span:.3f}s vs decoded duration "
                f"{decoded_duration:.3f}s (drift {drift:.3f}s)"
            )
```

with:

```python
    assistant_samples = _decode_opus_pcm(rollout.raw_audio_path)
    decoded_duration = len(assistant_samples) / SAMPLE_RATE
    if decoded_duration <= 0:
        raise ValueError(f"assistant audio decoded to no samples for {scenario_id}")

    first_output = rollout.first_output_received_seconds
    last_output = rollout.last_output_received_seconds
    wall_clock_span: float | None = None
    if first_output is not None and last_output is not None:
        wall_clock_span = last_output - first_output
        if wall_clock_span < 0:
            raise ValueError(
                f"assistant audio timestamps are out of order for {scenario_id}: "
                f"last_output_received_seconds ({last_output}) precedes "
                f"first_output_received_seconds ({first_output})"
            )
```

Then, further down, find this line (still inside
`build_self_distillation_example`):

```python
    assistant_start_sec = rollout.first_output_received_seconds or 0.0
```

Leave it unchanged -- `first_output` (the new local variable above) and
`rollout.first_output_received_seconds` are the same value; this line can
stay as-is or be changed to `assistant_start_sec = first_output or 0.0` for
consistency with the new local variable (either is correct; prefer the
local variable to avoid two names for the same thing).

Finally, find the alignment-building block near the end of the function:

```python
    # The last post_barge_text_frames chunk must be bounded by the real end
    # of audible assistant speech, not assistant_end_sec -- assistant_end_sec
    # is the end of the entire captured assistant audio, which includes a
    # trailing response_seconds window (default 12.0s) of mostly silence
    # after the assistant actually stops talking. Falling back to
    # assistant_end_sec when no frame is ever active is a safe degenerate
    # default (matches the old behavior) rather than crashing on audio that
    # apparently never had any real output.
    real_speech_end_sec = _last_active_frame_seconds(rollout.output_frames)
    if real_speech_end_sec is None:
        real_speech_end_sec = assistant_end_sec
    alignments = _word_alignments_from_frames(
        rollout.post_barge_text_frames, real_speech_end_sec=real_speech_end_sec
    )
```

Replace it with:

```python
    def _rescale(wall_clock_seconds: float) -> float:
        if wall_clock_span is None or first_output is None:
            return wall_clock_seconds
        return _rescale_to_placed_span(
            wall_clock_seconds,
            first_output_received_seconds=first_output,
            wall_clock_span=wall_clock_span,
            assistant_start_sec=assistant_start_sec,
            decoded_duration=decoded_duration,
        )

    # The last post_barge_text_frames chunk must be bounded by the real end
    # of audible assistant speech, not assistant_end_sec -- assistant_end_sec
    # is the end of the entire captured assistant audio, which includes a
    # trailing response_seconds window (default 12.0s) of mostly silence
    # after the assistant actually stops talking. Falling back to
    # assistant_end_sec when no frame is ever active is a safe degenerate
    # default (matches the old behavior) rather than crashing on audio that
    # apparently never had any real output. Both this value and each
    # post_barge_text_frames chunk's own timestamp are wall-clock times that
    # may span more real time than the audio's actual placed span (PersonaPlex
    # can generate slower than real-time) -- rescale both onto
    # [assistant_start_sec, assistant_end_sec] before using them as chunk
    # boundaries, or a chunk could be timestamped after the audio it's
    # describing has already ended in the reconstructed WAV.
    real_speech_end_sec = _last_active_frame_seconds(rollout.output_frames)
    real_speech_end_sec = (
        _rescale(real_speech_end_sec) if real_speech_end_sec is not None else assistant_end_sec
    )
    rescaled_frames = [(_rescale(chunk_start), text) for chunk_start, text in rollout.post_barge_text_frames]
    alignments = _word_alignments_from_frames(
        rescaled_frames, real_speech_end_sec=real_speech_end_sec
    )
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/test_training_self_distillation.py -v`
Expected: PASS, all tests in this file -- including the three new guard
tests from Step 1 and the two rescale tests from Step 3.

- [ ] **Step 9: Update the two A1-fix tests to exercise a genuine rescale**

The existing tests
`test_build_self_distillation_example_aligns_words_to_real_chunk_timestamps`
and `test_build_self_distillation_example_bounds_last_chunk_by_real_speech_end`
use fixtures where `wall_clock_span` happens to equal `decoded_duration`
(`first_output_received_seconds=0.5`, `last_output_received_seconds=15.5`,
decoded audio 15.0s -- exactly `15.5 - 0.5`) -- under the new rescale
formula, `fraction * decoded_duration` degenerates to `fraction *
wall_clock_span` in that case, so these tests currently pass without ever
exercising the rescale path at all. Change `last_output_received_seconds`
to `23.0` in both (keeping `first_output_received_seconds=0.5` and the
15.0s-decoded fixture unchanged), giving `wall_clock_span = 22.5` against
`decoded_duration = 15.0` -- a 1.5x ratio, the same direction and a similar
magnitude to the ~1.4x observed live on Spark. Every timestamp the tests
assert on then needs recomputing through the rescale.

In `test_build_self_distillation_example_aligns_words_to_real_chunk_timestamps`,
change the `rollout = Rollout(...)` block's timing fields and the final
assertions to:

```python
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
```

and the final assertions block to:

```python
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
```

(`abs_tol=1e-4` instead of the file's default `1e-6`: these expected values
are Python fraction expressions evaluated independently of the production
code's own floating-point operation order, so bit-identical agreement
isn't guaranteed even though both sides compute the same mathematical
result -- 1e-4 is still far tighter than the multi-second discrepancy the
old, unrescaled behavior would have produced, so it stays fully
discriminating.)

In `test_build_self_distillation_example_bounds_last_chunk_by_real_speech_end`,
change the `rollout = Rollout(...)` block similarly:

```python
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
```

and the final assertions block to:

```python
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
```

The `_rollout` helper's own defaults (used by other, unrelated tests in
this file -- `select_best_rollout`/`leading_silence_seconds` tests) are
untouched by this step; only the two `Rollout(...)` blocks constructed
directly inside these two specific tests change.

- [ ] **Step 10: Run tests to verify they pass**

Run: `uv run pytest tests/test_training_self_distillation.py -v`
Expected: PASS, all tests.

- [ ] **Step 11: Run the full test suite and lint**

Run: `uv run pytest -q`
Expected: PASS, no regressions in any other file.

Run: `uv run ruff check --exclude vendor .`
Expected: `All checks passed!`

- [ ] **Step 12: Commit**

```bash
git add src/tolkien_voice/training/self_distillation.py tests/test_training_self_distillation.py
git commit -m "fix: recalibrate self-distillation timing guard for sub-realtime generation"
```

---

## After this task: live re-verification (not automated, requires Spark)

Once merged, re-run the same small-scale live check that originally
surfaced this bug:

1. `scripts/_personaplex_ctl.py start` against the base runtime config.
2. Run the documented warmup probe (`run-v6-native-eval.sh`'s `warm_up()`
   pattern) once, manually -- `build-self-distillation-corpus.py` doesn't
   yet include this itself (tracked separately as a known follow-up item).
3. Run `build-self-distillation-corpus.py` against a small scenario subset
   (2-3 per character, `--group-size 3`).
4. `scripts/_personaplex_ctl.py stop`.
5. Confirm the run completes without the old guard's rejection, produces a
   non-empty manifest, and spot-check a resulting alignment JSON's word
   timestamps land within its WAV's actual duration.
