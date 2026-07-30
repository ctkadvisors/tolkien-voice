# Self-distillation realtime timing recalibration

Date: 2026-07-30

Status: Proposed for implementation

Repository: `ctkadvisors/tolkien-voice` (private)

## Context

The reward-filtered self-distillation branch (`feat/self-distillation-corpus`,
merged into `feat/tolkien-sts-distillation` at commit `dcf4db3`) was live-tested
against a real PersonaPlex instance on the Spark box per the implementation
plan's "After all tasks: live verification" section. Every attempt failed --
100% rejection rate -- at the `I1` timing-consistency guard added during that
branch's review cycle
(`docs/superpowers/specs/2026-07-30-reward-filtered-self-distillation-design.md`,
`build_self_distillation_example`'s drift check).

## What the guard checks and why it fails on real data

`I1`'s guard compares two quantities for the assistant's captured audio:

- `wall_clock_span = rollout.last_output_received_seconds - rollout.first_output_received_seconds`
  -- the wall-clock time between the *first* and *last* received websocket
  audio packet.
- `decoded_duration = len(assistant_samples) / SAMPLE_RATE` -- the total
  playback duration of the concatenated decoded PCM content.

It raises if these disagree by more than 0.15s. The check was built to guard
against an *unverified* hypothesis (documented in the original design doc):
that PersonaPlex's server-side model might run in lockstep with received user
frames, in which case the probe's client-side sender loop (which sleeps
`FRAME_SECONDS` after every send) would accumulate wall-clock drift relative
to nominal stream time, and that drift could misplace assistant audio on the
reconstructed corpus timeline.

Two live diagnostic runs against Spark's PersonaPlex instance (2026-07-30)
falsified the *specific mechanism* the guard was checking, but confirmed a
different and larger effect in the same direction:

- `interruption_underrun_count` and `interruption_max_packet_gap_ms` (the
  existing packet-arrival-gap tracking, 0.12s threshold) show **no large
  stalls** -- packets arrive steadily, no gaps over ~234ms.
- Despite steady arrival, the audio genuinely decoded is consistently
  **shorter** than the wall-clock time it took to arrive -- one diagnostic
  call showed 372 decoded 40ms frames (14.88s of content) spanning 20.546s of
  packet-arrival wall-clock time; a full corpus-build run showed 13.360s of
  decoded content spanning 18.66-18.68s (reproduced twice, nearly identical
  magnitude both times).

This pattern -- steady packet cadence, but each packet carrying less audio
than the wall-clock interval since the previous one -- is consistent with
PersonaPlex generating audio slower than real-time on this hardware
(observed ratio approximately 1.4x), not with clock drift or bursty
delivery. Sub-realtime generation is a property of the inference stack on
this specific hardware, not evidence of corpus-corrupting timing
misplacement, and checking for it as if it were a defect makes the guard
reject essentially every live-session example.

## Decision: what the corpus should actually encode

Two decisions, confirmed with the project owner before design:

1. **Audio placement stays anchored to real interruption timing, not
   stretched to match wall-clock delivery pacing.** The assistant's captured
   audio starts playing at `first_output_received_seconds` -- correctly
   preserving *when after the interruption new content actually began*, the
   signal this whole self-distillation effort exists to teach -- and plays
   continuously at its own natural (decoded) length from there. It does not
   try to reproduce inference-latency gaps within the response as audible
   dead air. Baking real generation-latency stalls into training audio would
   teach the model to pause unnaturally, matching an infra artifact on this
   specific evaluation hardware rather than genuine conversational timing.

   **This part of the system was already implemented correctly** --
   `build_self_distillation_example` already computes
   `assistant_end_sec = assistant_start_sec + decoded_duration`, i.e.
   "anchor start, play continuously for its own length." The bug is
   entirely in the *guard* that second-guesses this already-correct
   placement by comparing it against a wall-clock span that was never a
   valid invariant to check.

2. **Alignment chunk boundaries rescale onto the placed-audio's actual span,
   not raw wall-clock time.** `_last_active_frame_seconds` (added in the A1
   fix) and each `post_barge_text_frames` chunk's own timestamp are
   wall-clock values somewhere within
   `[first_output_received_seconds, last_output_received_seconds]` -- the
   *observed packet-arrival window*, which is now understood to run longer
   than the audio's actual placed span
   `[assistant_start_sec, assistant_end_sec]`. Left unrescaled, a chunk's
   wall-clock timestamp (or `_last_active_frame_seconds`'s answer) can fall
   *after* the placed audio has already ended in the reconstructed WAV.

   Fix: linearly rescale every such wall-clock timestamp onto the placed
   audio's span before computing chunk boundaries:

   ```
   rescaled = assistant_start_sec + (t - first_output_received_seconds)
              * (decoded_duration / wall_clock_span)
   ```

   This maps `first_output_received_seconds -> assistant_start_sec` and
   `last_output_received_seconds -> assistant_end_sec`, linearly
   interpolating everything between. A linear rescale (not exact per-packet
   cumulative-duration tracking) was chosen deliberately: this project's
   existing word-alignment (`_word_alignments`, used since `render.py`) is
   already a uniform-spacing approximation, not true forced alignment, so
   building exact cumulative-duration tracking machinery for a downstream
   consumer that's already approximate by design would be effort spent
   past the precision this system actually needs or uses elsewhere.

## Scope

**In scope:**
- Replace `I1`'s guard in `build_self_distillation_example`
  (`src/tolkien_voice/training/self_distillation.py`) with a much weaker
  sanity check (below), removing the absolute-tolerance drift comparison
  entirely.
- Rescale wall-clock timestamps onto the placed-audio span before they're
  used for alignment chunk boundaries (both `_last_active_frame_seconds`'s
  input/output and `_word_alignments_from_frames`'s per-chunk timestamps).

**Out of scope:**
- `compute_yield_delay_reward`/`compute_quality_reward`/`select_best_rollout`
  -- these operate on `barge_in_seconds`/`output_frames`/rewards directly,
  unaffected by how corpus audio gets *placed and labeled* after a rollout
  is already selected.
- `collect_rollout_group`'s sampling loop, `native_probe.py`'s probe
  mechanics, the orchestration script -- unaffected.
- Any change to `render.py`'s original TTS-based rendering path (unrelated,
  synthetic-audio pipeline, not live-probed).

## New guard

Replaces the removed `I1` check. Catches genuine corruption without
rejecting normal inference-speed variance:

```python
if decoded_duration <= 0:
    raise ValueError(f"assistant audio decoded to no samples for {scenario_id}")
if wall_clock_span < 0:
    raise ValueError(
        f"assistant audio timestamps are out of order for {scenario_id}: "
        f"last_output_received_seconds ({rollout.last_output_received_seconds}) "
        f"precedes first_output_received_seconds "
        f"({rollout.first_output_received_seconds})"
    )
```

`decoded_duration <= 0` catches a genuinely broken capture (e.g. an empty or
corrupt Opus stream -- the actual failure mode the original I1 check's
"never happens in practice" sibling case, `output_frames` all-silent,
already partially guards against elsewhere). `wall_clock_span < 0` is a pure
data-integrity check (packets arriving out of the order they were sent would
indicate a real bug in the probe or transport, not inference speed) --
unlike the removed check, it has no tolerance band to tune because it's
checking an impossibility, not a variable quantity.

## Rescale helper

New function in `self_distillation.py`:

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

    wall_clock_span may be 0 (a session with exactly one received packet) --
    in that degenerate case there is nothing to interpolate across, so the
    result is assistant_start_sec unconditionally rather than dividing by
    zero.
    """
    if wall_clock_span <= 0:
        return assistant_start_sec
    fraction = (wall_clock_seconds - first_output_received_seconds) / wall_clock_span
    return assistant_start_sec + fraction * decoded_duration
```

`_last_active_frame_seconds`'s return value and each
`post_barge_text_frames` chunk's timestamp both pass through this before
being used as `_word_alignments_from_frames` bounds. `_word_alignments_from_frames`'s
own internal logic (interior chunks bounded by the next chunk's timestamp,
last chunk bounded by the rescaled real-speech-end) is otherwise unchanged
from the A1 fix -- only the *inputs* it receives are now pre-rescaled rather
than raw wall-clock values.

**Rescaling happens at the call site in `build_self_distillation_example`,
not inside `_word_alignments_from_frames` or `_last_active_frame_seconds`.**
Both of those stay pure functions over whatever timestamps they're handed --
`build_self_distillation_example` rescales `rollout.post_barge_text_frames`
(mapping each `(t, text)` pair's `t`) and the value returned by
`_last_active_frame_seconds(rollout.output_frames)` before passing either
into `_word_alignments_from_frames`. This keeps both existing functions
unit-testable in isolation exactly as before, matching this project's
established small-pure-function convention, rather than teaching either of
them about `assistant_start_sec`/`decoded_duration`.

## Testing and validation

- `tests/test_training_self_distillation.py`: replace the two tests built
  around the old absolute-tolerance guard
  (`test_build_self_distillation_example_raises_on_assistant_timing_drift`)
  with tests for the new guard's two real invariants (`decoded_duration <= 0`,
  `wall_clock_span < 0`), plus a new **non-raising** test using the *actual
  magnitude of drift observed live* (roughly 1.4x realtime, e.g.
  `wall_clock_span` meaningfully larger than `decoded_duration`, the exact
  shape that made every live attempt fail) to prove the new guard does NOT
  reject this case -- this is the test that would have caught the original
  bug before it reached live verification.
- Rewrite `test_build_self_distillation_example_bounds_last_chunk_by_real_speech_end`
  and `test_build_self_distillation_example_aligns_words_to_real_chunk_timestamps`
  to use fixtures where `wall_clock_span` genuinely differs from
  `decoded_duration` (not the coincidentally-equal fixtures the A1 fix used,
  which never exercised the rescale path since `fraction * decoded_duration`
  degenerates to `fraction * wall_clock_span` when they're equal) --
  asserting the resulting alignment timestamps land within
  `[assistant_start_sec, assistant_end_sec]`, not past it.
- A new unit test for `_rescale_to_placed_span` directly: maps the span's
  start to `assistant_start_sec`, the span's end to `assistant_start_sec +
  decoded_duration`, interpolates linearly in between, and returns
  `assistant_start_sec` (not a division error) when `wall_clock_span <= 0`.

## Correction (found during this fix's own final whole-branch review)

The first implementation of this design (commit `1ca005d`, reviewed but not
merged) rescaled alignment timestamps onto the placed-audio span but left
`interruption_start_sec` computed from raw, unrescaled `rollout.barge_in_seconds`.
Because the assistant's audio now plays back *compressed* relative to
wall-clock time (that's the whole point of this fix -- `decoded_duration`
is shorter than `wall_clock_span`), positioning the interruption at its raw
wall-clock timestamp while the assistant's response plays out on a shorter,
compressed timeline can place the interruption *after* the assistant's
audio has already gone quiet in the reconstructed WAV -- teaching the
model to yield before the interruption it's supposedly responding to is
even audible. At the live-observed ~1.4-1.5x ratio this was measured at
1.5-2.5 seconds of apparent-yield-precedes-interruption distortion.

**Fix:** don't derive the interruption's placement from a rescale of
`barge_in_seconds` at all -- that would need the user channel's own
content resampled to stay fully coherent with a compressed timeline (a
real, unwanted expansion of scope: resampling live-recorded audio content
just to reposition an event marker). Instead, derive it from data already
computed for exactly this purpose: `Rollout.delay_seconds` (from
`measure_interruption_response`) is the *ground-truth* measured gap, in
real wall-clock seconds, between the interruption becoming audible and the
assistant going quiet -- entirely independent of how this corpus builder
later reconstructs audio. Placing the interruption at
`rescaled_real_speech_end_sec - delay_seconds` (rather than at a
rescaled/unrescaled `barge_in_seconds`) guarantees the reconstructed gap
between interruption and yield matches the real measured delay by
construction, using only repositioning (not resampling) of the
interruption clip.

`real_speech_end_sec` here is the same rescaled value already computed for
alignment bounding (`_last_active_frame_seconds`, rescaled via
`_rescale_to_placed_span`) -- this fix reuses it rather than computing a
second, separate "where does the assistant go quiet" value.

If `rollout.delay_seconds` is `None` (should not occur for any rollout
`select_best_rollout` would actually select, since the reward gate on
`yield_delay_reward` excludes never-yielded rollouts, but the field is
typed `float | None`), fall back to the original raw-`barge_in_seconds`
placement -- the same "degrade to the old, simpler behavior when the
better data isn't available" pattern already used for
`real_speech_end_sec is None -> assistant_end_sec`.

This still risks the interruption's derived placement overlapping the
opening clip's own tail on the user channel (if
`rescaled_real_speech_end_sec - delay_seconds` lands before
`opening_end_sec`) -- this is not a new risk introduced by this
correction; the original, unfixed code had the identical risk using raw
`barge_in_seconds`. `_mix`'s existing clip-raise already catches audible
overlap loudly rather than silently corrupting, which is the correct
behavior for a genuinely inconsistent rollout, not something this
correction needs to additionally guard against.

## Reproducibility entry point

- Same as the original design's live-verification section: run
  `scripts/build-self-distillation-corpus.py` against a small scenario
  subset via a live PersonaPlex instance
  (`scripts/_personaplex_ctl.py start`/`stop`) -- this time, run the
  documented warmup probe (`run-v6-native-eval.sh`'s `warm_up()` pattern)
  first, since the corpus-build script does not yet include it (tracked as
  a separate, already-known follow-up item, not addressed by this fix).
