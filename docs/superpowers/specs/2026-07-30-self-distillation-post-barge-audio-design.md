# Self-distillation post-barge-only audio placement

Date: 2026-07-30

Status: Proposed for implementation

Repository: `ctkadvisors/tolkien-voice` (private)

## Context

`docs/superpowers/specs/2026-07-30-self-distillation-realtime-timing-recalibration-design.md`
("the prior spec") fixed a guard that wrongly rejected 100% of live corpus
examples, then attempted to fix interruption placement by rescaling
wall-clock timestamps onto the assistant's compressed (sub-realtime)
playback span. Both pieces were implemented, independently task-reviewed,
and cleared individually -- but the branch's own final whole-branch review
found the combination introduces **silent corpus corruption**: the
interruption can land inside the opening clip's own playback window on
the same (user) channel, and normal speech amplitudes summed together
don't cross the clip-detection threshold, so nothing raises. Root cause,
confirmed with a live diagnostic probe on 2026-07-30: `first_output_received_seconds`
is anchored at session start (~0.0015s into the probe), not near when the
model starts substantively responding, and the model genuinely produces
real (if brief) vocal activity while the user's opening is still playing
-- 11 active output frames (rms >= 0.02) between t=0.68s and t=1.46s, well
before barge-in at t=3.89s. This is real full-duplex backchannel behavior,
not an artifact to filter out, and any design that tries to place the
*entire* captured session's audio on one reconstructed timeline has to
correctly model this overlap or risk exactly the corruption found above.

## Decision: stop trying to reconstruct the whole session

`Rollout.transcript` has been post-barge-only since it was introduced
(sub-project 2, to prevent a reward-gaming loophole where a policy speaks
well before barge-in and goes silent after). The corpus's semantic content
has therefore never included the pre-barge portion of the assistant's
turn. The audio reconstruction, however, has been trying to place the
*entire* captured session -- pre-barge backchannel, the long near-silent
listening stretch, the post-barge response, and the trailing
`response_seconds` silence window -- onto one timeline. That mismatch is
the root of every problem found across the prior spec's three review
rounds: whole-session compression math, ambiguity about what "real speech
end" anchors to, and now audio literally overlapping the opening.

Decision (confirmed with the project owner 2026-07-30): make the audio
match the transcript's existing scope. The assistant channel is silent
until shortly before the interruption, then contains the model's real
captured post-barge audio, extracted directly by content position rather
than reconstructed via wall-clock timing. This is a real change to what
training examples sound like (the assistant is not continuously
"speaking" for the whole clip) -- an intentional tradeoff for eliminating
essentially all of the timing complexity found so far, not an
oversight.

## Extraction: content position, not wall-clock time

`Rollout.output_frames` entries are appended to a list, in order, one per
`FRAME_SECONDS`-sized decoded chunk, during the live probe
(`native_probe.py`'s `probe_native_interruption`). An entry's *index* in
that list is therefore a direct count of decoded audio chunks -- a
content-position measure, immune to the sub-realtime wall-clock drift that
made every prior attempt at this fix fragile. Counting how many
`output_frames` entries have `received_seconds < barge_in_seconds` gives
the number of `FRAME_SECONDS`-sized chunks decoded before barge-in, which
converts directly to a sample offset into the *same* underlying decoded
buffer `_decode_opus_pcm` produces offline (same Opus bytes, same
`frame_samples = round(SAMPLE_RATE * FRAME_SECONDS)` chunking convention
already used identically in both `native_probe.py` and this module) --
no wall-clock-to-sample-position conversion, no rescale factor, no
timestamp arithmetic at all.

```
barge_in_frame_index = count of output_frames entries with received_seconds < barge_in_seconds
sample_offset_at_barge_in = barge_in_frame_index * frame_samples
```

Extract exactly `Rollout.delay_seconds` worth of samples starting there:

```
extraction_start = sample_offset_at_barge_in
extraction_end = extraction_start + round(delay_seconds * SAMPLE_RATE)
post_barge_samples = assistant_samples[extraction_start : extraction_end]
```

`delay_seconds` is the same ground-truth value `measure_interruption_response`
already computed and `select_best_rollout`'s reward gate already validated
-- reusing it here means the extracted segment's duration is exact, not
approximated via a second, independently-fallible "real speech end"
derivation (the prior spec's `_last_active_frame_seconds`, which scans the
*entire* `output_frames` list including the 12-second trailing silence
window and can be thrown off by a stray late blip -- a known, previously
flagged issue this design sidesteps rather than fixes).

**New guard, replacing the prior spec's absolute-tolerance drift check and
its narrower `decoded_duration <= 0` / `wall_clock_span < 0` successor:**
if `extraction_start >= len(assistant_samples)`, the decoded audio doesn't
extend as far as barge-in -- a genuine, meaningful data inconsistency
(nothing to extract), not an abstract wall-clock comparison:

```python
if extraction_start >= len(assistant_samples):
    raise ValueError(
        f"decoded assistant audio for {scenario_id} does not extend to "
        f"barge-in (decoded {len(assistant_samples)} samples, barge-in at "
        f"sample {extraction_start})"
    )
```

`extraction_end` is clamped to `len(assistant_samples)` (extract whatever
is available past that point) rather than raised on -- a session where the
assistant's real yield happened right at the edge of what got captured is
not itself evidence of corruption, unlike `extraction_start` landing
beyond the buffer entirely.

## Placement: direct, not rescaled

**Interruption** (user/right channel): placed so its audible onset lands
at `max(opening_end_sec, barge_in_seconds)` -- the `max` makes overlap
with the opening *structurally impossible* by construction, not
incidentally unlikely (unlike the prior spec's placement, which had no
such guarantee once derived from `real_speech_end_sec - delay_seconds`).
`barge_in_seconds` is a send-side timestamp (recorded in
`probe_native_interruption`'s sender loop, which paces frames in real
time and involves no GPU inference), so it is not subject to the
sub-realtime generation drift that motivated rescaling everything else in
the prior spec -- it can be used directly.

```
interruption_start_sec = max(opening_end_sec, barge_in_seconds) - onset_offset
```

**Assistant's post-barge segment** (assistant/left channel): placed
starting exactly at the interruption's own audible onset -- physically:
this is what the model said starting when the user interrupted, running
for `delay_seconds` until it yields. No separate "where does this segment
go" calculation; it shares the interruption's own onset point.

```
assistant_start_sec = interruption_start_sec + onset_offset
```

## Total example duration

`assistant_end_sec` for the purpose of sizing the output WAV is now
`assistant_start_sec + len(post_barge_samples) / SAMPLE_RATE` -- the
*extracted segment's* actual length (which may be shorter than
`delay_seconds` if `extraction_end` was clamped to the decoded buffer's
own end) -- not `assistant_start_sec + decoded_duration` (the prior
design's formula, which no longer applies since the full decoded buffer
is not what gets placed). The overall example duration remains
`max(opening_end_sec, interruption_end_sec, assistant_end_sec)`, unchanged
in form from the prior design, just fed this design's redefined
`assistant_end_sec`.

## Alignment: offset translation, not rescaling

Each `post_barge_text_frames` chunk's timestamp is on the same wall-clock
axis as `barge_in_seconds` (both derive from `loop.time() - started` in
`native_probe.py`). Since the assistant's segment is now placed via a
direct, uncompressed translation (not a scaled/compressed mapping), each
chunk's placed position is a simple offset, not a rescale:

```
placed_chunk_start = assistant_start_sec + (chunk_start - barge_in_seconds)
```

clamped to `[assistant_start_sec, assistant_start_sec + delay_seconds]`
(a chunk timestamped before barge-in or after the extracted segment's own
end has no meaningful position within this shorter clip). The last
chunk's bound is `assistant_start_sec + delay_seconds` (the extracted
segment's own natural end) rather than any independently-derived "real
speech end" value.

## What this removes

`_rescale_to_placed_span`, `_last_active_frame_seconds`'s use for audio
and alignment placement, and the `wall_clock_span`/`first_output`/`last_output`
consistency-guard machinery from the prior spec are no longer needed and
should be removed rather than left as dead code -- this design does not
compress or rescale anything, so nothing in this function needs to reason
about the ratio between wall-clock arrival span and decoded duration.
`_decode_opus_pcm` is unchanged (still needed to produce the full decoded
buffer this design then slices from). `leading_silence_seconds` is
unchanged (still used identically, to place the interruption clip's
audible onset correctly).

`Rollout.first_output_received_seconds`/`last_output_received_seconds`/
`output_frames` remain on the `Rollout` model (other consumers may still
reference them; `output_frames` specifically is still needed here, for
`barge_in_frame_index`) -- only their *use inside
`build_self_distillation_example`* changes.

## Testing and validation

- Unit tests for the frame-index-to-sample-offset extraction against a
  synthetic `output_frames` list with a known, hand-countable number of
  entries before/after a chosen `barge_in_seconds`.
- A test proving the interruption can never overlap the opening: construct
  a scenario where the prior design's formula *would* have produced
  overlap (reusing the concrete parameters from the whole-branch review's
  own reproduction -- 4.5s opening, `delay_seconds` large enough to have
  pulled the interruption's audible onset earlier than the opening's end
  under the old formula) and assert the new formula's `max(opening_end_sec, ...)`
  keeps the interruption's audible onset at or after `opening_end_sec`.
- A test for the new `extraction_start >= len(assistant_samples)` guard.
- Update or replace every existing test in `tests/test_training_self_distillation.py`
  that constructs a `Rollout` and calls `build_self_distillation_example`
  expecting the prior design's rescale-based placement/alignment behavior
  -- all of those need re-deriving under the new, simpler direct-offset
  model (most should become easier to reason about and verify by hand,
  since there is no compression factor to carry through the arithmetic
  anymore).

## Reproducibility entry point

Same as the prior spec's: run `scripts/build-self-distillation-corpus.py`
against a small scenario subset via a live PersonaPlex instance
(`scripts/_personaplex_ctl.py start`/`stop`, with the documented warmup
probe run first) -- this design's live-verification bar is unchanged: the
build should succeed (not reject on the sub-realtime shape the prior
guard wrongly rejected), and manual inspection of a resulting WAV should
confirm the interruption never overlaps the opening and the assistant's
segment starts audibly at the interruption's own onset.
