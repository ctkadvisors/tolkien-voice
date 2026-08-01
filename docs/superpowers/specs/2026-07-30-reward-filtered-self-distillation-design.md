# Reward-filtered self-distillation design

Date: 2026-07-30

Status: **CLOSED 2026-08-01 — implemented, failed the release gate,
approach abandoned.** The self-distillation-augmented adapters were
trained (runs/mvp-v6-{aragorn,sauron}-self-distill-v1-r64-384step) and
evaluated (eval-runs/mvp-v6-self-distill-v1/report.json): interruption
p95 got slightly WORSE (4085ms vs baseline's 3843ms), and persona
significance regressed into a new failure. Root cause of the whole
direction: post-interruption yield behavior is set by duplex pretraining
scale (millions of hours); a 20-example LoRA cannot move it — see
`2026-07-31-proxy-barge-in-mute-design.md`, whose proxy-level fix
achieved 93-191ms perceived interruption latency, live-verified. Kept as
the record of why this path is closed; several timing/alignment passages
below were superseded during implementation (see
`.superpowers/sdd/progress.md`) and should not be treated as accurate.

Repository: `ctkadvisors/tolkien-voice` (private)

## Context

Sub-project 1 (interruption reward module) and sub-project 2 (rollout
generation) are merged. The originally planned sub-project 3 was "GRPO
training loop (policy gradient, KL regularization)". Reading
`vendor/moshi-finetune/train.py` in full before starting sub-project 3
found this is not buildable as originally scoped without substantial,
risky surgery on pinned third-party code:

- `moshi-finetune`'s training loop is pure teacher-forced SFT: it calls
  `model(codes=codes, ...)`, compares the output logits against
  ground-truth `codes` via `compute_loss_with_mask`, and backpropagates.
  There is no sampling/generation code anywhere in it.
- `collect_rollout_group` (sub-project 2) gets its rollouts through the
  *deployed inference server* -- the same websocket protocol
  `collect-native-arm.py` uses -- which returns decoded audio bytes and
  transcript text. It does not and cannot expose the underlying Moshi
  token codes the training loop's loss function operates on, because the
  server-side protocol was never designed to.
- A true differentiable policy-gradient update needs the actual token
  sequence that was sampled, not just its audio/text rendering, so a loss
  can be computed against it. There is no path from what
  `collect_rollout_group` returns back to that token sequence without
  either running generation inside the training process itself (reusing
  moshi's own inference code, not moshi-finetune's, inside a modified
  training loop) or modifying the pinned, vendored PersonaPlex server to
  additionally expose raw tokens over the wire. Both are large, risky
  changes to code this project does not own the design of.

## Revised approach: reward-filtered self-distillation (rejection sampling)

Rather than true GRPO, this reuses 100% of what's already built and
proven and feeds it into the existing, already-validated SFT trainer:

1. Sample N rollouts per scenario (`collect_rollout_group`, sub-project 2,
   unchanged) -- **already verified live against Spark on 2026-07-30**:
   5 rollouts against one real scenario produced genuinely different
   transcripts, yield timings (53ms to 2.06s), and quality scores (0.0 to
   0.9). `RolloutGroup.is_degenerate` was `False`. The one previously
   unverified risk in sub-project 2's spec is resolved: PersonaPlex does
   produce diverse completions across repeated identical-prompt calls.
2. Select the best rollout per scenario by a gate-then-rank rule (not a
   weighted sum of two differently-scaled rewards): among rollouts that
   yielded within a target window, pick the one with the highest
   `quality_reward`. This mirrors the actual release gate's structure
   (interruption timing is a hard pass/fail threshold; persona quality is
   what's compared once timing passes) rather than inventing a new
   combined metric.
3. Build a training corpus from the selected rollouts using their
   **already-captured raw audio** (`probe_native_interruption`'s existing
   `raw_audio_path` parameter -- the model's own generated speech, not
   re-synthesized TTS), muxed with the original scenario's user-channel
   audio into the same two-channel WAV + alignment JSON format
   `render.py` already produces and `manifest.py`/`moshi-finetune`'s data
   loader already consume.
4. Retrain using the existing `train-mvp-v6-persona.sh` pipeline
   (unchanged) against this new, reward-filtered corpus, continuing from
   the v6 adapter.
5. Evaluate with the existing sealed native A/B pipeline
   (`run-v6-native-eval.sh`, unchanged).

This is the model learning from its own best-scoring behavior --
reinforcement in spirit, self-distillation/rejection-sampling in
mechanism -- rather than literal policy-gradient RL. It is a well
established technique for this class of problem (sometimes called RAFT /
best-of-N distillation) and, critically, requires no new training
infrastructure: steps 4-5 are scripts and a trainer that already exist
and are already proven end-to-end in this repository.

## Scope of this document (sub-project 3, revised)

Steps 1-3: sampling, selection, and corpus construction. Steps 4-5 reuse
existing scripts unmodified and are execution, not new design.

**Findings that expand scope beyond the original "no changes to
`collect_rollout_group`" line** (found by reading `rollout.py`,
`native_probe.py`, `render.py`, `schema.py`, `manifest.py` in full while
designing the corpus builder -- reported here rather than silently
narrowing the deliverable to fit the original out-of-scope line):

1. **`collect_rollout_group` never captures rollout audio at all.** It
   calls `probe_native_interruption` without `raw_audio_path`, so no
   audio from any of the N sampled rollouts is ever saved. A corpus
   builder needs that audio. Fix: pass a unique `raw_audio_path` per
   sampled rollout (index-suffixed, e.g.
   `{workdir}/{probe_id}-{index}.opus`) and add `raw_audio_path: Path`
   to `Rollout`.
2. **The previously-flagged `max_wait_seconds`/`response_seconds`
   conflation bug is still unfixed** (`rollout.py:117`,
   `result["max_wait_seconds"]` used as the observation window when the
   correct value is `response_seconds`, the parameter actually bounding
   the post-barge window `measure_interruption_response` searches).
   Sub-project 3 is the first actual consumer of these rewards for a
   training decision (which rollout to keep), so this must be fixed now,
   not deferred again: pass `response_seconds` (the local variable
   already in scope in `collect_rollout_group`) as
   `observation_window_seconds` instead of `result["max_wait_seconds"]`.
3. **Reconstructing the corpus example's timing needs one more
   timestamp `native_probe.py` computes but discards.**
   `probe_native_interruption` already tracks `audio_times` (the
   wall-clock-since-session-start receive time of each assistant audio
   packet) but only returns a length/gap summary of it
   (`interruption_underrun_count` etc.), not the value needed here:
   when the assistant's captured audio actually starts, on the same
   time axis as `barge_in_ms`. Fix: add
   `"first_output_received_seconds": audio_times[0] if audio_times else None`
   to its return dict (purely additive, same pattern as the
   `output_frames`/`max_wait_seconds` additions sub-project 1 already
   made to this function). Add `barge_in_seconds: float` and
   `first_output_received_seconds: float | None` to `Rollout`.

These are small, additive changes (new dict key, new optional-safe
pydantic fields, one bug-fix line) to already-existing, already-tested
functions -- not a redesign of sampling or reward computation, which
remain as sub-project 2 built them.

**Still out of scope:** any change to `moshi-finetune`,
`compute_yield_delay_reward`/`compute_quality_reward`'s formulas, or the
deployed PersonaPlex protocol.

## Architecture

Extend `Rollout` (`rollout.py`) with three fields needed to reconstruct a
correctly-timed corpus example -- everything a rollout's own probe call
already computes but previously discarded:

```python
class Rollout(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    transcript: str
    yield_delay_reward: float
    quality_reward: float
    delay_seconds: float | None
    raw_audio_path: Path
    barge_in_seconds: float
    first_output_received_seconds: float | None
```

New module, `src/tolkien_voice/training/self_distillation.py`:

```python
def select_best_rollout(
    group: RolloutGroup,
    *,
    yield_delay_reward_gate: float,
) -> Rollout | None:
    """Among rollouts with yield_delay_reward >= yield_delay_reward_gate
    (i.e. the model yielded within the gate's time budget -- note
    yield_delay_reward is negative seconds, so 'faster than the gate'
    means 'greater than or equal to', not 'less than'), returns the one
    with the highest quality_reward. Returns None if no rollout in the
    group clears the gate -- a scenario where every sample yielded too
    slowly contributes nothing to the corpus rather than contributing its
    least-bad failure.
    """


def leading_silence_seconds(wav_path: Path, *, activity_threshold: float = 0.01) -> float:
    """Seconds of near-silence before the first audible frame in wav_path.

    Reuses load_user_channel (native_probe.py, unchanged) to read the
    user-channel samples, then scans FRAME_SECONDS-sized chunks with the
    same 0.01 absolute-amplitude threshold probe_native_interruption
    already uses inline to find barge_in_seconds. Needed to place
    interruption_wav_path on the reconstructed user channel so its
    audible onset lands exactly at the rollout's own barge_in_seconds --
    the interruption clip almost always has some leading silence before
    speech starts, which would otherwise shift the whole channel out of
    sync with what the model actually heard.
    """


def build_self_distillation_example(
    rollout: Rollout,
    *,
    scenario_id: str,
    character: Literal["aragorn", "sauron"],
    opening_wav_path: Path,
    interruption_wav_path: Path,
    system_prompt: str,
    split: Split,
    output_directory: Path,
) -> RenderedExample:
    """Reconstructs the real recorded session timeline -- not a synthetic
    blueprint's heuristic timing -- on one shared time axis anchored at
    t=0 (session start, matching probe_native_interruption's `started`
    reference):

    - Right (user) channel: opening_wav_path at t=0; interruption_wav_path
      positioned so its audible onset (via leading_silence_seconds) lands
      at rollout.barge_in_seconds.
    - Left (assistant) channel: rollout.raw_audio_path decoded from Opus
      via sphn.OpusStreamReader (the same decoder native_probe.py itself
      uses to monitor output while probing) to 24kHz PCM, positioned
      starting at rollout.first_output_received_seconds. This is the
      model's own full captured turn -- both its pre-barge continuation
      and its post-barge yield -- placed at the exact time it was really
      produced, so the example teaches the real observed dynamic rather
      than a reconstructed approximation of it.

    Two-channel mixing, WAV writing (temp-file-then-rename, 0o600), and
    alignment-JSON writing (temp-file-then-rename, 0o600) duplicate
    render.py's existing pattern at small scale (~30 lines) rather than
    calling render_blueprint directly -- render_blueprint's _build_timeline
    is a from-text heuristic-timing engine for synthesized speech that
    doesn't apply here; a real recorded timeline needs no such heuristic.
    Word-level alignment reuses render.py's _assistant_alignments technique
    verbatim (uniform word-spacing across [rollout.first_output_received_seconds,
    end of decoded audio]) -- confirmed by reading render.py that this
    project's existing "alignment" is already this same uniform-spacing
    approximation, not true forced alignment, so no new alignment
    infrastructure is needed for parity.

    blueprint_hash is sha256(f"{scenario_id}:{rollout.transcript}") --
    self-distillation examples have no DialogueBlueprint to hash, but the
    schema requires a 64-char lowercase hex digest (schema.py's
    SHA256_PATTERN) that's unique per example (manifest.py's
    validate_rendered_examples enforces hash uniqueness across the
    corpus); hashing scenario_id+transcript satisfies both and stays
    traceable back to its source rollout.

    voice_license is the literal "self-distillation-v6-continuation" --
    this is the model's own generated audio, not a licensed third-party
    voice, and the field exists to track provenance, which this value
    does correctly (mirrors Provenance.voice_license's non-blank-string
    contract in schema.py, just without a real license to name).

    id is f"{scenario_id}-{rollout's selection index or a short digest}"
    matching RenderedExample's ^[a-z0-9][a-z0-9-]{2,127}$ pattern.
    """
```

## Open questions -- resolved

1. ~~Opus decoding~~ -- resolved above: `sphn.OpusStreamReader`, already a
   dependency, already used this way in `native_probe.py`.
2. ~~`blueprint_hash`/`voice_license` semantics~~ -- resolved above.
3. **`select_best_rollout`'s gate value:** `-1.0` (yield within 1 second),
   matching the project's own release gate (`interruption p95 must be at
   most 1,000 ms`) exactly rather than picking a new number.

## Corpus orchestration script

`scripts/build-self-distillation-corpus.py`: a new script, same shape as
`scripts/run-v6-native-eval.sh` + `scripts/collect-native-arm.py`'s
relationship -- shell owns the PersonaPlex container lifecycle via
`scripts/_personaplex_ctl.py` (unchanged), the Python script does the
work:

1. Load scenarios from the existing v6 eval prompt set
   (`evals/persona/probes.yaml`, unchanged -- 12 probes/character already
   built for the sealed eval, reused here as rollout-collection
   scenarios rather than building a separate scenario list).
2. For each scenario: `collect_rollout_group` (group_size configurable,
   default 6 -- large enough that the live diversity check's observed
   spread, e.g. quality_reward 0.0-0.9 in one group of 5, plausibly
   contains at least one rollout clearing the gate) against the running
   instance.
3. `select_best_rollout`; skip scenarios where it returns `None`, logging
   which ones (a scenario contributing nothing is a real, visible signal
   about that scenario's difficulty, not a silent drop).
4. `build_self_distillation_example` for each kept rollout, with
   `split=Split.TRAIN` always -- confirmed by reading
   `PersonaProbe`/`load_persona_manifest`
   (`src/tolkien_voice/evaluation.py`) that `probes.yaml` carries no
   split metadata to inherit; this corpus supplements v6's existing
   train/validation/held-out sets rather than replacing them, and
   held-out examples exist for evaluation only, so every new
   self-distillation example is training signal by construction.
5. `write_moshi_manifests` (`manifest.py`, unchanged) over the collected
   `RenderedExample` list -- same manifest format v6's corpus already
   produces, so `train-mvp-v6-persona.sh` consumes it with zero changes.

## Testing and validation

- `tests/test_training_rollout.py`: extend for the `Rollout` field
  additions and the `observation_window_seconds` fix -- a test asserting
  `collect_rollout_group` passes `response_seconds` (not
  `result["max_wait_seconds"]`) as the reward's observation window, and
  that each call receives a distinct `raw_audio_path`.
- `tests/test_training_native_probe.py`: extend for
  `first_output_received_seconds` -- present and equal to the first
  recorded audio-packet time when audio was received, `None` when it
  wasn't (mirrors the existing never-yielded test already covering
  `output_frames`/`interruption_response_ms`).
- `tests/test_training_self_distillation.py` (new): `select_best_rollout`
  (gate excludes/includes correctly, ranks by quality among survivors,
  returns `None` when nothing clears the gate); `leading_silence_seconds`
  (synthetic WAV fixtures with known silence lengths);
  `build_self_distillation_example` (synthetic opening/interruption WAVs
  + a synthetic Opus-encoded rollout audio fixture, asserting the
  returned `RenderedExample` passes `validate_rendered_examples` --
  reusing the real validator, the same real go/no-go check
  `write_moshi_manifests` itself runs, rather than a hand-rolled
  assertion list that could silently drift from what the trainer
  actually requires).
- **Live check, once Spark is reachable, not part of the automated
  suite:** run `build-self-distillation-corpus.py` against one running
  PersonaPlex instance for a small scenario subset (e.g. 2-3 scenarios
  per character) and manually listen to a resulting WAV to confirm the
  two channels are audible, in the right speaker positions, and
  correctly timed relative to each other -- the automated tests validate
  the file *format* is correct; only listening confirms the
  *reconstruction* is correct, since nothing else in this project's test
  suite has ground truth for "does this sound like the real
  interaction."

## Reproducibility entry point

- `scripts/build-self-distillation-corpus.py`, run against a live
  PersonaPlex instance via `scripts/_personaplex_ctl.py start` /
  `... stop`, same operational pattern as `scripts/run-v6-native-eval.sh`.
- Output: a manifest directory consumable by the existing, unmodified
  `train-mvp-v6-persona.sh` -- retraining and evaluation after this point
  need no new tooling.
