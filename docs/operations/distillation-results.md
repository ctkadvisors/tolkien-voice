# Tolkien-native STS distillation results

Date: 2026-07-27
Hardware: NVIDIA DGX Spark / GB10
Status: v4 failed release gates; v5 trained against a corrupted corpus and is
superseded, not evaluated; v6 (corpus generator fix) is prepared; no trained
adapter is deployed

This log records the native speech-to-speech distillation experiments after the
one-step plumbing smoke test. Private corpora, checkpoints, raw speech, sealed
arm assignments, and judge outputs remain outside Git under
`/home/chrisknuteson/tolkien-voice-data` on Spark.

## Fixed evaluation protocol

The release comparison is a blinded, randomized base-vs-adapter evaluation.
Each arm receives the same held-out spoken probes and interruption fixture.
The collector records native response audio, decoded transcript, first-audio
latency, interruption response, and underruns. A separately loaded Tolkien 27B
judge scores persona fidelity, lore accuracy, and cross-persona confusion
without receiving the arm identity. The sealed arm map is opened only when the
release report is generated.

An adapter may become active only if the report establishes a material gain
over base and all absolute speech/latency gates pass. Checkpoint selection and
deployment are separate operations; selection never updates the active adapter.

## v1: longer plumbing run

The first non-smoke run used a balanced 128-dialogue, 62.23-minute private
Aragorn/Sauron corpus, LoRA rank 16, 12-second crops, and 192 optimizer steps.
Step 64 was selected. The blinded result did not establish improvement:

| Metric | Base | v1 adapter |
| --- | ---: | ---: |
| Persona fidelity | 0.2250 | 0.2167 |
| Lore accuracy | 0.2333 | 0.1667 |
| Interruption p95 | not releaseable | 4,299 ms |

Result: failed and not deployed.

## v2: longer context and held-out selection

v2 used LoRA rank 32, 32-second crops, 256 steps, and a held-out validation
split. The first attempt exposed an upstream trainer defect: its finite
validation iterator was consumed after the first checkpoint, causing later
evaluations to report `NaN` despite finite training losses. The training
wrapper now creates a fresh held-out loader for every checkpoint. The corrected
run produced finite, monotonically improving held-out losses:

| Step | Eval loss | Perplexity |
| ---: | ---: | ---: |
| 64 | 3.9219105 | 15.15698 |
| 128 | 3.8761706 | 14.68398 |
| 192 | 3.8509741 | 14.42975 |
| 256 | 3.8456433 | 14.37653 |

Step 256 was selected with adapter SHA-256
`efc05af094cdf4e702ebcdf5f3175fa1dcd18d07d16fd15e42f54fd145c37c65`.
Runtime validation confirmed the exact pinned base revision and 545 merged
projection pairs.

The fresh native evaluation collected 12 observations and 24 audio artifacts
per arm. No transcript or interruption result was missing. The blinded report
failed:

| Metric | Base | v2 adapter | Required result |
| --- | ---: | ---: | --- |
| Persona fidelity | 0.3333 | 0.2500 | positive 95% delta |
| Lore accuracy | 0.3000 | 0.2083 | positive 95% delta |
| Cross-persona confusion | 0 | 0 | no regression |
| Interruption p95 | 2,681 ms | 3,822 ms | at most 1,000 ms |
| Underruns per observation | 1 | 1 | below release limit |

The paired bootstrap confidence intervals were `[-0.2917, -0.0833, 0.1333]`
for persona and `[-0.3167, -0.0917, 0.1500]` for lore. v2 therefore remained
inactive.

## v2 root cause

The generated alignment records identified the central defect: every example
had `text_conditions: null`, and the upstream trainer never inserted a system
prompt. Aragorn and Sauron examples therefore trained a single adapter without
any character-discriminating input. At runtime PersonaPlex did receive a
character prompt, but those prompt tokens were absent during training. The
observed generic, blended speech was the expected failure mode, not evidence of
successful multi-persona distillation.

## v3: runtime-prompt-conditioned training

v3 corrects that defect before adding more compute. Corpus preparation resolves
each example's character through the same registry used by inference, stores
the exact runtime system prompt and its digest, and rejects unconditioned or
malformed examples. The training tokenizer prepends that prompt to the native
interleaved sequence on the text channel and masks its audio codebooks. This
makes the runtime character selection an actual learned conditioning variable.

The run used LoRA rank 64, 32-second crops, 384 optimizer steps, BF16, gradient
checkpointing, and evaluations every 96 steps. All four independent held-out
passes were finite and improved monotonically:

| Step | Eval loss | Perplexity |
| ---: | ---: | ---: |
| 96 | 4.9785948 | 31.52872 |
| 192 | 4.1654987 | 17.94486 |
| 288 | 3.7702532 | 13.64455 |
| 384 | 3.7064388 | 13.05417 |

Step 384 was selected. Peak allocated GPU memory was 21.676 GiB. The selected
adapter SHA-256 is
`30cee9d2a0a9734fd197edcfc7c109e12d269acc4c346f4ba008b28bdc774632`.
Its metadata binds the artifact to the exact PersonaPlex base revision and the
pinned PersonaPlex, Moshi, and trainer source revisions.

The new sealed native comparison collected 12 observations and 24 private Opus
artifacts for each arm, with no empty transcripts or missing interruption
measurements. v3 improved both point estimates but did not establish a positive
paired difference at 95% confidence:

| Metric | Base | v3 adapter | Paired delta CI95 |
| --- | ---: | ---: | ---: |
| Persona fidelity | 0.2750 | 0.3833 | `[-0.1417, 0.1083, 0.3833]` |
| Lore accuracy | 0.2417 | 0.3917 | `[-0.1167, 0.1500, 0.4375]` |
| Cross-persona confusion | 0 | 0 | no regression |
| Interruption p95 | 4,295 ms | 3,960 ms | must be at most 1,000 ms |
| Underruns per observation | 1 | 1 | no regression |

The v3 point changes were +39% relative for persona and +62% relative for lore,
showing that runtime-prompt conditioning fixed a real part of v2's failure.
However, gains were inconsistent by probe. The adapter improved modern-topic
redirection, Sauron's Númenor response, and uncertainty behavior, while
regressing on the Andúril, Paths of the Dead, and Rings probes and failing both
character-break probes. It remained inactive.

## v4: weighted persona adapter bank

v4 keeps one native PersonaPlex STS backbone but trains independently selected
Aragorn and Sauron LoRAs. The controller passes the selected character into the
runtime before model load, which mounts that character's exact verified
adapter. Characters without a trained entry continue to use the base model.
Global and per-character adapter configuration are mutually exclusive.

For each character, v4 filters the original private corpus into a distinct
training and validation set. It gives one additional sampling weight to lore,
uncertainty, resistance, and repair examples, and three additional weights to
real interruption-timed examples. This targets the observed failures without
copying held-out probe prompts into training. The character adapters retain
the exact v3 runtime-prompt prefix, 32-second context, rank 64, BF16, and four
fresh held-out checkpoints. Selection still cannot activate an adapter.

Both persona runs selected step 384. Aragorn's held-out loss improved from
4.2604 at step 96 to 2.5713 at step 384; Sauron's improved from 5.2486 to
3.4555. The selected adapter SHA-256 values are
`4539d03543e468d93cd1e08ef4543f87779f7a38388862809c2a409745f41be0`
for Aragorn and
`f8671ce2b41ceb297a19b22de8af37dbde4652dd899b98f3497f3918103ea3a5`
for Sauron.

The fresh sealed evaluation again collected 12 observations and 24 artifacts
per arm with no missing measurements. Persona-specific adapters materially
improved the point estimates, but the run still failed the precommitted gates:

| Metric | Base | v4 adapter bank | Paired delta CI95 |
| --- | ---: | ---: | ---: |
| Persona fidelity | 0.3250 | 0.5083 | `[-0.0583, 0.1833, 0.4417]` |
| Lore accuracy | 0.3000 | 0.4792 | `[-0.0917, 0.1792, 0.4750]` |
| Cross-persona confusion | 0 | 0 | no regression |
| Interruption p95 | 2,812 ms | 3,598 ms | must be at most 1,000 ms |
| Underruns per observation | 1 | 1 | no regression |

The adapter bank produced +56% relative persona and +60% relative lore point
gains, but neither paired confidence interval excluded zero. The adapter bank
therefore remained inactive.

## v4 interruption root cause and v5 correction

An audit of the rendered duplex alignments found a training/evaluation
distribution error. Every nominal interruption example placed the user's
barge-in only 150 ms before the assistant's *natural end*. Those examples
taught near-end overlap rather than interruption during active speech, while
the evaluator interrupts shortly after response onset. Oversampling those
examples in v4 amplified the wrong behavior.

v5 preserves the successful persona-specific architecture and changes the
duplex rendering contract: an interruption now begins in the first third of
the assistant turn and never later than one second after it starts. The
assistant channel is truncated with only a 100 ms crossfade. A fresh rendering
of all 128 private dialogues contains 56.39 minutes of audio. Its 32
interruption examples all begin at 1.000 seconds, between 8% and 23% of the
assistant's natural turn. All 128 alignments carry the exact runtime persona
prompt. The old audio is not reused.

`train-mvp-v5-persona.sh` was subsequently run to completion for both
characters against that `mvp-v5-early-barge` corpus (`run_dir`s
`mvp-v5-aragorn-early-barge-r64-384step` and
`mvp-v5-sauron-early-barge-r64-384step`, both reaching `checkpoint_000384`
with a selected adapter). Those results are **not reported here and must not
be evaluated or activated**: an independent review found the `mvp-v5-early-barge`
corpus itself was corrupted at generation time (see "v5 corpus defect and v6
correction" below), so any metrics from that run measure the wrong thing. The
run directories are retained on disk for provenance but superseded.

## v5 corpus defect and v6 correction

A review ahead of committing further GPU time to v5 found the corpus
generator (`generate.py`), not the trainer, was still broken. Timing-kind
assignment (`index % len(TimingKind)`) and curriculum assignment (period-8
cycle) aliased perfectly, so every curriculum position was locked to exactly
one timing kind for the entire corpus -- concretely, every "lore" dialogue in
`mvp-v5-early-barge` was also an "interruption" dialogue, and the held-out
validation split (`index % 8 == 7`) was a single confounded cell: 100%
interruption x 100% "uncertainty", with "uncertainty" never appearing in
training at all. This is the most likely root explanation for the whole
v1-v4 pattern: held-out loss always improved cleanly (the validation slice
was internally consistent) while the blinded release gate always failed,
because checkpoint selection was driven by loss on a slice sharing almost
nothing with what was actually trained. It also meant v5's early-barge-in
fix was, in practice, applied almost entirely to lore/uncertainty content --
53% of all lore and uncertainty assistant speech was truncated to ~1.1
seconds and then oversampled 5x by the persona weighting scheme.

v6 fixes the generator: timing kind is now assigned per curriculum-round
(`index // len(_CATEGORY_ORDER)`) instead of per curriculum-slot, and the
train/validation split is stratified across both curriculum and timing.
`validate_substantive_corpus` now asserts both splits cover every curriculum
and every timing kind, and rejects confounded coverage outright, so this bug
class fails loudly at generation time rather than silently at evaluation
time. A fresh 128-dialogue corpus (`mvp-v6-decoupled`) was generated through
the real teacher model and rendered through the real synthesizer: every one
of the 8 curricula now has exactly 2 samples per timing kind (64 per
character), and the 8-row validation split spans all 8 curricula and all 4
timings. The persona-weighted training distribution was verified against the
real rendered audio: 126 weighted rows per character, max single-clip share
3.3-3.7% (v5/v4: 4.81%, concentrated entirely in lore), and lore's
interruption-truncated share down to 14-17% (from effectively 100%).

`train-mvp-v6-persona.sh` and `configs/training/mvp-v6-{aragorn,sauron}.yaml`
run the same architecture (rank 64, 32s crops, BF16, 384 steps, held-out
selection) against `mvp-v6-decoupled`, writing to fresh `run_dir`s
(`mvp-v6-{character}-decoupled-r64-384step`) so the superseded v5 runs are
left untouched.

Both v6 runs completed on 2026-07-27: aragorn in 28m45s (final held-out loss
2.255, checkpoint 384 selected, adapter SHA-256
`4bb7edcc3ac1bb11af12796a365af57d394a692e14e4615d99f7dc3b446fa1d7`), sauron
in 29m37s (final held-out eval_loss 2.818, eval_perplexity 7.053, checkpoint
384 selected, adapter SHA-256
`2485d3778f0011ed21588b6ab4df3f6cf14f3581c382405b2571240e3091caa8`). Both are
bound to base revision `fdaf4090a61cb315c138a1faee287ffd6c716309`. **Training
completing cleanly is not evidence the release gate will pass** -- v1 through
v4 all trained to completion with improving held-out loss and still failed
the blinded base-vs-adapter comparison. The next step is the same fixed
evaluation protocol used for v1-v4: `prepare-blind-eval.py` seals the arm
assignment, `collect-native-arm.py` collects real native speech from both
base and v6-adapter arms, `score-native-results.py` runs the sealed
Tolkien-judge scoring, and `report-ab-eval.py` applies the precommitted
gates. No claim is made here until that report exists.

## Reproducibility entry points

- `scripts/train-mvp-v2.sh`: corrected v2 reference run.
- `scripts/train-mvp-v3.sh`: prompt annotation, corpus audit, v3 training, and
  finite held-out checkpoint selection.
- `scripts/train-mvp-v5-persona.sh`: corrected-corpus audit, per-character
  weighted manifest generation, v5 training, and held-out selection. Ran
  against the corrupted `mvp-v5-early-barge` corpus; superseded, see above.
- `scripts/train-mvp-v6-persona.sh`: same pipeline against the corpus-generator
  fix (`mvp-v6-decoupled`), writing to fresh `run_dir`s.
- `scripts/prepare-blind-eval.py`: seeded arm sealing and randomized probes.
- `scripts/collect-native-arm.py`: native full-duplex speech collector.
- `scripts/score-native-results.py`: sealed Tolkien-judge scoring.
- `scripts/report-ab-eval.py`: precommitted release gates and confidence
  intervals.

No prompt-only demo, smoke checkpoint, or failed experimental adapter is
authorized for production activation.
