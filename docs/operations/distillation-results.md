# Tolkien-native STS distillation results

Date: 2026-07-22  
Hardware: NVIDIA DGX Spark / GB10  
Status: v3 failed release gates; v4 persona-adapter training prepared; no trained adapter is deployed

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

## Reproducibility entry points

- `scripts/train-mvp-v2.sh`: corrected v2 reference run.
- `scripts/train-mvp-v3.sh`: prompt annotation, corpus audit, v3 training, and
  finite held-out checkpoint selection.
- `scripts/prepare-blind-eval.py`: seeded arm sealing and randomized probes.
- `scripts/collect-native-arm.py`: native full-duplex speech collector.
- `scripts/score-native-results.py`: sealed Tolkien-judge scoring.
- `scripts/report-native-eval.py`: precommitted release gates and confidence
  intervals.

No prompt-only demo, smoke checkpoint, or failed experimental adapter is
authorized for production activation.
