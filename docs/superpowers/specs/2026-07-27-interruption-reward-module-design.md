# Interruption reward module design

Date: 2026-07-27

Status: Proposed for implementation

Repository: `ctkadvisors/tolkien-voice` (private)

## Context

Every trained adapter through v6 has failed the interruption p95 release
gate (must be at most 1,000ms; v6 measured 3,843ms on the adapter arm,
3,488ms on base). Three separate supervised-fine-tuning data defects were
found and fixed across v2 (missing prompt conditioning), v4 (reweighting
that amplified a timing bug), and v6 (curriculum/timing aliasing that
concentrated interruption training on one curriculum) -- interruption
timing never moved regardless. See `docs/operations/distillation-results.md`
for the full history.

Research into established practice (same date, see that document's
"Research notes for the next iteration" section) found that Kyutai's own
2026 follow-up work on Moshi/PersonaPlex (arXiv:2606.11167, "Multi-Faceted
Interactivity Alignment in Full-Duplex Speech Models") trains this exact
capability -- user interruption handling -- via GRPO reinforcement learning
with an interruption response-delay reward plus an LLM-judge quality
reward, on real two-party human conversation recordings, not supervised
fine-tuning on synthetic dialogue. This is consistent with interruption
being a capability the SFT-on-synthetic-data paradigm this project has
used is not well suited to move, independent of how carefully the
synthetic data is curated.

## Scope

This is **sub-project 1 of 4** in a larger GRPO-based interruption-training
initiative, decomposed because it spans independent subsystems too large
for one design/plan/implementation cycle:

1. **Data + reward pipeline** (this document)
2. Rollout generation (drive the live model to produce completions for
   training)
3. GRPO training loop (policy gradient, KL regularization, produces a
   new adapter -- likely continuing from the v6 SFT adapter rather than
   raw base, so the persona gate it already cleared isn't lost; this
   specific decision belongs to sub-project 3, not resolved here)
4. Evaluation & gating (largely already built -- extend the existing
   sealed native A/B pipeline from `scripts/run-v6-native-eval.sh`)

This document covers only the reward computation piece of sub-project 1:
build and validate the reward functions against real audio the project
*already has*, before any external dataset download or RL infrastructure
exists. Real-data acquisition (Seamless Interaction segment extraction) is
a following, separate step once this reward is proven -- not covered here.

**Out of scope for this document:** rollout generation, the GRPO loop
itself, adapter lineage/continuation decisions, Seamless Interaction
download and extraction, any change to the existing SFT training pipeline.

## Architecture

A new module, `src/tolkien_voice/training/interruption_reward.py`, matching
the project's existing pattern for training modules: small pure functions,
no hidden I/O beyond what's passed in, pydantic result types. Same shape as
`native_probe.py` and `scoring.py`.

Two independent scoring functions, not one collapsed scalar -- the final
weighting between them is a GRPO-training-time tuning decision (sub-project
3), not something to bake in at the reward-definition layer:

```python
def compute_yield_delay_reward(
    frames: list[tuple[float, float]],
    *,
    barge_in_seconds: float,
    observation_window_seconds: float,
    silence_threshold: float,
    sustained_frames: int,
) -> float:
    """R = -d, where d is seconds from barge-in to sustained silence.

    Reuses measure_interruption_response's detection logic directly rather
    than reimplementing it, so this can never silently drift from what the
    release gate actually measures.
    """

def compute_quality_reward(
    transcript: str,
    *,
    target_character: Literal["aragorn", "sauron"],
    probe_prompt: str,
    expected_traits: list[str],
    judge: JudgeClient,
    seed: int,
) -> float:
    """Returns JudgeClient.score(...).persona_score.

    Exists so a policy cannot trivially maximize the yield reward by
    learning to go silent immediately regardless of content -- persona_score
    specifically, not an average with lore_score, since "goes silent /
    generic" is what this reward exists to catch; lore accuracy is already
    the SFT side's concern.
    """

class InterruptionRewardResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    yield_delay_reward: float
    quality_reward: float
    delay_seconds: float | None  # None only if compute_yield_delay_reward clamped
    judge_rationale: str
```

## Data flow

Inputs are exactly what `collect-native-arm.py` already produces per
observation: the per-frame `(timestamp, rms)` list, `barge_in_seconds`, and
the decoded transcript. `observation_window_seconds` is the same value as
`probe_native_interruption`'s existing `max_wait_seconds` parameter for
that observation (not a new, separately-tuned quantity) -- the reward's
"never yielded" case and the probe's own timeout are the same window by
definition, so this must not drift into a second, independently-chosen
number. `probe_native_interruption` already computes
`output_frames` internally during collection but currently discards it --
this needs to be persisted alongside the raw `.opus` audio (a small,
additive change to that function's return value / the collector's output
schema) so the reward module can be validated against recordings that
already exist, with no new data collection required for validation.

## Reward formula and edge cases

- **Yield detected within the observation window:** `R = -delay_seconds`.
  Raw seconds, unnormalized -- matches Kyutai's own `R_int = -d` and relies
  on GRPO's group-relative advantage normalization downstream (sub-project
  3) rather than normalizing at the reward-function layer.
- **Yield never detected** (today's `measure_interruption_response` returns
  `None` in this case): clamp to `R = -observation_window_seconds` -- the
  worst measurable outcome within however long the probe watched, not an
  unbounded or undefined penalty. This must be explicit and tested; an
  unhandled `None` here would either crash a future RL loop or silently
  propagate as NaN, which is exactly the shape of bug this project has
  already been bitten by more than once.
- **Judge failure/timeout:** `compute_quality_reward` propagates
  `JudgeClient`'s existing `JudgeResponseError` rather than swallowing it
  into a default score -- a caller (later, the rollout loop in sub-project
  2) decides how to handle a bad rollout, this module does not silently
  substitute a guessed reward.

## Validation and testing

- `tests/test_training_interruption_reward.py`: unit tests against
  synthetic frame sequences, same style as
  `tests/test_training_native_probe.py` -- a fake `JudgeClient` via
  dependency injection, no network or GPU required. Explicit test cases for
  both edge cases above (never-yielded clamp, judge failure propagation).
- `scripts/validate-interruption-reward.py`: runs both reward functions
  against the real `base-arm.jsonl` / `adapter-arm-aragorn.jsonl` /
  `adapter-arm-sauron.jsonl` recordings already collected by
  `run-v6-native-eval.sh` (once frame persistence is added, re-run against
  those or a fresh collection), printing both reward components per
  observation alongside the already-known `interruption_response_ms` and
  `persona_score` from that same data.

  **Go/no-go check:** does the reward's ranking of "which responses handled
  interruption well" agree with what the sealed eval already established?
  If it doesn't, that's a stop-and-fix signal before any of this is wired
  to a system that spends GPU-hours on rollouts scored by it -- not
  something to proceed past on the assumption the formula must be right
  because it matches the paper.

## Reproducibility entry point

- `scripts/validate-interruption-reward.py`: the validation check described
  above. No training or GPU-heavy step in this sub-project; this can run
  entirely against existing recordings.
