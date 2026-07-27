# Rollout generation design

Date: 2026-07-27

Status: Proposed for implementation

Repository: `ctkadvisors/tolkien-voice` (private)

## Context

Sub-project 1 (interruption reward module, `docs/superpowers/specs/2026-07-27-interruption-reward-module-design.md`) built and validated the reward computation. This is sub-project 2 of the same 4-part GRPO initiative: rollout generation -- driving the live model to produce multiple sampled completions per scenario so a policy-gradient update (sub-project 3) has something to compute a relative advantage over.

Confirmed during brainstorming: this does **not** need new pipeline infrastructure. `generate.py`/`render.py` (v6, already fixed) already produce realistic scenarios; `native_probe.py`'s `probe_native_session`/`probe_native_interruption` and the `_personaplex_ctl.py` orchestrator (both from this session) already drive a live PersonaPlex instance and collect a response. GRPO doesn't need real recorded audio as an imitation target -- it scores the model's own sampled behavior -- so real Seamless Interaction data (confirmed: consented, compensated, purpose-built research participants, not genuine private conversations -- arXiv:2506.22554, "All participants signed an informed consent and were paid for their time") is a later calibration input, not a blocker for this sub-project.

## Scope

Sub-project 2 only. Two things, both reusing existing functions rather than building new collection mechanics:

1. Sample **N** completions for one scenario against a live PersonaPlex instance, instead of the existing pipeline's 1.
2. Package the N scored completions into a `RolloutGroup` for sub-project 3 to consume.

**Out of scope:** the GRPO policy-gradient update itself (sub-project 3), real Seamless Interaction data acquisition, any change to `generate.py`/`render.py`/`native_probe.py`'s existing collection functions.

## Open technical risk -- explicit, not assumed away

GRPO's relative-advantage math requires the N completions to actually differ. Nothing in PersonaPlex's websocket protocol (as used by `probe_native_session`/`probe_native_interruption`) exposes a temperature or seed parameter -- whether repeated identical-prompt calls produce genuinely different completions is unverified. This cannot be checked without a live PersonaPlex instance (Spark), so it is not assumed either way here. It is the first thing to verify once Spark is reachable, before trusting any rollout group this code produces (see Verification section) -- same pattern as sub-project 1's real-data validation gate.

## Architecture

A new module, `src/tolkien_voice/training/rollout.py`, matching the project's existing pattern (small pure/thin functions, pydantic result types).

```python
class Rollout(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    transcript: str
    yield_delay_reward: float
    quality_reward: float
    delay_seconds: float | None


class RolloutGroup(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    probe_id: str
    character: Literal["aragorn", "sauron"]
    rollouts: list[Rollout]

    @property
    def is_degenerate(self) -> bool:
        """True if every rollout's yield_delay_reward is identical.

        GRPO's advantage is (reward - group_mean); a degenerate group has
        zero variance and contributes nothing to a policy update. Surfacing
        this as a named property means a caller can filter degenerate
        groups explicitly rather than silently training on a zero-signal
        batch.
        """


async def collect_rollout_group(
    url: str,
    opening_wav_path: Path,
    interruption_wav_path: Path,
    *,
    probe_id: str,
    character: Literal["aragorn", "sauron"],
    probe_prompt: str,
    expected_traits: list[str],
    judge: JudgeClient,
    group_size: int,
    response_seconds: float = 12.0,
    max_wait_seconds: float = 12.0,
    seed: int,
) -> RolloutGroup:
    """Calls probe_native_interruption group_size times against the same
    scenario, scores each with compute_yield_delay_reward and
    compute_quality_reward (sub-project 1, unchanged), and returns a
    RolloutGroup. Does not itself start/stop the PersonaPlex instance --
    the caller (a future rollout-collection script, not part of this
    sub-project) owns that lifecycle via the existing
    scripts/_personaplex_ctl.py, exactly as run-v6-native-eval.sh already
    does for evaluation.
    """
```

## Data flow

`collect_rollout_group` is a thin loop: call `probe_native_interruption` (native_probe.py, unchanged) `group_size` times sequentially against one already-running PersonaPlex instance and one scenario, then `compute_quality_reward`/`compute_yield_delay_reward` (interruption_reward.py, unchanged) on each result, matching exactly what `validate-interruption-reward.py` already does per-row -- this function is that same scoring logic, called in a loop instead of once, over live calls instead of persisted JSONL rows.

Sequential, not concurrent: PersonaPlex is a single stateful stream per README ("One private, full-duplex speech session at a time"); concurrent calls against the same instance would violate that.

## Verification

- `tests/test_training_rollout.py`: unit tests using the same fake-module injection pattern as `native_probe.py`'s tests. Cover: `RolloutGroup.is_degenerate` true when all rewards equal, false when they differ; `collect_rollout_group` calls the probe exactly `group_size` times; a probe failure on one call doesn't silently drop it from the group (propagates, matching sub-project 1's "don't guess on failure" principle) rather than returning a short group.
- **Live check, not part of this sub-project's automated tests, first thing once Spark is reachable:** run `collect_rollout_group` for one real scenario with `group_size >= 4` against a live instance and confirm `is_degenerate` is false. If it's true, the sampling-diversity risk above is real and blocks sub-project 3 until resolved (would need investigating whether PersonaPlex's protocol has an undocumented sampling control, or whether diversity needs to come from varying the input instead, e.g. audio-level jitter).
