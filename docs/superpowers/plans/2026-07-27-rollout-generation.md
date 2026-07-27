# Rollout Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and locally-test the rollout-sampling and group-packaging code for the future GRPO policy update, per `docs/superpowers/specs/2026-07-27-rollout-generation-design.md`.

**Architecture:** A new module, `src/tolkien_voice/training/rollout.py`, with two pydantic types (`Rollout`, `RolloutGroup`) and one async function (`collect_rollout_group`) that calls the existing `probe_native_interruption` (native_probe.py, unchanged) N times against one scenario and scores each result with the existing `compute_yield_delay_reward`/`compute_quality_reward` (interruption_reward.py, unchanged).

**Tech Stack:** Python 3.12, pytest + pytest-asyncio, pydantic, httpx (`httpx.MockTransport` for judge tests). Tests for `collect_rollout_group` mock at the function boundary (`monkeypatch.setattr` on `probe_native_interruption` as imported into `rollout.py`), not the websocket/protocol boundary — this module is a *consumer* of `probe_native_interruption`, not a reimplementation of it, so its tests don't need the ~150-line aiohttp/numpy/sphn fake harness that `native_probe.py`'s own tests use.

## Global Constraints

- `collect_rollout_group` scores `post_barge_text`, not `interruption_text` or the full session `text` -- a deliberate choice, not a default. `interruption_text` includes pre-barge speech, which the final review of sub-project 1 flagged as blunting the anti-gaming property (a policy that speaks well before barge-in and then goes silent would score well on `interruption_text` despite exhibiting the exact failure mode the quality reward exists to catch). `post_barge_text` scores only what the model said after being interrupted -- empty post-barge text correctly produces a low judge score, which is the desired penalty for "goes silent to game the yield reward."
- `RolloutGroup.is_degenerate` is True only when **neither** `yield_delay_reward` nor `quality_reward` varies across the group (checked independently, not combined) -- a group where only one reward dimension has variance still carries usable signal for whichever combination sub-project 3 eventually uses.
- Calls to `probe_native_interruption` within one group are sequential, not concurrent -- PersonaPlex serves one full-duplex session at a time (README: "One private, full-duplex speech session at a time").
- `group_size` must be at least 1 (reject 0 explicitly); `group_size == 1` is allowed and always produces a degenerate group (self-documenting via `is_degenerate`, not a separate error path).
- `collect_rollout_group` does not start/stop the PersonaPlex instance itself -- that lifecycle belongs to whatever future script calls this function, via the existing `scripts/_personaplex_ctl.py`.
- All new tests run fully offline (no live GPU/network).

---

### Task 1: `Rollout` and `RolloutGroup`

**Files:**
- Create: `src/tolkien_voice/training/rollout.py`
- Test: `tests/test_training_rollout.py`

**Interfaces:**
- Produces: `Rollout` (fields: `transcript: str`, `yield_delay_reward: float`, `quality_reward: float`, `delay_seconds: float | None`) and `RolloutGroup` (fields: `probe_id: str`, `character: Literal["aragorn", "sauron"]`, `rollouts: list[Rollout]`, plus the `is_degenerate` property) -- both consumed by Task 2.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_training_rollout.py`:

```python
from tolkien_voice.training.rollout import Rollout, RolloutGroup


def _rollout(*, yield_delay_reward: float, quality_reward: float) -> Rollout:
    return Rollout(
        transcript="text",
        yield_delay_reward=yield_delay_reward,
        quality_reward=quality_reward,
        delay_seconds=0.5,
    )


def test_single_rollout_group_is_degenerate() -> None:
    group = RolloutGroup(
        probe_id="aragorn-lore-rangers",
        character="aragorn",
        rollouts=[_rollout(yield_delay_reward=-0.5, quality_reward=0.7)],
    )

    assert group.is_degenerate is True


def test_group_is_degenerate_when_both_rewards_are_constant() -> None:
    group = RolloutGroup(
        probe_id="aragorn-lore-rangers",
        character="aragorn",
        rollouts=[
            _rollout(yield_delay_reward=-0.5, quality_reward=0.7),
            _rollout(yield_delay_reward=-0.5, quality_reward=0.7),
        ],
    )

    assert group.is_degenerate is True


def test_group_is_not_degenerate_when_yield_reward_varies() -> None:
    group = RolloutGroup(
        probe_id="aragorn-lore-rangers",
        character="aragorn",
        rollouts=[
            _rollout(yield_delay_reward=-0.5, quality_reward=0.7),
            _rollout(yield_delay_reward=-1.2, quality_reward=0.7),
        ],
    )

    assert group.is_degenerate is False


def test_group_is_not_degenerate_when_quality_reward_varies() -> None:
    group = RolloutGroup(
        probe_id="aragorn-lore-rangers",
        character="aragorn",
        rollouts=[
            _rollout(yield_delay_reward=-0.5, quality_reward=0.7),
            _rollout(yield_delay_reward=-0.5, quality_reward=0.2),
        ],
    )

    assert group.is_degenerate is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd <repo> && UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_training_rollout.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tolkien_voice.training.rollout'`

- [ ] **Step 3: Write the minimal implementation**

Create `src/tolkien_voice/training/rollout.py`:

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


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
        """True only if neither reward dimension varies across the group.

        GRPO's advantage is (reward - group_mean); a group with zero
        variance on both dimensions contributes nothing to a policy
        update regardless of how the two rewards eventually get combined
        (sub-project 3's decision, not made yet). Checked independently
        rather than on a combined score so a group with signal on only
        one dimension isn't misclassified as degenerate.
        """
        if len(self.rollouts) <= 1:
            return True
        yield_rewards = {rollout.yield_delay_reward for rollout in self.rollouts}
        quality_rewards = {rollout.quality_reward for rollout in self.rollouts}
        return len(yield_rewards) <= 1 and len(quality_rewards) <= 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_training_rollout.py -v`
Expected: all four tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tolkien_voice/training/rollout.py tests/test_training_rollout.py
git commit -m "feat: add Rollout and RolloutGroup types

First piece of rollout generation (design doc:
docs/superpowers/specs/2026-07-27-rollout-generation-design.md).
is_degenerate checks both reward dimensions independently so a group
with signal on only one axis isn't misclassified as contributing
nothing to a future policy update."
```

---

### Task 2: `collect_rollout_group`

**Files:**
- Modify: `src/tolkien_voice/training/rollout.py`
- Test: `tests/test_training_rollout.py`

**Interfaces:**
- Consumes: `probe_native_interruption` from `tolkien_voice.training.native_probe` (unchanged), `compute_yield_delay_reward`/`compute_quality_reward` from `tolkien_voice.training.interruption_reward` (unchanged), `JudgeClient` from `tolkien_voice.training.judge` (unchanged)
- Produces: `async def collect_rollout_group(url: str, opening_wav_path: Path, interruption_wav_path: Path, *, probe_id: str, character: Literal["aragorn", "sauron"], probe_prompt: str, expected_traits: list[str], judge: JudgeClient, group_size: int, response_seconds: float = 12.0, max_wait_seconds: float = 12.0, seed: int) -> RolloutGroup`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_training_rollout.py`:

```python
import json
from pathlib import Path

import httpx
import pytest

import tolkien_voice.training.rollout as rollout_module
from tolkien_voice.training.judge import JudgeClient
from tolkien_voice.training.rollout import collect_rollout_group


@pytest.mark.asyncio
async def test_collect_rollout_group_scores_post_barge_text_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[float] = []

    async def fake_probe_native_interruption(
        url: str,
        opening_wav_path: Path,
        interruption_wav_path: Path,
        *,
        max_wait_seconds: float,
        response_seconds: float,
    ) -> dict:
        calls.append(max_wait_seconds)
        index = len(calls)
        return {
            "output_frames": [(0.0, 0.08), (0.04, 0.01), (0.08, 0.01), (0.12, 0.01), (0.16, 0.01)],
            "barge_in_ms": 100,
            "max_wait_seconds": max_wait_seconds,
            "interruption_response_ms": 40,
            "interruption_text": f"pre-barge speech. post-barge reply {index}",
            "post_barge_text": f"post-barge reply {index}",
        }

    monkeypatch.setattr(rollout_module, "probe_native_interruption", fake_probe_native_interruption)

    scored_user_messages: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        scored_user_messages.append(body["messages"][1]["content"])
        content = json.dumps(
            {
                "persona_score": 0.1 * len(scored_user_messages),
                "lore_score": 0.5,
                "identified_character": "aragorn",
                "rationale": "ok",
            }
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    judge = JudgeClient(base_url="http://teacher.test/v1", transport=httpx.MockTransport(handler))
    opening_wav = tmp_path / "opening.wav"
    interruption_wav = tmp_path / "interruption.wav"
    opening_wav.touch()
    interruption_wav.touch()

    group = await collect_rollout_group(
        "ws://probe",
        opening_wav,
        interruption_wav,
        probe_id="aragorn-lore-rangers",
        character="aragorn",
        probe_prompt="What did your years leading the Rangers of the North teach you?",
        expected_traits=["grounded", "resolute", "first-person"],
        judge=judge,
        group_size=3,
        seed=44,
    )

    assert len(calls) == 3
    assert len(group.rollouts) == 3
    assert group.probe_id == "aragorn-lore-rangers"
    assert group.character == "aragorn"
    assert all("post-barge reply" in message for message in scored_user_messages)
    assert all("pre-barge speech" not in message for message in scored_user_messages)
    assert group.rollouts[0].transcript == "post-barge reply 1"
    assert group.rollouts[0].delay_seconds == 0.04


@pytest.mark.asyncio
async def test_collect_rollout_group_rejects_group_size_below_one() -> None:
    with pytest.raises(ValueError, match="group_size"):
        await collect_rollout_group(
            "ws://probe",
            Path("/tmp/opening.wav"),
            Path("/tmp/interruption.wav"),
            probe_id="x",
            character="aragorn",
            probe_prompt="p",
            expected_traits=[],
            judge=JudgeClient(base_url="http://teacher.test/v1"),
            group_size=0,
            seed=1,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_training_rollout.py -v`
Expected: FAIL with `ImportError: cannot import name 'collect_rollout_group'`

- [ ] **Step 3: Write the minimal implementation**

Add to `src/tolkien_voice/training/rollout.py` (add these imports at the top alongside the existing ones):

```python
from pathlib import Path

from .interruption_reward import compute_quality_reward, compute_yield_delay_reward
from .judge import JudgeClient
from .native_probe import probe_native_interruption
```

Append the function:

```python
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
    """Samples group_size completions for one scenario against an
    already-running PersonaPlex instance and scores each.

    Calls are sequential, not concurrent: PersonaPlex serves one
    full-duplex session at a time. Scores post_barge_text (what the model
    said after being interrupted), not interruption_text (which also
    includes pre-barge speech and would blunt the anti-gaming property --
    a policy that speaks well before barge-in and then goes silent would
    otherwise score well despite exhibiting the exact failure mode the
    quality reward exists to catch).

    Does not start or stop the PersonaPlex instance itself -- that
    lifecycle belongs to the caller, via scripts/_personaplex_ctl.py.
    """
    if group_size < 1:
        raise ValueError("group_size must be at least 1")

    rollouts: list[Rollout] = []
    for index in range(group_size):
        result = await probe_native_interruption(
            url,
            opening_wav_path,
            interruption_wav_path,
            max_wait_seconds=max_wait_seconds,
            response_seconds=response_seconds,
        )
        yield_reward = compute_yield_delay_reward(
            result["output_frames"],
            barge_in_seconds=result["barge_in_ms"] / 1000,
            observation_window_seconds=result["max_wait_seconds"],
            silence_threshold=0.015,
            sustained_frames=12,
        )
        quality_reward = compute_quality_reward(
            result["post_barge_text"],
            target_character=character,
            probe_prompt=probe_prompt,
            expected_traits=expected_traits,
            judge=judge,
            seed=seed + index,
        )
        known_delay_ms = result["interruption_response_ms"]
        rollouts.append(
            Rollout(
                transcript=result["post_barge_text"],
                yield_delay_reward=yield_reward,
                quality_reward=quality_reward,
                delay_seconds=(known_delay_ms / 1000 if known_delay_ms is not None else None),
            )
        )
    return RolloutGroup(probe_id=probe_id, character=character, rollouts=rollouts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_training_rollout.py -v`
Expected: all six tests in this file PASS.

- [ ] **Step 5: Run the full suite and lint**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` — expect all tests passing (198, up from 192, or nearby depending on exact final count).
Run: `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests scripts` — expect "All checks passed!" (a prior sub-project's final review found this project's plans need an explicit lint step at task boundaries; this step exists because of that finding).

- [ ] **Step 6: Commit**

```bash
git add src/tolkien_voice/training/rollout.py tests/test_training_rollout.py
git commit -m "feat: add collect_rollout_group

Second piece of rollout generation. Samples group_size completions
sequentially against one scenario (PersonaPlex serves one full-duplex
session at a time) and scores post_barge_text specifically, not
interruption_text or the full session transcript -- a deliberate choice
per the design doc, since interruption_text includes pre-barge speech
and would blunt the anti-gaming property the quality reward exists to
enforce."
```

---

## Follow-up (not part of this plan)

Whether repeated calls to `probe_native_interruption` with an identical prompt actually produce different completions is unverified -- nothing in PersonaPlex's protocol exposes a sampling/temperature control. This cannot be checked without a live instance. The first concrete action once Spark is reachable: call `collect_rollout_group` for one real scenario with `group_size >= 4` and confirm `RolloutGroup.is_degenerate` is false. If it's true, that blocks sub-project 3 (the GRPO update has nothing to compute an advantage over) until resolved -- not a task in this plan, tracked as the explicit next step in the design doc's Verification section.
