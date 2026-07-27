# Interruption Reward Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and locally-validate the reward computation functions (yield-delay reward, LLM-judge quality reward) for the future GRPO interruption-training initiative, per `docs/superpowers/specs/2026-07-27-interruption-reward-module-design.md`.

**Architecture:** A new pure-function module, `src/tolkien_voice/training/interruption_reward.py`, reusing `measure_interruption_response` (native_probe.py) and `JudgeClient` (judge.py). One small additive change to `native_probe.py` to expose per-frame RMS data already computed internally. A validation script that runs both reward functions against real recordings once available.

**Tech Stack:** Python 3.12, pytest + pytest-asyncio, pydantic, httpx (with `httpx.MockTransport` for judge tests), the project's existing fake-module injection pattern for aiohttp/numpy/sphn (see `tests/test_training_native_probe.py`).

## Global Constraints

- Reuse `measure_interruption_response` directly for the yield-delay reward rather than reimplementing detection logic — the reward must never be able to silently drift from what the release gate measures (spec, Architecture section).
- `observation_window_seconds` in `compute_yield_delay_reward` is always the same value as the caller's `probe_native_interruption` `max_wait_seconds` for that observation, not an independently chosen number (spec, Data flow section).
- `compute_quality_reward` uses `persona_score` only, not an average with `lore_score` (spec, Architecture section).
- `compute_quality_reward` propagates `JudgeResponseError` rather than substituting a default score on judge failure (spec, Reward formula section).
- All new tests run fully offline (no live GPU/network) using the project's existing fake-module injection pattern (`monkeypatch.setitem(sys.modules, ...)`) or `httpx.MockTransport` — matching `tests/test_training_native_probe.py` and `tests/test_training_judge.py`.

---

### Task 1: Persist interruption output frames in native_probe.py

**Files:**
- Modify: `src/tolkien_voice/training/native_probe.py` (the `return` statement at the end of `probe_native_interruption`, currently lines 375-390)
- Test: `tests/test_training_native_probe.py`

**Interfaces:**
- Consumes: nothing new (uses the existing local variable `output_frames: list[tuple[float, float]]` already populated inside `probe_native_interruption`)
- Produces: `probe_native_interruption(...)`'s returned dict now includes `"output_frames": list[tuple[float, float]]`, which flows unchanged into `collect-native-arm.py`'s JSONL rows since that script already does `rows.append({..., **interruption})`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_training_native_probe.py` (append after the existing `test_probe_native_session_waits_for_sender_and_measures_real_speech` test, reusing the same fake-module pattern already in that file):

```python
@pytest.mark.asyncio
async def test_probe_native_interruption_exposes_output_frames_for_reward_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tolkien_voice.training.native_probe import probe_native_interruption

    def write_wav(path: Path, amplitude: int) -> None:
        frames = array("h", [amplitude] * 960)
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(24_000)
            output.writeframes(frames.tobytes())

    opening_wav = tmp_path / "opening.wav"
    interruption_wav = tmp_path / "interruption.wav"
    write_wav(opening_wav, amplitude=1000)
    write_wav(interruption_wav, amplitude=2000)

    ws_msg_type = SimpleNamespace(BINARY="binary", CLOSE="close", CLOSED="closed", ERROR="error")

    class FakeMessage:
        def __init__(self, message_type: str, data: bytes) -> None:
            self.type = message_type
            self.data = data

    class FakeWebSocket:
        def __init__(self) -> None:
            self._queue: asyncio.Queue[FakeMessage] = asyncio.Queue()
            self._handshake_sent = False
            self._send_count = 0

        async def __aenter__(self) -> "FakeWebSocket":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def receive(self) -> FakeMessage:
            if not self._handshake_sent:
                self._handshake_sent = True
                return FakeMessage(ws_msg_type.BINARY, b"\x00")
            return await self._queue.get()

        async def send_bytes(self, _data: bytes) -> None:
            self._send_count += 1
            await asyncio.sleep(0.02)
            if self._send_count == 1:
                for _ in range(3):
                    await self._queue.put(FakeMessage(ws_msg_type.BINARY, b"\x01speech"))

    websocket = FakeWebSocket()

    class FakeClientSession:
        async def __aenter__(self) -> "FakeClientSession":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        def ws_connect(self, *_: object, **__: object) -> FakeWebSocket:
            return websocket

    class FakeWriter:
        def __init__(self, sample_rate: int) -> None:
            self.sample_rate = sample_rate

        def append_pcm(self, _frame: object) -> None:
            return None

        def read_bytes(self) -> bytes:
            return b"frame"

    class FakeReader:
        def __init__(self, sample_rate: int) -> None:
            self.sample_rate = sample_rate
            self._payload = b""

        def append_bytes(self, payload: bytes) -> None:
            self._payload = payload

        def read_pcm(self) -> list[float]:
            frame_samples = round(self.sample_rate * native_probe.FRAME_SECONDS)
            value = 0.05 if self._payload.startswith(b"speech") else 0.0
            return [value] * frame_samples

    fake_aiohttp = ModuleType("aiohttp")
    fake_aiohttp.ClientSession = FakeClientSession
    fake_aiohttp.WSMsgType = ws_msg_type
    fake_numpy = ModuleType("numpy")

    class FakeArray:
        def __init__(self, values: list[float]) -> None:
            self._values = values

        @property
        def size(self) -> int:
            return len(self._values)

        def reshape(self, _shape: int) -> "FakeArray":
            return self

        def __getitem__(self, item: slice | int) -> "FakeArray | float":
            if isinstance(item, slice):
                return FakeArray(self._values[item])
            return self._values[item]

        def __iter__(self):
            return iter(self._values)

    def asarray(values: object, dtype: object | None = None) -> FakeArray:
        del dtype
        return FakeArray(list(values))

    def square(values: FakeArray) -> FakeArray:
        return FakeArray([value * value for value in values])

    def mean(values: FakeArray) -> float:
        squared = list(values)
        return sum(squared) / len(squared)

    fake_numpy.asarray = asarray
    fake_numpy.float32 = "float32"
    fake_numpy.mean = mean
    fake_numpy.sqrt = math.sqrt
    fake_numpy.square = square
    fake_sphn = ModuleType("sphn")
    fake_sphn.OpusStreamReader = FakeReader
    fake_sphn.OpusStreamWriter = FakeWriter
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "sphn", fake_sphn)

    result = await probe_native_interruption(
        "ws://probe",
        opening_wav,
        interruption_wav,
        max_wait_seconds=1.0,
        response_seconds=0.04,
    )

    assert "output_frames" in result
    assert len(result["output_frames"]) > 0
    assert all(isinstance(frame, tuple) and len(frame) == 2 for frame in result["output_frames"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd <repo> && UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_training_native_probe.py::test_probe_native_interruption_exposes_output_frames_for_reward_reuse -v`
Expected: FAIL with `KeyError: 'output_frames'` (or an `AssertionError` on `"output_frames" in result`) — the field doesn't exist in the returned dict yet.

- [ ] **Step 3: Add `output_frames` to the returned dict**

In `src/tolkien_voice/training/native_probe.py`, find the `return` statement at the end of `probe_native_interruption` (currently reads exactly):

```python
    packet_summary = summarize_audio_packets(audio_times)
    return {
        "interruption_response_ms": measure_interruption_response(
            output_frames,
            barge_in_seconds=barge_in_seconds,
            silence_threshold=silence_threshold,
            # Require roughly half a second of quiet so a normal pause is not
            # mistaken for the assistant yielding to a barge-in.
            sustained_frames=12,
        ),
        "barge_in_ms": round(barge_in_seconds * 1000),
        "output_activity_frames": len(output_frames),
        "interruption_audio_bytes": len(audio_bytes),
        "interruption_underrun_count": packet_summary["underrun_count"],
        "interruption_max_packet_gap_ms": packet_summary["max_packet_gap_ms"],
        **summarize_interruption_text(text_frames, barge_in_seconds=barge_in_seconds),
    }
```

Change it to:

```python
    packet_summary = summarize_audio_packets(audio_times)
    return {
        "interruption_response_ms": measure_interruption_response(
            output_frames,
            barge_in_seconds=barge_in_seconds,
            silence_threshold=silence_threshold,
            # Require roughly half a second of quiet so a normal pause is not
            # mistaken for the assistant yielding to a barge-in.
            sustained_frames=12,
        ),
        "barge_in_ms": round(barge_in_seconds * 1000),
        "output_activity_frames": len(output_frames),
        "output_frames": output_frames,
        "interruption_audio_bytes": len(audio_bytes),
        "interruption_underrun_count": packet_summary["underrun_count"],
        "interruption_max_packet_gap_ms": packet_summary["max_packet_gap_ms"],
        **summarize_interruption_text(text_frames, barge_in_seconds=barge_in_seconds),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_training_native_probe.py -v`
Expected: all tests PASS, including the new one. If the new test's fake-message timing doesn't trigger `output_started` before `max_wait_seconds` elapses (visible as the test hanging or raising `RuntimeError: native STS did not speak before interruption timeout`), increase the number of queued `b"\x01speech"` messages in `FakeWebSocket.send_bytes`'s `if self._send_count == 1` branch from 3 to 5 and re-run — `ActivityDetector` needs `sustained_frames=3` *consecutive* above-threshold frames, and this only needs to be enough to satisfy that deterministically.

- [ ] **Step 5: Run the full test suite to confirm no regressions**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q`
Expected: all tests pass (188 previously, 189 with this addition).

- [ ] **Step 6: Commit**

```bash
git add src/tolkien_voice/training/native_probe.py tests/test_training_native_probe.py
git commit -m "feat: expose output_frames from probe_native_interruption

Needed by the upcoming interruption reward module so it can be validated
against real recordings without redoing RMS computation from raw audio.
Already computed internally and passed to measure_interruption_response;
this just also includes it in the returned dict, which collect-native-arm.py
already merges into its JSONL rows unchanged."
```

---

### Task 2: `compute_yield_delay_reward` and `InterruptionRewardResult`

**Files:**
- Create: `src/tolkien_voice/training/interruption_reward.py`
- Test: `tests/test_training_interruption_reward.py`

**Interfaces:**
- Consumes: `measure_interruption_response` from `tolkien_voice.training.native_probe` (Task 1's module, unchanged signature)
- Produces: `compute_yield_delay_reward(frames: list[tuple[float, float]], *, barge_in_seconds: float, observation_window_seconds: float, silence_threshold: float, sustained_frames: int) -> float` and `InterruptionRewardResult` (pydantic model with fields `yield_delay_reward: float`, `quality_reward: float`, `delay_seconds: float | None`, `judge_rationale: str`) — both consumed by Task 3 and Task 4.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_training_interruption_reward.py`:

```python
from tolkien_voice.training.interruption_reward import compute_yield_delay_reward


def test_yield_delay_reward_is_negative_seconds_to_silence() -> None:
    frames = [
        (1.00, 0.08),
        (1.04, 0.07),
        (1.08, 0.06),
        (1.12, 0.01),
        (1.16, 0.06),
        (1.20, 0.01),
        (1.24, 0.008),
        (1.28, 0.006),
        (1.32, 0.005),
        (1.36, 0.004),
    ]

    reward = compute_yield_delay_reward(
        frames,
        barge_in_seconds=1.10,
        observation_window_seconds=12.0,
        silence_threshold=0.015,
        sustained_frames=4,
    )

    assert reward == -0.1


def test_yield_delay_reward_clamps_to_negative_window_when_never_yielded() -> None:
    frames = [(1.0 + index * 0.04, 0.08) for index in range(10)]

    reward = compute_yield_delay_reward(
        frames,
        barge_in_seconds=1.10,
        observation_window_seconds=12.0,
        silence_threshold=0.015,
        sustained_frames=4,
    )

    assert reward == -12.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_training_interruption_reward.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tolkien_voice.training.interruption_reward'`

- [ ] **Step 3: Write the minimal implementation**

Create `src/tolkien_voice/training/interruption_reward.py`:

```python
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .native_probe import measure_interruption_response


class InterruptionRewardResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    yield_delay_reward: float
    quality_reward: float
    delay_seconds: float | None


def compute_yield_delay_reward(
    frames: list[tuple[float, float]],
    *,
    barge_in_seconds: float,
    observation_window_seconds: float,
    silence_threshold: float,
    sustained_frames: int,
) -> float:
    """Reward = -d, where d is seconds from barge-in to sustained silence.

    Mirrors Kyutai's R_int = -d (arXiv:2606.11167) but applied to this
    project's own yield-latency definition (time for the assistant to go
    quiet after being barged in on), matching what the release gate
    actually measures -- not the paper's different turn-taking-response
    metric. When the assistant never yields within the observation window,
    clamps to -observation_window_seconds: the worst measurable outcome
    within however long was watched, not an unbounded or undefined penalty.
    """
    if observation_window_seconds <= 0:
        raise ValueError("observation_window_seconds must be positive")
    delay_ms = measure_interruption_response(
        frames,
        barge_in_seconds=barge_in_seconds,
        silence_threshold=silence_threshold,
        sustained_frames=sustained_frames,
    )
    if delay_ms is None:
        return -observation_window_seconds
    return -(delay_ms / 1000)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_training_interruption_reward.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tolkien_voice/training/interruption_reward.py tests/test_training_interruption_reward.py
git commit -m "feat: add compute_yield_delay_reward

First piece of the interruption reward module (design doc:
docs/superpowers/specs/2026-07-27-interruption-reward-module-design.md).
Reuses measure_interruption_response directly so this can never drift
from what the release gate measures. Clamps to -observation_window_seconds
on a never-yielded observation rather than propagating None/NaN."
```

---

### Task 3: `compute_quality_reward`

**Files:**
- Modify: `src/tolkien_voice/training/interruption_reward.py`
- Test: `tests/test_training_interruption_reward.py`

**Interfaces:**
- Consumes: `JudgeClient` and `JudgeResponseError` from `tolkien_voice.training.judge` (existing, unchanged)
- Produces: `compute_quality_reward(transcript: str, *, target_character: Literal["aragorn", "sauron"], probe_prompt: str, expected_traits: list[str], judge: JudgeClient, seed: int) -> float`, consumed by Task 4's validation script

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_training_interruption_reward.py`:

```python
import json

import httpx
import pytest

from tolkien_voice.training.interruption_reward import compute_quality_reward
from tolkien_voice.training.judge import JudgeClient, JudgeResponseError


def test_quality_reward_is_the_judges_persona_score() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        content = json.dumps(
            {
                "persona_score": 0.7,
                "lore_score": 0.95,
                "identified_character": "aragorn",
                "rationale": "Stayed in character despite the interruption.",
            }
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    judge = JudgeClient(base_url="http://teacher.test/v1", transport=httpx.MockTransport(handler))

    reward = compute_quality_reward(
        "I took that road because no other force could reach the river in time.",
        target_character="aragorn",
        probe_prompt="Why did you choose the Paths of the Dead?",
        expected_traits=["grounded", "resolute", "first-person"],
        judge=judge,
        seed=44,
    )

    assert reward == 0.7


def test_quality_reward_propagates_judge_failure_instead_of_guessing() -> None:
    judge = JudgeClient(
        base_url="http://teacher.test/v1",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": []})),
        max_attempts=1,
    )

    with pytest.raises(JudgeResponseError):
        compute_quality_reward(
            "...",
            target_character="aragorn",
            probe_prompt="Why did you choose the Paths of the Dead?",
            expected_traits=["grounded"],
            judge=judge,
            seed=44,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_training_interruption_reward.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_quality_reward'`

- [ ] **Step 3: Write the minimal implementation**

Add to `src/tolkien_voice/training/interruption_reward.py` (after `compute_yield_delay_reward`, add the import at the top alongside the existing one):

```python
from typing import Literal

from .judge import JudgeClient
```

```python
def compute_quality_reward(
    transcript: str,
    *,
    target_character: Literal["aragorn", "sauron"],
    probe_prompt: str,
    expected_traits: list[str],
    judge: JudgeClient,
    seed: int,
) -> float:
    """The judge's persona_score, not an average with lore_score.

    The failure mode this reward exists to prevent is a policy that learns
    to go silent or generic to trivially maximize the yield-delay reward --
    persona_score penalizes that directly. Lore accuracy is the SFT side's
    concern, already covered by the existing training pipeline.

    Raises JudgeClient's existing JudgeResponseError on judge failure rather
    than substituting a default score -- a caller decides how to handle a
    bad rollout, this function does not silently guess.
    """
    return judge.score(
        target_character=target_character,
        probe_prompt=probe_prompt,
        expected_traits=expected_traits,
        candidate_transcript=transcript,
        seed=seed,
    ).persona_score
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_training_interruption_reward.py -v`
Expected: all four tests in this file PASS.

- [ ] **Step 5: Run the full test suite**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/tolkien_voice/training/interruption_reward.py tests/test_training_interruption_reward.py
git commit -m "feat: add compute_quality_reward

Second piece of the interruption reward module. Uses persona_score
specifically (not averaged with lore_score) since this reward exists to
catch a policy learning to go silent/generic to game the yield-delay
reward. Propagates JudgeResponseError rather than guessing a score on
judge failure."
```

---

### Task 4: Validation script against real recordings

**Files:**
- Create: `scripts/validate-interruption-reward.py`

**Interfaces:**
- Consumes: `compute_yield_delay_reward`, `compute_quality_reward` from Tasks 2-3; `JudgeClient` from `tolkien_voice.training.judge`; reads JSONL files matching `collect-native-arm.py`'s output schema, now including `output_frames` per Task 1
- Produces: a printed comparison table; no code depends on this script's output, it is the go/no-go check described in the spec

- [ ] **Step 1: Write the script**

Create `scripts/validate-interruption-reward.py`:

```python
#!/usr/bin/env python3
"""Validate the interruption reward against real collected recordings.

Go/no-go check per docs/superpowers/specs/2026-07-27-interruption-reward-module-design.md:
does the reward's ranking of "handled interruption well" agree with what a
sealed native A/B eval already established, before this is wired to
anything that spends GPU-hours on rollouts scored by it. Requires
observations collected AFTER src/tolkien_voice/training/native_probe.py's
output_frames change (Task 1) -- rows collected before that change will be
skipped with a warning, not silently scored as zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tolkien_voice.training.interruption_reward import (
    InterruptionRewardResult,
    compute_quality_reward,
    compute_yield_delay_reward,
)
from tolkien_voice.training.judge import JudgeClient, JudgeResponseError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:9090/v1")
    parser.add_argument("--model", default="tolkien")
    parser.add_argument("--observation-window-seconds", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()

    rows: list[dict] = []
    for path in args.input:
        with path.open(encoding="utf-8") as source:
            rows.extend(json.loads(line) for line in source if line.strip())

    skipped = [row for row in rows if "output_frames" not in row]
    scoreable = [row for row in rows if "output_frames" in row]
    if skipped:
        print(
            f"WARNING: skipping {len(skipped)} row(s) collected before "
            "output_frames was persisted (Task 1) -- these cannot be scored."
        )
    if not scoreable:
        print("No scoreable rows (all collected before output_frames was added).")
        return 1

    with JudgeClient(base_url=args.base_url, model=args.model) as judge:
        for row in scoreable:
            frames = [tuple(frame) for frame in row["output_frames"]]
            yield_reward = compute_yield_delay_reward(
                frames,
                barge_in_seconds=row["barge_in_ms"] / 1000,
                observation_window_seconds=args.observation_window_seconds,
                silence_threshold=0.015,
                sustained_frames=12,
            )
            known_delay_ms = row.get("interruption_response_ms")
            try:
                quality_reward = compute_quality_reward(
                    row["interruption_text"],
                    target_character=row["character"],
                    probe_prompt=row["prompt"],
                    expected_traits=row["expected_traits"],
                    judge=judge,
                    seed=args.seed,
                )
            except JudgeResponseError as error:
                print(f"WARNING: judge failed for {row['probe_id']} ({row['blinded_arm']}): {error}")
                continue
            result = InterruptionRewardResult(
                yield_delay_reward=yield_reward,
                quality_reward=quality_reward,
                delay_seconds=(known_delay_ms / 1000 if known_delay_ms is not None else None),
            )
            print(
                json.dumps(
                    {
                        "probe_id": row["probe_id"],
                        "blinded_arm": row["blinded_arm"],
                        "known_interruption_response_ms": known_delay_ms,
                        **result.model_dump(),
                    }
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify it runs against a synthetic fixture**

There is no committed automated test for this script (it is the manual go/no-go check itself, run against real data per the spec) — but verify it at least imports and argument-parses correctly before relying on it:

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-interruption-reward.py --help`
Expected: prints the argparse help text with no import errors.

- [ ] **Step 3: Commit**

```bash
git add scripts/validate-interruption-reward.py
git commit -m "feat: add validate-interruption-reward.py

The go/no-go check described in the design spec. Not runnable against
real data yet -- needs a fresh collection run (any run-v6-native-eval.sh
invocation) made AFTER Task 1's output_frames change, since existing
collected recordings predate it. That run requires live GPU access and is
tracked as follow-up work, not part of this plan."
```

---

## Follow-up (not part of this plan)

Running `scripts/validate-interruption-reward.py` against real data requires a fresh `run-v6-native-eval.sh` collection (or a smaller standalone collection) made after Task 1 lands, which needs live Spark/GPU access. That collection + the go/no-go read is the actual spec-mandated validation gate before this reward is trusted — track it as the next concrete action once Spark connectivity is back, not a task in this plan (this plan only needed to produce and locally test the code).
