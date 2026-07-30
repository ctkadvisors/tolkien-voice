# Reward-Filtered Self-Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a corpus builder that samples N live completions per interruption scenario, keeps the one that both yields fast and stays in character, and reassembles the model's own captured audio (muxed with the scenario's real user audio, on the real recorded timeline) into a `moshi-finetune`-ready training example -- so the existing SFT trainer can retrain on the model's own best behavior instead of only synthetic TTS dialogue.

**Architecture:** Two small additive fixes to already-merged sub-project 1/2 code (native_probe.py gains one more returned timestamp; rollout.py starts capturing rollout audio and fixes a known reward-window bug), a new pure-function module (`self_distillation.py`) for selection and audio reconstruction, and a new orchestration script that ties sampling -> selection -> corpus-building -> manifest-writing together using entirely existing, unmodified building blocks (`collect_rollout_group`, `write_moshi_manifests`, `_personaplex_ctl.py`).

**Tech Stack:** Python 3.12, pydantic, `sphn` (Opus codec), pytest + pytest-asyncio, existing project conventions (temp-file-then-rename writes at 0o600/0o700, `extra="forbid", frozen=True` pydantic models).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-30-reward-filtered-self-distillation-design.md` -- read this first for the full rationale; this plan implements it verbatim.
- `select_best_rollout`'s gate is `-1.0` (yield within 1 second) -- exact value, matches the project's release gate.
- Sample rate is always `24_000`; `RenderedExample.channels` is always `2`; `RenderedExample.sample_rate` is always `24000` (schema.py enforces these as `Literal` types already -- do not pass other values).
- `blueprint_hash` for a self-distillation example is `hashlib.sha256(f"{scenario_id}:{rollout.transcript}".encode("utf-8")).hexdigest()`.
- `voice_license` for a self-distillation example is the literal string `"self-distillation-v6-continuation"`.
- `split` for a self-distillation example is always `Split.TRAIN`.
- All new/changed file writes that produce a WAV or JSON artifact follow the existing pattern: write to a `.tmp` sibling, `chmod(0o600)`, then `os.replace` to the final path (see `render.py:78-99` for the exact pattern to copy).
- Do not modify `vendor/moshi-finetune/`, `compute_yield_delay_reward`, `compute_quality_reward`, or the PersonaPlex websocket protocol.

---

### Task 1: Fix reward-window bug, capture rollout audio, add missing timestamp

**Files:**
- Modify: `src/tolkien_voice/training/native_probe.py:373-392` (`probe_native_interruption`'s return dict)
- Modify: `src/tolkien_voice/training/rollout.py` (`Rollout` model, `collect_rollout_group`)
- Test: `tests/test_training_native_probe.py` (extend existing test at line 279)
- Test: `tests/test_training_rollout.py` (extend `_rollout` helper and existing tests)

**Interfaces:**
- Consumes: nothing new -- extends existing, already-tested functions.
- Produces: `Rollout.raw_audio_path: Path`, `Rollout.barge_in_seconds: float`, `Rollout.first_output_received_seconds: float | None`, and `probe_native_interruption`'s return dict gains `"first_output_received_seconds"`. Task 2/3 consume these.

- [ ] **Step 1: Write the failing test for `first_output_received_seconds`**

Add this assertion to the existing test
`test_probe_native_interruption_exposes_output_frames_for_reward_reuse` in
`tests/test_training_native_probe.py`, immediately after the existing
`assert result["max_wait_seconds"] == 1.0` line (around line 427):

```python
    assert result["first_output_received_seconds"] == result["output_frames"][0][0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_training_native_probe.py::test_probe_native_interruption_exposes_output_frames_for_reward_reuse -v`
Expected: FAIL with `KeyError: 'first_output_received_seconds'`

- [ ] **Step 3: Add the field to `probe_native_interruption`'s return dict**

In `src/tolkien_voice/training/native_probe.py`, in the `return {` block at
the end of `probe_native_interruption` (currently lines 375-392), add one
key. The block currently reads:

```python
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
        "max_wait_seconds": max_wait_seconds,
        "interruption_audio_bytes": len(audio_bytes),
        "interruption_underrun_count": packet_summary["underrun_count"],
        "interruption_max_packet_gap_ms": packet_summary["max_packet_gap_ms"],
        **summarize_interruption_text(text_frames, barge_in_seconds=barge_in_seconds),
    }
```

Add `"first_output_received_seconds"` right after `"output_frames"`:

```python
        "output_frames": output_frames,
        "first_output_received_seconds": audio_times[0] if audio_times else None,
        "max_wait_seconds": max_wait_seconds,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_training_native_probe.py::test_probe_native_interruption_exposes_output_frames_for_reward_reuse -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tolkien_voice/training/native_probe.py tests/test_training_native_probe.py
git commit -m "feat: expose first_output_received_seconds from probe_native_interruption"
```

- [ ] **Step 6: Write the failing tests for `Rollout`'s new fields and the reward-window fix**

In `tests/test_training_rollout.py`, replace the `_rollout` helper (lines
9-18) with a version that supplies the three new required fields with
sensible defaults so existing tests that don't care about them keep
passing unchanged:

```python
def _rollout(
    *,
    yield_delay_reward: float,
    quality_reward: float,
    raw_audio_path: Path = Path("/tmp/rollout.opus"),
    barge_in_seconds: float = 1.1,
    first_output_received_seconds: float | None = 0.2,
) -> Rollout:
    return Rollout(
        transcript="text",
        yield_delay_reward=yield_delay_reward,
        quality_reward=quality_reward,
        delay_seconds=0.5,
        raw_audio_path=raw_audio_path,
        barge_in_seconds=barge_in_seconds,
        first_output_received_seconds=first_output_received_seconds,
    )
```

Then, in `test_collect_rollout_group_scores_post_barge_text_only`
(currently lines 71-138), update the fake probe function's signature and
returned dict to accept `raw_audio_path` and to report
`first_output_received_seconds`, and record the paths it's called with.
Replace the whole function body from `calls: list[float] = []` through
the `monkeypatch.setattr(...)` line with:

```python
    calls: list[float] = []
    raw_audio_paths: list[Path] = []

    async def fake_probe_native_interruption(
        url: str,
        opening_wav_path: Path,
        interruption_wav_path: Path,
        *,
        max_wait_seconds: float,
        response_seconds: float,
        raw_audio_path: Path,
    ) -> dict:
        calls.append(response_seconds)
        raw_audio_paths.append(raw_audio_path)
        index = len(calls)
        return {
            "output_frames": [(0.0, 0.08), (0.04, 0.01), (0.08, 0.01), (0.12, 0.01), (0.16, 0.01)],
            "barge_in_ms": 100,
            "max_wait_seconds": max_wait_seconds,
            "first_output_received_seconds": 0.0,
            "interruption_response_ms": 40,
            "interruption_text": f"pre-barge speech. post-barge reply {index}",
            "post_barge_text": f"post-barge reply {index}",
        }

    monkeypatch.setattr(rollout_module, "probe_native_interruption", fake_probe_native_interruption)
```

(`calls` now tracks `response_seconds`, not `max_wait_seconds`, because
Step 8 below makes `collect_rollout_group` pass `response_seconds` as the
reward's observation window -- this test's `assert len(calls) == 3` still
holds, it's just tracking the corrected parameter.)

At the end of the test function (after the existing
`assert group.rollouts[0].delay_seconds == 0.04` line), add:

```python
    assert len(set(raw_audio_paths)) == 3
    assert group.rollouts[0].raw_audio_path == raw_audio_paths[0]
    assert group.rollouts[0].barge_in_seconds == 0.1
    assert group.rollouts[0].first_output_received_seconds == 0.0
```

Apply the same signature change (add `raw_audio_path: Path` keyword-only
parameter) to `failing_probe_native_interruption` in
`test_collect_rollout_group_propagates_mid_group_failure` (currently
lines 147-165) -- it doesn't need to record or return anything new, just
accept the parameter so the call doesn't raise `TypeError`:

```python
    async def failing_probe_native_interruption(
        url: str,
        opening_wav_path: Path,
        interruption_wav_path: Path,
        *,
        max_wait_seconds: float,
        response_seconds: float,
        raw_audio_path: Path,
    ) -> dict:
        calls.append(max_wait_seconds)
        if len(calls) == 2:
            raise RuntimeError("probe failed")
        return {
            "output_frames": [(0.0, 0.08), (0.04, 0.01), (0.08, 0.01), (0.12, 0.01), (0.16, 0.01)],
            "barge_in_ms": 100,
            "max_wait_seconds": max_wait_seconds,
            "first_output_received_seconds": 0.0,
            "interruption_response_ms": 40,
            "interruption_text": "pre-barge speech. post-barge reply",
            "post_barge_text": "post-barge reply",
        }
```

- [ ] **Step 7: Run tests to verify they fail**

Run: `uv run pytest tests/test_training_rollout.py -v`
Expected: FAIL -- `Rollout` rejects the new kwargs (`extra="forbid"`) and/or
`collect_rollout_group` calls the fake without `raw_audio_path`, causing a
`TypeError`.

- [ ] **Step 8: Update `Rollout` and `collect_rollout_group`**

In `src/tolkien_voice/training/rollout.py`, replace the `Rollout` class
(lines 13-19):

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

Replace the body of `collect_rollout_group`'s loop (lines 97-137) with:

```python
    rollouts: list[Rollout] = []
    for index in range(group_size):
        raw_audio_path = raw_audio_directory / f"{probe_id}-{index}.opus"
        result = await probe_native_interruption(
            url,
            opening_wav_path,
            interruption_wav_path,
            max_wait_seconds=max_wait_seconds,
            response_seconds=response_seconds,
            raw_audio_path=raw_audio_path,
        )
        yield_reward = compute_yield_delay_reward(
            result["output_frames"],
            barge_in_seconds=result["barge_in_ms"] / 1000,
            observation_window_seconds=response_seconds,
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
                raw_audio_path=raw_audio_path,
                barge_in_seconds=result["barge_in_ms"] / 1000,
                first_output_received_seconds=result["first_output_received_seconds"],
            )
        )
    return RolloutGroup(probe_id=probe_id, character=character, rollouts=rollouts)
```

Note `observation_window_seconds=response_seconds` -- this is the bug fix
(was `result["max_wait_seconds"]`); the inline comment that used to sit
above this call explaining the bug is intentionally gone in the
replacement above, since the bug it described is now fixed. Also note the
new local variable `raw_audio_directory`: add it as a new keyword-only
parameter to `collect_rollout_group`'s signature, right after
`judge: JudgeClient,`:

```python
    judge: JudgeClient,
    raw_audio_directory: Path,
    group_size: int,
```

Leave the rest of the function's docstring (the paragraphs about
sequential calls, fail-fast behavior, and the now-resolved diversity
question) as-is -- still accurate background, not part of this fix.

- [ ] **Step 9: Run tests to verify they pass**

Run: `uv run pytest tests/test_training_rollout.py tests/test_training_native_probe.py -v`
Expected: PASS, all tests.

- [ ] **Step 10: Run the full test suite to check for ripple effects**

Run: `uv run pytest -q`
Expected: PASS. If any other test constructs a `Rollout` directly (grep
for `Rollout(` outside the files already touched), update it the same way
Step 6 updated `_rollout`.

- [ ] **Step 11: Commit**

```bash
git add src/tolkien_voice/training/rollout.py tests/test_training_rollout.py
git commit -m "fix: capture rollout audio, fix reward observation window, expose barge timing"
```

---

### Task 2: `select_best_rollout` and `leading_silence_seconds`

**Files:**
- Create: `src/tolkien_voice/training/self_distillation.py`
- Test: `tests/test_training_self_distillation.py`

**Interfaces:**
- Consumes: `Rollout`, `RolloutGroup` from `rollout.py` (Task 1's version, with the 3 new fields); `load_user_channel`, `FRAME_SECONDS` from `native_probe.py` (unchanged).
- Produces: `select_best_rollout(group, *, yield_delay_reward_gate) -> Rollout | None` and `leading_silence_seconds(wav_path, *, activity_threshold=0.01) -> float`, both consumed by Task 3.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_training_self_distillation.py`:

```python
from array import array
from pathlib import Path
import wave

from tolkien_voice.training.rollout import Rollout, RolloutGroup
from tolkien_voice.training.self_distillation import (
    leading_silence_seconds,
    select_best_rollout,
)


def _rollout(*, yield_delay_reward: float, quality_reward: float, index: int = 0) -> Rollout:
    return Rollout(
        transcript=f"reply {index}",
        yield_delay_reward=yield_delay_reward,
        quality_reward=quality_reward,
        delay_seconds=0.5,
        raw_audio_path=Path(f"/tmp/rollout-{index}.opus"),
        barge_in_seconds=1.1,
        first_output_received_seconds=0.2,
    )


def test_select_best_rollout_excludes_slow_yields_and_ranks_by_quality() -> None:
    group = RolloutGroup(
        probe_id="aragorn-lore-rangers",
        character="aragorn",
        rollouts=[
            _rollout(yield_delay_reward=-0.05, quality_reward=0.1, index=0),
            _rollout(yield_delay_reward=-0.9, quality_reward=0.9, index=1),
            _rollout(yield_delay_reward=-2.5, quality_reward=1.0, index=2),
        ],
    )

    best = select_best_rollout(group, yield_delay_reward_gate=-1.0)

    assert best is not None
    assert best.transcript == "reply 1"


def test_select_best_rollout_returns_none_when_nothing_clears_the_gate() -> None:
    group = RolloutGroup(
        probe_id="aragorn-lore-rangers",
        character="aragorn",
        rollouts=[
            _rollout(yield_delay_reward=-2.0, quality_reward=0.9, index=0),
            _rollout(yield_delay_reward=-3.0, quality_reward=0.5, index=1),
        ],
    )

    assert select_best_rollout(group, yield_delay_reward_gate=-1.0) is None


def test_select_best_rollout_gate_is_inclusive() -> None:
    group = RolloutGroup(
        probe_id="aragorn-lore-rangers",
        character="aragorn",
        rollouts=[_rollout(yield_delay_reward=-1.0, quality_reward=0.4, index=0)],
    )

    best = select_best_rollout(group, yield_delay_reward_gate=-1.0)

    assert best is not None
    assert best.transcript == "reply 0"


def _write_mono_wav(path: Path, *, silence_seconds: float, speech_seconds: float) -> None:
    sample_rate = 24_000
    samples = array("h")
    samples.extend([0] * round(silence_seconds * sample_rate))
    samples.extend([5000] * round(speech_seconds * sample_rate))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())


def test_leading_silence_seconds_finds_first_audible_frame(tmp_path: Path) -> None:
    wav_path = tmp_path / "interruption.wav"
    _write_mono_wav(wav_path, silence_seconds=0.3, speech_seconds=0.5)

    result = leading_silence_seconds(wav_path)

    assert 0.28 <= result <= 0.32


def test_leading_silence_seconds_is_zero_when_speech_starts_immediately(tmp_path: Path) -> None:
    wav_path = tmp_path / "interruption.wav"
    _write_mono_wav(wav_path, silence_seconds=0.0, speech_seconds=0.5)

    assert leading_silence_seconds(wav_path) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training_self_distillation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tolkien_voice.training.self_distillation'`

- [ ] **Step 3: Implement**

Create `src/tolkien_voice/training/self_distillation.py`:

```python
from __future__ import annotations

from pathlib import Path

from .native_probe import FRAME_SECONDS, load_user_channel
from .rollout import Rollout, RolloutGroup


def select_best_rollout(
    group: RolloutGroup,
    *,
    yield_delay_reward_gate: float,
) -> Rollout | None:
    """Among rollouts that yielded within yield_delay_reward_gate seconds
    (yield_delay_reward is negative seconds, so clearing the gate means
    yield_delay_reward >= yield_delay_reward_gate), returns the one with
    the highest quality_reward. Returns None if no rollout clears the
    gate -- mirrors the project's release gate structure: interruption
    timing is a hard pass/fail threshold, persona quality is what's
    compared once timing passes.
    """
    survivors = [
        rollout
        for rollout in group.rollouts
        if rollout.yield_delay_reward >= yield_delay_reward_gate
    ]
    if not survivors:
        return None
    return max(survivors, key=lambda rollout: rollout.quality_reward)


def leading_silence_seconds(wav_path: Path, *, activity_threshold: float = 0.01) -> float:
    """Seconds of near-silence before the first audible frame in wav_path.

    Uses the same absolute-amplitude threshold probe_native_interruption
    applies inline to find barge_in_seconds, so a corpus builder can place
    this same clip on a reconstructed timeline with its audible onset
    landing exactly where the live probe measured it.
    """
    sample_rate, samples, _ = load_user_channel(wav_path)
    frame_samples = round(sample_rate * FRAME_SECONDS)
    for offset in range(0, len(samples), frame_samples):
        frame = samples[offset : offset + frame_samples]
        if max((abs(sample) for sample in frame), default=0.0) >= activity_threshold:
            return offset / sample_rate
    return len(samples) / sample_rate
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_training_self_distillation.py -v`
Expected: PASS, all 5 tests.

- [ ] **Step 5: Commit**

```bash
git add src/tolkien_voice/training/self_distillation.py tests/test_training_self_distillation.py
git commit -m "feat: add select_best_rollout and leading_silence_seconds"
```

---

### Task 3: `build_self_distillation_example`

**Files:**
- Modify: `src/tolkien_voice/training/self_distillation.py`
- Test: `tests/test_training_self_distillation.py`

**Interfaces:**
- Consumes: `Rollout` (Task 1), `select_best_rollout`/`leading_silence_seconds` (Task 2), `RenderedExample`/`Split` from `schema.py` (unchanged), `validate_rendered_examples` from `manifest.py` (unchanged), `sphn.OpusStreamReader` (already a dependency).
- Produces: `build_self_distillation_example(...) -> RenderedExample`, consumed by Task 4.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_training_self_distillation.py`. This needs a real
Opus-encoded fixture for the rollout's raw audio -- encode it with
`sphn.OpusStreamWriter` directly (the project's real encoder, already a
dependency, avoiding a hand-rolled fake Opus payload):

```python
import json

import sphn

from tolkien_voice.training.manifest import validate_rendered_examples
from tolkien_voice.training.schema import Split
from tolkien_voice.training.self_distillation import build_self_distillation_example


def _write_opus_fixture(path: Path, *, seconds: float, amplitude: float) -> None:
    import numpy as np

    sample_rate = 24_000
    frame_samples = round(sample_rate * 0.04)
    writer = sphn.OpusStreamWriter(sample_rate)
    total_samples = round(seconds * sample_rate)
    audio_bytes = bytearray()
    for offset in range(0, total_samples, frame_samples):
        length = min(frame_samples, total_samples - offset)
        frame = np.full(frame_samples, amplitude, dtype=np.float32)
        frame[length:] = 0.0
        writer.append_pcm(frame)
        payload = writer.read_bytes()
        if payload:
            audio_bytes.extend(payload)
    path.write_bytes(bytes(audio_bytes))


def test_build_self_distillation_example_produces_a_valid_rendered_example(
    tmp_path: Path,
) -> None:
    opening_wav = tmp_path / "opening.wav"
    interruption_wav = tmp_path / "interruption.wav"
    _write_mono_wav(opening_wav, silence_seconds=0.0, speech_seconds=1.0)
    _write_mono_wav(interruption_wav, silence_seconds=0.2, speech_seconds=0.6)

    raw_audio_path = tmp_path / "rollout-0.opus"
    _write_opus_fixture(raw_audio_path, seconds=1.0, amplitude=0.2)

    rollout = Rollout(
        transcript="Go on, then.",
        yield_delay_reward=-0.3,
        quality_reward=0.8,
        delay_seconds=0.3,
        raw_audio_path=raw_audio_path,
        barge_in_seconds=1.0,
        first_output_received_seconds=0.1,
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

    validate_rendered_examples([example])
    assert example.split == Split.TRAIN
    assert example.voice_license == "self-distillation-v6-continuation"
    assert example.wav_path.is_file()
    assert example.alignment_path.is_file()
    payload = json.loads(example.alignment_path.read_text())
    assert payload["system_prompt"] == "<system>You are Aragorn.<system>"
    assert len(payload["alignments"]) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_training_self_distillation.py::test_build_self_distillation_example_produces_a_valid_rendered_example -v`
Expected: FAIL with `ImportError` (`build_self_distillation_example` doesn't exist yet).

- [ ] **Step 3: Implement**

Append to `src/tolkien_voice/training/self_distillation.py`. Add these
imports at the top (merge with the existing import block from Task 2):

```python
from __future__ import annotations

from array import array
import hashlib
import json
import math
import os
import re
import string
import wave

from .manifest import SAMPLE_RATE
from .native_probe import FRAME_SECONDS, load_user_channel
from .rollout import Rollout, RolloutGroup
from .schema import RenderedExample, Split

MAX_SAFE_SAMPLE = int(32_767 * 0.99)
VOICE_LICENSE = "self-distillation-v6-continuation"
```

(`select_best_rollout` and `leading_silence_seconds` from Task 2 stay as
they are; this just adds to the same file's import block and appends new
functions below them.)

```python
def _decode_opus_pcm(opus_path: Path) -> array:
    import numpy as np
    import sphn

    reader = sphn.OpusStreamReader(SAMPLE_RATE)
    reader.append_bytes(opus_path.read_bytes())
    decoded = np.asarray(reader.read_pcm(), dtype=np.float32).reshape(-1)
    clipped = np.clip(decoded, -1.0, 1.0)
    return array("h", (clipped * 32_767).astype(np.int16).tolist())


def _mono_pcm16(wav_path: Path) -> array:
    with wave.open(str(wav_path), "rb") as source:
        if source.getsampwidth() != 2:
            raise ValueError(f"expected PCM16: {wav_path}")
        channels = source.getnchannels()
        interleaved = array("h")
        interleaved.frombytes(source.readframes(source.getnframes()))
    return interleaved[1::2] if channels == 2 else interleaved


def _word_alignments(transcript: str, *, start_sec: float, end_sec: float) -> list[list]:
    punctuation = string.punctuation + "“”‘’"
    words = [word.strip(punctuation) for word in re.findall(r"\S+", transcript)]
    words = [word for word in words if word]
    if not words:
        return []
    duration = end_sec - start_sec
    alignments: list[list] = []
    for index, word in enumerate(words):
        word_start = start_sec + duration * index / len(words)
        word_end = start_sec + duration * (index + 1) / len(words)
        alignments.append(
            [word, [round(word_start, 6), round(word_end, 6)], "SPEAKER_MAIN"]
        )
    return alignments


def build_self_distillation_example(
    rollout: Rollout,
    *,
    scenario_id: str,
    character: str,
    opening_wav_path: Path,
    interruption_wav_path: Path,
    system_prompt: str,
    split: Split,
    output_directory: Path,
) -> RenderedExample:
    """Reconstructs the real recorded session timeline: opening_wav_path
    at t=0 and interruption_wav_path positioned so its audible onset lands
    at rollout.barge_in_seconds on the user (right) channel;
    rollout.raw_audio_path decoded from Opus and positioned starting at
    rollout.first_output_received_seconds on the assistant (left) channel.
    """
    del character  # not needed for construction; kept for caller symmetry with scenario metadata
    output_directory.mkdir(parents=True, exist_ok=True, mode=0o700)

    opening_samples = _mono_pcm16(opening_wav_path)
    interruption_samples = _mono_pcm16(interruption_wav_path)
    assistant_samples = _decode_opus_pcm(rollout.raw_audio_path)

    onset_offset = leading_silence_seconds(interruption_wav_path)
    interruption_start_sec = max(0.0, rollout.barge_in_seconds - onset_offset)
    assistant_start_sec = rollout.first_output_received_seconds or 0.0

    opening_end_sec = len(opening_samples) / SAMPLE_RATE
    interruption_end_sec = interruption_start_sec + len(interruption_samples) / SAMPLE_RATE
    assistant_end_sec = assistant_start_sec + len(assistant_samples) / SAMPLE_RATE
    total_seconds = max(opening_end_sec, interruption_end_sec, assistant_end_sec)
    total_frames = math.ceil(total_seconds * SAMPLE_RATE)

    left = array("i", [0] * total_frames)   # assistant
    right = array("i", [0] * total_frames)  # user

    def _mix(channel: array, samples: array, start_sec: float) -> None:
        start_frame = round(start_sec * SAMPLE_RATE)
        for offset, sample in enumerate(samples):
            target = start_frame + offset
            if target >= len(channel):
                break
            mixed = channel[target] + sample
            if abs(mixed) >= MAX_SAFE_SAMPLE:
                mixed = MAX_SAFE_SAMPLE if mixed > 0 else -MAX_SAFE_SAMPLE
            channel[target] = mixed

    _mix(right, opening_samples, 0.0)
    _mix(right, interruption_samples, interruption_start_sec)
    _mix(left, assistant_samples, assistant_start_sec)

    interleaved = array("h")
    for left_sample, right_sample in zip(left, right, strict=True):
        interleaved.extend((left_sample, right_sample))

    example_id = f"{scenario_id}-self-distill"
    wav_path = output_directory / f"{example_id}.wav"
    temporary_wav = wav_path.with_suffix(".wav.tmp")
    with wave.open(str(temporary_wav), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(interleaved.tobytes())
    os.replace(temporary_wav, wav_path)
    wav_path.chmod(0o600)

    alignments = _word_alignments(
        rollout.transcript, start_sec=assistant_start_sec, end_sec=assistant_end_sec
    )
    alignment_path = output_directory / f"{example_id}.json"
    temporary_alignment = alignment_path.with_suffix(".json.tmp")
    temporary_alignment.write_text(
        json.dumps(
            {
                "alignments": alignments,
                "system_prompt": system_prompt,
            },
            indent=2,
        )
        + "\n"
    )
    temporary_alignment.chmod(0o600)
    os.replace(temporary_alignment, alignment_path)

    blueprint_hash = hashlib.sha256(
        f"{scenario_id}:{rollout.transcript}".encode("utf-8")
    ).hexdigest()

    return RenderedExample(
        id=example_id,
        blueprint_hash=blueprint_hash,
        wav_path=wav_path,
        alignment_path=alignment_path,
        duration_sec=total_frames / SAMPLE_RATE,
        split=split,
        voice_license=VOICE_LICENSE,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_training_self_distillation.py -v`
Expected: PASS, all tests.

If `_word_alignments` produces zero alignments (e.g. `rollout.transcript`
in the test fixture is non-empty so this shouldn't happen, but if a
future caller passes an empty transcript, `validate_rendered_examples`
will fail with "alignment list is empty" -- that's the correct,
already-existing behavior; do not special-case it here, see Task 4's
`select_best_rollout` usage which only keeps rollouts with real
transcripts already scored above 0 by `compute_quality_reward`).

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest -q`
Expected: PASS, no regressions.

- [ ] **Step 6: Commit**

```bash
git add src/tolkien_voice/training/self_distillation.py tests/test_training_self_distillation.py
git commit -m "feat: add build_self_distillation_example"
```

---

### Task 4: Corpus-building orchestration script

**Files:**
- Create: `scripts/build-self-distillation-corpus.py`

**Interfaces:**
- Consumes: `collect_rollout_group` (`rollout.py`, unchanged interface plus Task 1's new `raw_audio_directory` parameter), `select_best_rollout`/`build_self_distillation_example` (Tasks 2-3), `write_moshi_manifests` (`manifest.py`, unchanged), `load_persona_manifest` (`tolkien_voice.evaluation`, unchanged), `CharacterRegistry` (`tolkien_voice.characters`, unchanged), `JudgeClient` (`tolkien_voice.training.judge`, unchanged).
- Produces: a manifest directory consumable by the existing
  `train-mvp-v6-persona.sh` unchanged.

- [ ] **Step 1: Write the script**

Create `scripts/build-self-distillation-corpus.py`, modeled directly on
`scripts/collect-native-arm.py`'s argument/URL-building pattern and
`scripts/validate-interruption-reward.py`'s judge-construction pattern:

```python
#!/usr/bin/env python3
"""Sample N completions per interruption scenario against a live
PersonaPlex instance, keep the highest-quality rollout that yielded fast
enough, and build a moshi-finetune-ready training corpus from the model's
own captured audio.

Does not start or stop PersonaPlex -- run scripts/_personaplex_ctl.py
start/stop around this, exactly as scripts/run-v6-native-eval.sh does for
evaluation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from urllib.parse import urlencode, urlparse, urlunparse

from tolkien_voice.characters import CharacterRegistry
from tolkien_voice.evaluation import load_persona_manifest
from tolkien_voice.training.judge import JudgeClient
from tolkien_voice.training.manifest import write_moshi_manifests
from tolkien_voice.training.rollout import collect_rollout_group
from tolkien_voice.training.schema import RenderedExample, Split
from tolkien_voice.training.self_distillation import (
    build_self_distillation_example,
    select_best_rollout,
)


def websocket_url(server_url: str, *, text_prompt: str, voice_prompt: str) -> str:
    parsed = urlparse(server_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    query = urlencode({"text_prompt": text_prompt, "voice_prompt": voice_prompt})
    return urlunparse((scheme, parsed.netloc, "/api/chat", "", query, ""))


async def build(args: argparse.Namespace) -> list[RenderedExample]:
    manifest = load_persona_manifest(args.manifest)
    registry = CharacterRegistry.from_directory(args.character_dir)
    examples: list[RenderedExample] = []
    skipped: list[str] = []
    with JudgeClient(base_url=args.judge_base_url, model=args.judge_model) as judge:
        for character in args.characters:
            resolved = registry.resolve(character)
            url = websocket_url(
                args.server_url,
                text_prompt=resolved.text_prompt,
                voice_prompt=resolved.voice_prompt,
            )
            for probe in manifest[character]:
                opening_wav = args.audio_dir / f"{probe.id}.wav"
                group = await collect_rollout_group(
                    url,
                    opening_wav,
                    args.interruption_audio,
                    probe_id=probe.id,
                    character=character,
                    probe_prompt=probe.prompt,
                    expected_traits=probe.expected_traits,
                    judge=judge,
                    raw_audio_directory=args.raw_audio_dir,
                    group_size=args.group_size,
                    seed=args.seed,
                )
                best = select_best_rollout(
                    group, yield_delay_reward_gate=args.yield_delay_reward_gate
                )
                if best is None:
                    skipped.append(probe.id)
                    continue
                example = build_self_distillation_example(
                    best,
                    scenario_id=probe.id,
                    character=character,
                    opening_wav_path=opening_wav,
                    interruption_wav_path=args.interruption_audio,
                    system_prompt=resolved.text_prompt,
                    split=Split.TRAIN,
                    output_directory=args.output_directory / "audio",
                )
                examples.append(example)
    if skipped:
        print(f"skipped (no rollout cleared the gate): {skipped}")
    return examples


def main() -> int:
    repository = Path(__file__).parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:8998")
    parser.add_argument("--manifest", type=Path, default=repository / "evals/persona/probes.yaml")
    parser.add_argument("--character-dir", type=Path, default=repository / "configs/characters")
    parser.add_argument("--characters", nargs="+", default=["aragorn", "sauron"])
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--interruption-audio", type=Path, required=True)
    parser.add_argument("--raw-audio-dir", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--judge-base-url", default="http://127.0.0.1:9090/v1")
    parser.add_argument("--judge-model", default="tolkien")
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--yield-delay-reward-gate", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    examples = asyncio.run(build(args))
    if not examples:
        raise SystemExit("no scenario produced a corpus example -- nothing to write")
    manifests = write_moshi_manifests(args.output_directory / "manifests", examples)
    print(
        json.dumps(
            {
                "examples": len(examples),
                "manifests": {split.value: str(path) for split, path in manifests.items()},
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Syntax/import check**

Run: `uv run python -c "import ast; ast.parse(open('scripts/build-self-distillation-corpus.py').read())"`
Expected: no output (valid syntax). Then:
Run: `uv run python -m py_compile scripts/build-self-distillation-corpus.py`
Expected: no output (imports resolve, no syntax errors).

- [ ] **Step 3: Run the full test suite one more time**

Run: `uv run pytest -q`
Expected: PASS, no regressions from any task in this plan.

- [ ] **Step 4: Commit**

```bash
git add scripts/build-self-distillation-corpus.py
git commit -m "feat: add self-distillation corpus-building orchestration script"
```

---

## After all tasks: live verification (not automated, requires Spark)

Once merged, before trusting this for a real retraining run:

1. `scripts/_personaplex_ctl.py start <config>`
2. Run `build-self-distillation-corpus.py` against 2-3 scenarios per
   character (small `--characters`/manually trimmed `--manifest` subset)
   with `--group-size 3` to keep it fast.
3. `scripts/_personaplex_ctl.py stop <config>`
4. Manually listen to a resulting WAV in `<output>/audio/*.wav` --
   confirm both channels are audible, in the correct left/right
   positions, and correctly timed relative to each other (the automated
   tests validate file format, not perceptual correctness).
5. Only after that listening check passes: run the full corpus build
   across all scenarios, then `train-mvp-v6-persona.sh` (unchanged)
   against the resulting manifest, continuing from the v6 adapter, then
   `run-v6-native-eval.sh` (unchanged) to see whether the retrained
   adapter clears the gates v6 failed.
