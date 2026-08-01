# Proxy-Enforced Barge-In Muting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce fast interruption response at the systems level -- the proxy in `api.py` detects sustained user speech and withholds the assistant's outgoing audio until the user goes quiet again, decoupled from whatever PersonaPlex itself would do on its own.

**Architecture:** A pure `BargeInDetector` state machine (activation on sustained loud frames, deactivation on sustained quiet frames -- asymmetric thresholds, matching this project's existing conventions for "speech starting" vs "speech stopping" detection) drives a `BargeInGate` that owns per-connection Opus-decode state. `api.py`'s existing bidirectional relay (`_browser_to_upstream` / `_upstream_to_browser`) feeds the gate from the user's audio and gates the assistant's audio through it, sending one new protocol control message to the browser the instant a barge-in is detected so it can flush its own small playback buffer immediately via the existing worklet reset mechanism.

**Tech Stack:** Python (FastAPI, `sphn` Opus codec), TypeScript (Vite/React web client, `vitest`).

## Global Constraints

- `sphn==0.1.12` moves from the `evaluation` optional-dependency group to core `dependencies` in `pyproject.toml`. `numpy` stays evaluation-only -- RMS over a 40ms PCM16 frame doesn't need it.
- Activation (start of user speech): `threshold=0.02, sustained_frames=3` (120ms at the project's existing `FRAME_SECONDS=0.04` frame size) -- matches this project's existing convention for detecting speech *starting* (`native_probe.py`'s `detect_speech_start`-style usage).
- Deactivation (user has gone quiet again): `threshold=0.015, sustained_frames=12` (480ms) -- matches this project's existing convention for detecting speech *stopping* (`native_probe.py`'s `measure_interruption_response` / `rollout.py` usage of the same pair).
- New protocol control action `bargeIn`, byte value `4` (`0b00000100`) -- the next unused value after the existing `start=0, endTurn=1, pause=2, restart=3`. Do not reuse `pause`.
- Text frames (`0x02`) are never gated by any of this -- only audio frames (`0x01`).
- No change to `session.py`'s session lifecycle/lease logic, `runtime.py`, or PersonaPlex itself.

---

### Task 1: `BargeInDetector` and `frame_rms`

**Files:**
- Create: `src/tolkien_voice/barge_in.py`
- Test: `tests/test_barge_in.py`

**Interfaces:**
- Consumes: nothing (pure, stdlib only).
- Produces: `frame_rms(samples: Sequence[float]) -> float`; `BargeInDetector(*, activation_threshold: float, activation_sustained_frames: int, deactivation_threshold: float, deactivation_sustained_frames: int)` with `.append(rms: float) -> bool` (edge-triggered: `True` exactly on the sample that flips `.active`) and `.active: bool` property. Both consumed by Task 2.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_barge_in.py`:

```python
from __future__ import annotations

import pytest

from tolkien_voice.barge_in import BargeInDetector, frame_rms


def test_frame_rms_of_silence_is_zero() -> None:
    assert frame_rms([0.0, 0.0, 0.0]) == 0.0


def test_frame_rms_of_constant_signal_equals_its_magnitude() -> None:
    assert frame_rms([0.1, -0.1, 0.1, -0.1]) == pytest.approx(0.1)


def test_frame_rms_of_empty_frame_is_zero() -> None:
    assert frame_rms([]) == 0.0


def _detector() -> BargeInDetector:
    return BargeInDetector(
        activation_threshold=0.02,
        activation_sustained_frames=3,
        deactivation_threshold=0.015,
        deactivation_sustained_frames=12,
    )


def test_detector_starts_idle() -> None:
    assert _detector().active is False


def test_detector_activates_after_sustained_loud_frames() -> None:
    detector = _detector()

    assert detector.append(0.03) is False
    assert detector.append(0.03) is False
    assert detector.append(0.03) is True
    assert detector.active is True


def test_detector_ignores_a_brief_dip_below_activation_threshold() -> None:
    detector = _detector()

    assert detector.append(0.03) is False
    assert detector.append(0.01) is False
    assert detector.append(0.03) is False
    assert detector.append(0.03) is False
    assert detector.append(0.03) is True


def test_detector_requires_sustained_quiet_to_deactivate() -> None:
    detector = _detector()
    for _ in range(3):
        detector.append(0.03)
    assert detector.active is True

    for _ in range(11):
        assert detector.append(0.01) is False
    assert detector.active is True
    assert detector.append(0.01) is True
    assert detector.active is False


def test_detector_ignores_a_brief_loud_blip_while_deactivating() -> None:
    detector = _detector()
    for _ in range(3):
        detector.append(0.03)
    assert detector.active is True

    for _ in range(5):
        detector.append(0.01)
    detector.append(0.03)  # blip resets the deactivation run
    for _ in range(11):
        assert detector.append(0.01) is False
    assert detector.append(0.01) is True
    assert detector.active is False


def test_detector_rejects_non_positive_construction_values() -> None:
    with pytest.raises(ValueError):
        BargeInDetector(
            activation_threshold=-0.01,
            activation_sustained_frames=3,
            deactivation_threshold=0.015,
            deactivation_sustained_frames=12,
        )
    with pytest.raises(ValueError):
        BargeInDetector(
            activation_threshold=0.02,
            activation_sustained_frames=0,
            deactivation_threshold=0.015,
            deactivation_sustained_frames=12,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_barge_in.py -v`
Expected: FAIL with `ModuleNotFoundError` -- `tolkien_voice.barge_in` doesn't exist yet.

- [ ] **Step 3: Implement**

Create `src/tolkien_voice/barge_in.py`:

```python
from __future__ import annotations

import math
from typing import Sequence

FRAME_SECONDS = 0.04


def frame_rms(samples: Sequence[float]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


class BargeInDetector:
    """Edge-triggered activity state machine with asymmetric thresholds:
    fast to activate (a few sustained loud frames means the user has
    started talking), slow to deactivate (many sustained quiet frames
    means the user has genuinely gone quiet, not just paused between
    words) -- deliberately different sensitivity in each direction, not
    a single reusable threshold applied twice.
    """

    def __init__(
        self,
        *,
        activation_threshold: float,
        activation_sustained_frames: int,
        deactivation_threshold: float,
        deactivation_sustained_frames: int,
    ) -> None:
        if activation_threshold < 0 or deactivation_threshold < 0:
            raise ValueError("thresholds must be non-negative")
        if activation_sustained_frames <= 0 or deactivation_sustained_frames <= 0:
            raise ValueError("sustained-frame counts must be positive")
        self.activation_threshold = activation_threshold
        self.activation_sustained_frames = activation_sustained_frames
        self.deactivation_threshold = deactivation_threshold
        self.deactivation_sustained_frames = deactivation_sustained_frames
        self.active = False
        self._run_length = 0

    def append(self, rms: float) -> bool:
        was_active = self.active
        if self.active:
            self._run_length = (
                self._run_length + 1 if rms <= self.deactivation_threshold else 0
            )
            if self._run_length >= self.deactivation_sustained_frames:
                self.active = False
                self._run_length = 0
        else:
            self._run_length = (
                self._run_length + 1 if rms >= self.activation_threshold else 0
            )
            if self._run_length >= self.activation_sustained_frames:
                self.active = True
                self._run_length = 0
        return self.active != was_active
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_barge_in.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Run the full suite and lint**

Run: `uv run pytest -q`
Expected: PASS, no regressions.

Run: `uv run ruff check --exclude vendor .`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/tolkien_voice/barge_in.py tests/test_barge_in.py
git commit -m "feat: add BargeInDetector activity state machine"
```

---

### Task 2: `BargeInGate` and `api.py` relay wiring

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/tolkien_voice/barge_in.py`
- Modify: `src/tolkien_voice/api.py`
- Test: `tests/test_barge_in.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `BargeInDetector`, `frame_rms` (Task 1, unchanged).
- Produces: `BargeInGate(*, reader: OpusPcmReader, detector: BargeInDetector, sample_rate: int)` with `.observe_user_audio(opus_payload: bytes) -> bool` (edge-triggered on activation) and `.muting: bool` property; `BARGE_IN_CONTROL_BYTE = 4`. `api.py`'s `_browser_to_upstream`/`_upstream_to_browser` gain a required third `gate: BargeInGate` parameter.

- [ ] **Step 1: Move `sphn` to core dependencies**

In `pyproject.toml`, move the `sphn==0.1.12` line out of `[project.optional-dependencies].evaluation` and into `[project].dependencies`, keeping the list alphabetically ordered:

```toml
dependencies = [
  "fastapi>=0.116,<1",
  "httpx>=0.28,<1",
  "pydantic>=2.11,<3",
  "pydantic-settings>=2.10,<3",
  "pyyaml>=6.0,<7",
  "ruamel-yaml>=0.18,<1",
  "sphn==0.1.12",
  "uvicorn[standard]>=0.35,<1",
  "websockets>=15,<16",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.4,<9",
  "pytest-asyncio>=1.1,<2",
  "pytest-cov>=6.2,<7",
  "ruff>=0.12,<1",
]
evaluation = [
  "aiohttp>=3.10.5,<3.11",
  "numpy>=1.26,<2.2",
]
```

Run: `uv sync`
Expected: resolves and installs `sphn` unconditionally (no `--extra evaluation` needed).

- [ ] **Step 2: Write the failing `BargeInGate` tests**

Append to `tests/test_barge_in.py`:

```python
from tolkien_voice.barge_in import BARGE_IN_CONTROL_BYTE, BargeInGate


class _FakeReader:
    def __init__(self) -> None:
        self.appended: list[bytes] = []
        self.next_pcm: list[float] = []

    def append_bytes(self, payload: bytes) -> None:
        self.appended.append(payload)

    def read_pcm(self) -> list[float]:
        return self.next_pcm


def _gate(reader: "_FakeReader") -> BargeInGate:
    return BargeInGate(
        reader=reader,
        detector=BargeInDetector(
            activation_threshold=0.02,
            activation_sustained_frames=3,
            deactivation_threshold=0.015,
            deactivation_sustained_frames=12,
        ),
        sample_rate=24_000,
    )


def test_gate_reports_activation_on_the_frame_that_crosses_the_threshold() -> None:
    reader = _FakeReader()
    gate = _gate(reader)
    frame_samples = round(24_000 * 0.04)
    reader.next_pcm = [0.03] * frame_samples

    assert gate.observe_user_audio(b"opus-1") is False
    assert gate.observe_user_audio(b"opus-2") is False
    assert gate.observe_user_audio(b"opus-3") is True
    assert gate.muting is True
    assert reader.appended == [b"opus-1", b"opus-2", b"opus-3"]


def test_gate_processes_multiple_frames_decoded_from_one_payload() -> None:
    reader = _FakeReader()
    gate = _gate(reader)
    frame_samples = round(24_000 * 0.04)
    reader.next_pcm = [0.03] * (frame_samples * 3)

    assert gate.observe_user_audio(b"opus") is True
    assert gate.muting is True


def test_gate_stays_quiet_below_the_activation_threshold() -> None:
    reader = _FakeReader()
    gate = _gate(reader)
    frame_samples = round(24_000 * 0.04)
    reader.next_pcm = [0.01] * frame_samples

    assert gate.observe_user_audio(b"opus-1") is False
    assert gate.observe_user_audio(b"opus-2") is False
    assert gate.muting is False


def test_barge_in_control_byte_matches_the_web_protocol_value() -> None:
    assert BARGE_IN_CONTROL_BYTE == 4
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_barge_in.py -v`
Expected: the four new tests FAIL with `ImportError` (`BargeInGate`, `BARGE_IN_CONTROL_BYTE` don't exist yet); the Task 1 tests still PASS.

- [ ] **Step 4: Implement `BargeInGate`**

Append to `src/tolkien_voice/barge_in.py` (add `Protocol` to the existing `typing` import):

```python
from typing import Protocol, Sequence
```

Then append:

```python
BARGE_IN_CONTROL_BYTE = 4


class OpusPcmReader(Protocol):
    def append_bytes(self, payload: bytes) -> None: ...

    def read_pcm(self) -> Sequence[float]: ...


class BargeInGate:
    """Owns one connection's Opus-decode state and BargeInDetector,
    translating incoming Opus payloads from the user's channel into
    activation/deactivation decisions for that connection.
    """

    def __init__(
        self,
        *,
        reader: OpusPcmReader,
        detector: BargeInDetector,
        sample_rate: int,
    ) -> None:
        self._reader = reader
        self._detector = detector
        self._frame_samples = round(sample_rate * FRAME_SECONDS)

    @property
    def muting(self) -> bool:
        return self._detector.active

    def observe_user_audio(self, opus_payload: bytes) -> bool:
        """Decodes one Opus payload from the user's channel and feeds its
        RMS-per-frame content into the detector. Returns True exactly when
        this call causes the detector to activate.
        """
        self._reader.append_bytes(opus_payload)
        decoded = list(self._reader.read_pcm())
        activated = False
        for offset in range(0, len(decoded), self._frame_samples):
            chunk = decoded[offset : offset + self._frame_samples]
            if not chunk:
                continue
            if self._detector.append(frame_rms(chunk)) and self._detector.active:
                activated = True
        return activated
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_barge_in.py -v`
Expected: PASS, all 14 tests (10 from Task 1, 4 new).

- [ ] **Step 6: Write the failing `api.py` relay tests**

Append to `tests/test_api.py` (add these imports to the existing import block):

```python
import asyncio

from tolkien_voice.api import _browser_to_upstream, _upstream_to_browser
from tolkien_voice.barge_in import BARGE_IN_CONTROL_BYTE, BargeInDetector, BargeInGate
```

Then append:

```python
class _RecordingReader:
    def __init__(self) -> None:
        self.next_pcm: list[float] = []

    def append_bytes(self, _payload: bytes) -> None:
        return None

    def read_pcm(self) -> list[float]:
        return self.next_pcm


class _FakeWebSocket:
    def __init__(self, messages: list[dict[str, object]] | None = None) -> None:
        self._messages = list(messages or [])
        self.sent: list[bytes] = []
        self.closed: tuple[int, str] | None = None

    async def receive(self) -> dict[str, object]:
        if not self._messages:
            return {"type": "websocket.disconnect"}
        return self._messages.pop(0)

    async def send_bytes(self, payload: bytes) -> None:
        self.sent.append(payload)

    async def close(self, code: int, reason: str) -> None:
        self.closed = (code, reason)


class _FakeUpstream:
    def __init__(self) -> None:
        self.sent: list[bytes] = []

    async def send(self, payload: bytes) -> None:
        self.sent.append(payload)


def _one_frame_gate() -> tuple[BargeInGate, _RecordingReader]:
    reader = _RecordingReader()
    gate = BargeInGate(
        reader=reader,
        detector=BargeInDetector(
            activation_threshold=0.02,
            activation_sustained_frames=1,
            deactivation_threshold=0.015,
            deactivation_sustained_frames=1,
        ),
        sample_rate=24_000,
    )
    reader.next_pcm = [0.03] * round(24_000 * 0.04)
    return gate, reader


@pytest.mark.asyncio
async def test_browser_to_upstream_sends_control_message_on_barge_in() -> None:
    gate, _ = _one_frame_gate()
    websocket = _FakeWebSocket(
        [{"type": "websocket.receive", "bytes": b"\x01opus-frame"}]
    )
    upstream = _FakeUpstream()

    await _browser_to_upstream(websocket, upstream, gate)

    assert upstream.sent == [b"\x01opus-frame"]
    assert websocket.sent == [bytes([0x03, BARGE_IN_CONTROL_BYTE])]


@pytest.mark.asyncio
async def test_browser_to_upstream_relays_normally_without_barge_in() -> None:
    quiet_reader = _RecordingReader()
    quiet_reader.next_pcm = [0.0] * round(24_000 * 0.04)
    gate = BargeInGate(
        reader=quiet_reader,
        detector=BargeInDetector(
            activation_threshold=0.02,
            activation_sustained_frames=3,
            deactivation_threshold=0.015,
            deactivation_sustained_frames=12,
        ),
        sample_rate=24_000,
    )
    websocket = _FakeWebSocket(
        [{"type": "websocket.receive", "bytes": b"\x01opus-frame"}]
    )
    upstream = _FakeUpstream()

    await _browser_to_upstream(websocket, upstream, gate)

    assert upstream.sent == [b"\x01opus-frame"]
    assert websocket.sent == []


@pytest.mark.asyncio
async def test_upstream_to_browser_withholds_audio_but_forwards_text_while_muting() -> None:
    gate, _ = _one_frame_gate()
    gate.observe_user_audio(b"opus")
    assert gate.muting is True

    async def upstream_frames():
        yield b"\x01assistant-audio"
        yield b"\x02assistant-text"

    websocket = _FakeWebSocket()

    await _upstream_to_browser(websocket, upstream_frames(), gate)

    assert websocket.sent == [b"\x02assistant-text"]


@pytest.mark.asyncio
async def test_upstream_to_browser_forwards_audio_when_not_muting() -> None:
    quiet_reader = _RecordingReader()
    gate = BargeInGate(
        reader=quiet_reader,
        detector=BargeInDetector(
            activation_threshold=0.02,
            activation_sustained_frames=3,
            deactivation_threshold=0.015,
            deactivation_sustained_frames=12,
        ),
        sample_rate=24_000,
    )
    assert gate.muting is False

    async def upstream_frames():
        yield b"\x01assistant-audio"

    websocket = _FakeWebSocket()

    await _upstream_to_browser(websocket, upstream_frames(), gate)

    assert websocket.sent == [b"\x01assistant-audio"]


@pytest.mark.asyncio
async def test_browser_to_upstream_tolerates_undecodable_audio_and_keeps_relaying() -> None:
    class _RaisingReader:
        def append_bytes(self, _payload: bytes) -> None:
            raise ValueError("corrupt opus payload")

        def read_pcm(self) -> list[float]:
            return []

    gate = BargeInGate(
        reader=_RaisingReader(),
        detector=BargeInDetector(
            activation_threshold=0.02,
            activation_sustained_frames=3,
            deactivation_threshold=0.015,
            deactivation_sustained_frames=12,
        ),
        sample_rate=24_000,
    )
    websocket = _FakeWebSocket(
        [{"type": "websocket.receive", "bytes": b"\x01broken-opus"}]
    )
    upstream = _FakeUpstream()

    await _browser_to_upstream(websocket, upstream, gate)

    assert upstream.sent == [b"\x01broken-opus"]
    assert websocket.sent == []
```

Note: `tests/test_api.py` doesn't currently import `pytest` -- add `import pytest` to its import block. `pyproject.toml` does not set `asyncio_mode = "auto"` (confirmed by `tests/test_training_native_probe.py`'s existing explicit `@pytest.mark.asyncio` usage on its one async test), which is why each of the five new `async def test_...` functions above is decorated with `@pytest.mark.asyncio`.

- [ ] **Step 7: Run tests to verify they fail**

Run: `uv run pytest tests/test_api.py -v`
Expected: the five new tests FAIL (`_browser_to_upstream`/`_upstream_to_browser` don't accept a `gate` parameter yet); existing `test_api.py` tests still PASS.

- [ ] **Step 8: Wire `BargeInGate` into `api.py`**

In `src/tolkien_voice/api.py`, add to the imports:

```python
import sphn

from tolkien_voice.barge_in import BARGE_IN_CONTROL_BYTE, BargeInDetector, BargeInGate
```

Add a module constant near `MAX_FRAME_BYTES`:

```python
SAMPLE_RATE = 24_000
```

In the `chat()` handler, replace:

```python
                browser_to_upstream = asyncio.create_task(
                    _browser_to_upstream(websocket, upstream)
                )
                upstream_to_browser = asyncio.create_task(
                    _upstream_to_browser(websocket, upstream)
                )
```

with:

```python
                gate = BargeInGate(
                    reader=sphn.OpusStreamReader(SAMPLE_RATE),
                    detector=BargeInDetector(
                        activation_threshold=0.02,
                        activation_sustained_frames=3,
                        deactivation_threshold=0.015,
                        deactivation_sustained_frames=12,
                    ),
                    sample_rate=SAMPLE_RATE,
                )
                browser_to_upstream = asyncio.create_task(
                    _browser_to_upstream(websocket, upstream, gate)
                )
                upstream_to_browser = asyncio.create_task(
                    _upstream_to_browser(websocket, upstream, gate)
                )
```

Replace the two relay functions:

```python
async def _browser_to_upstream(websocket: WebSocket, upstream: object) -> None:
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
        payload = message.get("bytes")
        if payload is None:
            await websocket.close(code=1003, reason="binary frames required")
            return
        if len(payload) > MAX_FRAME_BYTES:
            await websocket.close(code=1009, reason="frame too large")
            return
        await upstream.send(payload)  # type: ignore[attr-defined]


async def _upstream_to_browser(websocket: WebSocket, upstream: object) -> None:
    async for payload in upstream:  # type: ignore[attr-defined]
        if not isinstance(payload, bytes):
            await websocket.close(code=1011, reason="invalid upstream frame")
            return
        await websocket.send_bytes(payload)
```

with:

```python
async def _browser_to_upstream(
    websocket: WebSocket, upstream: object, gate: BargeInGate
) -> None:
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
        payload = message.get("bytes")
        if payload is None:
            await websocket.close(code=1003, reason="binary frames required")
            return
        if len(payload) > MAX_FRAME_BYTES:
            await websocket.close(code=1009, reason="frame too large")
            return
        if payload and payload[0] == 1:
            try:
                activated = gate.observe_user_audio(payload[1:])
            except Exception:
                # A single undecodable Opus payload must not break the
                # conversation -- VAD is a passive behavior layered on
                # top of the relay, not a new hard requirement for the
                # relay itself to keep functioning.
                activated = False
            if activated:
                await websocket.send_bytes(bytes([0x03, BARGE_IN_CONTROL_BYTE]))
        await upstream.send(payload)  # type: ignore[attr-defined]


async def _upstream_to_browser(
    websocket: WebSocket, upstream: object, gate: BargeInGate
) -> None:
    async for payload in upstream:  # type: ignore[attr-defined]
        if not isinstance(payload, bytes):
            await websocket.close(code=1011, reason="invalid upstream frame")
            return
        if payload and payload[0] == 1 and gate.muting:
            continue
        await websocket.send_bytes(payload)
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `uv run pytest tests/test_api.py tests/test_barge_in.py -v`
Expected: PASS, all tests.

- [ ] **Step 10: Run the full suite and lint**

Run: `uv run pytest -q`
Expected: PASS, no regressions.

Run: `uv run ruff check --exclude vendor .`
Expected: `All checks passed!`

- [ ] **Step 11: Commit**

```bash
git add pyproject.toml uv.lock src/tolkien_voice/barge_in.py src/tolkien_voice/api.py tests/test_barge_in.py tests/test_api.py
git commit -m "feat: enforce barge-in muting in the relay proxy"
```

---

### Task 3: Client protocol action and worklet flush

**Files:**
- Modify: `web/src/protocol/types.ts`
- Modify: `web/src/protocol/encoder.ts`
- Create: `web/src/protocol/controlActions.ts`
- Modify: `web/src/pages/Conversation/hooks/useServerAudio.ts`
- Test: `web/src/protocol/encoder.test.ts` (new)
- Test: `web/src/protocol/controlActions.test.ts` (new)

**Interfaces:**
- Consumes: nothing from Tasks 1-2 (server and client only need to agree on the raw byte value `4`, verified independently on each side -- `tests/test_barge_in.py::test_barge_in_control_byte_matches_the_web_protocol_value` on the Python side, `bargeIn: 0b00000100` in `types.ts` on this side).
- Produces: `shouldFlushPlayback(message: WSMessage) -> boolean`, consumed by `useServerAudio.ts`'s `onSocketMessage`.

- [ ] **Step 1: Write the failing encoder tests**

Create `web/src/protocol/encoder.test.ts`:

```typescript
import { describe, expect, it, vi } from "vitest";
import { decodeMessage, encodeMessage } from "./encoder";

describe("control message round-trip", () => {
  it("encodes and decodes the bargeIn action", () => {
    const encoded = encodeMessage({ type: "control", action: "bargeIn" });
    expect(encoded).toEqual(new Uint8Array([0x03, 0b00000100]));

    const decoded = decodeMessage(encoded);
    expect(decoded).toEqual({ type: "control", action: "bargeIn" });
  });

  it("logs and returns a ping instead of throwing on an unrecognized control action", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);

    const decoded = decodeMessage(new Uint8Array([0x03, 0b11111111]));

    expect(decoded).toEqual({ type: "ping" });
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd web && npx vitest run src/protocol/encoder.test.ts`
Expected: FAIL on both -- the round-trip test fails because `"bargeIn"` isn't a valid `CONTROL_MESSAGE` yet (`CONTROL_MESSAGES_MAP` doesn't have it, so this won't even type-check), and the unrecognized-action test fails because `decodeMessage` currently `throw`s instead of logging and returning `{type: "ping"}`.

- [ ] **Step 3: Add the `bargeIn` control action**

In `web/src/protocol/types.ts`, change:

```typescript
export const CONTROL_MESSAGES_MAP = {
  start: 0b00000000,
  endTurn: 0b00000001,
  pause: 0b00000010,
  restart: 0b00000011,
} as const;
```

to:

```typescript
export const CONTROL_MESSAGES_MAP = {
  start: 0b00000000,
  endTurn: 0b00000001,
  pause: 0b00000010,
  restart: 0b00000011,
  bargeIn: 0b00000100,
} as const;
```

- [ ] **Step 4: Fix `decodeMessage`'s unknown-action handling**

In `web/src/protocol/encoder.ts`, change:

```typescript
    case 0x03: {
      const action = Object.keys(CONTROL_MESSAGES_MAP).find(
        key => CONTROL_MESSAGES_MAP[key as CONTROL_MESSAGE] === payload[0],
      ) as CONTROL_MESSAGE | undefined;

      //TODO: log this and don't throw
      if (!action) {
        throw new Error("Unknown control message");
      }
      return {
        type: "control",
        action,
      };
    }
```

to:

```typescript
    case 0x03: {
      const action = Object.keys(CONTROL_MESSAGES_MAP).find(
        key => CONTROL_MESSAGES_MAP[key as CONTROL_MESSAGE] === payload[0],
      ) as CONTROL_MESSAGE | undefined;

      if (!action) {
        console.warn("Unknown control message action byte:", payload[0]);
        return { type: "ping" };
      }
      return {
        type: "control",
        action,
      };
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd web && npx vitest run src/protocol/encoder.test.ts`
Expected: PASS, both tests.

- [ ] **Step 6: Write the failing `shouldFlushPlayback` test**

Create `web/src/protocol/controlActions.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import { shouldFlushPlayback } from "./controlActions";

describe("shouldFlushPlayback", () => {
  it("is true for a bargeIn control message", () => {
    expect(shouldFlushPlayback({ type: "control", action: "bargeIn" })).toBe(true);
  });

  it("is false for other control actions", () => {
    expect(shouldFlushPlayback({ type: "control", action: "pause" })).toBe(false);
  });

  it("is false for non-control message types", () => {
    expect(shouldFlushPlayback({ type: "audio", data: new Uint8Array() })).toBe(false);
    expect(shouldFlushPlayback({ type: "ping" })).toBe(false);
  });
});
```

- [ ] **Step 7: Run test to verify it fails**

Run: `cd web && npx vitest run src/protocol/controlActions.test.ts`
Expected: FAIL -- `./controlActions` doesn't exist yet.

- [ ] **Step 8: Implement `shouldFlushPlayback`**

Create `web/src/protocol/controlActions.ts`:

```typescript
import { WSMessage } from "./types";

export const shouldFlushPlayback = (message: WSMessage): boolean =>
  message.type === "control" && message.action === "bargeIn";
```

- [ ] **Step 9: Run test to verify it passes**

Run: `cd web && npx vitest run src/protocol/controlActions.test.ts`
Expected: PASS, all three tests.

- [ ] **Step 10: Wire the flush into `useServerAudio.ts`**

In `web/src/pages/Conversation/hooks/useServerAudio.ts`, add to the imports:

```typescript
import { shouldFlushPlayback } from "../../../protocol/controlActions";
```

Change the `onSocketMessage` callback from:

```typescript
  const onSocketMessage = useCallback(
    (e: MessageEvent) => {
      const dataArray = new Uint8Array(e.data);
      const message = decodeMessage(dataArray);
      if (message.type === "audio") {
        decodeAudio(message.data);
        //For stats purposes for now
        totalAudioMessages.current++;
      }
    },
    [decodeAudio],
  );
```

to:

```typescript
  const onSocketMessage = useCallback(
    (e: MessageEvent) => {
      const dataArray = new Uint8Array(e.data);
      const message = decodeMessage(dataArray);
      if (message.type === "audio") {
        decodeAudio(message.data);
        //For stats purposes for now
        totalAudioMessages.current++;
      } else if (shouldFlushPlayback(message)) {
        worklet.current.port.postMessage({ type: "reset" });
      }
    },
    [decodeAudio],
  );
```

This reuses the exact same `{ type: "reset" }` message the worklet's existing `port.onmessage` handler already responds to (see `web/src/audio-processor.ts`) -- no worklet-side change needed.

No dedicated test is added for this one-line wiring change itself: `useServerAudio.ts` has no existing test file today (unlike `useUserAudio.ts`, which needs ~150 lines of `SocketContext`/`MediaContext`/decoder-worker mocking to test via `renderHook` -- disproportionate to add here for one conditional branch). The actual decision logic (`shouldFlushPlayback`) is fully covered by Step 9's tests; the action taken here (`worklet.current.port.postMessage({ type: "reset" })`) is byte-for-byte identical to the existing, already-shipped, already-untested call at this same file's line 128 -- not a new risk surface this plan is introducing.

- [ ] **Step 11: Run the full web test suite and lint**

Run: `cd web && npx vitest run`
Expected: PASS, no regressions.

Run: `cd web && npx tsc --noEmit`
Expected: no type errors.

- [ ] **Step 12: Commit**

```bash
git add web/src/protocol/types.ts web/src/protocol/encoder.ts web/src/protocol/controlActions.ts web/src/protocol/encoder.test.ts web/src/protocol/controlActions.test.ts web/src/pages/Conversation/hooks/useServerAudio.ts
git commit -m "feat: flush client playback on proxy-detected barge-in"
```

---

## After this plan

This makes the proxy itself withhold the assistant's audio and flush client playback the instant sustained user speech is detected -- independent of whether PersonaPlex's own learned behavior ever improves. It does **not** yet change how the release-gate evaluation measures interruption latency: `collect-native-arm.py` still connects directly to PersonaPlex, bypassing this proxy entirely, so passing the gate still won't reflect this fix. That eval-harness routing change (making `collect-native-arm.py` go through `POST /api/session` -> `ws://.../api/session/{id}/chat` -> `DELETE /api/session/{id}`, and updating `run-v6-native-eval.sh` / `run-v6-native-eval-self-distill.sh` to launch the `tolkien_voice` app itself rather than driving PersonaPlex directly via `_personaplex_ctl.py`) is real, additional architectural work -- how per-arm adapter configs (base/aragorn-adapter/sauron-adapter) actually flow through `PersonaPlexRuntime.start()`'s docker-command construction needs to be traced precisely before that plan can be written with the same confidence as this one -- and is intentionally a separate, follow-on plan rather than bolted onto this one.
