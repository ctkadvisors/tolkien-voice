# Proxy-enforced barge-in muting

Date: 2026-07-31

Status: Proposed for implementation

Repository: `ctkadvisors/tolkien-voice` (private)

## Context

Two rounds of training (the original v6-only curriculum, and this session's
reward-filtered self-distillation) have both tried to fix post-interruption
yield latency purely by teaching the model itself to react faster. Both
failed the release gate on this exact metric
(`interruption_p95_ms <= 1000ms`): the v6-only baseline measured 3843ms, and
the self-distillation-augmented checkpoints measured 4085ms -- slightly
worse, not better (`eval-runs/mvp-v6-decoupled/report.json`,
`eval-runs/mvp-v6-self-distill-v1/report.json`).

This is consistent with how duplex speech models actually work: this
project's own code cites Kyutai's Moshi paper (`interruption_reward.py`'s
`compute_yield_delay_reward` docstring references arXiv:2606.11167), and
Moshi's native duplex responsiveness comes from being pretrained end-to-end
on millions of hours of continuously interleaved dual-stream audio. A LoRA
fine-tune on 20 examples (~100 seconds of audio) is trying to move a
behavior baked in at a scale roughly six orders of magnitude larger -- not a
tuning problem, the wrong lever entirely.

Checking the actual production code confirms training was never the only
option: `src/tolkien_voice/api.py` already runs a transparent bidirectional
proxy between the browser and PersonaPlex's raw websocket
(`/api/session/{session_id}/chat` -> upstream `/api/chat`), currently pure
passthrough with zero inspection in either direction. This is exactly the
choke point for a standard technique used by many production voice
products: VAD-triggered hard muting of the assistant's outgoing audio the
instant sustained user speech is detected, enforced at the proxy,
completely decoupled from whether the underlying model itself decides to
go quiet.

## Decision: enforce muting at the proxy, not client-side-only

Confirmed with the project owner (2026-07-31): the fix must be validated by
the actual release-gate evaluation, which means it must change what bytes
flow over the wire at the same endpoint both a real browser and the eval
probe connect to. A client-side-only fix (the browser running its own local
VAD on its own mic input and flushing its own player buffer, no server
round-trip) would likely be *lower latency* for real users, but would be
invisible to `collect-native-arm.py`'s measurement (which reads raw
received bytes, not what a browser's JS chooses to play) unless the eval
itself simulated a full client-side audio pipeline -- a much larger, more
fragile change, and one that would only validate this one specific web
client rather than any integration talking to the proxy.

Server-side proxy enforcement is therefore the primary mechanism. Local
client-side VAD as an additional latency optimization is explicitly out of
scope for this work (see Scope).

## Architecture

The existing `/api/session/{session_id}/chat` relay in `api.py` gains a
`BargeInDetector` (new, `src/tolkien_voice/barge_in.py`) that watches the
browser-to-upstream audio stream. `_browser_to_upstream` decodes each
incoming Opus frame to PCM (via `sphn.OpusStreamReader`, promoted from an
evaluation-only dependency to a core one -- see Dependencies) and feeds it
to the detector as 40ms RMS frames, mirroring
`training.native_probe.ActivityDetector`'s threshold + sustained-frames
shape but reimplemented in core `tolkien_voice` rather than imported across
the training-package boundary (`api.py` is core; `training` is not a
dependency of core, and importing across that boundary for one small class
isn't worth blurring it). When the detector flips idle -> active
(sustained user speech = barge-in), `_upstream_to_browser` stops forwarding
PersonaPlex's audio frames (`0x01`) to the browser and the relay sends one
new control message telling the browser to flush its playback buffer. Text
frames (`0x02`) are never gated -- transcripts and UI state keep flowing
normally in both states. When the detector flips back to idle (user has
gone quiet), audio forwarding resumes on PersonaPlex's next frame; no
explicit "resume" signal is needed.

No change to how PersonaPlex itself is driven, or to session
start/stop/lease logic in `session.py` -- this only governs what reaches
the browser.

## Components

- **`BargeInDetector`** (new, `src/tolkien_voice/barge_in.py`): pure class,
  `append(rms: float) -> bool` returning whether state just flipped
  active/idle (mirrors the existing `ActivityDetector.append` shape).
  Threshold and sustained-frame-count are constructor parameters, not
  hardcoded, so they can be tuned without code changes. Default values:
  `threshold=0.02, sustained_frames=3` (120ms at the existing 40ms frame
  size) for detecting the *start* of user speech -- this project already
  has two established conventions for RMS-based activity detection
  (`native_probe.py`), and this is the one already used elsewhere in this
  codebase for detecting speech *starting* (as opposed to
  `threshold=0.015, sustained_frames=12`, used elsewhere for detecting the
  *assistant* going quiet -- a different, slower-by-design threshold for a
  different purpose). Reusing the existing speech-start convention rather
  than inventing a new pair of numbers.
- **Opus decode**: `sphn.OpusStreamReader` per connection (same library
  already pinned and used in `training`'s `evaluation` extra), instantiated
  fresh inside the `chat()` handler alongside the existing `upstream`
  connection -- same lifecycle, no cross-session state.
- **`api.py` changes**: `_browser_to_upstream` decodes each audio frame and
  feeds `BargeInDetector`; `_upstream_to_browser` checks detector state
  before forwarding an audio (`0x01`) frame; the relay setup sends the new
  control message the moment the detector flips active.
- **Protocol**: add a new control action to `CONTROL_MESSAGES_MAP` in
  `web/src/protocol/types.ts` (e.g. `bargeIn`) -- not reusing the existing
  `pause` action, which is presumably part of PersonaPlex's own
  session-control semantics and shouldn't be overloaded with an unrelated
  meaning. The server only needs to emit the matching raw byte value, not
  the TypeScript enum.
- **Client** (`web/src/audio-processor.ts` and whatever owns the worklet
  `port`): on receiving the new control message, post `{type: "reset"}` to
  the `AudioWorkletProcessor` port -- reusing the exact mechanism already
  implemented there for the existing (currently manually-triggered) reset
  case, not a new flush mechanism.

## Dependencies

`sphn` moves from the `evaluation` optional-dependency group to core
`dependencies` in `pyproject.toml`. `numpy` stays evaluation-only --
RMS over a 40ms PCM16 frame doesn't need it; plain Python/`array` math is
sufficient at this frame rate and size, and keeps the core dependency
addition to exactly the one library actually needed (the Opus codec),
not a general numeric-computing dependency the relay path doesn't
otherwise require.

## Error handling

- A malformed/undecodable Opus frame from the browser must not break the
  conversation: skip that frame's contribution to `BargeInDetector`, but
  still relay it upstream to PersonaPlex normally. VAD is a passive
  behavior on top of the existing relay, not a new hard requirement for
  the relay to keep functioning.
- `web/src/protocol/encoder.ts`'s `decodeMessage` currently `throw`s on an
  unrecognized control action byte, with an existing
  `//TODO: log this and don't throw` marking this as a known, unaddressed
  gap. Since this work adds a new reliance on control messages carrying
  real behavior, fix this now: log and ignore an unrecognized action
  rather than throwing and breaking the connection. Small, directly in the
  path of what's being built, not unrelated scope creep.
- `BargeInDetector` and the per-connection `sphn.OpusStreamReader` are both
  instantiated fresh inside `chat()`, scoped to that single websocket
  connection's lifetime -- no shared/module-level state, no leakage
  between concurrent or sequential sessions.

## Testing

- `BargeInDetector`: pure unit tests, feeding synthetic RMS sequences and
  asserting active/idle transitions -- same pattern as the existing
  `ActivityDetector` tests in the training suite.
- `api.py`: integration-style tests around the new mute/unmute behavior
  (audio frames withheld while active, control message sent on the active
  transition, text frames never gated, resumption on the idle transition)
  and around the new dependency wiring.
- Client (`vitest`, matching the existing suite structure such as
  `decoderWorker.test.ts`): `encoder.ts`/`types.ts` changes for the new
  control action, and the worklet's response to receiving it (posting
  `{type: "reset"}`), alongside the existing reset-path test if one exists
  or a new one mirroring it.

## Eval-harness update (required for this fix to be measurable at all)

`collect-native-arm.py` currently connects directly to PersonaPlex's raw
`/api/chat` -- no session, no lease, nothing from `tolkien_voice`'s own API
surface is exercised. To measure the real, fixed path, it must instead:
`POST /api/session` (start a session), connect to
`ws://.../api/session/{session_id}/chat`, run the probe exactly as today,
`DELETE /api/session/{session_id}` (end it) -- the exact sequence a real
browser client goes through. This means `run-v6-native-eval.sh` and
`run-v6-native-eval-self-distill.sh` must also start/stop the
`tolkien_voice` FastAPI app itself (currently they only start/stop
PersonaPlex via `_personaplex_ctl.py`), and the big-model-lease
coordination (`scripts/run-with-big-model-lease.sh`,
`TOLKIEN_VOICE_LEASE_WAIT_SECONDS`) must keep working correctly with both
processes involved -- the app process acquires/releases PersonaPlex access
on the eval's behalf the same way it would for a real user session, not a
second, competing lease path.

Both the original v6-only baseline's report and the self-distillation
report were generated by the OLD, direct-to-PersonaPlex methodology --
re-running either eval variant after this change will not be comparable to
those existing reports without re-establishing a new baseline (a base-arm
measurement) under the corrected, proxy-routed methodology at the same
time as measuring the fix. This is expected and correct, not a defect:
the old reports measured a materially different thing (the raw model,
bypassing the exact layer this fix lives in).

## Scope

**In scope:** `BargeInDetector`, `api.py` relay changes, the new protocol
control action and its encode/decode, the client-side worklet-reset
trigger, the `decodeMessage` unknown-action fix, promoting `sphn` to a core
dependency, and updating both native-eval scripts (and
`collect-native-arm.py`) to route through the real session/websocket path
instead of connecting directly to PersonaPlex.

**Out of scope:** client-side-only local VAD (a possible future latency
optimization layered on top, not needed to satisfy the release gate as
currently defined); any change to PersonaPlex, `session.py`'s session
lifecycle/lease logic itself, or the LoRA training pipeline; tuning the
detector's threshold/sustained-frame parameters beyond initial reasonable
defaults (real-world tuning against the live system is a follow-up, not
a blocker for a first correct implementation); re-running the full
self-distillation training effort -- those checkpoints and the finding
that LoRA fine-tuning alone doesn't move this metric stand as-is.
