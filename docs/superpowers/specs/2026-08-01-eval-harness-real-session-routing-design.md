# Route the release-gate eval through the real app session/websocket path

Date: 2026-08-01

Status: **WITHDRAWN 2026-08-01 — do not implement.** Three reasons: (1) it
fixes at most 1 of the release gate's 4 failures — the other 3 are
adapter-vs-base significance questions at n=12 that proxy routing cannot
touch, so the gate still fails afterward; (2) the question the A/B
apparatus exists to answer ("should an adapter go live?") was already
answered "no" twice, and the standing recommendation is to ship
base+prompts with no adapter at all; (3) the one thing routing would newly
measure — perceived interruption latency through the real session path —
was measured directly on 2026-08-01: 93/183/191 ms live probes through
`POST /api/session` → `ws /api/session/{id}/chat` (see
`.superpowers/sdd/progress.md`). The operational findings below (port
8788 hardcoding, the flock double-acquisition deadlock, `web_dist=None`)
remain accurate and useful reference if an app-launching harness is ever
genuinely needed.

Repository: `ctkadvisors/tolkien-voice` (private)

## Context

The proxy-enforced barge-in muting feature (spec:
`docs/superpowers/specs/2026-07-31-proxy-barge-in-mute-design.md`, merged
2026-08-01) fixes interruption latency at the systems level, decoupled from
whatever PersonaPlex's own learned behavior does. But the sealed native A/B
release-gate evaluation (`scripts/collect-native-arm.py`,
`scripts/run-v6-native-eval.sh`, `scripts/run-v6-native-eval-self-distill.sh`)
connects directly to PersonaPlex's raw `/api/chat` websocket, bypassing
`tolkien_voice`'s own relay proxy (`src/tolkien_voice/api.py`'s
`/api/session/{session_id}/chat`) entirely. The barge-in fix lives in that
proxy, so the eval as it exists today cannot measure it, and would not
measure any future fix that also lives in the proxy layer.

## Decision: launch the real app for eval, not PersonaPlex directly

Investigated (2026-08-01) whether the app's existing production machinery
already supports what the eval needs, before designing anything new:

- `eval_runtime.py`'s per-arm config rendering (`render_eval_runtime_config`)
  already produces exactly the right shape -- `adapter_path`/
  `adapter_metadata_path` set for an adapter arm, both stripped for the base
  arm -- consumed correctly by `PersonaPlexRuntime._validate_adapter` via
  `RuntimeSettings.adapter_path`/`adapter_bank` fallback logic. **No change
  needed here.**
- The app's own `SessionController.start()` → `PersonaPlexRuntime.start(character_id)`
  already handles PersonaPlex's full docker lifecycle (start, health-poll,
  stop) internally, keyed off whatever adapter the loaded `TOLKIEN_VOICE_CONFIG`
  specifies -- this is the exact "start PersonaPlex with this specific
  variant" mechanism the eval currently re-implements manually via
  `_personaplex_ctl.py`.

Given that, the decision is: **launch the real `tolkien_voice` app itself
for each eval arm**, and drive it through its real `POST /api/session` →
`ws://.../api/session/{id}/chat` → `DELETE /api/session/{id}` flow, letting
the app manage PersonaPlex internally exactly as it does for a real user.
This replaces `_personaplex_ctl.py start/stop` in the eval flow entirely --
not just supplements it.

## Two operational conflicts found, both need to be designed around

**Port collision.** Confirmed empirically (2026-08-01,
`curl http://127.0.0.1:8788/api/health` on Spark returned
`{"status":"ok","service":"tolkien-voice"}`): the real production app is
running right now on port 8788 (`configs/runtime.yaml`'s `controller_port`).
Launching an eval instance on the same port would collide with it.
Confirmed with the project owner: eval instances bind to a different port
(8789) instead, so eval runs never contend with a live production instance
on the same port.

`controller_port` in `RuntimeSettings` is validated but never actually
consumed for binding -- `src/tolkien_voice/main.py`'s `run()` hardcodes
`port=8788` directly in its `uvicorn.run()` call, ignoring the config field
entirely. Setting `controller_port: 8789` in an eval-rendered config
therefore does nothing on its own; the eval's own app-launcher must pass an
explicit port to `uvicorn.run()`.

**Lease double-acquisition deadlock.** `SparkModelLease` (the app's own
GPU-model-access coordination, used by `PersonaPlexRuntime` internally) and
`scripts/run-with-big-model-lease.sh` (used to wrap the eval's *current*
direct-to-PersonaPlex flow, and used by the training scripts) both lock the
exact same file: `/run/user/1000/tolkien-voice/big-model.lock`. `flock` is
not reentrant across separate processes -- if the eval script wrapped the
new app-launch step in `run-with-big-model-lease.sh` the way it wraps
today's `_personaplex_ctl.py` calls, the outer wrapper would hold the lock
indefinitely while the app's own `SparkModelLease.acquire()` (triggered by
`POST /api/session`) waited on the same lock forever. **`run-with-big-model-lease.sh`
must not wrap any part of this new flow** -- the app already re-implements
that exact coordination internally, correctly, whenever a session starts.

## `web_dist=None` for eval instances

`src/tolkien_voice/main.py:run()` calls `create_app(web_dist=ROOT/"web/dist")`,
which raises `RuntimeError` at startup if `web/dist/index.html` and
`web/dist/assets/` don't both exist -- the app refuses to start without a
built browser bundle. Eval never touches `/` or `/assets`, only the JSON
API and the chat websocket, so the eval's own app-launcher must call
`create_app(web_dist=None)` directly (skipping `run()` entirely, since
`run()` hardcodes both the wrong port and this web_dist requirement) --
not work around the requirement by building the frontend as an eval
prerequisite that has nothing to do with what eval actually exercises.

## Scope

**In scope:**
- `scripts/run-eval-app.py` (new): thin launcher taking `--port` (default
  8789), reading `TOLKIEN_VOICE_CONFIG` from the environment exactly as
  `main.py`'s `run()` does, calling
  `uvicorn.run(create_app(web_dist=None), host="127.0.0.1", port=args.port)`
  directly.
- `scripts/collect-native-arm.py`: gains `--app-url` (default
  `http://127.0.0.1:8789`); replaces its direct-to-PersonaPlex websocket
  connection with `POST {app_url}/api/session` → poll `GET {app_url}/api/session`
  until `state == "ready"`, bounded by a timeout (matching the
  `runtime_start_timeout_seconds` convention already used elsewhere for
  PersonaPlex-readiness waits -- the poll must fail loudly on timeout, not
  hang indefinitely) → connect to `ws://.../api/session/{session_id}/chat`
  → run the probe exactly as today (no change to probe logic itself, only
  to how the connection is established) → `DELETE {app_url}/api/session/{session_id}`.
- `scripts/run-v6-native-eval.sh` and `scripts/run-v6-native-eval-self-distill.sh`:
  `collect_one` launches `run-eval-app.py` (background, PID captured) with
  `TOLKIEN_VOICE_CONFIG` pointed at the per-arm config, waits for
  `GET /api/health` to respond before proceeding, runs
  `collect-native-arm.py` against it, then stops the app process (SIGTERM)
  -- replacing the current `_personaplex_ctl.py start`/`stop` calls
  entirely. No `run-with-big-model-lease.sh` wrapping anywhere in this new
  flow.

**Out of scope:** any change to `eval_runtime.py` (already correct, verified
above), `SessionController`/`PersonaPlexRuntime`/`SparkModelLease`
(already correct, this plan only calls them through their existing public
HTTP/session API, never modifies their internals), the barge-in-mute
feature itself (already shipped, separate branch, merged), and actually
running this against live Spark infrastructure to produce a new eval
report -- that is a distinct, explicit decision after this plan's code is
merged, not part of this plan's own deliverable.

## Testing

- `scripts/run-eval-app.py` has no logic of its own worth unit testing (a
  thin `argparse` + `uvicorn.run` wrapper) -- validated by actually running
  it, not by a pytest target.
- `collect-native-arm.py`'s new session-lifecycle logic (start / poll-ready
  / connect / end) must be unit-testable with fakes for the HTTP client and
  websocket connection, following the same fake-based testing convention
  already established for `probe_native_session` in
  `tests/test_training_native_probe.py` -- real behavior verified against
  fakes, not the actual network/process.
- Real end-to-end validation (does the rewritten eval flow actually produce
  a valid report against live PersonaPlex on Spark) is explicitly excluded
  from this plan's own scope and requires a separate go-ahead once the code
  is merged.
