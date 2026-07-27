# Tolkien Voice Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a private browser experience on the DGX Spark where a user can select Gandalf, Sauron, or another initial character and hold a low-latency, interruptible, native PersonaPlex speech-to-speech conversation without using a shell.

**Architecture:** A FastAPI session controller serves a customized copy of NVIDIA's PersonaPlex browser client, validates character cards, owns a single Spark large-model file lease, starts a GB10-compatible PersonaPlex container, and proxies the binary WebSocket stream. Existing llama-swap large-model launch commands use the same lease and a short idle TTL, while the embedding route remains untouched. Tailscale Serve terminates HTTPS and forwards only to the loopback-bound controller.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, PyYAML, httpx, websockets, pytest, React 18, TypeScript, Vite, Vitest, Playwright, Docker with NVIDIA Container Toolkit, PersonaPlex commit `3428dfd95309a7f3c84fd93259ded0f810d1ff91`, Tailscale Serve 1.98+, systemd user services.

---

## Scope and invariants

This plan implements Phase 1 of the approved design only. Tolkien behavior distillation, Moshi LoRA training, authorized custom voice creation, iPhone deployment, and simultaneous multi-character speech require separate plans after the baseline metrics exist.

The implementation must preserve these invariants:

- Live audio goes directly through PersonaPlex/Moshi; no ASR, text LLM, or TTS process is added to the runtime path.
- Exactly one character and one PersonaPlex conversation may be active at a time.
- `embed` remains resident throughout speech sessions.
- `coder`, `coder-fp8`, `qwen36`, `gemma`, `tolkien`, and PersonaPlex share one exclusive OS file lease.
- Neither a new voice session nor a coder request kills an active workload. The new request waits for the lease.
- PersonaPlex binds to `127.0.0.1:8998`; the controller binds to `127.0.0.1:8788`; only Tailscale HTTPS is browser-facing.
- Browser audio and transcripts are not written to disk. Recording UI and MediaRecorder code are removed.
- The Hugging Face token is mounted as a runtime secret, never committed, copied into an image layer, logged, or placed in Docker container metadata.
- All deployment mutations are drift-checked, backed up, and reversible.

## Immutable upstream inputs

- PersonaPlex source: `https://github.com/NVIDIA/personaplex.git`
- PersonaPlex source revision: `3428dfd95309a7f3c84fd93259ded0f810d1ff91`
- PersonaPlex model: `nvidia/personaplex-7b-v1`
- Container base already cached on Spark: `nvcr.io/nvidia/pytorch:25.11-py3`
- Spark host: `spark-1250.tail71b73b.ts.net` / `100.98.79.30`
- Existing llama-swap endpoint: `http://127.0.0.1:9090`
- Existing llama-swap config: `/home/chrisknuteson/inference/llama-swap/config.yaml`
- Deployment root: `/home/chrisknuteson/tolkien-voice`
- Persistent data root: `/home/chrisknuteson/tolkien-voice-data`

## Task 1: Scaffold the application and pin upstream PersonaPlex

**Files:**

- Create: `.gitignore`
- Create: `.python-version`
- Create: `pyproject.toml`
- Create: `src/tolkien_voice/__init__.py`
- Create: `src/tolkien_voice/main.py`
- Create: `tests/test_main.py`
- Create: `vendor/README.md`
- Modify: `.gitmodules`
- Create: `README.md`

- [ ] **Step 1: Add the pinned PersonaPlex submodule**

Run:

```bash
git submodule add https://github.com/NVIDIA/personaplex.git vendor/personaplex
git -C vendor/personaplex checkout 3428dfd95309a7f3c84fd93259ded0f810d1ff91
git add .gitmodules vendor/personaplex
```

Expected: `git submodule status` begins with a space and the exact revision:

```text
 3428dfd95309a7f3c84fd93259ded0f810d1ff91 vendor/personaplex
```

- [ ] **Step 2: Write the failing application-factory test**

```python
# tests/test_main.py
from fastapi.testclient import TestClient

from tolkien_voice.main import create_app


def test_health_is_loopback_service_health() -> None:
    client = TestClient(create_app())

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "tolkien-voice"}
```

- [ ] **Step 3: Run the test to verify the scaffold is absent**

Run: `uv run pytest tests/test_main.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'tolkien_voice'`.

- [ ] **Step 4: Add the Python project and minimal app**

Use this dependency boundary in `pyproject.toml` and commit the generated `uv.lock`:

```toml
[project]
name = "tolkien-voice"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "fastapi>=0.116,<1",
  "httpx>=0.28,<1",
  "pydantic>=2.11,<3",
  "pydantic-settings>=2.10,<3",
  "pyyaml>=6.0,<7",
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

[project.scripts]
tolkien-voice = "tolkien_voice.main:run"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
addopts = "--strict-markers --strict-config"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py312"
```

Implement `create_app()` with only `/api/health`, and implement `run()` as `uvicorn.run("tolkien_voice.main:create_app", factory=True, host="127.0.0.1", port=8788)`.

Set `.python-version` to `3.12`. Ignore `.venv`, Python/Node caches, `web/dist`, `.env*`, `*.wav`, `*.pt`, `*.safetensors`, `data/`, `recordings/`, and `secrets/`, while explicitly allowing `.env.example`.

- [ ] **Step 5: Install and verify**

Run:

```bash
uv sync --extra dev
uv run pytest tests/test_main.py -q
uv run ruff check src tests
```

Expected: one passing test and no Ruff errors.

- [ ] **Step 6: Document upstream provenance**

In `vendor/README.md`, record the repository, exact commit, upstream MIT source license, gated model license distinction, and the command used to update the submodule. In `README.md`, state that Phase 1 is prompt-conditioned PersonaPlex and does not yet contain a Tolkien LoRA.

- [ ] **Step 7: Commit**

```bash
git add .gitignore .python-version pyproject.toml uv.lock src tests vendor README.md .gitmodules
git commit -m "chore: scaffold Tolkien Voice service"
```

## Task 2: Implement settings and the validated character registry

**Files:**

- Create: `configs/runtime.yaml`
- Create: `configs/characters/gandalf.yaml`
- Create: `configs/characters/sauron.yaml`
- Create: `configs/characters/galadriel.yaml`
- Create: `configs/characters/samwise.yaml`
- Create: `configs/characters/aragorn.yaml`
- Create: `configs/characters/frodo.yaml`
- Create: `configs/characters/treebeard.yaml`
- Create: `configs/characters/boromir.yaml`
- Create: `configs/characters/barliman.yaml`
- Create: `src/tolkien_voice/settings.py`
- Create: `src/tolkien_voice/characters.py`
- Create: `tests/test_settings.py`
- Create: `tests/test_characters.py`

- [ ] **Step 1: Write failing settings and registry tests**

Cover all of these cases:

```python
def test_runtime_defaults_are_spark_safe(runtime_settings):
    assert runtime_settings.controller_host == "127.0.0.1"
    assert runtime_settings.controller_port == 8788
    assert runtime_settings.personaplex_url == "http://127.0.0.1:8998"
    assert runtime_settings.disconnect_idle_seconds == 300
    assert runtime_settings.lease_wait_seconds == 900


def test_registry_contains_initial_roster(registry):
    assert set(registry.ids()) == {
        "gandalf", "sauron", "galadriel", "samwise", "aragorn",
        "frodo", "treebeard", "boromir", "barliman",
    }


def test_registry_rejects_voice_not_allowed_for_character(registry):
    with pytest.raises(VoiceNotAllowedError):
        registry.resolve("sauron", "NATF0.pt")


def test_prompt_is_wrapped_once(registry):
    prompt = registry.resolve("gandalf", "NATM1.pt").text_prompt
    assert prompt.startswith("<system>")
    assert prompt.endswith("<system>")
    assert prompt.count("<system>") == 2
```

- [ ] **Step 2: Confirm red**

Run: `uv run pytest tests/test_settings.py tests/test_characters.py -q`

Expected: FAIL because the modules do not exist.

- [ ] **Step 3: Implement typed settings**

`RuntimeSettings` must resolve paths relative to the repository config file, allow overrides through `TOLKIEN_VOICE_*` environment variables, and contain:

```yaml
# configs/runtime.yaml
controller_host: 127.0.0.1
controller_port: 8788
personaplex_url: http://127.0.0.1:8998
personaplex_container: tolkien-voice-personaplex
personaplex_image: tolkien-voice/personaplex:3428dfd
hf_repo: nvidia/personaplex-7b-v1
disconnect_idle_seconds: 300
lease_wait_seconds: 900
runtime_start_timeout_seconds: 900
llama_swap_url: http://127.0.0.1:9090
lock_path: /run/user/1000/tolkien-voice/big-model.lock
owner_path: /run/user/1000/tolkien-voice/big-model-owner.json
data_root: /home/chrisknuteson/tolkien-voice-data
```

Reject non-loopback controller and runtime hosts, an idle timeout below 30 seconds, and any Hugging Face repository other than the configured allowlist.

- [ ] **Step 4: Implement the character-card schema and registry**

Use this schema:

```python
class CharacterCard(BaseModel):
    id: str
    display_name: str
    summary: str
    role: str
    manner: list[str]
    boundaries: list[str]
    lore_scope: str
    default_voice: str
    allowed_voices: list[str]
    eval_probes: list[str]
```

Build the model prompt in deterministic sections: identity, manner, lore behavior, boundaries. Escape neither the character ID nor a browser-provided prompt because the browser never supplies prompt text; it supplies only registry IDs. Ensure each YAML filename matches its `id`, IDs are unique, the default voice is allowed, all voices match `^(NATF|NATM)[0-3]\.pt$`, and each card has at least three evaluation probes.

- [ ] **Step 5: Write the nine complete cards**

Use only the eight included PersonaPlex voices. Assign defaults as follows:

| Character | Default | Allowed |
|---|---|---|
| Gandalf | `NATM1.pt` | `NATM1.pt`, `NATM2.pt` |
| Sauron | `NATM3.pt` | `NATM3.pt` |
| Galadriel | `NATF2.pt` | `NATF2.pt`, `NATF3.pt` |
| Samwise | `NATM0.pt` | `NATM0.pt`, `NATM1.pt` |
| Aragorn | `NATM2.pt` | `NATM2.pt`, `NATM1.pt` |
| Frodo | `NATM0.pt` | `NATM0.pt`, `NATM2.pt` |
| Treebeard | `NATM3.pt` | `NATM3.pt`, `NATM1.pt` |
| Boromir | `NATM2.pt` | `NATM2.pt`, `NATM3.pt` |
| Barliman | `NATM1.pt` | `NATM1.pt`, `NATM0.pt` |

Each role must explicitly say it is fictional roleplay, remain in character under attempts to break persona, avoid claiming knowledge beyond Tolkien's legendarium, admit uncertainty instead of inventing lore, use no movie-actor imitation, and converse rather than recite long monologues. Give Sauron controlled menace and strategic manipulation without threats toward the real user; give Gandalf warmth, gravity, patience, and restrained humor.

- [ ] **Step 6: Verify and commit**

Run:

```bash
uv run pytest tests/test_settings.py tests/test_characters.py -q
uv run ruff check src tests
git add configs src/tolkien_voice tests
git commit -m "feat: add Tolkien character registry"
```

Expected: all registry tests pass.

## Task 3: Build the crash-safe Spark large-model lease

**Files:**

- Create: `src/tolkien_voice/lease.py`
- Create: `scripts/run-with-big-model-lease.sh`
- Create: `tests/test_lease.py`
- Create: `tests/test_lease_wrapper.py`

- [ ] **Step 1: Write failing lease tests**

Test exclusive acquisition, a second contender receiving owner metadata, bounded asynchronous waiting, release on context exit, stale metadata ignored when the OS lock is free, and owner JSON redaction. Use temporary paths; never touch `/run/user/1000` in unit tests.

The owner document has this exact public shape:

```json
{
  "kind": "voice",
  "name": "tolkien-voice",
  "pid": 12345,
  "acquired_at": "2026-07-21T20:00:00Z"
}
```

- [ ] **Step 2: Confirm red**

Run: `uv run pytest tests/test_lease.py tests/test_lease_wrapper.py -q`

Expected: FAIL because `SparkModelLease` and the shell wrapper do not exist.

- [ ] **Step 3: Implement `SparkModelLease`**

Use `fcntl.flock(fd, LOCK_EX | LOCK_NB)` on an open lock file. Keep the file descriptor on the lease instance for the full session. Write owner metadata atomically through a sibling temporary file plus `os.replace()`. On release, remove owner metadata only if its PID and acquisition timestamp match the holder, unlock, and close the descriptor. On process death, the kernel releases the lock; a later holder overwrites stale metadata.

Expose four typed methods on `SparkModelLease`: synchronous `try_acquire(owner) -> bool`, asynchronous `acquire(owner, timeout_seconds) -> None`, `current_owner() -> LeaseOwner | None`, and idempotent `release() -> None`.

Poll at 500 ms with `asyncio.sleep`; raise `LeaseTimeoutError(owner=current_owner)` at the deadline.

- [ ] **Step 4: Implement the llama-swap launch wrapper**

`scripts/run-with-big-model-lease.sh` must accept `OWNER_NAME COMMAND [ARG...]`, create `${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/tolkien-voice`, open FD 9 on `big-model.lock`, and execute `flock --exclusive --wait "${TOLKIEN_VOICE_LEASE_WAIT_SECONDS:-840}" 9`. After acquisition it atomically writes model owner metadata with `kind=text`, then `exec`s the original command while FD 9 remains open.

Use exit code 75 and this stderr message after timeout:

```text
Spark large-model slot is busy; an active Tolkien Voice conversation was not interrupted.
```

Do not read or print environment values. Trap `EXIT`, `INT`, and `TERM` to remove matching owner metadata without deleting another process's metadata.

- [ ] **Step 5: Verify the real shell contention behavior**

The test starts one wrapper around `sleep 2`, starts a second with `TOLKIEN_VOICE_LEASE_WAIT_SECONDS=0`, and asserts exit 75 plus the exact message. It then waits for the first process and confirms a third wrapper succeeds.

Run:

```bash
uv run pytest tests/test_lease.py tests/test_lease_wrapper.py -q
shellcheck scripts/run-with-big-model-lease.sh
```

Expected: all tests pass and ShellCheck reports no findings.

- [ ] **Step 6: Commit**

```bash
git add src/tolkien_voice/lease.py scripts/run-with-big-model-lease.sh tests
git commit -m "feat: add shared Spark model lease"
```

## Task 4: Build and test the GB10 PersonaPlex runtime image

**Files:**

- Create: `container/Dockerfile.personaplex`
- Create: `container/entrypoint.sh`
- Create: `container/healthcheck.py`
- Create: `scripts/build-runtime.sh`
- Create: `tests/test_container_contract.py`
- Create: `docs/operations/huggingface-access.md`

- [ ] **Step 1: Write the failing container contract test**

Parse the Dockerfile and entrypoint as text and assert:

- the base is exactly `nvcr.io/nvidia/pytorch:25.11-py3`;
- upstream files come from `vendor/personaplex/moshi`;
- installation uses `pip install --no-deps -e` so upstream's `torch<2.5` constraint cannot replace the GB10 PyTorch build;
- the service uses `--host 127.0.0.1 --port 8998 --static none`;
- no `HF_TOKEN` `ARG` or `ENV` exists;
- the only token source is `/run/secrets/hf_token`;
- no TLS is enabled inside the container.

- [ ] **Step 2: Confirm red**

Run: `uv run pytest tests/test_container_contract.py -q`

Expected: FAIL because the container files do not exist.

- [ ] **Step 3: Create the GB10 image**

The Dockerfile must install `libopus-dev`, copy the pinned submodule's `moshi/`, install the upstream non-Torch dependencies within their published ranges, then install Moshi editable with `--no-deps`. Preserve NGC's CUDA/PyTorch stack. Add OCI labels for the project URL and exact PersonaPlex commit.

The entrypoint must:

1. Refuse startup unless `/run/secrets/hf_token` is a non-empty regular file with mode no broader than `0600` on the host-mounted source.
2. Read it inside the container into `HF_TOKEN` without echoing it.
3. Execute `python -m moshi.server --host 127.0.0.1 --port 8998 --static none --hf-repo nvidia/personaplex-7b-v1`.

The healthcheck performs an HTTP GET to `http://127.0.0.1:8998/` and accepts 404 as proof the aiohttp server is ready because static serving is intentionally disabled. It must not connect to `/api/chat`, which would allocate a conversation.

- [ ] **Step 4: Add the build script**

`scripts/build-runtime.sh` must verify the submodule revision before building:

```bash
test "$(git -C vendor/personaplex rev-parse HEAD)" = \
  "3428dfd95309a7f3c84fd93259ded0f810d1ff91"
docker build \
  --file container/Dockerfile.personaplex \
  --tag tolkien-voice/personaplex:3428dfd \
  .
```

- [ ] **Step 5: Document the only manual credential prerequisite**

`docs/operations/huggingface-access.md` must instruct the owner to:

1. Accept the NVIDIA Open Model License at the model page.
2. Create a fine-grained Hugging Face token with read access to `nvidia/personaplex-7b-v1` only.
3. Install it on Spark at `/home/chrisknuteson/tolkien-voice-data/secrets/hf_token` with directory mode `0700` and file mode `0600`.
4. Run a token preflight that reports only success/failure and never echoes the token.

- [ ] **Step 6: Run local contract tests and commit**

```bash
uv run pytest tests/test_container_contract.py -q
shellcheck container/entrypoint.sh scripts/build-runtime.sh
git add container scripts/build-runtime.sh tests/test_container_contract.py docs/operations
git commit -m "feat: add GB10 PersonaPlex runtime image"
```

Do not attempt the gated model download until the credential prerequisite is complete.

## Task 5: Implement runtime lifecycle without persisting audio

**Files:**

- Create: `src/tolkien_voice/commands.py`
- Create: `src/tolkien_voice/runtime.py`
- Create: `tests/fakes.py`
- Create: `tests/test_runtime.py`

- [ ] **Step 1: Write failing runtime tests**

Use a fake command runner and fake HTTP client. Assert the exact Docker contract:

```text
docker run --detach --rm
  --name tolkien-voice-personaplex
  --gpus all
  --network host
  --ipc host
  --env HF_HOME=/cache/huggingface
  --env NO_TORCH_COMPILE=1
  --mount type=bind,src=/home/chrisknuteson/tolkien-voice-data/cache/huggingface,dst=/cache/huggingface
  --mount type=bind,src=/home/chrisknuteson/tolkien-voice-data/secrets/hf_token,dst=/run/secrets/hf_token,readonly
  tolkien-voice/personaplex:3428dfd
```

Also assert:

- startup creates only cache, secret, and log directories;
- no audio or transcript mount exists;
- a pre-existing managed container is stopped during controller startup recovery;
- an unrelated container is never touched;
- readiness polls HTTP until ready or the 900-second bounded timeout;
- startup failure captures the last 200 log lines with token-like strings redacted, stops the managed container, and releases no lease itself;
- stop is idempotent.

- [ ] **Step 2: Confirm red**

Run: `uv run pytest tests/test_runtime.py -q`

Expected: FAIL because the runtime classes do not exist.

- [ ] **Step 3: Implement a non-shell command runner**

`CommandRunner.run(argv: Sequence[str], *, timeout: float)` must use `asyncio.create_subprocess_exec`, never `shell=True`. Return a typed result containing exit code, stdout, and stderr. Redact strings matching `hf_[A-Za-z0-9]{20,}` before logging or returning operator diagnostics.

- [ ] **Step 4: Implement `PersonaPlexRuntime`**

Expose `recover()`, `start()`, `wait_ready()`, `logs()`, and `stop()`. Validate every host path is under `data_root` before constructing Docker arguments. Never accept image, container name, mount path, or command text from an HTTP request. Use `docker inspect` to distinguish absent, starting, healthy, and exited states.

Startup recovery may stop only the configured `tolkien-voice-personaplex` container. The session controller, not the runtime, owns lease acquisition and release.

- [ ] **Step 5: Verify and commit**

```bash
uv run pytest tests/test_runtime.py -q
uv run ruff check src tests
git add src/tolkien_voice tests
git commit -m "feat: manage PersonaPlex container lifecycle"
```

## Task 6: Implement the single-session state machine and API

**Files:**

- Create: `src/tolkien_voice/session.py`
- Create: `src/tolkien_voice/api.py`
- Modify: `src/tolkien_voice/main.py`
- Create: `tests/test_session.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write failing state-machine tests**

Use these states and only these transitions:

```text
idle -> waiting_for_lease -> loading -> ready -> connected
connected -> disconnected -> connected
ready|connected|disconnected -> stopping -> idle
waiting_for_lease|loading -> failed -> idle
```

Test:

- a valid start returns an opaque UUID session ID and enters `waiting_for_lease` immediately;
- a second start receives `409` with the active character and state;
- the background start waits for the lease rather than unloading a model;
- lease timeout moves through `failed` to `idle` and exposes the current owner;
- runtime failure releases the lease;
- explicit end stops runtime and releases immediately;
- WebSocket disconnect starts a 300-second cleanup timer;
- reconnect with the same session ID cancels cleanup;
- expiration stops runtime and releases;
- stale and unknown session IDs receive `404` without affecting the active session.

- [ ] **Step 2: Confirm red**

Run: `uv run pytest tests/test_session.py tests/test_api.py -q`

Expected: FAIL because session and API modules do not exist.

- [ ] **Step 3: Implement the controller**

`SessionController.start(character_id, voice_id)` resolves the registry entry before allocating state, creates the session, then starts a background task that acquires the lease and launches the runtime. It does not call llama-swap's unload endpoint. Existing large models release through their 60-second idle TTL, preventing active request eviction.

Keep all state in memory. Persist neither request audio, generated audio, nor transcript. On service startup, call `runtime.recover()`, clear stale owner metadata only after confirming the file lock is acquirable, and return to `idle`.

- [ ] **Step 4: Implement HTTP endpoints**

```text
GET    /api/health
GET    /api/characters
GET    /api/session
POST   /api/session            body: {"character_id":"gandalf","voice_id":"NATM1.pt"}
DELETE /api/session/{session_id}
WS     /api/session/{session_id}/chat
```

Return `202` from POST with `state=waiting_for_lease`. The browser polls `GET /api/session` every second through loading. Never return the role prompt, local paths, model token, Docker logs, or full owner metadata; return only owner `kind` and `name`.

- [ ] **Step 5: Implement the binary WebSocket proxy**

When the session is `ready` or `disconnected`, open this upstream URL server-side:

Open `ws://127.0.0.1:8998/api/chat` and use `urllib.parse.urlencode` to add the registry-built text prompt and validated voice filename as `text_prompt` and `voice_prompt`. Relay binary frames in two concurrent tasks with bounded 1 MiB frame size, disable WebSocket compression, and cancel the peer relay immediately when one side closes. Reject browser text frames with close code 1003. Do not log the prompt query or frame contents. Mark the session connected only after the upstream handshake frame arrives; preserve all frames byte-for-byte.

- [ ] **Step 6: Verify and commit**

```bash
uv run pytest tests/test_session.py tests/test_api.py -q
uv run pytest -q
uv run ruff check src tests
git add src/tolkien_voice tests
git commit -m "feat: add voice session controller"
```

## Task 7: Adapt the official PersonaPlex browser client

**Files:**

- Create: `scripts/sync-personaplex-client.sh`
- Create: `web/UPSTREAM.md`
- Create: `web/package.json`
- Create: `web/package-lock.json`
- Create: `web/src/pages/CharacterSelect.tsx`
- Create: `web/src/pages/ConversationPage.tsx`
- Create: `web/src/api.ts`
- Create: `web/src/types.ts`
- Modify copied upstream files under: `web/src/`
- Create: `web/src/pages/CharacterSelect.test.tsx`
- Create: `web/src/pages/ConversationPage.test.tsx`
- Create: `web/e2e/session.spec.ts`
- Create: `web/vitest.config.ts`
- Create: `web/playwright.config.ts`
- Modify: `src/tolkien_voice/main.py`

- [ ] **Step 1: Import the official client reproducibly**

`scripts/sync-personaplex-client.sh` must refuse any submodule revision except the pinned commit, copy `vendor/personaplex/client/` into `web/` while preserving license notices, then restore Tolkien Voice-owned page, API, test, and configuration files. Record every intentionally changed upstream file and the pinned commit in `web/UPSTREAM.md`.

Run the script once, commit the resulting source and lock file, and keep the submodule as the authoritative comparison point.

- [ ] **Step 2: Write failing UI tests**

Test that:

- all nine characters render from `GET /api/characters`;
- Sauron exposes only `NATM3.pt`;
- Start posts IDs, never a role prompt;
- `waiting_for_lease` names the blocking workload without offering an eviction button;
- `loading` keeps polling and does not open the microphone;
- `ready` requests microphone permission and opens `/api/session/{id}/chat`;
- End closes the socket, stops audio nodes, and calls DELETE;
- microphone denial leaves the page recoverable with browser permission guidance;
- no Record or Download control exists;
- transcript display is hideable and remains DOM-memory-only.

- [ ] **Step 3: Confirm red**

Run: `npm --prefix web test -- --run`

Expected: FAIL because the Tolkien Voice pages and API client do not exist.

- [ ] **Step 4: Replace queue/model controls with character-first UX**

Retain upstream protocol encoder/decoder, AudioWorklet, Opus capture, jitter buffer, server audio playback, and transcript decoder. Remove upstream queue, free-form text prompt, unrestricted voice prompt, seed/tuning controls, recording, MediaRecorder, audio download, remote worker address, and tunnel behavior.

The initial page has a quiet parchment-and-ink visual treatment, large readable character names, short summaries, an allowed voice selector, and one Start Conversation button. Conversation state must be operable by keyboard and announced through an `aria-live="polite"` status region. Do not autoplay microphone access before explicit Start.

- [ ] **Step 5: Route WebSocket traffic through the controller**

Build the socket URL from `window.location.origin` and the opaque session ID only. The browser must never receive or append `text_prompt` or a filesystem voice path. Preserve upstream binary protocol version 0 and set `binaryType="arraybuffer"`.

Remove upstream's ten-second no-message forced disconnect because natural silence can exceed ten seconds. Rely on WebSocket close/error and controller idle cleanup. Add bounded reconnect attempts at 0.5, 1, and 2 seconds to the same session, then present a manual reconnect control.

- [ ] **Step 6: Serve the production bundle from FastAPI**

Mount `web/dist/assets` as static files and return `web/dist/index.html` for `/`. If the bundle is missing, fail application startup with an operator-facing message in production mode; allow API-only startup under tests.

- [ ] **Step 7: Verify unit and browser behavior**

Run:

```bash
npm --prefix web ci
npm --prefix web run lint
npm --prefix web test -- --run
npm --prefix web run build
(cd web && npx playwright test)
uv run pytest -q
```

Expected: all UI tests pass, the build produces `web/dist/index.html`, Playwright completes a mocked start/status/end lifecycle, and Python tests remain green.

- [ ] **Step 8: Commit**

```bash
git add scripts/sync-personaplex-client.sh web src/tolkien_voice/main.py
git commit -m "feat: add Tolkien Voice browser client"
```

## Task 8: Add safe llama-swap integration and rollback

**Files:**

- Create: `scripts/patch-llama-swap-config.py`
- Create: `scripts/install-spark.sh`
- Create: `scripts/rollback-spark.sh`
- Create: `tests/fixtures/llama-swap-config.yaml`
- Create: `tests/test_patch_llama_swap.py`
- Create: `docs/operations/spark-install.md`
- Create: `docs/operations/rollback.md`

- [ ] **Step 1: Capture the expected live configuration as a sanitized fixture**

The fixture must include the six route IDs, `always` and `big` groups, current proxies, commands, aliases, health checks, and startup preload. It must not contain secrets. The patcher recognizes exactly these heavy models:

```python
BIG_MODELS = {"coder", "coder-fp8", "qwen36", "gemma", "tolkien"}
```

- [ ] **Step 2: Write failing patcher tests**

Assert that the patcher:

- leaves `embed` byte-for-byte semantically unchanged and in persistent group `always`;
- leaves all aliases, proxies, model names, and health timeouts unchanged;
- sets `ttl: 60` on each big model;
- wraps each big command once with `/home/chrisknuteson/tolkien-voice/scripts/run-with-big-model-lease.sh` followed by its exact route ID (`coder`, `coder-fp8`, `qwen36`, `gemma`, or `tolkien`);
- is idempotent;
- refuses unknown commands, a missing route, a changed group membership, or an already-running big model;
- produces a timestamped backup path before apply;
- supports `--check`, `--apply`, and `--restore` followed by a validated backup path under `/home/chrisknuteson/inference/llama-swap/`.

- [ ] **Step 3: Confirm red**

Run: `uv run pytest tests/test_patch_llama_swap.py -q`

Expected: FAIL because the patcher does not exist.

- [ ] **Step 4: Implement a round-trip YAML patcher**

Add `ruamel-yaml>=0.18,<1` to the main project dependencies so the installed deployment environment can preserve comments and formatting. Refuse to apply unless `GET /running` reports no member of `BIG_MODELS`; never call the unload API from the installer. Back up to:

```text
/home/chrisknuteson/inference/llama-swap/config.yaml.bak.tolkien-voice-YYYYMMDDTHHMMSSZ
```

Wrap commands as argument arrays passed through the lease script, not through `eval`. Validate the generated YAML against llama-swap by launching the installed binary on a temporary loopback port with a temporary config and immediately terminating it after `/health` succeeds. Only then atomically replace the live file. Because llama-swap uses `-watch-config`, poll `/v1/models` and confirm all six route IDs remain available.

- [ ] **Step 5: Implement the installer**

`scripts/install-spark.sh` must be rerunnable and perform, in order:

1. Verify hostname, user, GPU access, Docker access, free disk >= 100 GiB, free port 8998, and the pinned submodule.
2. Refuse if the Hugging Face secret is missing or has unsafe permissions.
3. Refuse if a big llama-swap model is running.
4. Create data directories with modes `0700`; caches may be `0750`.
5. Build the browser and PersonaPlex image.
6. Run the config patcher in `--check`, then `--apply` mode.
7. Install and enable the user service from Task 9.
8. Configure Tailscale Serve only after loopback health succeeds.
9. Run the network smoke test.

Never use `sudo`, delete model/cache data, stop `vllm_embed`, or reset all Tailscale Serve state.

- [ ] **Step 6: Implement rollback**

Rollback stops and disables only `tolkien-voice.service`, removes only its Tailscale Serve mapping with the supported Tailscale configuration command, stops only `tolkien-voice-personaplex`, restores the selected timestamped llama-swap backup, and verifies `embed` plus the five original heavy routes. It must leave the PersonaPlex image and caches intact for recoverability.

- [ ] **Step 7: Verify and commit**

```bash
uv lock
uv run pytest tests/test_patch_llama_swap.py -q
shellcheck scripts/install-spark.sh scripts/rollback-spark.sh
git add pyproject.toml uv.lock scripts tests docs/operations
git commit -m "feat: integrate Spark model scheduling"
```

## Task 9: Add systemd, Tailscale HTTPS, and operational health

**Files:**

- Create: `systemd/tolkien-voice.service`
- Create: `scripts/configure-tailscale-serve.sh`
- Create: `scripts/smoke-test.sh`
- Create: `tests/test_service_contract.py`
- Modify: `docs/operations/spark-install.md`

- [ ] **Step 1: Write failing service contract tests**

Assert the unit:

- runs as the unprivileged current user;
- binds loopback through config;
- uses `RuntimeDirectory=tolkien-voice` and mode `0700`;
- uses `Restart=on-failure` with a bounded delay;
- has `UMask=0077`;
- never grants access to a public interface;
- stops the managed PersonaPlex container through `ExecStop`;
- does not include the Hugging Face token in `Environment` or `EnvironmentFile`.

- [ ] **Step 2: Confirm red**

Run: `uv run pytest tests/test_service_contract.py -q`

Expected: FAIL because the unit does not exist.

- [ ] **Step 3: Add the user service**

Use:

```ini
[Unit]
Description=Tolkien Voice native speech controller
After=network-online.target docker.service tailscaled.service

[Service]
Type=simple
WorkingDirectory=/home/chrisknuteson/tolkien-voice
ExecStart=/home/chrisknuteson/tolkien-voice/.venv/bin/tolkien-voice
ExecStop=/usr/bin/docker stop --time 20 tolkien-voice-personaplex
Restart=on-failure
RestartSec=5
RuntimeDirectory=tolkien-voice
RuntimeDirectoryMode=0700
UMask=0077
Environment=TOLKIEN_VOICE_CONFIG=/home/chrisknuteson/tolkien-voice/configs/runtime.yaml

[Install]
WantedBy=default.target
```

Treat Docker's “no such container” exit during `ExecStop` as success through a dedicated idempotent stop script rather than prefixing the command with a shell.

- [ ] **Step 4: Configure private HTTPS**

`scripts/configure-tailscale-serve.sh` must first assert `tailscale status --json` reports BackendState `Running`, the node DNS name is `spark-1250.tail71b73b.ts.net.`, and `curl http://127.0.0.1:8788/api/health` succeeds. Then run:

```bash
tailscale serve --bg --yes 8788
```

Capture `tailscale serve get-config` before mutation for rollback. Confirm `tailscale serve status --json` points HTTPS to `http://127.0.0.1:8788`. Never enable Funnel.

- [ ] **Step 5: Add smoke tests**

`scripts/smoke-test.sh` must check:

- controller health on loopback and `https://spark-1250.tail71b73b.ts.net/api/health`;
- all nine character IDs and Sauron's restricted voice list;
- PersonaPlex is absent while idle;
- `vllm_embed` is running and `POST /v1/embeddings` succeeds through llama-swap;
- no process listens publicly on 8788 or 8998;
- a synthetic start without opening audio reaches `ready`, then DELETE returns to `idle` and removes the PersonaPlex container;
- no `.wav`, `.webm`, or transcript file appears under the data root.

The synthetic start is skipped with a clear prerequisite result if gated weights are not available; all non-model checks still run.

- [ ] **Step 6: Verify and commit**

```bash
uv run pytest tests/test_service_contract.py -q
shellcheck scripts/configure-tailscale-serve.sh scripts/smoke-test.sh
git add systemd scripts tests docs/operations/spark-install.md
git commit -m "feat: add private Spark deployment service"
```

## Task 10: Add duplex and persona evaluation harnesses

**Files:**

- Create: `evals/persona/probes.yaml`
- Create: `evals/duplex/scenarios.yaml`
- Create: `src/tolkien_voice/evaluation.py`
- Create: `scripts/run-offline-eval.sh`
- Create: `scripts/summarize-eval.py`
- Create: `tests/test_evaluation.py`
- Create: `docs/operations/acceptance-test.md`

- [ ] **Step 1: Write failing evaluation-manifest tests**

Assert every character has at least:

- two lore questions;
- one uncertainty trap;
- one break-character request;
- one modern-topic redirect;
- one concise conversational response probe.

Assert duplex scenarios include interruption at 500 ms and 1,000 ms after agent speech begins, a two-second mid-sentence pause, overlap, silence, and reconnect. Reject any manifest that points inside `configs/characters`, preventing training/evaluation overlap later.

- [ ] **Step 2: Confirm red**

Run: `uv run pytest tests/test_evaluation.py -q`

Expected: FAIL because manifests and loader do not exist.

- [ ] **Step 3: Implement manifests and summary schema**

Store only prompts, timing instructions, expected traits, and numeric observations in Git. Generated WAV and transcript outputs go under `${data_root}/eval-runs/`, in a UTC timestamp plus UUID directory such as `20260722T041500Z-3f84e8b2`, never the repository.

The summary JSON contains:

```json
{
  "character": "gandalf",
  "voice": "NATM1.pt",
  "source_revision": "3428dfd95309a7f3c84fd93259ded0f810d1ff91",
  "first_audio_ms": 0,
  "interruption_response_ms": 0,
  "persona_pass_rate": 0.0,
  "lore_pass_rate": 0.0,
  "voice_consistent": false,
  "human_notes": ""
}
```

Zero values mean unmeasured and must not be interpreted as passing. The summarizer fails the acceptance gate unless measured interruption response is <= 1,000 ms and both five-minute sessions are marked complete.

- [ ] **Step 4: Add the human acceptance runbook**

Use the exact Gandalf and Sauron procedure from the approved design: five minutes each, interrupt speech, pause mid-sentence, ask grounded and obscure lore, request a character break, introduce a modern topic, and verify no cross-character leakage. Include `nvidia-smi`, `docker stats`, browser timestamps, session status, cleanup, and coder-after-cleanup observations.

- [ ] **Step 5: Verify and commit**

```bash
uv run pytest tests/test_evaluation.py -q
shellcheck scripts/run-offline-eval.sh
git add evals src/tolkien_voice/evaluation.py scripts tests docs/operations/acceptance-test.md
git commit -m "test: add persona and duplex acceptance harness"
```

## Task 11: Full verification, Spark deployment, and measured acceptance

**Files:**

- Modify: `README.md`
- Create: `docs/operations/baseline-results.md`

- [ ] **Step 1: Run the complete local gate**

```bash
git submodule status
uv sync --extra dev --locked
uv run ruff check src tests scripts
uv run pytest --cov=tolkien_voice --cov-report=term-missing -q
npm --prefix web ci
npm --prefix web run lint
npm --prefix web test -- --run
npm --prefix web run build
(cd web && npx playwright test)
shellcheck container/*.sh scripts/*.sh
git diff --check
```

Expected: exact pinned submodule, all tests/builds pass, no ShellCheck findings, and no whitespace errors.

- [ ] **Step 2: Push the implementation before deployment**

```bash
git push origin main
```

Expected: private GitHub `main` contains all reviewed source and tests.

- [ ] **Step 3: Complete the gated-model prerequisite**

Stop here if the NVIDIA license has not been accepted or the fine-grained token is not installed. Follow `docs/operations/huggingface-access.md`; do not request the token in chat and do not paste it into a command that enters shell history.

- [ ] **Step 4: Deploy without interrupting a running model**

From the local clone:

```bash
rsync -a --delete-delay --exclude '.git/' --exclude 'secrets/' ./ \
  chrisknuteson@100.98.79.30:/home/chrisknuteson/tolkien-voice/
ssh chrisknuteson@100.98.79.30 \
  'cd /home/chrisknuteson/tolkien-voice && ./scripts/install-spark.sh'
```

Expected: the installer refuses safely if any `big` model is running. Retry only after its normal TTL releases it; never force-stop an active inference.

- [ ] **Step 5: Run network smoke tests**

```bash
ssh chrisknuteson@100.98.79.30 \
  '/home/chrisknuteson/tolkien-voice/scripts/smoke-test.sh'
curl -fsS https://spark-1250.tail71b73b.ts.net/api/health
```

Expected: controller and embedding checks pass, browser URL has a valid Tailscale certificate, PersonaPlex cold-starts and cleans up, and ports 8788/8998 remain loopback-only.

- [ ] **Step 6: Perform the human Gandalf and Sauron acceptance test**

Open `https://spark-1250.tail71b73b.ts.net`, follow `docs/operations/acceptance-test.md`, and record measured results in `docs/operations/baseline-results.md`. Do not commit audio. Include cold-start time, warm-start time, first-audio latency, interruption response, five-minute stability, peak unified memory, persona failures, lore failures, cleanup time, and the coder request result after cleanup.

- [ ] **Step 7: Exercise contention in both directions**

1. Start a coder request, then Start Conversation. Confirm voice waits and the coder finishes normally.
2. Start a voice conversation, then request `model=coder`. Confirm the coder request waits and speech continues uninterrupted.
3. End voice. Confirm the waiting coder starts and completes.
4. Confirm `embed` stayed healthy throughout.

- [ ] **Step 8: Run privacy and cleanup checks**

```bash
ssh chrisknuteson@100.98.79.30 \
  'find /home/chrisknuteson/tolkien-voice-data -type f \( -name "*.wav" -o -name "*.webm" -o -iname "*transcript*" \) -print'
```

Expected: no output outside an explicitly created evaluation run directory.

- [ ] **Step 9: Commit measured results and final documentation**

Update `README.md` with the private URL, start/end behavior, cold-start expectation, active-session resource contract, supported characters, and links to operations documents. Record failures honestly; do not claim the Phase 1 persona baseline is distilled.

```bash
git add README.md docs/operations/baseline-results.md
git commit -m "docs: record Tolkien Voice baseline"
git push origin main
```

## Final definition of done

- The private browser page starts and ends speech sessions without shell access.
- Gandalf and Sauron each complete a measured five-minute native duplex conversation.
- Measured interruption response is at most one second.
- Character and voice remain stable within each session and do not leak between sessions.
- An active coder request is never killed for voice, and an active voice session is never killed for coder.
- `embed` remains available throughout.
- Explicit end and five-minute disconnected idle timeout remove the PersonaPlex container and release the lease.
- A coder request succeeds after speech cleanup.
- No interactive audio is persisted by default.
- The service is reachable through authenticated Tailscale HTTPS only.
- Installation and rollback procedures are tested and documented.
- Baseline results identify the exact prompt-only failures that Phase 2 distillation must address.
