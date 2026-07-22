# Tolkien-Native STS Distillation MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce and evaluate a real LoRA checkpoint trained from Tolkien 27B v4 teacher conversations on top of the PersonaPlex 7B v1 native STS checkpoint.

**Architecture:** The existing Tolkien model generates short structured dialogue blueprints offline. A deterministic renderer turns them into licensed 24 kHz stereo conversations and timestamped transcripts, the pinned official Moshi trainer adapts the PersonaPlex temporal model with Mimi frozen, and the runtime loads the resulting LoRA for native full-duplex A/B evaluation.

**Tech Stack:** Python 3.12, Pydantic, HTTPX, NumPy, FFmpeg, macOS `say` for the private feasibility corpus, PyTorch/CUDA on GB10, NVIDIA PersonaPlex commit `3428dfd95309a7f3c84fd93259ded0f810d1ff91`, Kyutai Moshi-Finetune commit `2acc879fe7c48f885a18f6cc9548bccb2674d87b`, pytest, Docker.

---

## File map

- `src/tolkien_voice/training/schema.py`: validated blueprint, utterance, timing, provenance and rendered-example contracts.
- `src/tolkien_voice/training/teacher.py`: OpenAI-compatible Tolkien teacher client and strict JSON response parser.
- `src/tolkien_voice/training/generate.py`: balanced generation CLI with deterministic splits and leakage checks.
- `src/tolkien_voice/training/render.py`: TTS command execution, stereo timeline mixing and word alignment output.
- `src/tolkien_voice/training/manifest.py`: rendered-dataset validation and Moshi JSONL writer.
- `src/tolkien_voice/training/compatibility.py`: PersonaPlex config/checkpoint compatibility report.
- `configs/training/personaplex-7b-v1.json`: full architecture consumed by Moshi-Finetune.
- `configs/training/smoke.yaml`: tiny LoRA configuration generated with absolute data/run paths.
- `container/Dockerfile.train`: reproducible GB10 training image with pinned upstreams.
- `scripts/generate-dialogues.sh`, `scripts/render-dataset.sh`, `scripts/train-smoke.sh`: guarded operator entrypoints.
- `tests/test_training_*.py`: unit/contract coverage for every transformation.
- `evals/sts/held-out.yaml`: immutable Aragorn/Sauron control and duplex probes.

## Task 1: Make trained STS the repository contract

**Files:**

- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-07-21-tolkien-voice-design.md`
- Modify: `docs/superpowers/plans/2026-07-21-tolkien-voice.md`
- Create: `docs/superpowers/specs/2026-07-22-tolkien-sts-distillation-mvp.md`
- Create: `docs/superpowers/plans/2026-07-22-tolkien-sts-distillation-mvp.md`

- [ ] Mark the prompt-only design and plan as superseded.
- [ ] State that prompt-only PersonaPlex is the control arm, never the MVP.
- [ ] Put the optimizer-step, adapter-load and A/B gates in the README.
- [ ] Run `rg -n "Phase 1|baseline|LoRA|trained" README.md docs/superpowers` and verify no active document calls the prompt-only demo an MVP.
- [ ] Commit with `docs: restore trained STS as first MVP`.

## Task 2: Define fail-closed dialogue and dataset contracts

**Files:**

- Create: `src/tolkien_voice/training/__init__.py`
- Create: `src/tolkien_voice/training/schema.py`
- Create: `tests/test_training_schema.py`

- [ ] Write failing tests proving a valid blueprint accepts alternating short turns, and rejects missing teacher revision, unknown characters, empty text, assistant monologues over 45 words, invalid timing kinds and held-out hash leakage.
- [ ] Run `uv run pytest tests/test_training_schema.py -q`; expect import failure.
- [ ] Implement `DialogueBlueprint`, `Utterance`, `Provenance`, `RenderedExample` and `Split` models. Compute canonical SHA-256 hashes from normalized utterance text and expose `assert_no_leakage(train, held_out)`.
- [ ] Run the focused test, then `uv run pytest -q` and `uv run ruff check src tests`.
- [ ] Commit with `feat: define Tolkien STS dataset contracts`.

## Task 3: Generate dialogues from Tolkien 27B

**Files:**

- Create: `src/tolkien_voice/training/teacher.py`
- Create: `src/tolkien_voice/training/generate.py`
- Create: `tests/test_training_teacher.py`
- Create: `tests/test_training_generate.py`
- Create: `evals/sts/held-out.yaml`
- Create: `scripts/generate-dialogues.sh`

- [ ] Write failing HTTP transport tests for a successful strict-JSON response, malformed JSON, timeout, wrong character, excessive response length and retries that preserve the generation seed.
- [ ] Implement an injected HTTPX transport targeting `/v1/chat/completions`, default model `tolkien`, temperature `0.7`, and response format requesting JSON. Never put source-book passages into prompts; use the teacher checkpoint's learned behavior plus the versioned character card.
- [ ] Implement balanced generation for Aragorn and Sauron across `normal`, `pause`, `overlap` and `interruption`. Reserve held-out scenario hashes before generation and reject collisions.
- [ ] Write JSONL atomically beneath the external data root with mode `0600`; include teacher model `tolkien-27b-v4-q8`, card hash, seed and template revision.
- [ ] Run focused tests, all Python tests and Ruff.
- [ ] Commit with `feat: generate STS dialogues from Tolkien teacher`.

## Task 4: Render private synthetic duplex audio

**Files:**

- Create: `src/tolkien_voice/training/render.py`
- Create: `src/tolkien_voice/training/manifest.py`
- Create: `tests/test_training_render.py`
- Create: `tests/test_training_manifest.py`
- Create: `scripts/render-dataset.sh`

- [ ] Write failing tests using a fake TTS executable that emits deterministic mono PCM. Cover channel placement, 24 kHz resampling, pause insertion, overlap, interruption truncation, clipping rejection and word timestamps.
- [ ] Implement a configurable subprocess TTS backend. The feasibility command is `/usr/bin/say -o {output} --data-format=LEF32@24000 {text}` with distinct installed voices for user, Aragorn and Sauron. Record `private-apple-system-voice` provenance and prohibit publishing the generated audio.
- [ ] Mix assistant audio into the left channel and user audio into the right channel using NumPy; write PCM16 stereo WAV and adjacent Moshi alignment JSON with `SPEAKER_MAIN` on assistant words.
- [ ] Validate sample rate, two channels, finite samples, peak below 0.99, non-silent channels, unique IDs, adjacent JSON and split isolation before emitting `train.jsonl` and `validation.jsonl`.
- [ ] Run focused tests, the full Python suite and Ruff.
- [ ] Commit with `feat: render Tolkien duplex training audio`.

## Task 5: Pin and prove PersonaPlex training compatibility

**Files:**

- Modify: `.gitmodules`
- Modify: `vendor/README.md`
- Add submodule: `vendor/moshi-finetune`
- Create: `configs/training/personaplex-7b-v1.json`
- Create: `src/tolkien_voice/training/compatibility.py`
- Create: `tests/test_training_compatibility.py`

- [ ] Add `kyutai-labs/moshi-finetune` at commit `2acc879fe7c48f885a18f6cc9548bccb2674d87b`.
- [ ] Write a failing config test requiring the exact PersonaPlex dimensions from the pinned NVIDIA loader: `dim=4096`, `num_layers=32`, `n_q=16`, `dep_q=16`, 17 stream delays, 32,000 text tokens and the published model/Mimi/tokenizer filenames.
- [ ] Implement a metadata-only compatibility command that creates the LoRA-shaped model on `meta`, compares every non-LoRA parameter key and shape with `model.safetensors`, and emits JSON. Any missing or mismatched base tensor exits non-zero.
- [ ] Run the unit test locally and the metadata probe inside the training container on Spark.
- [ ] Commit with `feat: prove PersonaPlex LoRA compatibility`.

## Task 6: Execute a one-step GB10 training proof

**Files:**

- Create: `container/Dockerfile.train`
- Create: `configs/training/smoke.yaml`
- Create: `scripts/build-training-image.sh`
- Create: `scripts/train-smoke.sh`
- Create: `tests/test_training_container_contract.py`

- [ ] Write failing contract tests requiring a pinned NGC base, pinned Moshi and trainer revisions, no credentials in layers, a non-root output mount and the shared large-model lease wrapper.
- [ ] Build a training image without replacing the NGC PyTorch build. Install Moshi and Moshi-Finetune without downgrading Torch, then run their import tests during the image build.
- [ ] Generate `smoke.yaml` with `duration_sec=12`, `batch_size=1`, `max_steps=1`, gradient checkpointing, BF16, LoRA rank 8, `do_eval=false`, `save_adapters=true` and external absolute paths.
- [ ] Acquire the shared lease, run the compatibility probe, execute one optimizer step, and save the adapter atomically. Capture elapsed time, peak unified memory, tokens/sec, loss and exact revisions in `run-metadata.json`.
- [ ] Verify a non-empty `lora.safetensors` exists and contains LoRA tensors; do not mark the active runtime adapter yet.
- [ ] Commit with `feat: add GB10 Tolkien STS training proof`.

## Task 7: Load the trained adapter in native PersonaPlex inference

**Files:**

- Modify: `vendor/personaplex/moshi/moshi/models/loaders.py`
- Modify: `vendor/personaplex/moshi/moshi/server.py`
- Modify: `container/entrypoint.sh`
- Modify: `src/tolkien_voice/runtime.py`
- Create: `tests/test_adapter_runtime_contract.py`

- [ ] Write failing tests requiring `--lora-weight`, an allowlisted adapter path beneath the external data root, a base-revision match and adapter identity in `/api/health`.
- [ ] Port the official Moshi LoRA replacement/loading behavior into the pinned PersonaPlex fork with the smallest reviewed patch; preserve voice and role prompt support.
- [ ] Refuse partial/mismatched adapter loads. Log hashes and revisions, never tensor contents or credentials.
- [ ] Start native inference with the smoke adapter and complete the existing synthetic binary WebSocket round trip.
- [ ] Run Python, web and container contract suites.
- [ ] Commit with `feat: serve Tolkien LoRA through native STS`.

## Task 8: A/B gate the adapter instead of grading a vibe

**Files:**

- Create: `src/tolkien_voice/training/ab_eval.py`
- Create: `tests/test_training_ab_eval.py`
- Modify: `src/tolkien_voice/evaluation.py`
- Create: `scripts/evaluate-adapter.sh`
- Modify: `docs/operations/acceptance-test.md`

- [ ] Write failing tests for randomized base/adapter labels, paired result joins, confidence intervals, persona confusion, interruption latency and underrun regression gates.
- [ ] Run the same held-out audio probes through unchanged PersonaPlex and the adapter. Save transcripts, timing and audio diagnostics outside Git.
- [ ] Use Tolkien 27B v4 as a blinded offline judge for persona/lore dimensions and report deterministic objective metrics separately.
- [ ] Fail the release unless the adapter exists, loads, improves held-out persona/lore, stays below 5% character confusion, interrupts within 1,000 ms and remains within the 10% latency/underrun budget.
- [ ] Run all tests and write the exact measured report to `docs/operations/training-smoke-results.md` without claiming more than the sample supports.
- [ ] Commit with `feat: gate Tolkien STS adapter against base`.

## Task 9: Repair the human audio path and run acceptance

**Files:**

- Modify: `web/src/pages/Conversation/hooks/useUserAudio.ts`
- Modify: `web/src/pages/Conversation/hooks/useServerAudio.ts`
- Modify: `web/src/pages/Conversation/components/UserAudio/UserAudio.tsx`
- Modify: `web/src/pages/Conversation/components/Controls/Controls.tsx`
- Create: `web/src/pages/Conversation/hooks/useUserAudio.test.ts`
- Create: `web/src/pages/Conversation/hooks/useServerAudio.test.ts`

- [ ] Write failing browser tests for one selected microphone stream shared by meter and encoder, visible live RMS/peak, recorder-start failure, input-device switching, explicit barge-in and bounded jitter buffering.
- [ ] Implement the minimal capture/playout corrections and expose diagnostics in the conversation UI.
- [ ] Run Vitest, production build and Playwright; then deploy through the existing guarded Spark install path.
- [ ] Complete five-minute Aragorn and Sauron human sessions against both base and adapter and record the release-gate report.
- [ ] Commit with `fix: make trained STS conversation measurable and interruptible`.

## Execution rule

Tasks run inline in order. A failed PersonaPlex compatibility probe stops the
training run and produces a concrete incompatibility report; it does not
authorize replacing the requested trained model with prompt conditioning.
