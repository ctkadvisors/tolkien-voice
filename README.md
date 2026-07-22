# Tolkien Voice

Tolkien Voice is a private, native full-duplex speech model distilled from the
existing Tolkien 27B v4 teacher. The first accepted MVP is a trained LoRA on
the PersonaPlex/Moshi STS backbone that beats the unchanged checkpoint on
held-out Tolkien persona and lore evaluations while preserving duplex timing.

The repository now contains the prompt-only PersonaPlex control arm, a private
teacher-generated duplex corpus pipeline, a GB10 LoRA trainer, and a trained
one-step adapter that loads into native PersonaPlex inference. This proves the
requested architecture end to end, but it is **not yet the accepted MVP**:
the smoke adapter has not passed held-out base-vs-adapter persona/lore or human
duplex quality gates. The corrective design and implementation plan are
[Tolkien-native STS distillation](docs/superpowers/specs/2026-07-22-tolkien-sts-distillation-mvp.md)
and its [execution plan](docs/superpowers/plans/2026-07-22-tolkien-sts-distillation-mvp.md).

The live path is browser audio → PersonaPlex/Moshi → browser audio. It does not insert ASR, a text LLM, or TTS between the speaker and the character.

The previous prompt-only design and plan are retained as a record of the
runtime plumbing, but are explicitly superseded.

## Existing control-arm capabilities

- Nine server-owned character cards with restricted included PersonaPlex voices; Sauron is fixed to `NATM3.pt`.
- One private, full-duplex speech session at a time through the upstream PersonaPlex binary audio protocol.
- A shared kernel lease with llama-swap's five heavy text routes, so a voice session waits rather than killing coder/Qwen/Gemma/Tolkien inference.
- Idle cleanup after five disconnected minutes, exact-container shutdown, and no session audio or transcript persistence.
- Loopback controller/runtime listeners, with private Tailscale HTTPS support and a localhost SSH-tunnel fallback.
- Held-out persona and duplex evaluation manifests that will gate the Tolkien-teacher distillation adapter.

## Local development

Python 3.12, Node 22, `uv`, and npm are required.

```bash
git submodule update --init --recursive
UV_CACHE_DIR=/tmp/uv-cache-tolkien uv sync --extra dev --locked
UV_CACHE_DIR=/tmp/uv-cache-tolkien uv run pytest -q
npm --prefix web ci
npm --prefix web run dev
```

The controller can be run after building the browser:

```bash
npm --prefix web run build
UV_CACHE_DIR=/tmp/uv-cache-tolkien uv run tolkien-voice
```

PersonaPlex startup additionally requires NVIDIA GPU Docker access and the gated model credential described in [Hugging Face access](docs/operations/huggingface-access.md).

## Spark operations

- [Install and routine checks](docs/operations/spark-install.md)
- [Human speech acceptance test](docs/operations/acceptance-test.md)
- [Rollback](docs/operations/rollback.md)
- [Measured baseline](docs/operations/baseline-results.md)
- [Training smoke and adapter-load results](docs/operations/training-smoke-results.md)

The install script is fail-closed and must run as `chrisknuteson` on `spark-1250`. It never unloads an active heavy model, stops `vllm_embed`, resets all Tailscale Serve state, or requests a credential in chat. Model weights and the credential remain prerequisites; source installation alone does not produce a working speech session.

The current Spark has the pinned checkpoint cached, completed a real LoRA
optimizer step, strictly loaded the resulting adapter, and passed an
adapter-backed native binary audio round trip. See the measured reports for
the exact evidence and limitations. Human conversational latency and persona
quality are intentionally not claimed until the A/B and private speech
acceptance runs are completed.
