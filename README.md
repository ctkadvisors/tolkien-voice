# Tolkien Voice

Tolkien Voice is a private, native full-duplex speech experience built on NVIDIA PersonaPlex. Phase 1 conditions one shared PersonaPlex checkpoint with versioned role and included voice prompts for Gandalf, Sauron, and the initial character roster.

This baseline does **not** contain a Tolkien fine-tune or LoRA. The existing Tolkien 27B model will be used later as an offline distillation teacher after prompt-only speech behavior is measured.

The live path is browser audio → PersonaPlex/Moshi → browser audio. It does not insert ASR, a text LLM, or TTS between the speaker and the character.

See the [approved design](docs/superpowers/specs/2026-07-21-tolkien-voice-design.md) and [Phase 1 implementation plan](docs/superpowers/plans/2026-07-21-tolkien-voice.md).

## Phase 1 capabilities

- Nine server-owned character cards with restricted included PersonaPlex voices; Sauron is fixed to `NATM3.pt`.
- One private, full-duplex speech session at a time through the upstream PersonaPlex binary audio protocol.
- A shared kernel lease with llama-swap's five heavy text routes, so a voice session waits rather than killing coder/Qwen/Gemma/Tolkien inference.
- Idle cleanup after five disconnected minutes, exact-container shutdown, and no session audio or transcript persistence.
- Loopback controller/runtime listeners exposed to the tailnet through private HTTPS only.
- Held-out persona and duplex evaluation manifests for the future Tolkien-teacher distillation phase.

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

The install script is fail-closed and must run as `chrisknuteson` on `spark-1250`. It never unloads an active heavy model, stops `vllm_embed`, resets all Tailscale Serve state, or requests a credential in chat. Model weights and the credential remain prerequisites; source installation alone does not produce a working speech session.

## License

[MIT](LICENSE) © 2026 Chris Knuteson. Keep the copyright and license notice in copies or substantial portions.

`web/` is adapted from the upstream PersonaPlex client and keeps its own [MIT license](web/LICENSE). The `vendor/` submodules carry their upstream licenses, and the `nvidia/personaplex-7b-v1` weights are distributed separately under the NVIDIA Open Model License. Tolkien and the character names belong to their owners; this is an unofficial fan project.
