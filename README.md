# Tolkien Voice

Tolkien Voice is a private, native full-duplex speech experience built on NVIDIA PersonaPlex. Phase 1 conditions one shared PersonaPlex checkpoint with versioned role and included voice prompts for Gandalf, Sauron, and the initial character roster.

This baseline does **not** contain a Tolkien fine-tune or LoRA. The existing Tolkien 27B model will be used later as an offline distillation teacher after prompt-only speech behavior is measured.

The live path is browser audio → PersonaPlex/Moshi → browser audio. It does not insert ASR, a text LLM, or TTS between the speaker and the character.

See the [approved design](docs/superpowers/specs/2026-07-21-tolkien-voice-design.md) and [Phase 1 implementation plan](docs/superpowers/plans/2026-07-21-tolkien-voice.md).
