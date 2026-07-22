# PersonaPlex browser-client provenance

The audio protocol, AudioWorklet, Opus encoder/decoder, visualizers, and supporting hooks began from NVIDIA's PersonaPlex client at commit `3428dfd95309a7f3c84fd93259ded0f810d1ff91`. The upstream MIT license remains in `web/LICENSE` and in source headers where provided.

Tolkien Voice intentionally owns and preserves these adaptations when `scripts/sync-personaplex-client.sh` refreshes unchanged upstream files:

- application entry, API client, types, design system, and build/test configuration;
- character selector and session-state pages;
- conversation composition and media context;
- WebSocket lifecycle, reconnect behavior, server-audio hook, user-audio hook, and microphone component.

The changes remove the remote queue, free-form role prompt, unrestricted voices, remote worker selection, tuning controls, MediaRecorder, recording downloads, and the ten-second silence disconnect. The browser sends only a session ID to Tolkien Voice; role and voice prompts are resolved server-side.

