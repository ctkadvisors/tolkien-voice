# Phase 1 baseline results

Captured on 2026-07-22 before the Tolkien Voice deployment mutation.

## Spark state

| Observation | Baseline |
| --- | --- |
| Host | `spark-1250` |
| GPU | NVIDIA GB10 |
| NVIDIA driver | 580.159.03 |
| Kernel | 6.17.0-1026-nvidia |
| Unified system memory | 121 GiB total, 112 GiB available |
| Root NVMe | 3.7 TiB total, 2.8 TiB available |
| llama-swap | active on port 9090 |
| Running llama-swap route | `embed` only, ready and persistent |
| Heavy text routes advertised | `coder`, `coder-fp8`, `qwen36`, `gemma`, `tolkien` |
| Tailscale Serve config | empty |
| Voice ports 8788/8998 | no listener |
| PersonaPlex credential | absent |
| PersonaPlex cached weights | absent |

The existing `vllm_embed` container was up and healthy. Its Docker-published port 8201 was bound on all host interfaces before this project; Tolkien Voice does not change that pre-existing exposure. The new controller and speech runtime are separately constrained to loopback.

## Local release gate

Before deployment, the clean release gate recorded:

- Python: 73 passed, one macOS-only `flock` integration skip, 84% line coverage.
- Web unit tests: 8 passed across three Vitest files.
- Browser lifecycle: one Playwright/Chrome flow passed.
- Production build: Vite 8.1.5 completed successfully.
- Static checks: Ruff, ESLint, TypeScript, ShellCheck, and whitespace validation passed.
- Dependency audit: zero npm vulnerabilities, including development tooling.

## Speech runtime bring-up

Captured on 2026-07-22 after installing the gated credential and pinned PersonaPlex image:

| Observation | Measured result |
| --- | --- |
| Cold start, including first checkpoint download | 6 minutes 45 seconds |
| Cached warm start | 1 minute 40 seconds |
| Hugging Face cache after download | 16 GiB |
| PersonaPlex steady-state GPU-backed memory | 19,063 MiB |
| Persistent embedding GPU-backed memory | 4,748 MiB |
| System memory available with both loaded | approximately 90 GiB |
| Gandalf controller-to-PersonaPlex handshake | passed, native binary kind `0` |
| Sauron controller-to-PersonaPlex handshake | passed, native binary kind `0` |
| Gandalf synthetic speech round trip | 90 native audio frames / 20,074 bytes and 29 text frames from a 9-second input window |
| Sauron prompt routing | Sauron persona plus `NATM3.pt` confirmed in runtime log |
| Text-route contention during voice | correctly blocked with exit `75`; voice was not evicted |
| Voice cleanup | exact PersonaPlex container removed, lease released, `vllm_embed` remained running |

The first private evaluation run was created at `/home/chrisknuteson/tolkien-voice-data/eval-runs/20260722T164658Z-3638b8dc` with separate Gandalf and Sauron summaries.

The synthetic round trip streamed six seconds of the included service speech fixture followed by three seconds of silence. It completed in 10.3 seconds and generated both Opus audio and text frames; the decoded text identified the speaker as Gandalf. This proves speech-in/speech-out inference through the controller, but not subjective audio quality.

First-audio latency, interruption response, five-minute stability, persona pass rate, lore pass rate, and voice consistency remain **unmeasured** until a human speaks through both sessions. Protocol and synthetic checks do not substitute for listening acceptance.

Zero values in generated evaluation summaries mean unmeasured and cannot pass acceptance. The first measured baseline must follow [acceptance-test.md](acceptance-test.md) for both Gandalf and Sauron and be stored under `/home/chrisknuteson/tolkien-voice-data/eval-runs/`, outside Git.

## Acceptance targets

- Gandalf and Sauron each complete a measured 300-second full-duplex session.
- Interruption response is measured and no greater than 1,000 ms.
- Two-second mid-sentence pauses do not cause premature takeover.
- Character and selected voice remain stable without cross-character leakage.
- Ending the session removes PersonaPlex and releases the shared large-model lease.
- Existing coder inference can start after speech cleanup without the embedding service being stopped.
- No session audio or transcript is persisted by default.
