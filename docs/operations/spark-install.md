# Spark installation

Tolkien Voice is installed only on `spark-1250`, as the unprivileged `chrisknuteson` user, at `/home/chrisknuteson/tolkien-voice`. The installer is deliberately fail-closed: it will not evict a running text model or voice session, stop the persistent embedding server, change public listeners, or install with an unvalidated llama-swap configuration.

## Prerequisites

1. Accept the NVIDIA PersonaPlex license and install the fine-grained credential using [huggingface-access.md](huggingface-access.md). Never paste the credential into an argument or chat.
2. Ensure at least 100 GiB is free under `/home/chrisknuteson`.
3. Clone this private repository at the exact install path and initialize the pinned submodule:

   ```bash
   git submodule update --init --recursive
   ```

4. Wait for any active coder, Qwen, Gemma, Tolkien text, or Tolkien Voice session to finish. The installer checks both llama-swap and the shared kernel lock and exits without unloading either workload.

## Install

Run from the repository:

```bash
./scripts/install-spark.sh
```

The ordered installation gate verifies the machine, GPU, Docker, disk, ports, source revision, gated-model access, and empty heavy-model slot. It creates private data directories, builds the browser and pinned runtime image, validates and backs up llama-swap, enables the user service, exposes only private Tailscale HTTPS, and runs a smoke test.

The llama-swap backup path is printed during install. Record it for rollback. Backups are named `config.yaml.bak.tolkien-voice-YYYYMMDDTHHMMSSZ` beside the live llama-swap configuration.

The controller binds only `127.0.0.1:8788`; PersonaPlex binds only `127.0.0.1:8998` while a conversation is active. The installer captures the pre-install Serve configuration under `/home/chrisknuteson/tolkien-voice-data/tailscale/`, refuses to replace a nonempty unrecognized mapping, and maps private tailnet HTTPS to the controller. It never creates an internet-facing mapping.

### Current localhost tunnel fallback

On the 2026-07-22 deployment, Tailscale refused Serve changes until the machine owner grants the unprivileged account operator access. No existing Serve configuration was changed. Until that one-time host action is approved, open the microphone-capable dashboard from the workstation through a localhost tunnel:

```bash
ssh -fNT -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 8788:127.0.0.1:8788 chrisknuteson@100.98.79.30
open http://localhost:8788/
```

The eventual one-time host action is:

```bash
sudo tailscale set --operator=chrisknuteson
```

Afterward, rerun `./scripts/configure-tailscale-serve.sh`. Do not provide a sudo password to this project or paste it into chat.

## Routine checks

```bash
systemctl --user status tolkien-voice.service
curl -fsS http://127.0.0.1:8788/api/health
tailscale serve status
./scripts/smoke-test.sh
```

PersonaPlex is expected to be absent while idle. The persistent `vllm_embed` container remains ready, and only one heavy text or speech runtime may own the shared lease.

The smoke test validates the nine-character public registry, Sauron's single allowed voice, embedding service continuity, loopback-only listeners, and cleanup. If gated weights have not yet been cached, it reports a clear skip for the synthetic voice startup while still enforcing all non-model checks.
