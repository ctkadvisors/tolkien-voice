# Rollback

Rollback is scoped to Tolkien Voice. It stops and disables only `tolkien-voice.service`, removes only its Tailscale Serve mapping, stops only `tolkien-voice-personaplex`, and restores a validated Tolkien Voice llama-swap backup. Images, model caches, evaluation data, and the persistent embedding runtime are retained.

Choose the backup printed by the installer, then run:

```bash
./scripts/rollback-spark.sh \
  /home/chrisknuteson/inference/llama-swap/config.yaml.bak.tolkien-voice-YYYYMMDDTHHMMSSZ
```

The restore command accepts only a timestamped backup in `/home/chrisknuteson/inference/llama-swap/`. It refuses to proceed while a heavy model is active, validates the selected configuration with the installed llama-swap binary, atomically restores it, and confirms all six routes remain advertised.

After rollback:

```bash
curl -fsS http://127.0.0.1:9090/v1/models
docker inspect --format '{{.State.Status}}' vllm_embed
tailscale serve status
```

If rollback refuses because a model is active, let that workload finish and rerun it. Do not manually unload the model.
