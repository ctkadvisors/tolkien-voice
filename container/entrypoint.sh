#!/usr/bin/env bash
set -Eeuo pipefail

token_file=/run/secrets/hf_token
if [[ ! -f "$token_file" || ! -s "$token_file" ]]; then
  echo "PersonaPlex requires a non-empty Hugging Face token secret at $token_file" >&2
  exit 78
fi

token_mode=$(stat -c '%a' "$token_file")
if (( (8#$token_mode & 077) != 0 )); then
  echo "Hugging Face token secret must not be accessible by group or other users" >&2
  exit 78
fi

HF_TOKEN=
IFS= read -r HF_TOKEN <"$token_file" || [[ -n "$HF_TOKEN" ]]
if [[ -z "$HF_TOKEN" ]]; then
  echo "Hugging Face token secret is empty" >&2
  exit 78
fi
export HF_TOKEN

readonly pinned_base_revision=fdaf4090a61cb315c138a1faee287ffd6c716309
base_revision=${TOLKIEN_VOICE_BASE_REVISION:-$pinned_base_revision}
if [[ "$base_revision" != "$pinned_base_revision" ]]; then
  echo "PersonaPlex base revision does not match the pinned runtime" >&2
  exit 78
fi

server_args=(
  --host 127.0.0.1
  --port 8998
  --static none
  --hf-repo nvidia/personaplex-7b-v1
)
if [[ -z "${TOLKIEN_VOICE_LORA_WEIGHT:-}" || -z "${TOLKIEN_VOICE_LORA_SHA256:-}" ]]; then
  echo "A verified LoRA adapter and SHA-256 are required" >&2
  exit 78
fi

export HF_HUB_OFFLINE=1
exec python /opt/tolkien-runtime/personaplex_runtime.py "${server_args[@]}"
