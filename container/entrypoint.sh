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

exec python -m moshi.server \
  --host 127.0.0.1 \
  --port 8998 \
  --static none \
  --hf-repo nvidia/personaplex-7b-v1
