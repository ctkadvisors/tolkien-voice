#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 /external/dialogues.jsonl /external/rendered-directory /data/container-audio-root" >&2
  exit 2
fi

blueprints=$1
output_directory=$2
manifest_audio_root=$3
case "$output_directory" in
  "$PWD"/*)
    echo "refusing to write generated training audio inside the repository" >&2
    exit 2
    ;;
esac

exec uv run python -m tolkien_voice.training.render \
  --blueprints "$blueprints" \
  --output-directory "$output_directory" \
  --manifest-audio-root "$manifest_audio_root" \
  --workers 4
