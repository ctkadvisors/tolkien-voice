#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /external/data-root/dialogues.jsonl" >&2
  exit 2
fi

destination=$1
case "$destination" in
  "$PWD"/*)
    echo "refusing to write generated training data inside the repository" >&2
    exit 2
    ;;
esac

exec uv run python -m tolkien_voice.training.generate \
  --base-url http://127.0.0.1:9090/v1 \
  --model tolkien \
  --output "$destination" \
  --characters aragorn sauron
