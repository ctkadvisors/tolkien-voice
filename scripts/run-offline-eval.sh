#!/usr/bin/env bash
set -Eeuo pipefail

readonly data_root=${TOLKIEN_VOICE_DATA_ROOT:-/home/chrisknuteson/tolkien-voice-data}

curl -fsS http://127.0.0.1:8788/api/health >/dev/null
run_dir=$(python -m tolkien_voice.evaluation \
  --data-root "$data_root" \
  --characters gandalf sauron)

echo "Created private acceptance run: $run_dir"
echo "Follow docs/operations/acceptance-test.md, then fill the two summary JSON files."
echo "Summarize with: ./scripts/summarize-eval.py $run_dir/*-summary.json"
