#!/usr/bin/env bash
# Variant of run-v6-native-eval.sh pointed at the self-distillation-augmented
# adapters (mvp-v6-{aragorn,sauron}-self-distill-v1-r64-384step) instead of
# the original v6-only baseline adapters. Same sealed native STS A/B
# evaluation process, same precommitted release gates -- this checks whether
# the self-distillation-augmented checkpoints actually clear the gates the
# original v6-only baseline failed. Entirely separate output directory
# (eval-runs/mvp-v6-self-distill-v1 by default) so the original baseline's
# eval-runs/mvp-v6-decoupled results are never touched.
set -Eeuo pipefail

readonly repository=/home/chrisknuteson/tolkien-voice
readonly data_root=/home/chrisknuteson/tolkien-voice-data
readonly python_runtime="$repository/.venv/bin/python"
readonly run_name=${1:-mvp-v6-self-distill-v1}
readonly probe_set=${2:-mvp-v1}
readonly eval_root="$data_root/eval-runs/$run_name"
readonly config_dir="$eval_root/configs"
readonly probe_audio_dir="$data_root/eval-inputs/$probe_set"
readonly interruption_audio="$probe_audio_dir/barge-in.wav"

cd "$repository"
mkdir -p "$eval_root" "$config_dir"
chmod 0700 "$eval_root"

[[ -s "$interruption_audio" ]] || {
  echo "probe audio is missing: $probe_audio_dir" >&2
  exit 66
}

running_config=""
cleanup() {
  if [[ -n "$running_config" ]]; then
    echo "--- stopping PersonaPlex (cleanup): $running_config ---" >&2
    "$python_runtime" scripts/_personaplex_ctl.py stop --config "$running_config" || true
    running_config=""
  fi
}
trap cleanup EXIT

echo "=== render eval-runtime configs ===" >&2
"$python_runtime" -m tolkien_voice.training.eval_runtime \
  --source configs/runtime.yaml \
  --output "$config_dir/base.yaml"
"$python_runtime" -m tolkien_voice.training.eval_runtime \
  --source configs/runtime.yaml \
  --output "$config_dir/aragorn-adapter.yaml" \
  --run-root "$data_root/runs/mvp-v6-aragorn-self-distill-v1-r64-384step"
"$python_runtime" -m tolkien_voice.training.eval_runtime \
  --source configs/runtime.yaml \
  --output "$config_dir/sauron-adapter.yaml" \
  --run-root "$data_root/runs/mvp-v6-sauron-self-distill-v1-r64-384step"

echo "=== prepare blinded assignment ===" >&2
"$python_runtime" scripts/prepare-blind-eval.py --output-directory "$eval_root"
base_arm=$("$python_runtime" -c "
import json
key = json.load(open('$eval_root/blind-key.json'))
print(next(arm for arm, variant in key.items() if variant == 'base'))
")
adapter_arm=$("$python_runtime" -c "
import json
key = json.load(open('$eval_root/blind-key.json'))
print(next(arm for arm, variant in key.items() if variant == 'adapter'))
")
echo "base arm label: $base_arm" >&2
echo "adapter arm label: $adapter_arm" >&2

raw_audio_dir="$eval_root/audio"
mkdir -p "$raw_audio_dir"
chmod 0700 "$raw_audio_dir"

warm_up() {
  local config=$1
  echo "--- warming up: $config ---" >&2
  TOLKIEN_VOICE_CONFIG="$config" "$python_runtime" -c "
import asyncio
from pathlib import Path
from tolkien_voice.characters import CharacterRegistry
from tolkien_voice.training.native_probe import probe_native_session

async def main():
    registry = CharacterRegistry.from_directory('configs/characters')
    resolved = registry.resolve('aragorn')
    from urllib.parse import urlencode
    query = urlencode({'text_prompt': resolved.text_prompt, 'voice_prompt': resolved.voice_prompt})
    url = f'ws://127.0.0.1:8998/api/chat?{query}'
    await probe_native_session(
        url,
        Path('$probe_audio_dir/aragorn-concise.wav'),
        response_seconds=6,
    )

asyncio.run(main())
"
}

collect_one() {
  local config=$1 arm=$2 output=$3
  shift 3
  echo "--- starting PersonaPlex: $config ---" >&2
  "$python_runtime" scripts/_personaplex_ctl.py start --config "$config"
  running_config="$config"
  warm_up "$config"
  "$python_runtime" scripts/collect-native-arm.py \
    --audio-dir "$probe_audio_dir" \
    --interruption-audio "$interruption_audio" \
    --raw-audio-dir "$raw_audio_dir" \
    --arm "$arm" \
    --output "$output" \
    "$@"
  echo "--- stopping PersonaPlex ---" >&2
  "$python_runtime" scripts/_personaplex_ctl.py stop --config "$config"
  running_config=""
}

echo "=== collecting base arm (both characters) ===" >&2
collect_one "$config_dir/base.yaml" "$base_arm" "$eval_root/base-arm.jsonl" \
  --characters aragorn sauron

echo "=== collecting adapter arm: aragorn ===" >&2
collect_one "$config_dir/aragorn-adapter.yaml" "$adapter_arm" "$eval_root/adapter-arm-aragorn.jsonl" \
  --characters aragorn

echo "=== collecting adapter arm: sauron ===" >&2
collect_one "$config_dir/sauron-adapter.yaml" "$adapter_arm" "$eval_root/adapter-arm-sauron.jsonl" \
  --characters sauron

echo "=== scoring (sealed Tolkien judge) ===" >&2
"$python_runtime" scripts/score-native-results.py \
  --input "$eval_root/base-arm.jsonl" \
  --input "$eval_root/adapter-arm-aragorn.jsonl" \
  --input "$eval_root/adapter-arm-sauron.jsonl" \
  --assignments "$eval_root/assignments.jsonl" \
  --output "$eval_root/scored.jsonl"

echo "=== applying release gates ===" >&2
gate_status=0
"$python_runtime" scripts/report-ab-eval.py \
  --observations "$eval_root/scored.jsonl" \
  --blind-key "$eval_root/blind-key.json" \
  --output "$eval_root/report.json" || gate_status=$?

echo "=== report (exit reflects gate pass/fail, not a script error) ===" >&2
cat "$eval_root/report.json"
exit "$gate_status"
