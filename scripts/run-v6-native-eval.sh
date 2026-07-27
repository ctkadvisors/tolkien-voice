#!/usr/bin/env bash
# Orchestrates the sealed native STS A/B evaluation for the v6 adapters:
# prepares a blinded arm assignment, collects base and per-character adapter
# arms by starting/stopping PersonaPlex directly (bypassing the controller,
# same as collect-native-arm.py talks to it directly), scores with the local
# Tolkien judge, and applies the precommitted release gates. Mirrors the
# process described in docs/operations/distillation-results.md for v1-v4;
# no committed script previously tied these steps together end to end.
set -Eeuo pipefail

readonly repository=/home/chrisknuteson/tolkien-voice
readonly data_root=/home/chrisknuteson/tolkien-voice-data
readonly python_runtime="$repository/.venv/bin/python"
readonly run_name=${1:-mvp-v6-decoupled}
readonly eval_root="$data_root/eval-runs/$run_name"
readonly config_dir="$eval_root/configs"
readonly probe_audio_dir="$data_root/eval-inputs/mvp-v1"
readonly interruption_audio="$probe_audio_dir/barge-in.wav"

cd "$repository"
mkdir -p "$eval_root" "$config_dir"
chmod 0700 "$eval_root"

[[ -s "$interruption_audio" ]] || {
  echo "probe audio is missing: $probe_audio_dir" >&2
  exit 66
}

echo "=== render eval-runtime configs ===" >&2
"$python_runtime" -m tolkien_voice.training.eval_runtime \
  --source configs/runtime.yaml \
  --output "$config_dir/base.yaml"
"$python_runtime" -m tolkien_voice.training.eval_runtime \
  --source configs/runtime.yaml \
  --output "$config_dir/aragorn-adapter.yaml" \
  --run-root "$data_root/runs/mvp-v6-aragorn-decoupled-r64-384step"
"$python_runtime" -m tolkien_voice.training.eval_runtime \
  --source configs/runtime.yaml \
  --output "$config_dir/sauron-adapter.yaml" \
  --run-root "$data_root/runs/mvp-v6-sauron-decoupled-r64-384step"

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

collect_one() {
  local config=$1 arm=$2 output=$3
  shift 3
  echo "--- starting PersonaPlex: $config ---" >&2
  "$python_runtime" scripts/_personaplex_ctl.py start --config "$config"
  local cleanup_needed=1
  trap 'if [[ $cleanup_needed -eq 1 ]]; then echo "--- stopping PersonaPlex (cleanup) ---" >&2; "$python_runtime" scripts/_personaplex_ctl.py stop --config "$config" || true; fi' RETURN
  "$python_runtime" scripts/collect-native-arm.py \
    --audio-dir "$probe_audio_dir" \
    --interruption-audio "$interruption_audio" \
    --raw-audio-dir "$raw_audio_dir" \
    --arm "$arm" \
    --output "$output" \
    "$@"
  echo "--- stopping PersonaPlex ---" >&2
  "$python_runtime" scripts/_personaplex_ctl.py stop --config "$config"
  cleanup_needed=0
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
