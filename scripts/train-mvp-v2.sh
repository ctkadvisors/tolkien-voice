#!/usr/bin/env bash
set -Eeuo pipefail

readonly repository=/home/chrisknuteson/tolkien-voice
readonly data_root=/home/chrisknuteson/tolkien-voice-data
readonly dataset_root="$data_root/datasets/mvp-v1"
readonly run_name=mvp-v2-r32-256step-32sec-r2
readonly run_root="$data_root/runs/$run_name"
readonly model_cache="$data_root/cache/huggingface/hub/models--nvidia--personaplex-7b-v1"
readonly final_adapter="$run_root/checkpoints/checkpoint_000256/consolidated/lora.safetensors"
readonly python_runtime="$repository/.venv/bin/python"

cd "$repository"
[[ ! -e "$run_root" ]] || {
  echo "training run already exists: $run_root" >&2
  exit 65
}
snapshot=$(find -L "$model_cache/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit)
[[ -n "$snapshot" && -s "$snapshot/model.safetensors" ]] || {
  echo "PersonaPlex checkpoint is not cached" >&2
  exit 66
}
for manifest in \
  "$dataset_root/manifests/train.jsonl" \
  "$dataset_root/manifests/validation.jsonl"; do
  [[ -s "$manifest" ]] || {
    echo "MVP training manifest is missing: $manifest" >&2
    exit 66
  }
done
PYTHONPATH="$repository/src" "$python_runtime" -m tolkien_voice.training.audit \
  --blueprints "$dataset_root/dialogues.jsonl" \
  --minimum-per-character 64 \
  --characters aragorn sauron

mkdir -p "$data_root/runs"
scripts/run-with-big-model-lease.sh tolkien-sts-train-mvp-v2 \
  docker run --rm \
    --gpus all \
    --ipc host \
    --network none \
    --user "$(id -u):$(id -g)" \
    --volume "$repository:/workspace:ro" \
    --volume "$data_root:/data" \
    --volume "$snapshot:/models:ro" \
    --volume "$model_cache/blobs:/blobs:ro" \
    --workdir /workspace \
    --env PYTHONPATH=/workspace/src:/opt/moshi-finetune \
    --env CUDA_VISIBLE_DEVICES=0 \
    tolkien-voice/trainer:2acc879 \
    torchrun --nproc-per-node 1 \
      -m tolkien_voice.training.personaplex_train \
      --config /workspace/configs/training/mvp-v2.yaml

[[ -s "$final_adapter" ]] || {
  echo "MVP v2 training did not reach the expected final checkpoint" >&2
  exit 70
}
selected_adapter=$(
  PYTHONPATH="$repository/src" "$python_runtime" \
    -m tolkien_voice.training.select_checkpoint \
    --run-root "$run_root" \
    --adapter-path-only
)
[[ -s "$selected_adapter" ]] || {
  echo "MVP v2 checkpoint selection returned no adapter" >&2
  exit 70
}
selected_checkpoint=$(basename "$(dirname "$(dirname "$selected_adapter")")")
selected_step=${selected_checkpoint#checkpoint_}
selected_step=$((10#$selected_step))

sha256sum "$selected_adapter" >"$run_root/selected-adapter.sha256"
chmod 0600 "$run_root/selected-adapter.sha256"
PYTHONPATH="$repository/src" "$python_runtime" -m tolkien_voice.training.run_metadata \
  --run-root "$run_root" \
  --adapter "$selected_adapter" \
  --base-revision "$(basename "$snapshot")" \
  --personaplex-revision 3428dfd95309a7f3c84fd93259ded0f810d1ff91 \
  --moshi-revision 061cc4c630d9e11722e08b7d02b1836ba58f30e8 \
  --trainer-revision 2acc879fe7c48f885a18f6cc9548bccb2674d87b \
  --metric-step "$selected_step" \
  --output "$run_root/selected-checkpoint-metadata.json"
