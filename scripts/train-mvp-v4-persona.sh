#!/usr/bin/env bash
set -Eeuo pipefail

[[ $# -eq 1 ]] || {
  echo "usage: $0 aragorn|sauron" >&2
  exit 2
}
readonly character=$1
[[ "$character" = aragorn || "$character" = sauron ]] || {
  echo "unsupported persona: $character" >&2
  exit 2
}

readonly repository=/home/chrisknuteson/tolkien-voice
readonly data_root=/home/chrisknuteson/tolkien-voice-data
readonly source_dataset="$data_root/datasets/mvp-v1"
readonly persona_dataset="$data_root/datasets/mvp-v4-persona/$character"
readonly run_name="mvp-v4-$character-r64-384step"
readonly run_root="$data_root/runs/$run_name"
readonly config="configs/training/mvp-v4-$character.yaml"
readonly model_cache="$data_root/cache/huggingface/hub/models--nvidia--personaplex-7b-v1"
readonly final_adapter="$run_root/checkpoints/checkpoint_000384/consolidated/lora.safetensors"
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
for source in \
  "$source_dataset/dialogues.jsonl" \
  "$source_dataset/manifests/train.jsonl" \
  "$source_dataset/manifests/validation.jsonl"; do
  [[ -s "$source" ]] || {
    echo "MVP source corpus is missing: $source" >&2
    exit 66
  }
done

PYTHONPATH="$repository/src" "$python_runtime" -m tolkien_voice.training.audit \
  --blueprints "$source_dataset/dialogues.jsonl" \
  --minimum-per-character 64 \
  --characters aragorn sauron
PYTHONPATH="$repository/src" "$python_runtime" -m tolkien_voice.training.conditioning \
  --blueprints "$source_dataset/dialogues.jsonl" \
  --audio-directory "$source_dataset/audio" \
  --character-directory "$repository/configs/characters"
PYTHONPATH="$repository/src" "$python_runtime" -m tolkien_voice.training.persona_manifest \
  --blueprints "$source_dataset/dialogues.jsonl" \
  --source-train "$source_dataset/manifests/train.jsonl" \
  --source-eval "$source_dataset/manifests/validation.jsonl" \
  --character "$character" \
  --output-directory "$persona_dataset"

mkdir -p "$data_root/runs"
scripts/run-with-big-model-lease.sh "tolkien-sts-train-mvp-v4-$character" \
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
      --config "/workspace/$config"

[[ -s "$final_adapter" ]] || {
  echo "MVP v4 $character training did not reach the expected final checkpoint" >&2
  exit 70
}
selected_adapter=$(
  PYTHONPATH="$repository/src" "$python_runtime" \
    -m tolkien_voice.training.select_checkpoint \
    --run-root "$run_root" \
    --adapter-path-only
)
[[ -s "$selected_adapter" ]] || {
  echo "MVP v4 $character checkpoint selection returned no adapter" >&2
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
