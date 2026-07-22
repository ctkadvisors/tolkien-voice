#!/usr/bin/env bash
set -Eeuo pipefail

readonly repository=/home/chrisknuteson/tolkien-voice
readonly data_root=/home/chrisknuteson/tolkien-voice-data
readonly run_name=smoke-one-step
readonly run_root="$data_root/runs/$run_name"
readonly model_cache="$data_root/cache/huggingface/hub/models--nvidia--personaplex-7b-v1"

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
[[ -s "$data_root/datasets/smoke/manifests/train.jsonl" ]] || {
  echo "smoke training manifest is missing" >&2
  exit 66
}

mkdir -p "$data_root/runs"
scripts/run-with-big-model-lease.sh tolkien-sts-train \
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
      --config /workspace/configs/training/smoke.yaml

adapter="$run_root/checkpoints/checkpoint_000001/consolidated/lora.safetensors"
[[ -s "$adapter" ]] || {
  echo "one-step training did not produce a LoRA adapter" >&2
  exit 70
}
sha256sum "$adapter" >"$run_root/adapter.sha256"
chmod 0600 "$run_root/adapter.sha256"
PYTHONPATH="$repository/src" python3 -m tolkien_voice.training.run_metadata \
  --run-root "$run_root" \
  --adapter "$adapter" \
  --base-revision "$(basename "$snapshot")" \
  --personaplex-revision 3428dfd95309a7f3c84fd93259ded0f810d1ff91 \
  --moshi-revision 061cc4c630d9e11722e08b7d02b1836ba58f30e8 \
  --trainer-revision 2acc879fe7c48f885a18f6cc9548bccb2674d87b
