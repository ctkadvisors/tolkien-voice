#!/usr/bin/env bash
set -Eeuo pipefail

readonly expected_trainer=2acc879fe7c48f885a18f6cc9548bccb2674d87b
actual_trainer=$(git -C vendor/moshi-finetune rev-parse HEAD)
if [[ "$actual_trainer" != "$expected_trainer" ]]; then
  echo "Moshi-Finetune is $actual_trainer; expected $expected_trainer" >&2
  exit 65
fi

docker build \
  --file container/Dockerfile.train \
  --tag tolkien-voice/trainer:2acc879 \
  .
