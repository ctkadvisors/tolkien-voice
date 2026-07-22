#!/usr/bin/env bash
set -Eeuo pipefail

readonly expected_trainer=2acc879fe7c48f885a18f6cc9548bccb2674d87b
readonly revision_marker=vendor/moshi-finetune-revision.txt
if actual_trainer=$(git -C vendor/moshi-finetune rev-parse HEAD 2>/dev/null); then
  :
elif [[ -r "$revision_marker" ]]; then
  IFS= read -r actual_trainer <"$revision_marker"
else
  echo "Moshi-Finetune revision cannot be verified" >&2
  exit 65
fi
if [[ "$actual_trainer" != "$expected_trainer" ]]; then
  echo "Moshi-Finetune is $actual_trainer; expected $expected_trainer" >&2
  exit 65
fi

docker build \
  --file container/Dockerfile.train \
  --tag tolkien-voice/trainer:2acc879 \
  .
