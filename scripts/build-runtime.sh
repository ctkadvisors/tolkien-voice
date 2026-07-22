#!/usr/bin/env bash
set -Eeuo pipefail

readonly expected_revision=3428dfd95309a7f3c84fd93259ded0f810d1ff91
actual_revision=$(git -C vendor/personaplex rev-parse HEAD)
if [[ "$actual_revision" != "$expected_revision" ]]; then
  echo "PersonaPlex submodule is $actual_revision; expected $expected_revision" >&2
  exit 65
fi

docker build \
  --file container/Dockerfile.personaplex \
  --tag tolkien-voice/personaplex:3428dfd \
  .

