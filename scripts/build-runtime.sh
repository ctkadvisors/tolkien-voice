#!/usr/bin/env bash
set -Eeuo pipefail

readonly expected_revision=3428dfd95309a7f3c84fd93259ded0f810d1ff91
readonly revision_marker=vendor/personaplex-revision.txt
if actual_revision=$(git -C vendor/personaplex rev-parse HEAD 2>/dev/null); then
  :
elif [[ -r "$revision_marker" ]]; then
  IFS= read -r actual_revision <"$revision_marker"
else
  echo "PersonaPlex revision cannot be verified." >&2
  exit 65
fi
if [[ "$actual_revision" != "$expected_revision" ]]; then
  echo "PersonaPlex submodule is $actual_revision; expected $expected_revision" >&2
  exit 65
fi
[[ -f vendor/personaplex/moshi/pyproject.toml ]] || {
  echo "PersonaPlex source is incomplete." >&2
  exit 65
}

docker build \
  --file container/Dockerfile.personaplex \
  --tag tolkien-voice/personaplex:3428dfd \
  .
