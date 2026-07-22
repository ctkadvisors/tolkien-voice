#!/usr/bin/env bash
set -Eeuo pipefail

readonly container=tolkien-voice-personaplex

if docker inspect "$container" >/dev/null 2>&1; then
  docker stop --time 20 "$container" >/dev/null
fi
