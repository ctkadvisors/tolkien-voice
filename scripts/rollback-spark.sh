#!/usr/bin/env bash
set -Eeuo pipefail

readonly install_root=/home/chrisknuteson/tolkien-voice
readonly service=tolkien-voice.service
readonly container=tolkien-voice-personaplex
readonly patcher=$install_root/scripts/patch-llama-swap-config.py

if (( $# != 1 )); then
  echo "usage: rollback-spark.sh LLAMA_SWAP_BACKUP" >&2
  exit 64
fi
readonly backup=$1

systemctl --user stop "$service" 2>/dev/null || true
systemctl --user disable "$service" 2>/dev/null || true
"$install_root/scripts/configure-tailscale-serve.sh" --remove

if docker inspect "$container" >/dev/null 2>&1; then
  docker stop --time 20 "$container" >/dev/null
fi

"$patcher" --restore "$backup"

echo "Tolkien Voice rollback complete. Model images and caches were retained."
