#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# < 2 )); then
  echo "usage: run-with-big-model-lease.sh OWNER_NAME COMMAND [ARG...]" >&2
  exit 64
fi

owner_name=$1
shift
if [[ ! "$owner_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]; then
  echo "invalid lease owner name" >&2
  exit 64
fi

runtime_base=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
lease_dir="$runtime_base/tolkien-voice"
lock_file="$lease_dir/big-model.lock"
owner_file="$lease_dir/big-model-owner.json"
wait_seconds=${TOLKIEN_VOICE_LEASE_WAIT_SECONDS:-840}

umask 077
mkdir -p "$lease_dir"
exec 9>"$lock_file"

if ! flock --exclusive --wait "$wait_seconds" 9; then
  echo "Spark large-model slot is busy; an active Tolkien Voice conversation was not interrupted." >&2
  exit 75
fi

acquired_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
owner_tmp="$owner_file.tmp.$$"
printf '{"kind":"text","name":"%s","pid":%d,"acquired_at":"%s"}\n' \
  "$owner_name" "$$" "$acquired_at" >"$owner_tmp"
mv -f "$owner_tmp" "$owner_file"

cleanup() {
  if [[ -f "$owner_file" ]] && grep -q "\"pid\":$$" "$owner_file"; then
    rm -f "$owner_file"
  fi
}
trap cleanup EXIT

"$@"
