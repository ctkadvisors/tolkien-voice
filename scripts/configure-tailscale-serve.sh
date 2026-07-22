#!/usr/bin/env bash
set -Eeuo pipefail

readonly expected_dns=spark-1250.tail71b73b.ts.net.
readonly controller=http://127.0.0.1:8788
readonly state_dir=/home/chrisknuteson/tolkien-voice-data/tailscale
readonly baseline=$state_dir/serve-before-tolkien-voice.json

assert_tailnet_identity() {
  tailscale status --json | python3 -c '
import json, sys
status = json.load(sys.stdin)
if status.get("BackendState") != "Running":
    raise SystemExit("Tailscale backend is not running")
if status.get("Self", {}).get("DNSName") != "spark-1250.tail71b73b.ts.net.":
    raise SystemExit("Unexpected Tailscale node identity")
'
}

serve_state() {
  tailscale serve status --json
}

state_is_empty() {
  serve_state | python3 -c 'import json, sys; raise SystemExit(0 if not json.load(sys.stdin) else 1)'
}

state_points_to_controller() {
  serve_state | python3 -c '
import json, sys
payload = json.load(sys.stdin)
values = []
def walk(value):
    if isinstance(value, dict):
        for child in value.values():
            walk(child)
    elif isinstance(value, list):
        for child in value:
            walk(child)
    elif isinstance(value, str):
        values.append(value.rstrip("/"))
walk(payload)
raise SystemExit(0 if "http://127.0.0.1:8788" in values else 1)
'
}

assert_tailnet_identity

if [[ ${1:-} == --remove ]]; then
  if state_is_empty; then
    exit 0
  fi
  if ! state_points_to_controller; then
    echo "Refusing to remove an unrecognized Tailscale Serve configuration." >&2
    exit 1
  fi
  tailscale serve --https=443 off
  if state_points_to_controller; then
    echo "Tolkien Voice Tailscale mapping is still active." >&2
    exit 1
  fi
  echo "Removed the Tolkien Voice private HTTPS mapping."
  exit 0
fi

if (( $# != 0 )); then
  echo "usage: configure-tailscale-serve.sh [--remove]" >&2
  exit 64
fi

curl -fsS "$controller/api/health" >/dev/null
install -d -m 0700 "$state_dir"

if state_points_to_controller; then
  echo "Tolkien Voice private HTTPS mapping is already configured."
  exit 0
fi
if ! state_is_empty; then
  echo "Refusing to replace an existing Tailscale Serve configuration." >&2
  exit 1
fi
if [[ ! -e "$baseline" ]]; then
  tailscale serve get-config "$baseline" --all
  chmod 0600 "$baseline"
fi

tailscale serve --bg --yes 8788
if ! state_points_to_controller; then
  echo "Tailscale Serve did not point HTTPS at $controller." >&2
  exit 1
fi

echo "Tolkien Voice is available at https://${expected_dns%.}/"
