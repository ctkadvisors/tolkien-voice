#!/usr/bin/env bash
set -Eeuo pipefail

readonly controller=http://127.0.0.1:8788
readonly private_url=https://spark-1250.tail71b73b.ts.net
readonly runtime_container=tolkien-voice-personaplex
readonly data_root=/home/chrisknuteson/tolkien-voice-data
readonly cache_root=$data_root/cache/huggingface

curl -fsS "$controller/api/health" >/dev/null
curl -fsS "$private_url/api/health" >/dev/null

python3 - <<'PY'
import json
from urllib.request import urlopen

with urlopen("http://127.0.0.1:8788/api/characters", timeout=10) as response:
    cards = json.load(response)["characters"]
expected = {
    "gandalf", "sauron", "galadriel", "samwise", "aragorn",
    "frodo", "treebeard", "boromir", "barliman",
}
if {card["id"] for card in cards} != expected:
    raise SystemExit("Character registry did not expose the expected nine IDs")
sauron = next(card for card in cards if card["id"] == "sauron")
if sauron["allowed_voices"] != ["NATM3.pt"]:
    raise SystemExit("Sauron voice restriction changed")
PY

if docker inspect "$runtime_container" >/dev/null 2>&1; then
  echo "PersonaPlex must be absent while the controller is idle." >&2
  exit 1
fi

curl -fsS http://127.0.0.1:9090/v1/embeddings \
  -H 'Content-Type: application/json' \
  --data '{"model":"embed","input":"Tolkien Voice smoke test"}' \
  | python3 -c 'import json, sys; assert json.load(sys.stdin)["data"][0]["embedding"]'
[[ $(docker inspect --format '{{.State.Status}}' vllm_embed) == running ]] || {
  echo "Persistent embedding runtime is not running." >&2
  exit 1
}

listeners=$(ss -H -ltn | awk '{print $4}')
for public_listener in \
  0.0.0.0:8788 0.0.0.0:8998 '[::]:8788' '[::]:8998' '*:8788' '*:8998'; do
  if grep -Fxq "$public_listener" <<<"$listeners"; then
    echo "Public listener detected: $public_listener" >&2
    exit 1
  fi
done

artifact_count() {
  find "$data_root" -type f \
    \( -iname '*.wav' -o -iname '*.webm' -o -iname '*transcript*' \) \
    -not -path "$data_root/eval-runs/*" -print | wc -l | tr -d ' '
}
readonly artifacts_before
artifacts_before=$(artifact_count)

if [[ ! -d "$cache_root/models--nvidia--personaplex-7b-v1" ]]; then
  echo "SKIP: PersonaPlex weights are not cached; non-model smoke checks passed."
  exit 0
fi

session_json=$(curl -fsS -X POST "$controller/api/session" \
  -H 'Content-Type: application/json' \
  --data '{"character_id":"gandalf","voice_id":"NATM1.pt"}')
session_id=$(python3 -c 'import json, sys; print(json.load(sys.stdin)["session_id"])' \
  <<<"$session_json")

cleanup_session() {
  curl -fsS -X DELETE "$controller/api/session/$session_id" >/dev/null 2>&1 || true
}
trap cleanup_session EXIT

state=waiting_for_lease
for _attempt in {1..180}; do
  state=$(curl -fsS "$controller/api/session" \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["state"])')
  case "$state" in
    ready) break ;;
    failed|idle)
      echo "Synthetic voice session failed before readiness: $state" >&2
      exit 1
      ;;
  esac
  sleep 5
done
[[ "$state" == ready ]] || {
  echo "Synthetic voice session did not reach ready." >&2
  exit 1
}

curl -fsS -X DELETE "$controller/api/session/$session_id" >/dev/null
trap - EXIT

for _attempt in {1..30}; do
  if ! docker inspect "$runtime_container" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if docker inspect "$runtime_container" >/dev/null 2>&1; then
  echo "PersonaPlex container remained after DELETE cleanup." >&2
  exit 1
fi

readonly artifacts_after
artifacts_after=$(artifact_count)
[[ "$artifacts_after" == "$artifacts_before" ]] || {
  echo "Unexpected audio or transcript artifact appeared under $data_root." >&2
  exit 1
}

echo "Tolkien Voice smoke test passed."
