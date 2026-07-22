#!/usr/bin/env bash
set -Eeuo pipefail

readonly expected_host=spark-1250
readonly expected_user=chrisknuteson
readonly expected_revision=3428dfd95309a7f3c84fd93259ded0f810d1ff91
readonly install_root=/home/chrisknuteson/tolkien-voice
readonly data_root=/home/chrisknuteson/tolkien-voice-data
readonly secret_file=$data_root/secrets/hf_token
readonly patcher=$install_root/scripts/patch-llama-swap-config.py

if [[ $(id -u) -eq 0 ]]; then
  echo "Install Tolkien Voice as $expected_user, not root." >&2
  exit 77
fi

if [[ $(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P) != "$install_root" ]]; then
  echo "Tolkien Voice must be installed at $install_root." >&2
  exit 78
fi

preflight_host() {
  [[ $(hostname -s) == "$expected_host" ]] || {
    echo "Expected host $expected_host." >&2
    return 1
  }
  [[ $(id -un) == "$expected_user" ]] || {
    echo "Expected user $expected_user." >&2
    return 1
  }
  command -v uv >/dev/null
  command -v npm >/dev/null
  command -v docker >/dev/null
  command -v nvidia-smi >/dev/null
  command -v flock >/dev/null
  command -v systemctl >/dev/null
  command -v tailscale >/dev/null
  nvidia-smi -L >/dev/null
  docker info >/dev/null

  # df reports KiB; require at least 100 GiB before weights and image layers.
  local required_kib=$((100 * 1024 * 1024))
  local available_kib
  available_kib=$(df -Pk "$data_root" 2>/dev/null | awk 'NR == 2 {print $4}')
  if [[ -z "$available_kib" ]]; then
    available_kib=$(df -Pk /home/chrisknuteson | awk 'NR == 2 {print $4}')
  fi
  (( available_kib >= required_kib )) || {
    echo "Tolkien Voice requires at least 100 GiB free." >&2
    return 1
  }

  if ss -H -ltn | awk '{print $4}' | grep -Eq '(^|:)8998$'; then
    echo "Loopback runtime port 8998 is already in use." >&2
    return 1
  fi
  [[ $(git -C "$install_root/vendor/personaplex" rev-parse HEAD) == "$expected_revision" ]] || {
    echo "PersonaPlex submodule revision is not pinned correctly." >&2
    return 1
  }
}

preflight_secret() {
  [[ -s "$secret_file" ]] || {
    echo "Missing gated-model credential; see docs/operations/huggingface-access.md." >&2
    return 1
  }
  [[ $(stat -c '%a' "$secret_file") == 600 ]] || {
    echo "The gated-model credential must have mode 0600." >&2
    return 1
  }
  python3 - "$secret_file" <<'PY'
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

token = Path(sys.argv[1]).read_text().strip()
request = Request(
    "https://huggingface.co/nvidia/personaplex-7b-v1/resolve/main/config.json",
    method="HEAD",
    headers={"Authorization": f"Bearer {token}"},
)
try:
    with urlopen(request, timeout=20) as response:
        if response.status not in {200, 302, 307}:
            raise RuntimeError(f"unexpected status {response.status}")
except (HTTPError, URLError, RuntimeError) as error:
    print("PersonaPlex license/token access check failed.", file=sys.stderr)
    raise SystemExit(1) from error
print("PersonaPlex gated-model access is available.")
PY
}

preflight_model_slot() {
  local runtime_dir=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/tolkien-voice
  install -d -m 0700 "$runtime_dir"
  exec 9>"$runtime_dir/big-model.lock"
  if ! flock --exclusive --nonblock 9; then
    echo "The Spark large-model slot is active; install deferred without interruption." >&2
    return 1
  fi
  "$patcher" --check
}

prepare_data_directories() {
  install -d -m 0700 "$data_root" "$data_root/secrets" "$data_root/logs" \
    "$data_root/eval-runs"
  install -d -m 0750 "$data_root/cache" "$data_root/cache/huggingface"
  chmod 0600 "$secret_file"
}

build_assets() {
  uv sync --extra dev --locked
  npm --prefix "$install_root/web" ci
  npm --prefix "$install_root/web" run build
  "$install_root/scripts/build-runtime.sh"
}

install_user_service() {
  install -d -m 0700 /home/chrisknuteson/.config/systemd/user
  install -m 0644 "$install_root/systemd/tolkien-voice.service" \
    /home/chrisknuteson/.config/systemd/user/tolkien-voice.service
  systemctl --user daemon-reload
  systemctl --user enable --now tolkien-voice.service
}

preflight_host
preflight_secret
preflight_model_slot
prepare_data_directories
build_assets
"$patcher" --check
"$patcher" --apply
flock --unlock 9
install_user_service
"$install_root/scripts/configure-tailscale-serve.sh"
"$install_root/scripts/smoke-test.sh"

echo "Tolkien Voice installation passed all checks."
