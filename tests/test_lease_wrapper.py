import os
from pathlib import Path
import shutil
import subprocess
import time

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run-with-big-model-lease.sh"


def test_wrapper_has_bounded_wait_and_safe_busy_message() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "TOLKIEN_VOICE_LEASE_WAIT_SECONDS:-840" in source
    assert "flock --exclusive --wait" in source
    assert "an active Tolkien Voice conversation was not interrupted" in source
    assert "eval" not in source


@pytest.mark.skipif(shutil.which("flock") is None, reason="util-linux flock is tested on Spark")
def test_wrapper_contends_and_recovers(tmp_path: Path) -> None:
    environment = {**os.environ, "XDG_RUNTIME_DIR": str(tmp_path)}
    first = subprocess.Popen(
        ["bash", str(SCRIPT), "coder", "sleep", "1"],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    owner_path = tmp_path / "tolkien-voice" / "big-model-owner.json"
    for _ in range(100):
        if owner_path.exists():
            break
        time.sleep(0.01)

    blocked = subprocess.run(
        ["bash", str(SCRIPT), "qwen36", "true"],
        env={**environment, "TOLKIEN_VOICE_LEASE_WAIT_SECONDS": "0"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert blocked.returncode == 75
    assert (
        blocked.stderr.strip()
        == "Spark large-model slot is busy; an active Tolkien Voice conversation was not interrupted."
    )
    assert first.wait(timeout=3) == 0
    recovered = subprocess.run(
        ["bash", str(SCRIPT), "tolkien", "true"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert recovered.returncode == 0
