import hashlib
import json
from pathlib import Path

from tolkien_voice.training.run_metadata import write_run_metadata


def test_run_metadata_binds_adapter_base_revisions_and_metrics(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    adapter = run_root / "lora.safetensors"
    adapter.write_bytes(b"adapter bytes")
    metrics = run_root / "metrics.train.jsonl"
    metrics.write_text(
        json.dumps({"step": 1, "loss": 3.482, "peak_allocated_mem": 16.8, "wps": 265.6})
        + "\n"
    )

    path = write_run_metadata(
        run_root=run_root,
        adapter_path=adapter,
        base_revision="base-revision",
        personaplex_revision="personaplex-revision",
        moshi_revision="moshi-revision",
        trainer_revision="trainer-revision",
    )

    payload = json.loads(path.read_text())
    assert payload["adapter_sha256"] == hashlib.sha256(b"adapter bytes").hexdigest()
    assert payload["base_revision"] == "base-revision"
    assert payload["personaplex_revision"] == "personaplex-revision"
    assert payload["moshi_revision"] == "moshi-revision"
    assert payload["trainer_revision"] == "trainer-revision"
    assert payload["metrics"] == {
        "step": 1,
        "loss": 3.482,
        "peak_allocated_mem": 16.8,
        "wps": 265.6,
    }
    assert path.stat().st_mode & 0o777 == 0o600
