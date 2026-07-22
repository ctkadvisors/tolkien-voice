from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "training" / "mvp-v1.yaml"
SCRIPT = ROOT / "scripts" / "train-mvp.sh"


def test_mvp_training_config_is_not_a_smoke_run() -> None:
    payload = yaml.safe_load(CONFIG.read_text())

    assert payload["data"]["train_data"] == "/data/datasets/mvp-v1/manifests/train.jsonl"
    assert payload["data"]["eval_data"] == "/data/datasets/mvp-v1/manifests/validation.jsonl"
    assert payload["duration_sec"] == 12
    assert payload["max_steps"] >= 128
    assert payload["lora"]["rank"] >= 16
    assert payload["do_eval"] is True
    assert 0 < payload["eval_freq"] <= payload["max_steps"]
    assert payload["ckpt_freq"] == payload["eval_freq"]
    assert payload["num_ckpt_keep"] >= 3
    assert payload["overwrite_run_dir"] is False
    assert "smoke" not in payload["run_dir"]


def test_mvp_training_script_fails_closed_and_records_exact_adapter() -> None:
    source = SCRIPT.read_text()

    assert "set -Eeuo pipefail" in source
    assert 'readonly dataset_root="$data_root/datasets/mvp-v1"' in source
    assert '"$dataset_root/manifests/train.jsonl"' in source
    assert '"$dataset_root/manifests/validation.jsonl"' in source
    assert "tolkien_voice.training.audit" in source
    assert "scripts/run-with-big-model-lease.sh" in source
    assert "--network none" in source
    assert "configs/training/mvp-v1.yaml" in source
    assert "checkpoint_000192/consolidated/lora.safetensors" in source
    assert "run_metadata" in source
    assert "active-adapter" not in source
