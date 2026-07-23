from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "training" / "mvp-v1.yaml"
SCRIPT = ROOT / "scripts" / "train-mvp.sh"
V2_CONFIG = ROOT / "configs" / "training" / "mvp-v2.yaml"
V2_SCRIPT = ROOT / "scripts" / "train-mvp-v2.sh"
V3_CONFIG = ROOT / "configs" / "training" / "mvp-v3.yaml"
V3_SCRIPT = ROOT / "scripts" / "train-mvp-v3.sh"
V4_SCRIPT = ROOT / "scripts" / "train-mvp-v4-persona.sh"
V4_CONFIGS = {
    character: ROOT / "configs" / "training" / f"mvp-v4-{character}.yaml"
    for character in ("aragorn", "sauron")
}


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


def test_mvp_v2_training_config_uses_longer_context_and_lower_lr() -> None:
    payload = yaml.safe_load(V2_CONFIG.read_text())

    assert payload["data"]["train_data"] == "/data/datasets/mvp-v1/manifests/train.jsonl"
    assert payload["data"]["eval_data"] == "/data/datasets/mvp-v1/manifests/validation.jsonl"
    assert payload["duration_sec"] == 32
    assert payload["max_steps"] == 256
    assert payload["lora"]["rank"] == 32
    assert payload["lora"]["ft_embed"] is False
    assert payload["optim"]["lr"] == 5.0e-7
    assert payload["text_padding_weight"] == 0.1
    assert payload["first_codebook_weight_multiplier"] == 100.0
    assert payload["batch_size"] == 1
    assert payload["gradient_checkpointing"] is True
    assert payload["param_dtype"] == "bfloat16"
    assert payload["do_eval"] is True
    assert payload["eval_freq"] == 64
    assert payload["ckpt_freq"] == 64
    assert payload["num_ckpt_keep"] >= 4
    assert payload["overwrite_run_dir"] is False
    assert payload["run_dir"] == "/data/runs/mvp-v2-r32-256step-32sec-r2"


def test_mvp_v2_script_audits_and_selects_best_heldout_checkpoint() -> None:
    source = V2_SCRIPT.read_text()

    assert "set -Eeuo pipefail" in source
    assert 'readonly dataset_root="$data_root/datasets/mvp-v1"' in source
    assert 'readonly run_name=mvp-v2-r32-256step-32sec-r2' in source
    assert "tolkien_voice.training.audit" in source
    assert "scripts/run-with-big-model-lease.sh" in source
    assert "--network none" in source
    assert "configs/training/mvp-v2.yaml" in source
    assert "checkpoint_000256/consolidated/lora.safetensors" in source
    assert "tolkien_voice.training.select_checkpoint" in source
    assert "selected-checkpoint-metadata.json" in source
    assert "active-adapter" not in source


def test_mvp_v3_training_is_prompt_conditioned_and_higher_capacity() -> None:
    payload = yaml.safe_load(V3_CONFIG.read_text())

    assert payload["data"]["train_data"] == "/data/datasets/mvp-v1/manifests/train.jsonl"
    assert payload["data"]["eval_data"] == "/data/datasets/mvp-v1/manifests/validation.jsonl"
    assert payload["duration_sec"] == 32
    assert payload["max_steps"] == 384
    assert payload["lora"]["rank"] == 64
    assert payload["lora"]["ft_embed"] is False
    assert payload["optim"]["lr"] == 1.0e-6
    assert payload["text_padding_weight"] == 0.1
    assert payload["do_eval"] is True
    assert payload["eval_freq"] == 96
    assert payload["ckpt_freq"] == 96
    assert payload["num_ckpt_keep"] >= 4
    assert payload["overwrite_run_dir"] is False
    assert payload["run_dir"] == "/data/runs/mvp-v3-r64-384step-32sec-conditioned"


def test_mvp_v3_script_annotates_runtime_prompts_and_selects_checkpoint() -> None:
    source = V3_SCRIPT.read_text()

    assert "set -Eeuo pipefail" in source
    assert 'readonly run_name=mvp-v3-r64-384step-32sec-conditioned' in source
    assert "tolkien_voice.training.conditioning" in source
    assert "--character-directory" in source
    assert "tolkien_voice.training.audit" in source
    assert "configs/training/mvp-v3.yaml" in source
    assert "checkpoint_000384/consolidated/lora.safetensors" in source
    assert "tolkien_voice.training.select_checkpoint" in source
    assert "selected-checkpoint-metadata.json" in source
    assert "active-adapter" not in source


def test_mvp_v4_configs_train_separate_weighted_persona_adapters() -> None:
    for character, path in V4_CONFIGS.items():
        payload = yaml.safe_load(path.read_text())
        root = f"/data/datasets/mvp-v4-persona/{character}"

        assert payload["data"]["train_data"] == f"{root}/train.jsonl"
        assert payload["data"]["eval_data"] == f"{root}/validation.jsonl"
        assert payload["duration_sec"] == 32
        assert payload["max_steps"] == 384
        assert payload["lora"]["rank"] == 64
        assert payload["optim"]["lr"] == 1.0e-6
        assert payload["eval_freq"] == 96
        assert payload["ckpt_freq"] == 96
        assert payload["run_dir"] == f"/data/runs/mvp-v4-{character}-r64-384step"


def test_mvp_v4_script_builds_weighted_manifests_and_never_activates() -> None:
    source = V4_SCRIPT.read_text()

    assert "set -Eeuo pipefail" in source
    assert '[[ "$character" = aragorn || "$character" = sauron ]]' in source
    assert "tolkien_voice.training.persona_manifest" in source
    assert "tolkien_voice.training.conditioning" in source
    assert 'configs/training/mvp-v4-$character.yaml' in source
    assert "checkpoint_000384/consolidated/lora.safetensors" in source
    assert "tolkien_voice.training.select_checkpoint" in source
    assert "selected-checkpoint-metadata.json" in source
    assert "active-adapter" not in source
