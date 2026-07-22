from pathlib import Path

import yaml


REPOSITORY = Path(__file__).parents[1]


def test_training_image_pins_upstreams_without_replacing_ngc_torch() -> None:
    dockerfile = (REPOSITORY / "container/Dockerfile.train").read_text()

    assert dockerfile.startswith("FROM nvcr.io/nvidia/pytorch:25.11-py3")
    assert "2acc879fe7c48f885a18f6cc9548bccb2674d87b" in dockerfile
    assert "061cc4c630d9e11722e08b7d02b1836ba58f30e8" in dockerfile
    assert "torch==2.6" not in dockerfile
    assert "HF_TOKEN" not in dockerfile
    assert "hf_token" not in dockerfile
    assert "libopus-dev" in dockerfile


def test_smoke_config_is_one_small_lora_optimizer_step() -> None:
    config = yaml.safe_load((REPOSITORY / "configs/training/smoke.yaml").read_text())

    assert config["duration_sec"] == 12
    assert config["batch_size"] == 1
    assert config["max_steps"] == 1
    assert config["gradient_checkpointing"] is True
    assert config["param_dtype"] == "bfloat16"
    assert config["lora"] == {"enable": True, "rank": 8, "scaling": 2.0, "ft_embed": False}
    assert config["do_eval"] is False
    assert config["save_adapters"] is True
    assert config["run_dir"].startswith("/data/runs/")
    assert config["data"]["train_data"].startswith("/data/datasets/")


def test_training_script_uses_shared_lease_and_non_root_container_user() -> None:
    script = (REPOSITORY / "scripts/train-smoke.sh").read_text()

    assert "run-with-big-model-lease.sh tolkien-sts-train" in script
    assert '--user "$(id -u):$(id -g)"' in script
    assert "--gpus all" in script
    assert "--env CUDA_VISIBLE_DEVICES=0" in script
    assert "/home/chrisknuteson/tolkien-voice-data" in script
    assert "docker rm" not in script


def test_training_script_mounts_hugging_face_blobs_for_snapshot_symlinks() -> None:
    script = (REPOSITORY / "scripts/train-smoke.sh").read_text()

    assert '--volume "$model_cache/blobs:/blobs:ro"' in script


def test_training_image_builder_accepts_only_git_or_pinned_revision_marker() -> None:
    script = (REPOSITORY / "scripts/build-training-image.sh").read_text()
    marker = (REPOSITORY / "vendor/moshi-finetune-revision.txt").read_text().strip()

    assert marker == "2acc879fe7c48f885a18f6cc9548bccb2674d87b"
    assert "vendor/moshi-finetune-revision.txt" in script
    assert "Moshi-Finetune revision cannot be verified" in script
