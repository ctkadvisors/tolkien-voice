import json
from pathlib import Path

from tolkien_voice.training.compatibility import compare_parameter_shapes


REPOSITORY = Path(__file__).parents[1]


def test_personaplex_training_config_matches_pinned_nvidia_loader() -> None:
    config = json.loads(
        (REPOSITORY / "configs/training/personaplex-7b-v1.json").read_text()
    )

    assert "version" not in config
    assert config["model_type"] == "personaplex"
    assert config["moshi_name"] == "model.safetensors"
    assert config["mimi_name"] == "tokenizer-e351c8d8-checkpoint125.safetensors"
    assert config["tokenizer_name"] == "tokenizer_spm_32k_3.model"
    assert config["dim"] == 4096
    assert config["num_layers"] == 32
    assert config["n_q"] == 16
    assert config["dep_q"] == 16
    assert config["text_card"] == 32_000
    assert config["card"] == 2048
    assert len(config["delays"]) == 17


def test_shape_comparison_maps_frozen_lora_weights_to_base_names() -> None:
    expected = {
        "transformer.layer.weight": (4, 4),
        "text_linear.frozen_W.weight": (32_000, 4),
        "text_linear.lora_A.weight": (8, 4),
        "text_linear.lora_B.weight": (32_000, 8),
    }
    checkpoint = {
        "transformer.layer.weight": (4, 4),
        "text_linear.weight": (32_000, 4),
    }

    report = compare_parameter_shapes(expected, checkpoint)

    assert report.compatible is True
    assert report.missing == ()
    assert report.unexpected == ()
    assert report.mismatched == ()
    assert report.lora_tensor_count == 2


def test_shape_comparison_reports_missing_unexpected_and_mismatched() -> None:
    expected = {
        "a.weight": (2, 2),
        "b.weight": (3, 3),
    }
    checkpoint = {
        "a.weight": (2, 4),
        "c.weight": (1, 1),
    }

    report = compare_parameter_shapes(expected, checkpoint)

    assert report.compatible is False
    assert report.missing == ("b.weight",)
    assert report.unexpected == ("c.weight",)
    assert report.mismatched == ("a.weight: expected (2, 2), checkpoint (2, 4)",)


def test_shape_comparison_accepts_legacy_fused_attention_weights() -> None:
    expected = {
        "depformer.layers.0.self_attn.in_projs.0.frozen_W.weight": (12, 4),
        "depformer.layers.0.self_attn.in_projs.1.frozen_W.weight": (12, 4),
        "depformer.layers.0.self_attn.out_projs.0.frozen_W.weight": (4, 4),
        "depformer.layers.0.self_attn.out_projs.1.frozen_W.weight": (4, 4),
        "depformer.layers.0.self_attn.in_projs.0.lora_A.weight": (2, 4),
        "depformer.layers.0.self_attn.in_projs.0.lora_B.weight": (12, 2),
    }
    checkpoint = {
        "depformer.layers.0.self_attn.in_proj_weight": (24, 4),
        "depformer.layers.0.self_attn.out_proj.weight": (8, 4),
    }

    report = compare_parameter_shapes(expected, checkpoint)

    assert report.compatible is True
    assert report.missing == ()
    assert report.unexpected == ()
    assert report.mismatched == ()
