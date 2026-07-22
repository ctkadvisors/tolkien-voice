from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Mapping


ShapeMap = Mapping[str, tuple[int, ...]]


@dataclass(frozen=True)
class CompatibilityReport:
    compatible: bool
    base_tensor_count: int
    lora_tensor_count: int
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    mismatched: tuple[str, ...]


def compare_parameter_shapes(expected_lora_model: ShapeMap, checkpoint: ShapeMap) -> CompatibilityReport:
    lora_tensor_count = sum(
        ".lora_A." in name or ".lora_B." in name for name in expected_lora_model
    )
    expected_base: dict[str, tuple[int, ...]] = {}
    for name, shape in expected_lora_model.items():
        if ".lora_A." in name or ".lora_B." in name:
            continue
        canonical = name.replace(".frozen_W.", ".")
        expected_base[canonical] = tuple(shape)

    expected_names = set(expected_base)
    checkpoint_names = set(checkpoint)
    resolved_expected = expected_names & checkpoint_names
    consumed_checkpoint = set(resolved_expected)
    mismatched_items = [
        f"{name}: expected {expected_base[name]}, checkpoint {tuple(checkpoint[name])}"
        for name in sorted(resolved_expected)
        if expected_base[name] != tuple(checkpoint[name])
    ]

    split_pattern = re.compile(
        r"^(?P<prefix>.+\.self_attn)\.(?P<kind>in_projs|out_projs)\."
        r"(?P<index>\d+)\.weight$"
    )
    split_groups: dict[tuple[str, str], list[str]] = {}
    for name in expected_names - resolved_expected:
        match = split_pattern.fullmatch(name)
        if match:
            split_groups.setdefault((match["prefix"], match["kind"]), []).append(name)
    for (prefix, kind), names in split_groups.items():
        source = f"{prefix}.in_proj_weight" if kind == "in_projs" else f"{prefix}.out_proj.weight"
        if source not in checkpoint:
            continue
        ordered_names = sorted(names, key=lambda name: int(split_pattern.fullmatch(name)["index"]))
        first_shape = expected_base[ordered_names[0]]
        if any(expected_base[name] != first_shape for name in ordered_names):
            mismatched_items.append(f"{source}: split target shapes are not uniform")
            continue
        fused_shape = (first_shape[0] * len(ordered_names), *first_shape[1:])
        actual_shape = tuple(checkpoint[source])
        if fused_shape != actual_shape:
            mismatched_items.append(
                f"{source}: expected fused {fused_shape}, checkpoint {actual_shape}"
            )
            continue
        resolved_expected.update(ordered_names)
        consumed_checkpoint.add(source)

    missing = tuple(sorted(expected_names - resolved_expected))
    unexpected = tuple(sorted(checkpoint_names - consumed_checkpoint))
    mismatched = tuple(sorted(mismatched_items))
    compatible = not missing and not unexpected and not mismatched and lora_tensor_count > 0
    return CompatibilityReport(
        compatible=compatible,
        base_tensor_count=len(expected_base),
        lora_tensor_count=lora_tensor_count,
        missing=missing,
        unexpected=unexpected,
        mismatched=mismatched,
    )


def expected_lora_shapes(config_path: Path, *, rank: int) -> dict[str, tuple[int, ...]]:
    import torch
    from moshi.models import loaders

    raw = json.loads(config_path.read_text())
    metadata_keys = {
        "mimi_name",
        "model_type",
        "moshi_name",
        "tokenizer_name",
        "version",
    }
    lm_config = {key: value for key, value in raw.items() if key not in metadata_keys}
    model = loaders.get_moshi_lm(
        None,
        lm_kwargs=lm_config,
        device="meta",
        dtype=torch.bfloat16,
        lm_kwargs_overrides={"lora": True, "lora_rank": rank, "lora_scaling": 2.0},
    )
    return {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}


def checkpoint_shapes(checkpoint_path: Path) -> dict[str, tuple[int, ...]]:
    from safetensors import safe_open

    result: dict[str, tuple[int, ...]] = {}
    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        for name in checkpoint.keys():
            result[name] = tuple(checkpoint.get_slice(name).get_shape())
    return result


def _build_parser() -> argparse.ArgumentParser:
    repository = Path(__file__).parents[3]
    parser = argparse.ArgumentParser(
        description="Compare PersonaPlex checkpoint metadata to a LoRA-shaped Moshi model"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=repository / "configs/training/personaplex-7b-v1.json",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    expected = expected_lora_shapes(args.config, rank=args.rank)
    actual = checkpoint_shapes(args.checkpoint)
    report = compare_parameter_shapes(expected, actual)
    print(json.dumps(asdict(report), indent=2))
    return 0 if report.compatible else 1


if __name__ == "__main__":
    raise SystemExit(main())
