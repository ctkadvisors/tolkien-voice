from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def mimi_codebooks_for_config(config: dict[str, Any]) -> int:
    if config.get("model_type") == "personaplex":
        return 8
    n_q = int(config["n_q"])
    dep_q = int(config["dep_q"])
    return max(dep_q, n_q - dep_q)


def load_personaplex_mimi(loaders, checkpoint_info, device: str):
    raw_config = checkpoint_info.raw_config or {}
    return loaders.get_mimi(
        checkpoint_info.mimi_weights,
        num_codebooks=mimi_codebooks_for_config(raw_config),
        device=device,
    )


def install_personaplex_mimi_patch() -> None:
    from moshi.models import loaders

    original_get_mimi = loaders.CheckpointInfo.get_mimi

    def get_mimi(checkpoint_info, device: str = "cpu"):
        raw_config = checkpoint_info.raw_config or {}
        if raw_config.get("model_type") != "personaplex":
            return original_get_mimi(checkpoint_info, device=device)
        return load_personaplex_mimi(loaders, checkpoint_info, device)

    loaders.CheckpointInfo.get_mimi = get_mimi


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a LoRA on the PersonaPlex stereo model")
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    install_personaplex_mimi_patch()
    import train as upstream_train

    upstream_train.train(str(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
