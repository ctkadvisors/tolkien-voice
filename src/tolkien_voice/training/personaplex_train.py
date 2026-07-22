from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

from .conditioning import build_system_prompt_prefix, system_prompt_tokens


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


class _ReiterableEvalLoader:
    def __init__(self, factory: Callable[[], Iterable[Any]]) -> None:
        self._factory = factory

    def __iter__(self) -> Iterator[Any]:
        return iter(self._factory())


def install_reiterable_eval_loader_patch(upstream_train: Any) -> None:
    """Rebuild the pinned trainer's finite validation iterator for every eval."""
    original_build_data_loader = upstream_train.build_data_loader

    def build_data_loader(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("is_eval") is not True:
            return original_build_data_loader(*args, **kwargs)
        return _ReiterableEvalLoader(
            lambda: original_build_data_loader(*args, **kwargs)
        )

    upstream_train.build_data_loader = build_data_loader


def install_system_prompt_prefix_patch(upstream_train: Any) -> None:
    """Prepend each runtime character prompt to every training/eval chunk."""
    original_interleaved_tokenizer = upstream_train.InterleavedTokenizer

    class PromptedInterleavedTokenizer(original_interleaved_tokenizer):
        def __call__(self, wav: Any, start_sec: float, path: str) -> Any:
            sample = super().__call__(wav, start_sec, path)
            tokens = system_prompt_tokens(Path(path).with_suffix(".json"), self.interleaver.tokenizer)
            rows = build_system_prompt_prefix(
                num_codebooks=int(sample.codes.shape[1]),
                prompt_tokens=tokens,
                zero_token_id=int(self.interleaver.zero_padding),
            )
            import torch

            prefix = torch.tensor(
                [rows],
                device=sample.codes.device,
                dtype=sample.codes.dtype,
            )
            sample.codes = torch.cat([prefix, sample.codes], dim=2)
            return sample

    upstream_train.InterleavedTokenizer = PromptedInterleavedTokenizer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a LoRA on the PersonaPlex stereo model")
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    install_personaplex_mimi_patch()
    import train as upstream_train

    install_reiterable_eval_loader_patch(upstream_train)
    install_system_prompt_prefix_patch(upstream_train)
    upstream_train.train(str(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
