from __future__ import annotations

import re
from typing import Callable, MutableMapping


_LORA_KEY = re.compile(r"^(?P<prefix>.+)\.lora_(?P<side>A|B)\.weight$")
_SPLIT_PROJECTION = re.compile(
    r"^(?P<attention>.+\.self_attn)\.(?P<kind>in_projs|out_projs)\.(?P<index>\d+)$"
)


def merge_lora_state_dict(
    base_state: MutableMapping[str, object],
    adapter_state: MutableMapping[str, object],
    *,
    scaling: float,
    concatenate: Callable[[list[object]], object] | None = None,
) -> int:
    """Strictly merge a current-Moshi LoRA into a legacy PersonaPlex state dict."""
    if scaling <= 0:
        raise ValueError("LoRA scaling must be positive")
    if concatenate is None:
        import torch

        def concatenate(values: list[object]) -> object:
            return torch.cat(values, dim=0)

    pairs: dict[str, dict[str, object]] = {}
    for name, tensor in adapter_state.items():
        match = _LORA_KEY.fullmatch(name)
        if match is None:
            raise ValueError(f"unexpected non-LoRA adapter tensor: {name}")
        pairs.setdefault(match["prefix"], {})[match["side"]] = tensor

    direct_updates: dict[str, object] = {}
    split_updates: dict[str, dict[int, object]] = {}
    for prefix, sides in pairs.items():
        if "A" not in sides:
            raise ValueError(f"missing lora_A tensor for {prefix}")
        if "B" not in sides:
            raise ValueError(f"missing lora_B tensor for {prefix}")
        delta = (sides["B"] @ sides["A"]) * scaling
        split = _SPLIT_PROJECTION.fullmatch(prefix)
        if split is None:
            target = f"{prefix}.weight"
            if target in direct_updates:
                raise ValueError(f"duplicate LoRA target: {target}")
            direct_updates[target] = delta
            continue
        target = (
            f"{split['attention']}.in_proj_weight"
            if split["kind"] == "in_projs"
            else f"{split['attention']}.out_proj.weight"
        )
        index = int(split["index"])
        if index in split_updates.setdefault(target, {}):
            raise ValueError(f"duplicate split LoRA target: {prefix}")
        split_updates[target][index] = delta

    updates = dict(direct_updates)
    for target, indexed in split_updates.items():
        indices = sorted(indexed)
        if indices != list(range(len(indices))):
            raise ValueError(f"non-contiguous split LoRA projections for {target}: {indices}")
        updates[target] = concatenate([indexed[index] for index in indices])

    for target, delta in updates.items():
        if target not in base_state:
            raise ValueError(f"LoRA target is absent from PersonaPlex base: {target}")
        if tuple(base_state[target].shape) != tuple(delta.shape):
            raise ValueError(
                f"LoRA target shape mismatch for {target}: "
                f"base={tuple(base_state[target].shape)} delta={tuple(delta.shape)}"
            )

    for target, delta in updates.items():
        base_state[target] = base_state[target] + delta
    return len(pairs)
