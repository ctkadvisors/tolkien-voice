from __future__ import annotations

import hashlib
import inspect
import logging
import os
from pathlib import Path
import runpy

from aiohttp import web
from safetensors.torch import load_file

from personaplex_lora_merge import merge_lora_state_dict


PINNED_BASE_REVISION = "fdaf4090a61cb315c138a1faee287ffd6c716309"
logger = logging.getLogger("tolkien_voice.personaplex_runtime")


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required runtime environment is missing: {name}")
    return value


def _verify_adapter() -> tuple[Path, str]:
    path = Path(_required_environment("TOLKIEN_VOICE_LORA_WEIGHT")).resolve()
    if not path.is_relative_to(Path("/adapter")):
        raise RuntimeError("LoRA adapter path is outside the allowlisted container directory")
    if not path.is_file() or path.suffix != ".safetensors":
        raise RuntimeError("LoRA adapter is not a safetensors file")
    expected = _required_environment("TOLKIEN_VOICE_LORA_SHA256")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise RuntimeError("LoRA adapter SHA-256 mismatch")
    if _required_environment("TOLKIEN_VOICE_BASE_REVISION") != PINNED_BASE_REVISION:
        raise RuntimeError("LoRA adapter base revision does not match PersonaPlex")
    return path, actual


def _patch_model_loader(adapter_path: Path) -> None:
    from moshi.models import loaders

    original = loaders.get_moshi_lm
    signature = inspect.signature(original)

    def get_moshi_lm(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        dtype = bound.arguments["dtype"]
        model = original(*args, **kwargs)
        device = next(model.parameters()).device
        adapter_state = load_file(str(adapter_path), device=str(device))
        for key, tensor in adapter_state.items():
            if tensor.dtype.is_floating_point:
                adapter_state[key] = tensor.to(device=device, dtype=dtype)
        model_state = model.state_dict()
        merged_count = merge_lora_state_dict(
            model_state,
            adapter_state,
            scaling=2.0,
        )
        incompatible = model.load_state_dict(model_state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"partial LoRA load refused: {incompatible}")
        model.eval()
        logger.info("merged %d verified LoRA projection pairs", merged_count)
        return model

    loaders.get_moshi_lm = get_moshi_lm


def _patch_health_endpoint(adapter_sha256: str) -> None:
    original_application = web.Application

    def application(*args, **kwargs):
        app = original_application(*args, **kwargs)

        async def health(_):
            return web.json_response(
                {
                    "status": "ok",
                    "adapter_sha256": adapter_sha256,
                    "base_revision": PINNED_BASE_REVISION,
                }
            )

        app.router.add_get("/api/health", health)
        return app

    web.Application = application


def main() -> None:
    adapter_path, adapter_sha256 = _verify_adapter()
    _patch_model_loader(adapter_path)
    _patch_health_endpoint(adapter_sha256)
    logger.info(
        "starting PersonaPlex with adapter sha256=%s base_revision=%s",
        adapter_sha256,
        PINNED_BASE_REVISION,
    )
    runpy.run_module("moshi.server", run_name="__main__")


if __name__ == "__main__":
    main()
