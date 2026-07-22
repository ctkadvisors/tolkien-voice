from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from tests.fakes import FakeCommandRunner, FakeHttpClient
from tests.test_runtime import install_fake_token, settings_for
from tolkien_voice.runtime import PERSONAPLEX_BASE_REVISION, PersonaPlexRuntime


ROOT = Path(__file__).parents[1]
LORA_MERGE = ROOT / "container/personaplex_lora_merge.py"


class Matrix:
    def __init__(self, values: list[list[float]]) -> None:
        self.values = values

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.values), len(self.values[0]))

    def __matmul__(self, other: "Matrix") -> "Matrix":
        return Matrix(
            [
                [
                    sum(
                        self.values[row][item] * other.values[item][column]
                        for item in range(self.shape[1])
                    )
                    for column in range(other.shape[1])
                ]
                for row in range(self.shape[0])
            ]
        )

    def __mul__(self, scalar: float) -> "Matrix":
        return Matrix([[value * scalar for value in row] for row in self.values])

    def __add__(self, other: "Matrix") -> "Matrix":
        return Matrix(
            [
                [left + right for left, right in zip(left_row, right_row, strict=True)]
                for left_row, right_row in zip(self.values, other.values, strict=True)
            ]
        )


def zeros(rows: int, columns: int) -> Matrix:
    return Matrix([[0.0] * columns for _ in range(rows)])


def concatenate(values: list[Matrix]) -> Matrix:
    return Matrix([row for value in values for row in value.values])


def _load_lora_merge():
    spec = importlib.util.spec_from_file_location("personaplex_lora_merge", LORA_MERGE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lora_merge_applies_normal_and_split_attention_deltas() -> None:
    module = _load_lora_merge()
    base = {
        "linear.weight": zeros(2, 2),
        "block.self_attn.in_proj_weight": zeros(4, 2),
    }
    adapter = {
        "linear.lora_A.weight": Matrix([[1.0, 2.0]]),
        "linear.lora_B.weight": Matrix([[3.0], [4.0]]),
        "block.self_attn.in_projs.0.lora_A.weight": Matrix([[1.0, 0.0]]),
        "block.self_attn.in_projs.0.lora_B.weight": Matrix([[1.0], [2.0]]),
        "block.self_attn.in_projs.1.lora_A.weight": Matrix([[0.0, 1.0]]),
        "block.self_attn.in_projs.1.lora_B.weight": Matrix([[3.0], [4.0]]),
    }

    count = module.merge_lora_state_dict(
        base,
        adapter,
        scaling=2.0,
        concatenate=concatenate,
    )

    assert count == 3
    assert base["linear.weight"].values == [[6.0, 12.0], [8.0, 16.0]]
    assert base["block.self_attn.in_proj_weight"].values == [
        [2.0, 0.0],
        [4.0, 0.0],
        [0.0, 6.0],
        [0.0, 8.0],
    ]


def test_lora_merge_rejects_partial_adapter() -> None:
    module = _load_lora_merge()

    with pytest.raises(ValueError, match="missing lora_B"):
        module.merge_lora_state_dict(
            {"linear.weight": zeros(2, 2)},
            {"linear.lora_A.weight": zeros(1, 2)},
            scaling=2.0,
            concatenate=concatenate,
        )


@pytest.mark.asyncio
async def test_runtime_mounts_only_a_verified_allowlisted_adapter(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    install_fake_token(settings)
    adapter = settings.data_root / "runs/smoke/checkpoints/step/consolidated/lora.safetensors"
    adapter.parent.mkdir(parents=True)
    adapter.write_bytes(b"verified adapter")
    digest = hashlib.sha256(adapter.read_bytes()).hexdigest()
    metadata = settings.data_root / "runs/smoke/run-metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "base_revision": PERSONAPLEX_BASE_REVISION,
                "adapter_sha256": digest,
            }
        )
    )
    model_cache = (
        settings.data_root
        / "cache/huggingface/hub/models--nvidia--personaplex-7b-v1"
    )
    (model_cache / "refs").mkdir(parents=True)
    (model_cache / "refs/main").write_text(PERSONAPLEX_BASE_REVISION)
    snapshot = model_cache / "snapshots" / PERSONAPLEX_BASE_REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"base")
    settings = settings.model_copy(
        update={"adapter_path": adapter, "adapter_metadata_path": metadata}
    )
    runner = FakeCommandRunner()
    runtime = PersonaPlexRuntime(settings, runner=runner, http=FakeHttpClient([200]))

    await runtime.start()

    command = runner.calls[0][0]
    assert f"type=bind,src={adapter},dst=/adapter/lora.safetensors,readonly" in command
    assert "TOLKIEN_VOICE_LORA_WEIGHT=/adapter/lora.safetensors" in command
    assert f"TOLKIEN_VOICE_LORA_SHA256={digest}" in command
    assert f"TOLKIEN_VOICE_BASE_REVISION={PERSONAPLEX_BASE_REVISION}" in command
    assert f"type=bind,src={settings.data_root},dst=/data" not in command


def test_runtime_rejects_adapter_outside_data_root(tmp_path: Path) -> None:
    settings = settings_for(tmp_path).model_copy(
        update={
            "adapter_path": tmp_path / "outside.safetensors",
            "adapter_metadata_path": tmp_path / "outside.json",
        }
    )

    with pytest.raises(ValueError, match="escapes data root"):
        PersonaPlexRuntime(settings)._validate_adapter()


def test_server_and_entrypoint_expose_adapter_identity() -> None:
    runtime_wrapper = (ROOT / "container/personaplex_runtime.py").read_text()
    entrypoint = (ROOT / "container/entrypoint.sh").read_text()

    assert 'app.router.add_get("/api/health"' in runtime_wrapper
    assert '"adapter_sha256"' in runtime_wrapper
    assert '"base_revision"' in runtime_wrapper
    assert '"merged_projection_pairs"' in runtime_wrapper
    assert '"variant": "base"' in runtime_wrapper
    assert '"variant": "adapter"' in runtime_wrapper
    assert "merge_lora_state_dict" in runtime_wrapper
    assert "strict=True" in runtime_wrapper
    assert "TOLKIEN_VOICE_LORA_WEIGHT" in entrypoint
    assert "TOLKIEN_VOICE_LORA_SHA256" in entrypoint
    assert "TOLKIEN_VOICE_BASE_REVISION" in entrypoint
    assert "both be set or both be absent" in entrypoint
    assert "A verified LoRA adapter and SHA-256 are required" not in entrypoint


def test_native_probe_requires_handshake_and_audio_response() -> None:
    probe = (ROOT / "src/tolkien_voice/training/native_probe.py").read_text()

    assert 'handshake.data != b"\\x00"' in probe
    assert 'await websocket.send_bytes(b"\\x01" + payload)' in probe
    assert 'raise RuntimeError("native STS probe returned no audio")' in probe
    assert '"first_audio_ms"' in probe
    assert '"underrun_count"' in probe
