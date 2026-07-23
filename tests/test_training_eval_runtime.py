from pathlib import Path

import yaml

from tolkien_voice.training.eval_runtime import render_eval_runtime_config


def write_base(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "controller_host": "127.0.0.1",
                "controller_port": 8788,
                "data_root": "/private/data",
                "adapter_path": "/private/data/old/lora.safetensors",
                "adapter_metadata_path": "/private/data/old/metadata.json",
            }
        )
    )


def test_base_eval_config_removes_every_adapter_setting(tmp_path: Path) -> None:
    source = tmp_path / "runtime.yaml"
    destination = tmp_path / "eval" / "base.yaml"
    write_base(source)

    render_eval_runtime_config(source=source, output=destination)

    payload = yaml.safe_load(destination.read_text())
    assert "adapter_path" not in payload
    assert "adapter_metadata_path" not in payload
    assert "adapter_bank" not in payload
    assert destination.stat().st_mode & 0o777 == 0o600


def test_adapter_eval_config_uses_selected_checkpoint_metadata(tmp_path: Path) -> None:
    source = tmp_path / "runtime.yaml"
    destination = tmp_path / "eval" / "aragorn.yaml"
    run_root = tmp_path / "runs" / "aragorn"
    adapter = run_root / "checkpoints" / "checkpoint_000288" / "consolidated" / "lora.safetensors"
    adapter.parent.mkdir(parents=True)
    adapter.write_bytes(b"adapter")
    metadata = run_root / "selected-checkpoint-metadata.json"
    metadata.write_text(
        '{"adapter_path":"checkpoints/checkpoint_000288/consolidated/lora.safetensors"}'
    )
    write_base(source)

    render_eval_runtime_config(source=source, output=destination, run_root=run_root)

    payload = yaml.safe_load(destination.read_text())
    assert payload["adapter_path"] == str(adapter.resolve())
    assert payload["adapter_metadata_path"] == str(metadata.resolve())
