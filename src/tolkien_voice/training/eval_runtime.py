from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml


def render_eval_runtime_config(
    *, source: Path, output: Path, run_root: Path | None = None
) -> Path:
    payload: Any = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("runtime config must be a mapping")
    for key in ("adapter_path", "adapter_metadata_path", "adapter_bank"):
        payload.pop(key, None)

    if run_root is not None:
        root = run_root.expanduser().resolve()
        metadata_path = root / "selected-checkpoint-metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            relative_adapter = Path(metadata["adapter_path"])
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("selected checkpoint metadata is missing or invalid") from error
        if relative_adapter.is_absolute():
            raise ValueError("selected adapter path must be relative to its run root")
        adapter_path = (root / relative_adapter).resolve()
        if not adapter_path.is_relative_to(root):
            raise ValueError("selected adapter path escapes its run root")
        if not adapter_path.is_file() or adapter_path.stat().st_size == 0:
            raise ValueError(f"selected adapter is missing or empty: {adapter_path}")
        payload["adapter_path"] = str(adapter_path)
        payload["adapter_metadata_path"] = str(metadata_path.resolve())

    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a private base or selected-adapter STS evaluation config"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args(argv)
    path = render_eval_runtime_config(
        source=args.source,
        output=args.output,
        run_root=args.run_root,
    )
    print(json.dumps({"runtime_config": str(path), "adapter": args.run_root is not None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
