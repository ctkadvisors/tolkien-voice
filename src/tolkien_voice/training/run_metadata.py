from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric(path: Path, step: int | None = None) -> dict[str, object]:
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"training metrics are empty: {path}")
    payloads = [json.loads(row) for row in rows]
    if step is None:
        payload = payloads[-1]
    else:
        selected = [payload for payload in payloads if payload.get("step") == step]
        if len(selected) != 1:
            raise ValueError(f"expected one training metric for step {step}; got {len(selected)}")
        payload = selected[0]
    return {
        key: payload[key]
        for key in ("step", "loss", "peak_allocated_mem", "wps")
        if key in payload
    }


def write_run_metadata(
    *,
    run_root: Path,
    adapter_path: Path,
    base_revision: str,
    personaplex_revision: str,
    moshi_revision: str,
    trainer_revision: str,
    metric_step: int | None = None,
    output_path: Path | None = None,
) -> Path:
    run_root = run_root.resolve()
    adapter_path = adapter_path.resolve()
    if not adapter_path.is_relative_to(run_root):
        raise ValueError("adapter path escapes the training run")
    if not adapter_path.is_file() or adapter_path.stat().st_size == 0:
        raise ValueError(f"adapter is missing or empty: {adapter_path}")
    payload = {
        "schema_version": 1,
        "adapter_path": str(adapter_path.relative_to(run_root)),
        "adapter_sha256": _sha256(adapter_path),
        "base_revision": base_revision,
        "personaplex_revision": personaplex_revision,
        "moshi_revision": moshi_revision,
        "trainer_revision": trainer_revision,
        "metrics": _metric(run_root / "metrics.train.jsonl", metric_step),
    }
    destination = (output_path or run_root / "run-metadata.json").resolve()
    if not destination.is_relative_to(run_root):
        raise ValueError("metadata output path escapes the training run")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".run-metadata.", dir=run_root)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bind a Tolkien LoRA to its training proof")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--personaplex-revision", required=True)
    parser.add_argument("--moshi-revision", required=True)
    parser.add_argument("--trainer-revision", required=True)
    parser.add_argument("--metric-step", type=int)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    path = write_run_metadata(
        run_root=args.run_root,
        adapter_path=args.adapter,
        base_revision=args.base_revision,
        personaplex_revision=args.personaplex_revision,
        moshi_revision=args.moshi_revision,
        trainer_revision=args.trainer_revision,
        metric_step=args.metric_step,
        output_path=args.output,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
