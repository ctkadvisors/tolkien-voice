from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class SelectedCheckpoint:
    step: int
    eval_loss: float
    adapter_path: Path


def select_best_checkpoint(run_root: Path) -> SelectedCheckpoint:
    run_root = run_root.resolve()
    metrics_path = run_root / "metrics.eval.jsonl"
    if not metrics_path.is_file():
        raise ValueError(f"held-out metrics are missing: {metrics_path}")

    candidates: list[SelectedCheckpoint] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        step = row.get("step")
        eval_loss = row.get("eval_loss")
        if (
            not isinstance(step, int)
            or isinstance(step, bool)
            or not isinstance(eval_loss, (int, float))
            or isinstance(eval_loss, bool)
            or not math.isfinite(float(eval_loss))
        ):
            continue
        adapter = (
            run_root
            / "checkpoints"
            / f"checkpoint_{step:06d}"
            / "consolidated"
            / "lora.safetensors"
        )
        if not adapter.is_file() or adapter.stat().st_size == 0:
            continue
        candidates.append(
            SelectedCheckpoint(
                step=step,
                eval_loss=float(eval_loss),
                adapter_path=adapter,
            )
        )

    if not candidates:
        raise ValueError(f"training run has no finite held-out checkpoint: {run_root}")
    return min(candidates, key=lambda candidate: (candidate.eval_loss, candidate.step))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select the lowest finite held-out PersonaPlex checkpoint"
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--adapter-path-only",
        action="store_true",
        help="Print only the selected adapter path for shell automation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    selected = select_best_checkpoint(args.run_root)
    if args.adapter_path_only:
        print(selected.adapter_path)
    else:
        print(
            json.dumps(
                {
                    "step": selected.step,
                    "eval_loss": selected.eval_loss,
                    "adapter_path": str(selected.adapter_path),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
