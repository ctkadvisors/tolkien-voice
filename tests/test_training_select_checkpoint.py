import json
from pathlib import Path

import pytest

from tolkien_voice.training.select_checkpoint import select_best_checkpoint


def _write_checkpoint(run_root: Path, step: int, content: bytes = b"adapter") -> Path:
    path = (
        run_root
        / "checkpoints"
        / f"checkpoint_{step:06d}"
        / "consolidated"
        / "lora.safetensors"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _write_eval_metrics(run_root: Path, rows: list[dict[str, object]]) -> None:
    (run_root / "metrics.eval.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_selects_lowest_finite_eval_loss_and_matching_adapter(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    first = _write_checkpoint(run_root, 64, b"first")
    best = _write_checkpoint(run_root, 128, b"best")
    _write_checkpoint(run_root, 192, b"nan")
    _write_eval_metrics(
        run_root,
        [
            {"step": 64, "eval_loss": 3.1, "perplexity": 9.0},
            {"step": 128, "eval_loss": 2.7, "perplexity": 7.0},
            {"step": 192, "eval_loss": float("nan"), "perplexity": float("nan")},
        ],
    )

    selected = select_best_checkpoint(run_root)

    assert selected.step == 128
    assert selected.eval_loss == 2.7
    assert selected.adapter_path == best
    assert selected.adapter_path != first


def test_ignores_metric_when_checkpoint_is_missing(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    fallback = _write_checkpoint(run_root, 64)
    _write_eval_metrics(
        run_root,
        [
            {"step": 64, "eval_loss": 3.1},
            {"step": 128, "eval_loss": 2.0},
        ],
    )

    selected = select_best_checkpoint(run_root)

    assert selected.step == 64
    assert selected.adapter_path == fallback


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"step": 64, "eval_loss": float("nan")}],
        [{"step": "64", "eval_loss": 3.0}],
    ],
)
def test_fails_closed_without_valid_finite_checkpoint(
    tmp_path: Path, rows: list[dict[str, object]]
) -> None:
    run_root = tmp_path / "run"
    _write_checkpoint(run_root, 64)
    _write_eval_metrics(run_root, rows)

    with pytest.raises(ValueError, match="finite held-out checkpoint"):
        select_best_checkpoint(run_root)
