from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .schema import CurriculumKind, DialogueBlueprint, Split, TimingKind


TARGET_CURRICULA = frozenset(
    {
        CurriculumKind.LORE,
        CurriculumKind.UNCERTAINTY,
        CurriculumKind.RESISTANCE,
        CurriculumKind.REPAIR,
    }
)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"manifest is empty or invalid: {path}")
    return rows


def _rows_by_id(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _read_rows(path):
        audio_path = row.get("path")
        duration = row.get("duration")
        if not isinstance(audio_path, str) or not isinstance(duration, (int, float)):
            raise ValueError(f"manifest row is invalid: {path}")
        item_id = Path(audio_path).stem
        if item_id in result:
            raise ValueError(f"duplicate source manifest ID: {item_id}")
        result[item_id] = row
    return result


def _write_private(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.chmod(0o600)
    os.replace(temporary, path)


def build_persona_manifests(
    *,
    blueprints_path: Path,
    source_train_path: Path,
    source_eval_path: Path,
    character: str,
    output_directory: Path,
) -> dict[str, int | str]:
    blueprints = [
        DialogueBlueprint.model_validate_json(line)
        for line in blueprints_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [item for item in blueprints if item.character == character]
    if not selected:
        raise ValueError(f"corpus has no examples for character {character!r}")
    source_train = _rows_by_id(source_train_path)
    source_eval = _rows_by_id(source_eval_path)
    training = [item for item in selected if item.split == Split.TRAIN]
    validation = [item for item in selected if item.split == Split.VALIDATION]
    if not training or not validation:
        raise ValueError(f"character {character!r} requires train and validation examples")

    weighted_rows: list[dict[str, Any]] = []
    for item in training:
        try:
            row = source_train[item.id]
        except KeyError as error:
            raise ValueError(f"source train manifest is missing {item.id}") from error
        repeats = 1
        if item.curriculum in TARGET_CURRICULA:
            repeats += 1
        if item.timing == TimingKind.INTERRUPTION:
            repeats += 3
        weighted_rows.extend(dict(row) for _ in range(repeats))

    eval_rows: list[dict[str, Any]] = []
    for item in validation:
        try:
            eval_rows.append(dict(source_eval[item.id]))
        except KeyError as error:
            raise ValueError(f"source eval manifest is missing {item.id}") from error

    _write_private(output_directory / "train.jsonl", weighted_rows)
    _write_private(output_directory / "validation.jsonl", eval_rows)
    return {
        "character": character,
        "source_train_examples": len(training),
        "weighted_train_rows": len(weighted_rows),
        "validation_examples": len(validation),
        "interruption_examples": sum(
            item.timing == TimingKind.INTERRUPTION for item in training
        ),
        "target_curriculum_examples": sum(
            item.curriculum in TARGET_CURRICULA for item in training
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build private character-specific weighted PersonaPlex manifests"
    )
    parser.add_argument("--blueprints", type=Path, required=True)
    parser.add_argument("--source-train", type=Path, required=True)
    parser.add_argument("--source-eval", type=Path, required=True)
    parser.add_argument("--character", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = build_persona_manifests(
        blueprints_path=args.blueprints,
        source_train_path=args.source_train,
        source_eval_path=args.source_eval,
        character=args.character,
        output_directory=args.output_directory,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
