from __future__ import annotations

import argparse
import json
from pathlib import Path

from .generate import summarize_curriculum, validate_substantive_corpus
from .schema import DialogueBlueprint


def load_blueprints(path: Path) -> list[DialogueBlueprint]:
    blueprints: list[DialogueBlueprint] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                blueprints.append(DialogueBlueprint.model_validate_json(line))
            except ValueError as error:
                raise ValueError(f"invalid blueprint on line {line_number}: {error}") from error
    if not blueprints:
        raise ValueError("blueprint file is empty")
    return blueprints


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a substantive Tolkien STS corpus")
    parser.add_argument("--blueprints", type=Path, required=True)
    parser.add_argument("--minimum-per-character", type=int, default=64)
    parser.add_argument("--characters", nargs="+", default=["aragorn", "sauron"])
    args = parser.parse_args(argv)
    blueprints = load_blueprints(args.blueprints)
    validate_substantive_corpus(
        blueprints,
        minimum_per_character=args.minimum_per_character,
        characters=tuple(args.characters),
    )
    print(json.dumps(summarize_curriculum(blueprints), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
