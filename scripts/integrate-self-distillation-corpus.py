#!/usr/bin/env python3
"""Builds a new, derived training-dataset directory layering a
self-distillation corpus's examples on top of an existing v6 corpus,
without modifying the v6 corpus. Does not invoke training -- run
scripts/train-mvp-v6-persona.sh (or a script pointed at the derived
output) as a separate, explicit step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tolkien_voice.training.self_distillation_corpus import build_derived_dataset


def main() -> int:
    repository = Path(__file__).parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-source-dataset", type=Path, required=True)
    parser.add_argument("--self-distillation-output", type=Path, required=True)
    parser.add_argument(
        "--probes-manifest", type=Path, default=repository / "evals/persona/probes.yaml"
    )
    parser.add_argument(
        "--character-directory", type=Path, default=repository / "configs/characters"
    )
    parser.add_argument("--teacher-revision", required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()

    summary = build_derived_dataset(
        v6_source_dataset=args.v6_source_dataset,
        self_distillation_output=args.self_distillation_output,
        probes_manifest_path=args.probes_manifest,
        character_directory=args.character_directory,
        teacher_revision=args.teacher_revision,
        seed=args.seed,
        output_directory=args.output_directory,
    )
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
