#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from tolkien_voice.evaluation import load_persona_manifest
from tolkien_voice.training.ab_eval import build_blinded_assignments


def main() -> int:
    repository = Path(__file__).parents[1]
    parser = argparse.ArgumentParser(description="Prepare private blinded STS assignments")
    parser.add_argument("--manifest", type=Path, default=repository / "evals/persona/probes.yaml")
    parser.add_argument("--characters", nargs="+", default=["aragorn", "sauron"])
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_persona_manifest(args.manifest)
    probe_ids = [probe.id for character in args.characters for probe in manifest[character]]
    assignments, key = build_blinded_assignments(probe_ids, seed=args.seed)
    args.output_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    assignments_path = args.output_directory / "assignments.jsonl"
    with assignments_path.open("w", encoding="utf-8") as output:
        for assignment in assignments:
            output.write(json.dumps(asdict(assignment), sort_keys=True) + "\n")
    assignments_path.chmod(0o600)
    key_path = args.output_directory / "blind-key.json"
    key_path.write_text(json.dumps(key, indent=2, sort_keys=True) + "\n")
    key_path.chmod(0o600)
    print(json.dumps({"assignments": len(assignments), "directory": str(args.output_directory)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
