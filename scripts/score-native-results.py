#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tolkien_voice.training.judge import JudgeClient
from tolkien_voice.training.scoring import score_native_rows


def load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open(encoding="utf-8") as source:
            rows.extend(json.loads(line) for line in source if line.strip())
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Blind-score native STS transcripts")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:9090/v1")
    parser.add_argument("--model", default="tolkien")
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()
    rows = load_rows(args.input)
    with JudgeClient(base_url=args.base_url, model=args.model) as judge:
        observations = score_native_rows(rows, judge=judge, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for observation in observations:
            output.write(observation.model_dump_json() + "\n")
    temporary.chmod(0o600)
    os.replace(temporary, args.output)
    print(json.dumps({"observations": len(observations), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
