#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tolkien_voice.evaluation import EvaluationSummary, evaluate_acceptance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate a Tolkien Voice acceptance run")
    parser.add_argument("summaries", nargs="+", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summaries = [
        EvaluationSummary.model_validate_json(path.read_text()) for path in args.summaries
    ]
    result = evaluate_acceptance(summaries)
    print(
        json.dumps(
            {
                "passed": result.passed,
                "failures": list(result.failures),
                "summaries": [summary.model_dump() for summary in summaries],
            },
            indent=2,
        )
    )
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
