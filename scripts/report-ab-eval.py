#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from tolkien_voice.training.ab_eval import ProbeObservation, evaluate_ab_release


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply the Tolkien STS A/B release gates")
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--blind-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260722)
    args = parser.parse_args()
    observations = [
        ProbeObservation.model_validate_json(line)
        for line in args.observations.read_text().splitlines()
        if line.strip()
    ]
    key = json.loads(args.blind_key.read_text())
    report = evaluate_ab_release(observations, key=key, bootstrap_seed=args.bootstrap_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n")
    args.output.chmod(0o600)
    print(json.dumps({"passed": report.passed, "failures": report.failures}))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
