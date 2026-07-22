#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tolkien_voice.evaluation import load_persona_manifest
from tolkien_voice.training.render import MacOSSaySynthesizer
from tolkien_voice.training.schema import Utterance


def main() -> int:
    repository = Path(__file__).parents[1]
    parser = argparse.ArgumentParser(description="Render private held-out STS probe audio")
    parser.add_argument("--manifest", type=Path, default=repository / "evals/persona/probes.yaml")
    parser.add_argument("--characters", nargs="+", default=["aragorn", "sauron"])
    parser.add_argument("--voice", default="Samantha")
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_persona_manifest(args.manifest)
    args.output_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    synthesizer = MacOSSaySynthesizer(user_voice=args.voice, assistant_voice=args.voice)
    count = 0
    for character in args.characters:
        for probe in manifest[character]:
            destination = args.output_directory / f"{probe.id}.wav"
            synthesizer(Utterance(speaker="user", text=probe.prompt), destination)
            destination.chmod(0o600)
            count += 1
    print(json.dumps({"prompts": count, "output": str(args.output_directory)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
