#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from tolkien_voice.training.native_probe import probe_native_session


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure a ready native STS session")
    parser.add_argument("--url", required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--response-seconds", type=float, default=12)
    parser.add_argument("--raw-audio", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = asyncio.run(
        probe_native_session(
            args.url,
            args.wav,
            response_seconds=args.response_seconds,
            raw_audio_path=args.raw_audio,
        )
    )
    payload = json.dumps(result, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        args.output.write_text(payload + "\n")
        args.output.chmod(0o600)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
