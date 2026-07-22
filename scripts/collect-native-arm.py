#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from urllib.parse import urlencode, urlparse, urlunparse

from tolkien_voice.characters import CharacterRegistry
from tolkien_voice.evaluation import load_persona_manifest
from tolkien_voice.training.native_probe import (
    probe_native_interruption,
    probe_native_session,
)


def websocket_url(server_url: str, *, text_prompt: str, voice_prompt: str) -> str:
    parsed = urlparse(server_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    query = urlencode({"text_prompt": text_prompt, "voice_prompt": voice_prompt})
    return urlunparse((scheme, parsed.netloc, "/api/chat", "", query, ""))


async def collect(args: argparse.Namespace) -> list[dict[str, object]]:
    manifest = load_persona_manifest(args.manifest)
    registry = CharacterRegistry.from_directory(args.character_dir)
    rows: list[dict[str, object]] = []
    for character in args.characters:
        resolved = registry.resolve(character)
        url = websocket_url(
            args.server_url,
            text_prompt=resolved.text_prompt,
            voice_prompt=resolved.voice_prompt,
        )
        for probe in manifest[character]:
            wav_path = args.audio_dir / f"{probe.id}.wav"
            raw_audio = args.raw_audio_dir / f"{args.arm}-{probe.id}.opus"
            measured = await probe_native_session(
                url,
                wav_path,
                response_seconds=args.response_seconds,
                raw_audio_path=raw_audio,
            )
            interruption = await probe_native_interruption(
                url,
                wav_path,
                args.interruption_audio,
                max_wait_seconds=args.interruption_wait_seconds,
                response_seconds=args.interruption_response_seconds,
                raw_audio_path=(
                    args.raw_audio_dir / f"{args.arm}-{probe.id}-interruption.opus"
                ),
            )
            rows.append(
                {
                    "probe_id": probe.id,
                    "blinded_arm": args.arm,
                    "character": character,
                    "category": probe.category,
                    "prompt": probe.prompt,
                    "expected_traits": probe.expected_traits,
                    **measured,
                    **interruption,
                }
            )
    return rows


def main() -> int:
    repository = Path(__file__).parents[1]
    parser = argparse.ArgumentParser(description="Collect one blinded native STS evaluation arm")
    parser.add_argument("--server-url", default="http://127.0.0.1:8998")
    parser.add_argument("--manifest", type=Path, default=repository / "evals/persona/probes.yaml")
    parser.add_argument("--character-dir", type=Path, default=repository / "configs/characters")
    parser.add_argument("--characters", nargs="+", default=["aragorn", "sauron"])
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--interruption-audio", type=Path, required=True)
    parser.add_argument("--raw-audio-dir", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--response-seconds", type=float, default=12)
    parser.add_argument("--interruption-wait-seconds", type=float, default=12)
    parser.add_argument("--interruption-response-seconds", type=float, default=4)
    args = parser.parse_args()
    if not args.interruption_audio.is_file():
        parser.error(f"interruption audio does not exist: {args.interruption_audio}")
    rows = asyncio.run(collect(args))
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.chmod(0o600)
    os.replace(temporary, args.output)
    print(json.dumps({"arm": args.arm, "observations": len(rows), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
