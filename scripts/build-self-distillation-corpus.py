#!/usr/bin/env python3
"""Sample N completions per interruption scenario against a live
PersonaPlex instance, keep the highest-quality rollout that yielded fast
enough, and build a moshi-finetune-ready training corpus from the model's
own captured audio.

Does not start or stop PersonaPlex -- run scripts/_personaplex_ctl.py
start/stop around this, exactly as scripts/run-v6-native-eval.sh does for
evaluation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from urllib.parse import urlencode, urlparse, urlunparse

from tolkien_voice.characters import CharacterRegistry
from tolkien_voice.evaluation import load_persona_manifest
from tolkien_voice.training.judge import JudgeClient
from tolkien_voice.training.manifest import write_moshi_manifests
from tolkien_voice.training.rollout import collect_rollout_group
from tolkien_voice.training.schema import RenderedExample, Split
from tolkien_voice.training.self_distillation import (
    build_self_distillation_example,
    select_best_rollout,
)


def websocket_url(server_url: str, *, text_prompt: str, voice_prompt: str) -> str:
    parsed = urlparse(server_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    query = urlencode({"text_prompt": text_prompt, "voice_prompt": voice_prompt})
    return urlunparse((scheme, parsed.netloc, "/api/chat", "", query, ""))


async def build(args: argparse.Namespace) -> list[RenderedExample]:
    manifest = load_persona_manifest(args.manifest)
    registry = CharacterRegistry.from_directory(args.character_dir)
    examples: list[RenderedExample] = []
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []
    with JudgeClient(base_url=args.judge_base_url, model=args.judge_model) as judge:
        for character in args.characters:
            resolved = registry.resolve(character)
            url = websocket_url(
                args.server_url,
                text_prompt=resolved.text_prompt,
                voice_prompt=resolved.voice_prompt,
            )
            for probe in manifest[character]:
                opening_wav = args.audio_dir / f"{probe.id}.wav"
                group = await collect_rollout_group(
                    url,
                    opening_wav,
                    args.interruption_audio,
                    probe_id=probe.id,
                    character=character,
                    probe_prompt=probe.prompt,
                    expected_traits=probe.expected_traits,
                    judge=judge,
                    raw_audio_directory=args.raw_audio_dir,
                    group_size=args.group_size,
                    seed=args.seed,
                )
                best = select_best_rollout(
                    group, yield_delay_reward_gate=args.yield_delay_reward_gate
                )
                if best is None:
                    skipped.append(probe.id)
                    continue
                # A single scenario's rendered audio being rejected (e.g. an
                # all-zero channel from a genuinely near-empty extracted
                # segment) must not discard every other scenario's already-
                # completed, expensive live sampling -- catch here and keep
                # going, rather than letting write_moshi_manifests's own
                # validation abort the whole run at the very end.
                try:
                    example = build_self_distillation_example(
                        best,
                        scenario_id=probe.id,
                        character=character,
                        opening_wav_path=opening_wav,
                        interruption_wav_path=args.interruption_audio,
                        system_prompt=resolved.text_prompt,
                        split=Split.TRAIN,
                        output_directory=args.output_directory / "audio",
                    )
                except ValueError as error:
                    failed.append((probe.id, str(error)))
                    continue
                examples.append(example)
    if skipped:
        print(f"skipped (no rollout cleared the gate): {skipped}")
    if failed:
        print(f"failed (rendering error, scenario dropped): {failed}")
    return examples


def main() -> int:
    repository = Path(__file__).parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:8998")
    parser.add_argument("--manifest", type=Path, default=repository / "evals/persona/probes.yaml")
    parser.add_argument("--character-dir", type=Path, default=repository / "configs/characters")
    parser.add_argument("--characters", nargs="+", default=["aragorn", "sauron"])
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--interruption-audio", type=Path, required=True)
    parser.add_argument("--raw-audio-dir", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--judge-base-url", default="http://127.0.0.1:9090/v1")
    parser.add_argument("--judge-model", default="tolkien")
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--yield-delay-reward-gate", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    examples = asyncio.run(build(args))
    if not examples:
        raise SystemExit("no scenario produced a corpus example -- nothing to write")
    manifests = write_moshi_manifests(args.output_directory / "manifests", examples)
    print(
        json.dumps(
            {
                "examples": len(examples),
                "manifests": {split.value: str(path) for split, path in manifests.items()},
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
