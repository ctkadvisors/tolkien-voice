from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Protocol

from tolkien_voice.characters import CharacterRegistry

from .schema import DialogueBlueprint


class TextTokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


def build_system_prompt_prefix(
    *, num_codebooks: int, prompt_tokens: list[int], zero_token_id: int
) -> list[list[int]]:
    if num_codebooks < 2:
        raise ValueError("prompt prefix requires text and audio codebooks")
    if not prompt_tokens or len(prompt_tokens) > 512:
        raise ValueError("system prompt must encode to between 1 and 512 tokens")
    prefix = [[zero_token_id] * len(prompt_tokens) for _ in range(num_codebooks)]
    prefix[0] = list(prompt_tokens)
    return prefix


def system_prompt_tokens(alignment_path: Path, tokenizer: TextTokenizer) -> list[int]:
    payload = json.loads(alignment_path.read_text(encoding="utf-8"))
    prompt = payload.get("system_prompt") if isinstance(payload, dict) else None
    if (
        not isinstance(prompt, str)
        or not prompt.strip()
        or not prompt.startswith("<system>")
        or not prompt.endswith("<system>")
    ):
        raise ValueError(f"training alignment has no runtime system prompt: {alignment_path}")
    tokens = tokenizer.encode(prompt)
    if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
        raise ValueError(f"system prompt tokenizer returned invalid tokens: {alignment_path}")
    build_system_prompt_prefix(
        num_codebooks=2,
        prompt_tokens=tokens,
        zero_token_id=-1,
    )
    return tokens


def annotate_system_prompts(
    *,
    blueprints_path: Path,
    audio_directory: Path,
    character_directory: Path,
) -> int:
    registry = CharacterRegistry.from_directory(character_directory)
    blueprints = [
        DialogueBlueprint.model_validate_json(line)
        for line in blueprints_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not blueprints:
        raise ValueError("blueprint corpus is empty")

    prepared: list[tuple[Path, dict[str, Any]]] = []
    for blueprint in blueprints:
        path = audio_directory / f"{blueprint.id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"training alignment is not an object: {path}")
        if payload.get("blueprint_hash") != blueprint.canonical_hash:
            raise ValueError(f"training alignment blueprint hash mismatch: {path}")
        prompt = registry.resolve(blueprint.character).text_prompt
        payload["system_prompt"] = prompt
        payload["system_prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        prepared.append((path, payload))

    for path, payload in prepared:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(payload, output, indent=2)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            temporary.chmod(0o600)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return len(prepared)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bind exact runtime character prompts to rendered STS examples"
    )
    parser.add_argument("--blueprints", type=Path, required=True)
    parser.add_argument("--audio-directory", type=Path, required=True)
    parser.add_argument("--character-directory", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    count = annotate_system_prompts(
        blueprints_path=args.blueprints,
        audio_directory=args.audio_directory,
        character_directory=args.character_directory,
    )
    print(json.dumps({"annotated_examples": count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
