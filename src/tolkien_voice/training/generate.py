from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import yaml

from tolkien_voice.characters import CharacterRegistry

from .schema import DialogueBlueprint, Provenance, Split, TimingKind, normalize_text
from .teacher import TeacherClient, generate_blueprint


PROMPT_TEMPLATE_REVISION = "sts-dialogue-v1"
TEACHER_REVISION = "v4"
TEACHER_ARTIFACT = "tolkien-27b-v4-q8"
VOICE_LICENSE = "private-apple-system-voice"
TRAINING_SCENARIOS = (
    "A traveler asks for practical counsel before taking a difficult road.",
    "A tired companion questions whether continuing the journey is worthwhile.",
    "Two strangers share a quiet fire and speak briefly about courage.",
    "A listener challenges the character's judgment and interrupts the answer.",
    "A curious traveler asks about an uncertain point of old history.",
    "A companion changes the subject after a tense silence.",
    "Someone asks what loyalty demands when hope is thin.",
    "A stranger uses modern slang and the character redirects without breaking persona.",
)


def text_hash(value: str) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def held_out_prompt_hashes(prompts: Iterable[str]) -> set[str]:
    return {text_hash(prompt) for prompt in prompts}


def load_held_out_prompts(path: Path) -> list[str]:
    payload = yaml.safe_load(path.read_text())
    characters = payload.get("characters") if isinstance(payload, dict) else None
    if not isinstance(characters, dict):
        raise ValueError("held-out manifest requires a characters mapping")
    prompts: list[str] = []
    for probes in characters.values():
        if not isinstance(probes, list):
            raise ValueError("held-out character probes must be a list")
        for probe in probes:
            if not isinstance(probe, dict) or not isinstance(probe.get("prompt"), str):
                raise ValueError("each held-out probe requires a prompt string")
            prompts.append(probe["prompt"].strip())
    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError("held-out prompts must not be empty")
    return prompts


def write_blueprints_atomic(
    destination: Path,
    blueprints: list[DialogueBlueprint],
    *,
    reserved_prompt_hashes: set[str] | None = None,
) -> None:
    reserved = reserved_prompt_hashes or set()
    collisions = [item.id for item in blueprints if text_hash(item.scenario) in reserved]
    if collisions:
        raise ValueError(f"generated scenario collides with held-out prompt: {', '.join(collisions)}")
    ids = [item.id for item in blueprints]
    if len(ids) != len(set(ids)):
        raise ValueError("dialogue IDs must be unique")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for item in blueprints:
            handle.write(item.model_dump_json() + "\n")
    temporary.chmod(0o600)
    os.replace(temporary, destination)


def generate_corpus(
    *,
    client: TeacherClient,
    character_directory: Path,
    characters: tuple[str, ...],
    count_per_character: int,
    seed: int,
) -> list[DialogueBlueprint]:
    if count_per_character < 1:
        raise ValueError("count_per_character must be positive")
    registry = CharacterRegistry.from_directory(character_directory)
    result: list[DialogueBlueprint] = []
    timing_kinds = tuple(TimingKind)
    for character in characters:
        resolved = registry.resolve(character)
        card_path = character_directory / f"{character}.yaml"
        card_hash = hashlib.sha256(card_path.read_bytes()).hexdigest()
        for index in range(count_per_character):
            item_seed = seed + len(result)
            timing = timing_kinds[index % len(timing_kinds)]
            scenario = TRAINING_SCENARIOS[index % len(TRAINING_SCENARIOS)]
            split = Split.VALIDATION if index % 5 == 4 else Split.TRAIN
            provenance = Provenance(
                teacher_model=TEACHER_ARTIFACT,
                teacher_revision=TEACHER_REVISION,
                character_card_sha256=card_hash,
                prompt_template_revision=PROMPT_TEMPLATE_REVISION,
                seed=item_seed,
                voice_license=VOICE_LICENSE,
            )
            result.append(
                generate_blueprint(
                    client,
                    dialogue_id=f"{character}-{timing.value}-{index:04d}",
                    character=character,
                    character_prompt=resolved.text_prompt,
                    scenario=scenario,
                    timing=timing,
                    split=split,
                    provenance=provenance,
                )
            )
    return result


def _build_parser() -> argparse.ArgumentParser:
    repository = Path(__file__).parents[3]
    parser = argparse.ArgumentParser(description="Generate private Tolkien STS dialogue data")
    parser.add_argument("--base-url", default="http://127.0.0.1:9090/v1")
    parser.add_argument("--model", default="tolkien")
    parser.add_argument("--character-dir", type=Path, default=repository / "configs/characters")
    parser.add_argument("--held-out", type=Path, default=repository / "evals/sts/held-out.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--characters", nargs="+", default=["aragorn", "sauron"])
    parser.add_argument("--count-per-character", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    held_out = load_held_out_prompts(args.held_out)
    with TeacherClient(base_url=args.base_url, model=args.model) as client:
        blueprints = generate_corpus(
            client=client,
            character_directory=args.character_dir,
            characters=tuple(args.characters),
            count_per_character=args.count_per_character,
            seed=args.seed,
        )
    write_blueprints_atomic(
        args.output,
        blueprints,
        reserved_prompt_hashes=held_out_prompt_hashes(held_out),
    )
    print(json.dumps({"output": str(args.output), "dialogues": len(blueprints)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
