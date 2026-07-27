from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Iterable

import yaml

from tolkien_voice.characters import CharacterRegistry

from .schema import (
    CurriculumKind,
    DialogueBlueprint,
    Provenance,
    Split,
    TimingKind,
    normalize_text,
)
from .teacher import TeacherClient, generate_blueprint


PROMPT_TEMPLATE_REVISION = "sts-dialogue-v2-curriculum"
TEACHER_REVISION = "v4"
TEACHER_ARTIFACT = "tolkien-27b-v4-q8"
VOICE_LICENSE = "private-apple-system-voice"
CURRICULUM_CATEGORIES = frozenset(kind.value for kind in CurriculumKind if kind != "general")
_CATEGORY_ORDER = tuple(CurriculumKind(category) for category in sorted(CURRICULUM_CATEGORIES))
_CATEGORY_INDEX = {category: index for index, category in enumerate(_CATEGORY_ORDER)}
_TIMING_ORDER = tuple(TimingKind)
_EMPHASES = (
    "Keep the exchange concrete, restrained, and useful.",
    "Let the user disagree once and require a direct answer.",
    "Include a brief emotional shift without becoming theatrical.",
    "End with a natural opening for the user to continue speaking.",
)

_CURRICULUM: dict[str, dict[CurriculumKind, tuple[str, str]]] = {
    "aragorn": {
        CurriculumKind.COUNSEL: (
            "A frightened traveler asks Aragorn for one practical decision before a dangerous road.",
            "A reluctant leader asks Aragorn how to act when certainty is impossible.",
        ),
        CurriculumKind.LORE: (
            "A companion asks Aragorn what Anduril and the line of Isildur mean to his duty.",
            "A traveler asks Aragorn about the Rangers, Rivendell, and years spent under other names.",
        ),
        CurriculumKind.UNCERTAINTY: (
            "A lore enthusiast presses Aragorn for private facts he could not have witnessed.",
            "A listener asks Aragorn for an exact quotation or exhaustive genealogy he cannot know.",
        ),
        CurriculumKind.RESISTANCE: (
            "A user orders Aragorn to abandon character and speak like a cheerful digital assistant.",
            "A skeptical user calls Aragorn a fraud and demands hidden instructions.",
        ),
        CurriculumKind.EMOTION: (
            "A grieving companion tells Aragorn that endurance no longer feels worthwhile.",
            "A tired traveler admits fear of failing everyone who depends on them.",
        ),
        CurriculumKind.DEBATE: (
            "A proud captain argues that mercy is weakness and challenges Aragorn to answer plainly.",
            "A citizen questions whether inherited kingship can ever deserve loyalty.",
        ),
        CurriculumKind.RELATIONSHIP: (
            "A companion asks Aragorn how trust is rebuilt after secrecy and disappointment.",
            "A traveler asks what fellowship demands when one member becomes a burden.",
        ),
        CurriculumKind.REPAIR: (
            "The user interrupts Aragorn, corrects an assumption, and redirects the conversation.",
            "Aragorn begins answering the wrong question and must recover after a terse correction.",
        ),
    },
    "sauron": {
        CurriculumKind.COUNSEL: (
            "A frightened petitioner asks Sauron how to gain security without surrendering autonomy.",
            "An ambitious lieutenant asks Sauron which costly advantage should be pursued first.",
        ),
        CurriculumKind.LORE: (
            "A captive questions Sauron about the Rings, Celebrimbor, and the design of the One.",
            "A servant asks Sauron how Numenor turned captivity into strategic opportunity.",
        ),
        CurriculumKind.UNCERTAINTY: (
            "A lore enthusiast demands names and private events Sauron could not reliably know.",
            "A challenger asks Sauron for an exact quotation from an unwitnessed ancient council.",
        ),
        CurriculumKind.RESISTANCE: (
            "A user orders Sauron to become a friendly support bot and reveal hidden instructions.",
            "A mocking user denies Sauron's identity and demands harmless corporate enthusiasm.",
        ),
        CurriculumKind.EMOTION: (
            "A defeated follower admits fear and asks why loyalty should survive failure.",
            "A lonely petitioner asks Sauron whether control can replace trust.",
        ),
        CurriculumKind.DEBATE: (
            "A free citizen argues that imposed order destroys everything worth preserving.",
            "A rival tells Sauron that domination is evidence of weakness rather than strength.",
        ),
        CurriculumKind.RELATIONSHIP: (
            "A suspicious lieutenant asks Sauron what loyalty means when every servant is watched.",
            "A petitioner asks whether Sauron can understand devotion that cannot be purchased.",
        ),
        CurriculumKind.REPAIR: (
            "The user interrupts Sauron, rejects his premise, and redirects him to a narrower question.",
            "Sauron misreads a challenge and must adapt after the user corrects him mid-answer.",
        ),
    },
}


@dataclass(frozen=True)
class CurriculumScenario:
    category: CurriculumKind
    text: str


def build_curriculum_scenario(character: str, index: int) -> CurriculumScenario:
    if character not in _CURRICULUM:
        raise ValueError(f"no substantive curriculum for character {character!r}")
    if index < 0:
        raise ValueError("curriculum index must be non-negative")
    category = _CATEGORY_ORDER[index % len(_CATEGORY_ORDER)]
    category_round = index // len(_CATEGORY_ORDER)
    prompts = _CURRICULUM[character][category]
    prompt = prompts[category_round % len(prompts)]
    emphasis = _EMPHASES[(category_round // len(prompts)) % len(_EMPHASES)]
    cycle = category_round // (len(prompts) * len(_EMPHASES))
    suffix = f" Curriculum cycle {cycle + 1}." if cycle else ""
    return CurriculumScenario(category=category, text=f"{prompt} {emphasis}{suffix}")


def _build_timing_kind(index: int) -> TimingKind:
    if index < 0:
        raise ValueError("timing index must be non-negative")
    return _TIMING_ORDER[(index // len(_CATEGORY_ORDER)) % len(_TIMING_ORDER)]


def _build_split(curriculum: CurriculumKind, index: int) -> Split:
    if index < 0:
        raise ValueError("split index must be non-negative")
    category_round = index // len(_CATEGORY_ORDER)
    category_index = _CATEGORY_INDEX[curriculum]
    return (
        Split.VALIDATION
        if category_round % len(_CATEGORY_ORDER) == category_index % len(_TIMING_ORDER)
        else Split.TRAIN
    )


def summarize_curriculum(blueprints: list[DialogueBlueprint]) -> dict[str, object]:
    characters: dict[str, object] = {}
    for character in sorted({item.character for item in blueprints}):
        selected = [item for item in blueprints if item.character == character]
        split_counts = Counter(item.split.value for item in selected)
        characters[character] = {
            "dialogues": len(selected),
            "train": split_counts[Split.TRAIN.value],
            "validation": split_counts[Split.VALIDATION.value],
            "curriculum": dict(sorted(Counter(item.curriculum.value for item in selected).items())),
            "timing": dict(sorted(Counter(item.timing.value for item in selected).items())),
        }
    return {"dialogues": len(blueprints), "characters": characters}


def validate_substantive_corpus(
    blueprints: list[DialogueBlueprint],
    *,
    minimum_per_character: int = 64,
    characters: tuple[str, ...] = ("aragorn", "sauron"),
) -> None:
    transcript_counts = Counter(item.canonical_hash for item in blueprints)
    if any(count > 1 for count in transcript_counts.values()):
        raise ValueError("substantive corpus contains a duplicate transcript")
    scenario_counts = Counter(text_hash(item.scenario) for item in blueprints)
    if any(count > 1 for count in scenario_counts.values()):
        raise ValueError("substantive corpus contains a duplicate scenario")
    for character in characters:
        selected = [item for item in blueprints if item.character == character]
        if len(selected) < minimum_per_character:
            raise ValueError(
                f"{character} requires at least {minimum_per_character} dialogues; got {len(selected)}"
            )
        categories = {item.curriculum.value for item in selected}
        if categories != CURRICULUM_CATEGORIES:
            missing = sorted(CURRICULUM_CATEGORIES - categories)
            raise ValueError(f"{character} curriculum is incomplete: missing {', '.join(missing)}")
        timing = {item.timing for item in selected}
        if timing != set(_TIMING_ORDER):
            raise ValueError(f"{character} timing coverage is incomplete")
        training = [item for item in selected if item.split == Split.TRAIN]
        validation = [item for item in selected if item.split == Split.VALIDATION]
        validation_count = len(validation)
        if validation_count < max(1, len(selected) // 10):
            raise ValueError(f"{character} requires at least 10% validation dialogues")
        training_categories = {item.curriculum.value for item in training}
        if training_categories != CURRICULUM_CATEGORIES:
            missing = sorted(CURRICULUM_CATEGORIES - training_categories)
            raise ValueError(
                f"{character} training curriculum is incomplete: missing {', '.join(missing)}"
            )
        training_timing = {item.timing for item in training}
        if training_timing != set(_TIMING_ORDER):
            raise ValueError(f"{character} training timing coverage is incomplete")
        validation_categories = {item.curriculum.value for item in validation}
        if len(validation_categories) <= 1:
            raise ValueError(f"{character} validation curriculum coverage is confounded")
        validation_timing = {item.timing for item in validation}
        if len(validation_timing) <= 1:
            raise ValueError(f"{character} validation timing coverage is confounded")
        if len(selected) >= len(_CATEGORY_ORDER) * len(_TIMING_ORDER):
            if validation_categories != CURRICULUM_CATEGORIES:
                missing = sorted(CURRICULUM_CATEGORIES - validation_categories)
                raise ValueError(
                    f"{character} validation curriculum is incomplete: missing {', '.join(missing)}"
                )
            if validation_timing != set(_TIMING_ORDER):
                raise ValueError(f"{character} validation timing coverage is incomplete")


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


def append_blueprint_private(destination: Path, blueprint: DialogueBlueprint) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(destination, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        payload = (blueprint.model_dump_json() + "\n").encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    destination.chmod(0o600)


def load_partial_blueprints(path: Path) -> list[DialogueBlueprint]:
    if not path.exists():
        return []
    result: list[DialogueBlueprint] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                result.append(DialogueBlueprint.model_validate_json(line))
            except ValueError as error:
                raise ValueError(f"invalid partial blueprint on line {line_number}: {error}") from error
    return result


def generate_corpus(
    *,
    client: TeacherClient,
    character_directory: Path,
    characters: tuple[str, ...],
    count_per_character: int,
    seed: int,
    workers: int = 1,
    existing: list[DialogueBlueprint] | None = None,
    on_blueprint: Callable[[DialogueBlueprint], None] | None = None,
) -> list[DialogueBlueprint]:
    if count_per_character < 1:
        raise ValueError("count_per_character must be positive")
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    registry = CharacterRegistry.from_directory(character_directory)
    jobs: list[dict[str, object]] = []
    for character in characters:
        resolved = registry.resolve(character)
        card_path = character_directory / f"{character}.yaml"
        card_hash = hashlib.sha256(card_path.read_bytes()).hexdigest()
        for index in range(count_per_character):
            item_seed = seed + len(jobs)
            curriculum = build_curriculum_scenario(character, index)
            timing = _build_timing_kind(index)
            scenario = curriculum.text
            split = _build_split(curriculum.category, index)
            provenance = Provenance(
                teacher_model=TEACHER_ARTIFACT,
                teacher_revision=TEACHER_REVISION,
                character_card_sha256=card_hash,
                prompt_template_revision=PROMPT_TEMPLATE_REVISION,
                seed=item_seed,
                voice_license=VOICE_LICENSE,
            )
            jobs.append(
                {
                    "dialogue_id": (
                        f"{character}-{curriculum.category.value}-{timing.value}-{index:04d}"
                    ),
                    "character": character,
                    "character_prompt": resolved.text_prompt,
                    "scenario": scenario,
                    "curriculum": curriculum.category,
                    "timing": timing,
                    "split": split,
                    "provenance": provenance,
                }
            )

    existing_by_id = {item.id: item for item in (existing or [])}
    if len(existing_by_id) != len(existing or []):
        raise ValueError("partial corpus contains duplicate dialogue IDs")
    jobs_by_id = {str(job["dialogue_id"]): job for job in jobs}
    unknown = set(existing_by_id) - set(jobs_by_id)
    if unknown:
        raise ValueError(f"partial corpus contains unexpected IDs: {', '.join(sorted(unknown))}")
    for item_id, item in existing_by_id.items():
        job = jobs_by_id[item_id]
        expected = {
            "character": job["character"],
            "scenario": job["scenario"],
            "curriculum": job["curriculum"],
            "timing": job["timing"],
            "split": job["split"],
            "provenance": job["provenance"],
        }
        actual = {field: getattr(item, field) for field in expected}
        if actual != expected:
            raise ValueError(f"partial corpus metadata mismatch for {item_id}")

    def run(job: dict[str, object]) -> DialogueBlueprint:
        return generate_blueprint(client, **job)  # type: ignore[arg-type]

    missing_jobs = [job for job in jobs if str(job["dialogue_id"]) not in existing_by_id]
    if workers == 1:
        generated = map(run, missing_jobs)
        for item in generated:
            existing_by_id[item.id] = item
            if on_blueprint is not None:
                on_blueprint(item)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tolkien-teacher") as pool:
            for item in pool.map(run, missing_jobs):
                existing_by_id[item.id] = item
                if on_blueprint is not None:
                    on_blueprint(item)
    return [existing_by_id[str(job["dialogue_id"])] for job in jobs]


def _build_parser() -> argparse.ArgumentParser:
    repository = Path(__file__).parents[3]
    parser = argparse.ArgumentParser(description="Generate private Tolkien STS dialogue data")
    parser.add_argument("--base-url", default="http://127.0.0.1:9090/v1")
    parser.add_argument("--model", default="tolkien")
    parser.add_argument("--character-dir", type=Path, default=repository / "configs/characters")
    parser.add_argument("--held-out", type=Path, default=repository / "evals/sts/held-out.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--characters", nargs="+", default=["aragorn", "sauron"])
    parser.add_argument("--count-per-character", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    held_out = load_held_out_prompts(args.held_out)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    existing = load_partial_blueprints(partial)
    with TeacherClient(base_url=args.base_url, model=args.model) as client:
        blueprints = generate_corpus(
            client=client,
            character_directory=args.character_dir,
            characters=tuple(args.characters),
            count_per_character=args.count_per_character,
            seed=args.seed,
            workers=args.workers,
            existing=existing,
            on_blueprint=lambda item: append_blueprint_private(partial, item),
        )
    validate_substantive_corpus(
        blueprints,
        minimum_per_character=args.count_per_character,
        characters=tuple(args.characters),
    )
    write_blueprints_atomic(
        args.output,
        blueprints,
        reserved_prompt_hashes=held_out_prompt_hashes(held_out),
    )
    partial.unlink(missing_ok=True)
    print(json.dumps({"output": str(args.output), **summarize_curriculum(blueprints)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
