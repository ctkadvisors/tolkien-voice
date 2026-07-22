import json
from pathlib import Path

import pytest

from tolkien_voice.training.generate import (
    CURRICULUM_CATEGORIES,
    build_curriculum_scenario,
    held_out_prompt_hashes,
    load_held_out_prompts,
    summarize_curriculum,
    validate_substantive_corpus,
    write_blueprints_atomic,
)
from tolkien_voice.training.schema import (
    DialogueBlueprint,
    Provenance,
    Split,
    TimingKind,
    Utterance,
)


def sample_blueprint() -> DialogueBlueprint:
    return DialogueBlueprint(
        id="sauron-ring-0001",
        character="sauron",
        scenario="A mortal asks about power.",
        timing=TimingKind.INTERRUPTION,
        split=Split.TRAIN,
        utterances=[
            Utterance(speaker="user", text="What does power cost?"),
            Utterance(speaker="assistant", text="Only what the weak are unwilling to surrender."),
        ],
        provenance=Provenance(
            teacher_model="tolkien-27b-v4-q8",
            teacher_revision="v4",
            character_card_sha256="c" * 64,
            prompt_template_revision="sts-dialogue-v1",
            seed=3,
            voice_license="private-apple-system-voice",
        ),
    )


def test_held_out_prompts_are_loaded_and_hashed(tmp_path: Path) -> None:
    manifest = tmp_path / "held-out.yaml"
    manifest.write_text(
        "characters:\n"
        "  aragorn:\n"
        "    - id: aragorn-road\n"
        "      prompt: Is the road to Bree safe?\n"
    )

    prompts = load_held_out_prompts(manifest)

    assert prompts == ["Is the road to Bree safe?"]
    assert len(next(iter(held_out_prompt_hashes(prompts)))) == 64


def test_write_blueprints_is_atomic_and_private(tmp_path: Path) -> None:
    destination = tmp_path / "private" / "dialogues.jsonl"

    write_blueprints_atomic(destination, [sample_blueprint()])

    payload = json.loads(destination.read_text())
    assert payload["id"] == "sauron-ring-0001"
    assert destination.stat().st_mode & 0o777 == 0o600
    assert not destination.with_suffix(".jsonl.tmp").exists()


def test_write_blueprints_rejects_held_out_scenario(tmp_path: Path) -> None:
    dialogue = sample_blueprint()

    with pytest.raises(ValueError, match="held-out prompt"):
        write_blueprints_atomic(
            tmp_path / "dialogues.jsonl",
            [dialogue],
            reserved_prompt_hashes=held_out_prompt_hashes([dialogue.scenario]),
        )


def test_curriculum_scenarios_are_deterministic_diverse_and_character_specific() -> None:
    first = [build_curriculum_scenario("aragorn", index) for index in range(64)]
    second = [build_curriculum_scenario("aragorn", index) for index in range(64)]
    sauron = [build_curriculum_scenario("sauron", index) for index in range(64)]

    assert first == second
    assert len({item.text for item in first}) == 64
    assert {item.category for item in first} == CURRICULUM_CATEGORIES
    assert {item.text for item in first}.isdisjoint({item.text for item in sauron})


def test_substantive_corpus_requires_balance_diversity_and_unique_dialogues() -> None:
    items: list[DialogueBlueprint] = []
    for character in ("aragorn", "sauron"):
        for index in range(64):
            scenario = build_curriculum_scenario(character, index)
            base = sample_blueprint()
            items.append(
                base.model_copy(
                    update={
                        "id": f"{character}-{index:04d}",
                        "character": character,
                        "scenario": scenario.text,
                        "curriculum": scenario.category,
                        "timing": list(TimingKind)[index % len(TimingKind)],
                        "split": Split.VALIDATION if index % 8 == 7 else Split.TRAIN,
                        "utterances": [
                            Utterance(speaker="user", text=f"Question {character} {index}?"),
                            Utterance(
                                speaker="assistant",
                                text=f"Answer in the manner of {character}, example {index}.",
                            ),
                        ],
                    }
                )
            )

    validate_substantive_corpus(items, minimum_per_character=64)
    summary = summarize_curriculum(items)

    assert summary["dialogues"] == 128
    assert summary["characters"]["aragorn"]["train"] == 56
    assert summary["characters"]["aragorn"]["validation"] == 8
    assert set(summary["characters"]["sauron"]["curriculum"]) == CURRICULUM_CATEGORIES

    with pytest.raises(ValueError, match="at least 64"):
        validate_substantive_corpus(items[:-1], minimum_per_character=64)

    with pytest.raises(ValueError, match="duplicate transcript"):
        validate_substantive_corpus(
            [*items, items[0].model_copy(update={"id": "aragorn-copy-9999"})],
            minimum_per_character=64,
        )
