import json
from pathlib import Path

import pytest

from tolkien_voice.training.generate import (
    held_out_prompt_hashes,
    load_held_out_prompts,
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
