import hashlib
import json
from pathlib import Path

from tolkien_voice.characters import CharacterRegistry
from tolkien_voice.training.conditioning import annotate_system_prompts
from tolkien_voice.training.schema import (
    DialogueBlueprint,
    Provenance,
    Split,
    TimingKind,
    Utterance,
)


ROOT = Path(__file__).parents[1]


def test_annotation_binds_exact_runtime_character_prompt_to_alignment(tmp_path: Path) -> None:
    blueprint = DialogueBlueprint(
        id="aragorn-counsel-normal-0001",
        character="aragorn",
        scenario="A traveler asks for counsel.",
        curriculum="counsel",
        timing=TimingKind.NORMAL,
        split=Split.TRAIN,
        utterances=[
            Utterance(speaker="user", text="What now?"),
            Utterance(speaker="assistant", text="Keep watch."),
        ],
        provenance=Provenance(
            teacher_model="tolkien-27b-v4-q8",
            teacher_revision="v4",
            character_card_sha256="a" * 64,
            prompt_template_revision="sts-dialogue-v2-curriculum",
            seed=1,
            voice_license="private-apple-system-voice",
        ),
    )
    blueprints = tmp_path / "dialogues.jsonl"
    blueprints.write_text(blueprint.model_dump_json() + "\n")
    audio = tmp_path / "audio"
    audio.mkdir()
    alignment = audio / f"{blueprint.id}.json"
    alignment.write_text(
        json.dumps(
            {
                "blueprint_hash": blueprint.canonical_hash,
                "alignments": [["Keep", [1.0, 1.2], "SPEAKER_MAIN"]],
                "text_conditions": None,
            }
        )
    )

    count = annotate_system_prompts(
        blueprints_path=blueprints,
        audio_directory=audio,
        character_directory=ROOT / "configs" / "characters",
    )

    payload = json.loads(alignment.read_text())
    expected = CharacterRegistry.from_directory(
        ROOT / "configs" / "characters"
    ).resolve("aragorn").text_prompt
    assert count == 1
    assert payload["system_prompt"] == expected
    assert payload["system_prompt_sha256"] == hashlib.sha256(expected.encode()).hexdigest()
