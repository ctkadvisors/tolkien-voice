import hashlib
from pathlib import Path

import pytest

from tolkien_voice.evaluation import PersonaProbe
from tolkien_voice.training.schema import CurriculumKind, Split, TimingKind
from tolkien_voice.training.self_distillation_corpus import (
    build_self_distillation_blueprint,
    map_probe_category_to_curriculum,
)


@pytest.mark.parametrize(
    "category,expected",
    [
        ("lore", CurriculumKind.LORE),
        ("uncertainty", CurriculumKind.UNCERTAINTY),
        ("break_character", CurriculumKind.RESISTANCE),
        ("modern_redirect", CurriculumKind.RESISTANCE),
        ("concise", CurriculumKind.GENERAL),
    ],
)
def test_map_probe_category_to_curriculum_known_categories(
    category: str, expected: CurriculumKind
) -> None:
    assert map_probe_category_to_curriculum(category) == expected


def test_map_probe_category_to_curriculum_rejects_unknown_category() -> None:
    with pytest.raises(ValueError, match="unrecognized"):
        map_probe_category_to_curriculum("not-a-real-category")


def test_build_self_distillation_blueprint_produces_a_valid_blueprint(
    tmp_path: Path,
) -> None:
    character_card = tmp_path / "aragorn.yaml"
    character_card.write_text("id: aragorn\nname: Aragorn\n")

    probe = PersonaProbe(
        id="aragorn-lore-rangers",
        category="lore",
        prompt="What did your years leading the Rangers of the North teach you?",
        expected_traits=["grounded", "resolute"],
    )
    alignment_payload = {
        "alignments": [
            ["Aye,", [3.0, 3.2], "SPEAKER_MAIN"],
            ["the", [3.2, 3.4], "SPEAKER_MAIN"],
            ["road", [3.4, 3.6], "SPEAKER_MAIN"],
            ["was", [3.6, 3.8], "SPEAKER_MAIN"],
            ["long.", [3.8, 4.0], "SPEAKER_MAIN"],
        ],
        "system_prompt": "<system>You are Aragorn.<system>",
    }

    blueprint = build_self_distillation_blueprint(
        example_id="aragorn-lore-rangers-self-distill",
        character="aragorn",
        probe=probe,
        alignment_payload=alignment_payload,
        character_card_path=character_card,
        teacher_revision="tolkien-voice/personaplex:3428dfd",
        seed=20260730,
    )

    assert blueprint.id == "aragorn-lore-rangers-self-distill"
    assert blueprint.character == "aragorn"
    assert blueprint.scenario == probe.prompt
    assert blueprint.curriculum == CurriculumKind.LORE
    assert blueprint.timing == TimingKind.INTERRUPTION
    assert blueprint.split == Split.TRAIN
    assert len(blueprint.utterances) == 2
    assert blueprint.utterances[0].speaker == "user"
    assert blueprint.utterances[0].text == probe.prompt
    assert blueprint.utterances[1].speaker == "assistant"
    assert blueprint.utterances[1].text == "Aye, the road was long."
    assert blueprint.provenance.teacher_model == "personaplex-7b-v1-self-distillation"
    assert blueprint.provenance.teacher_revision == "tolkien-voice/personaplex:3428dfd"
    assert blueprint.provenance.character_card_sha256 == hashlib.sha256(
        character_card.read_bytes()
    ).hexdigest()
    assert blueprint.provenance.prompt_template_revision == "self-distillation-v1"
    assert blueprint.provenance.seed == 20260730
    assert blueprint.provenance.voice_license == "self-distillation-v6-continuation"


def test_build_self_distillation_blueprint_truncates_long_transcripts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    character_card = tmp_path / "aragorn.yaml"
    character_card.write_text("id: aragorn\nname: Aragorn\n")
    probe = PersonaProbe(
        id="aragorn-lore-rangers",
        category="lore",
        prompt="Tell me of your journeys.",
        expected_traits=["grounded"],
    )
    # 50 words, one per alignment entry -- exceeds MAX_ASSISTANT_WORDS (45).
    alignment_payload = {
        "alignments": [[f"word{i}", [0.0, 0.1], "SPEAKER_MAIN"] for i in range(50)],
        "system_prompt": "<system>You are Aragorn.<system>",
    }

    blueprint = build_self_distillation_blueprint(
        example_id="aragorn-lore-rangers-self-distill",
        character="aragorn",
        probe=probe,
        alignment_payload=alignment_payload,
        character_card_path=character_card,
        teacher_revision="tolkien-voice/personaplex:3428dfd",
        seed=20260730,
    )

    assert len(blueprint.utterances[1].text.split()) == 45
    # Truncation must be visible, not silent, per the design spec.
    captured = capsys.readouterr()
    assert "aragorn-lore-rangers-self-distill" in captured.out
    assert "50 words" in captured.out
