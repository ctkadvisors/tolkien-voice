from pathlib import Path

import pytest
from pydantic import ValidationError

from tolkien_voice.training.schema import (
    DialogueBlueprint,
    Provenance,
    RenderedExample,
    Split,
    TimingKind,
    Utterance,
    assert_no_leakage,
)


def provenance() -> Provenance:
    return Provenance(
        teacher_model="tolkien-27b-v4-q8",
        teacher_revision="v4",
        character_card_sha256="a" * 64,
        prompt_template_revision="sts-dialogue-v1",
        seed=42,
        voice_license="private-apple-system-voice",
    )


def blueprint(
    *,
    dialogue_id: str = "aragorn-road-0001",
    character: str = "aragorn",
    split: Split = Split.TRAIN,
    assistant_text: str = "The road is quiet, but quiet roads still deserve watchful eyes.",
) -> DialogueBlueprint:
    return DialogueBlueprint(
        id=dialogue_id,
        character=character,
        scenario="A traveler asks about the road ahead.",
        timing=TimingKind.NORMAL,
        split=split,
        utterances=[
            Utterance(speaker="user", text="Is the road safe tonight?"),
            Utterance(speaker="assistant", text=assistant_text),
            Utterance(speaker="user", text="Will you walk with me?"),
            Utterance(speaker="assistant", text="For a while. Dawn will find us farther east."),
        ],
        provenance=provenance(),
    )


def test_valid_blueprint_has_stable_canonical_hash() -> None:
    first = blueprint()
    second = blueprint(
        dialogue_id="aragorn-road-0002",
        assistant_text="  The ROAD is quiet, but quiet roads still deserve watchful eyes. ",
    )

    assert first.canonical_hash == second.canonical_hash
    assert len(first.canonical_hash) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("teacher_revision", ""),
        ("character_card_sha256", "not-a-sha"),
        ("prompt_template_revision", ""),
        ("voice_license", ""),
    ],
)
def test_provenance_is_required(field: str, value: str) -> None:
    payload = provenance().model_dump()
    payload[field] = value

    with pytest.raises(ValidationError):
        Provenance.model_validate(payload)


def test_unknown_character_is_rejected() -> None:
    with pytest.raises(ValidationError, match="character"):
        blueprint(character="tom-bombadil")


def test_empty_utterance_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Utterance(speaker="user", text="   ")


def test_long_assistant_monologue_is_rejected() -> None:
    long_reply = " ".join(f"word{index}" for index in range(46))

    with pytest.raises(ValidationError, match="45 words"):
        blueprint(assistant_text=long_reply)


def test_non_alternating_speakers_are_rejected() -> None:
    payload = blueprint().model_dump()
    payload["utterances"][1]["speaker"] = "user"

    with pytest.raises(ValidationError, match="alternate"):
        DialogueBlueprint.model_validate(payload)


def test_dialogue_must_end_with_character_response() -> None:
    payload = blueprint().model_dump()
    payload["utterances"] = payload["utterances"][:-1]

    with pytest.raises(ValidationError, match="end with the assistant"):
        DialogueBlueprint.model_validate(payload)


def test_invalid_timing_kind_is_rejected() -> None:
    payload = blueprint().model_dump()
    payload["timing"] = "ramble"

    with pytest.raises(ValidationError):
        DialogueBlueprint.model_validate(payload)


def test_held_out_hash_leakage_is_rejected() -> None:
    train = blueprint()
    held_out = blueprint(dialogue_id="held-out-copy", split=Split.HELD_OUT)

    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage([train], [held_out])


def test_rendered_example_requires_moshi_audio_contract() -> None:
    rendered = RenderedExample(
        id="aragorn-road-0001",
        blueprint_hash="b" * 64,
        wav_path=Path("audio/aragorn-road-0001.wav"),
        alignment_path=Path("audio/aragorn-road-0001.json"),
        duration_sec=12.5,
        split=Split.TRAIN,
        voice_license="private-apple-system-voice",
    )

    assert rendered.sample_rate == 24_000
    assert rendered.channels == 2

    with pytest.raises(ValidationError):
        RenderedExample.model_validate({**rendered.model_dump(), "channels": 1})
