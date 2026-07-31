from array import array
import hashlib
import json
from pathlib import Path
import wave

import pytest

from tolkien_voice.evaluation import PersonaProbe
from tolkien_voice.training.schema import CurriculumKind, DialogueBlueprint, Split, TimingKind
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


def _write_v6_source_dataset(root: Path) -> None:
    (root / "audio").mkdir(parents=True)
    (root / "manifests").mkdir()
    blueprint = {
        "schema_version": "1.0",
        "id": "aragorn-existing-001",
        "character": "aragorn",
        "scenario": "An existing v6 scenario.",
        "curriculum": "general",
        "timing": "normal",
        "split": "train",
        "utterances": [
            {"speaker": "user", "text": "Hello there."},
            {"speaker": "assistant", "text": "Well met, traveler."},
        ],
        "provenance": {
            "teacher_model": "existing-teacher",
            "teacher_revision": "v1",
            "character_card_sha256": "a" * 64,
            "prompt_template_revision": "v1",
            "seed": 1,
            "voice_license": "existing-license",
        },
    }
    (root / "dialogues.jsonl").write_text(json.dumps(blueprint) + "\n")
    (root / "audio" / "aragorn-existing-001.wav").write_bytes(b"fake-wav-bytes")
    train_row = {"path": str((root / "audio" / "aragorn-existing-001.wav").resolve()), "duration": 3.0}
    (root / "manifests" / "train.jsonl").write_text(json.dumps(train_row) + "\n")
    validation_row = {"path": str((root / "audio" / "aragorn-existing-001.wav").resolve()), "duration": 3.0}
    (root / "manifests" / "validation.jsonl").write_text(json.dumps(validation_row) + "\n")


def _write_self_distillation_output(root: Path) -> None:
    (root / "audio").mkdir(parents=True)
    (root / "manifests").mkdir()
    wav_path = root / "audio" / "aragorn-lore-rangers-self-distill.wav"
    samples = array("h", [100] * 2400)
    with wave.open(str(wav_path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(samples.tobytes())
    alignment_path = root / "audio" / "aragorn-lore-rangers-self-distill.json"
    alignment_path.write_text(
        json.dumps(
            {
                "alignments": [["Aye,", [3.0, 3.2], "SPEAKER_MAIN"], ["indeed.", [3.2, 3.4], "SPEAKER_MAIN"]],
                "system_prompt": "<system>You are Aragorn.<system>",
            }
        )
    )
    train_row = {"path": str(wav_path.resolve()), "duration": 0.05}
    (root / "manifests" / "train.jsonl").write_text(json.dumps(train_row) + "\n")


def _write_probes_manifest(path: Path) -> None:
    path.write_text(
        """
version: 1
held_out: true
characters:
  aragorn:
    - id: aragorn-lore-rangers
      category: lore
      prompt: What did your years leading the Rangers of the North teach you?
      expected_traits: [grounded, resolute]
"""
    )


def test_build_derived_dataset_merges_without_touching_the_source(tmp_path: Path) -> None:
    v6_source = tmp_path / "mvp-v6-decoupled"
    _write_v6_source_dataset(v6_source)
    self_distillation_output = tmp_path / "self-distill-output"
    _write_self_distillation_output(self_distillation_output)
    probes_path = tmp_path / "probes.yaml"
    _write_probes_manifest(probes_path)
    character_dir = tmp_path / "characters"
    character_dir.mkdir()
    (character_dir / "aragorn.yaml").write_text("id: aragorn\nname: Aragorn\n")
    output_directory = tmp_path / "derived"

    original_v6_dialogues = (v6_source / "dialogues.jsonl").read_text()
    original_v6_validation = (v6_source / "manifests" / "validation.jsonl").read_text()

    from tolkien_voice.training.self_distillation_corpus import build_derived_dataset

    summary = build_derived_dataset(
        v6_source_dataset=v6_source,
        self_distillation_output=self_distillation_output,
        probes_manifest_path=probes_path,
        character_directory=character_dir,
        teacher_revision="tolkien-voice/personaplex:3428dfd",
        seed=20260730,
        output_directory=output_directory,
    )

    assert summary["self_distillation_examples"] == 1
    # Source v6 dataset is completely untouched.
    assert (v6_source / "dialogues.jsonl").read_text() == original_v6_dialogues
    assert (v6_source / "manifests" / "validation.jsonl").read_text() == original_v6_validation

    # Derived dialogues.jsonl has both the original blueprint and the new one.
    derived_dialogues = [
        json.loads(line)
        for line in (output_directory / "dialogues.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert {item["id"] for item in derived_dialogues} == {
        "aragorn-existing-001",
        "aragorn-lore-rangers-self-distill",
    }
    new_blueprint = next(
        item for item in derived_dialogues if item["id"] == "aragorn-lore-rangers-self-distill"
    )
    assert new_blueprint["timing"] == "interruption"
    assert new_blueprint["curriculum"] == "lore"

    # Validation is byte-identical to the source.
    assert (
        output_directory / "manifests" / "validation.jsonl"
    ).read_text() == original_v6_validation

    # Train manifest has both rows; the self-distillation row's path now
    # points into the derived dataset's own copied audio, the v6 row's
    # path is untouched (still pointing at the original v6 corpus).
    derived_train_rows = [
        json.loads(line)
        for line in (output_directory / "manifests" / "train.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(derived_train_rows) == 2
    self_distill_row = next(
        row for row in derived_train_rows if "self-distill" in row["path"]
    )
    assert Path(self_distill_row["path"]).parent == (
        output_directory / "self_distillation_audio"
    ).resolve()
    assert Path(self_distill_row["path"]).is_file()
    v6_row = next(row for row in derived_train_rows if "self-distill" not in row["path"])
    assert Path(v6_row["path"]) == (v6_source / "audio" / "aragorn-existing-001.wav").resolve()

    # The copied alignment JSON must carry blueprint_hash matching the
    # synthesized blueprint's own canonical_hash -- conditioning.py's
    # annotate_system_prompts raises on any mismatch (or absence), and the
    # self-distillation pipeline never wrote this field itself since no
    # DialogueBlueprint existed yet at that stage.
    copied_alignment = json.loads(
        Path(self_distill_row["path"]).with_suffix(".json").read_text()
    )
    assert copied_alignment["blueprint_hash"] == DialogueBlueprint.model_validate(
        new_blueprint
    ).canonical_hash


def test_build_derived_dataset_raises_on_missing_v6_source_file(tmp_path: Path) -> None:
    v6_source = tmp_path / "mvp-v6-decoupled"
    (v6_source / "audio").mkdir(parents=True)
    (v6_source / "manifests").mkdir()
    # dialogues.jsonl deliberately missing.
    self_distillation_output = tmp_path / "self-distill-output"
    _write_self_distillation_output(self_distillation_output)
    probes_path = tmp_path / "probes.yaml"
    _write_probes_manifest(probes_path)
    character_dir = tmp_path / "characters"
    character_dir.mkdir()
    (character_dir / "aragorn.yaml").write_text("id: aragorn\n")

    from tolkien_voice.training.self_distillation_corpus import build_derived_dataset

    with pytest.raises(ValueError, match="dialogues.jsonl"):
        build_derived_dataset(
            v6_source_dataset=v6_source,
            self_distillation_output=self_distillation_output,
            probes_manifest_path=probes_path,
            character_directory=character_dir,
            teacher_revision="tolkien-voice/personaplex:3428dfd",
            seed=20260730,
            output_directory=tmp_path / "derived",
        )
