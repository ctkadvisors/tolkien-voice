from __future__ import annotations

import json
from pathlib import Path

from tolkien_voice.training.persona_manifest import build_persona_manifests
from tolkien_voice.training.schema import (
    CurriculumKind,
    DialogueBlueprint,
    Provenance,
    Split,
    TimingKind,
    Utterance,
)


def blueprint(
    item_id: str,
    *,
    character: str,
    split: Split,
    curriculum: CurriculumKind,
    timing: TimingKind,
) -> DialogueBlueprint:
    return DialogueBlueprint(
        id=item_id,
        character=character,
        scenario=f"Scenario for {item_id}",
        curriculum=curriculum,
        timing=timing,
        split=split,
        utterances=[
            Utterance(speaker="user", text="What should be done?"),
            Utterance(speaker="assistant", text="We shall answer with care."),
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


def write_jsonl(path: Path, rows: list[object]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_persona_manifest_filters_and_weights_targeted_examples(tmp_path: Path) -> None:
    blueprints = [
        blueprint(
            "aragorn-normal",
            character="aragorn",
            split=Split.TRAIN,
            curriculum=CurriculumKind.COUNSEL,
            timing=TimingKind.NORMAL,
        ),
        blueprint(
            "aragorn-lore-interrupt",
            character="aragorn",
            split=Split.TRAIN,
            curriculum=CurriculumKind.LORE,
            timing=TimingKind.INTERRUPTION,
        ),
        blueprint(
            "aragorn-validation",
            character="aragorn",
            split=Split.VALIDATION,
            curriculum=CurriculumKind.LORE,
            timing=TimingKind.NORMAL,
        ),
        blueprint(
            "sauron-normal",
            character="sauron",
            split=Split.TRAIN,
            curriculum=CurriculumKind.COUNSEL,
            timing=TimingKind.NORMAL,
        ),
    ]
    blueprint_path = tmp_path / "dialogues.jsonl"
    write_jsonl(blueprint_path, [json.loads(item.model_dump_json()) for item in blueprints])
    source_train = tmp_path / "source-train.jsonl"
    source_eval = tmp_path / "source-eval.jsonl"
    write_jsonl(
        source_train,
        [
            {"path": f"/data/audio/{item.id}.wav", "duration": 10.0}
            for item in blueprints
            if item.split == Split.TRAIN
        ],
    )
    write_jsonl(
        source_eval,
        [{"path": "/data/audio/aragorn-validation.wav", "duration": 10.0}],
    )

    summary = build_persona_manifests(
        blueprints_path=blueprint_path,
        source_train_path=source_train,
        source_eval_path=source_eval,
        character="aragorn",
        output_directory=tmp_path / "out",
    )

    train_rows = [
        json.loads(line) for line in (tmp_path / "out/train.jsonl").read_text().splitlines()
    ]
    eval_rows = [
        json.loads(line)
        for line in (tmp_path / "out/validation.jsonl").read_text().splitlines()
    ]
    assert [Path(row["path"]).stem for row in train_rows].count("aragorn-normal") == 1
    assert [Path(row["path"]).stem for row in train_rows].count(
        "aragorn-lore-interrupt"
    ) == 5
    assert [Path(row["path"]).stem for row in eval_rows] == ["aragorn-validation"]
    assert summary == {
        "character": "aragorn",
        "source_train_examples": 2,
        "weighted_train_rows": 6,
        "validation_examples": 1,
        "interruption_examples": 1,
        "target_curriculum_examples": 1,
    }
    assert (tmp_path / "out/train.jsonl").stat().st_mode & 0o777 == 0o600
