from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from tolkien_voice.evaluation import PersonaProbe, load_persona_manifest

from .schema import (
    CurriculumKind,
    DialogueBlueprint,
    MAX_ASSISTANT_WORDS,
    Provenance,
    Split,
    TimingKind,
    Utterance,
)

VOICE_LICENSE = "self-distillation-v6-continuation"
TEACHER_MODEL = "personaplex-7b-v1-self-distillation"
PROMPT_TEMPLATE_REVISION = "self-distillation-v1"

_CATEGORY_TO_CURRICULUM: dict[str, CurriculumKind] = {
    "lore": CurriculumKind.LORE,
    "uncertainty": CurriculumKind.UNCERTAINTY,
    "break_character": CurriculumKind.RESISTANCE,
    "modern_redirect": CurriculumKind.RESISTANCE,
    "concise": CurriculumKind.GENERAL,
}


def map_probe_category_to_curriculum(category: str) -> CurriculumKind:
    """Maps a probes.yaml probe category onto a training CurriculumKind,
    per the mapping table in
    docs/superpowers/specs/2026-07-31-self-distillation-persona-corpus-integration-design.md.
    Raises rather than silently defaulting to GENERAL for an unrecognized
    category, so a future new probe category can't map wrong unnoticed.
    """
    try:
        return _CATEGORY_TO_CURRICULUM[category]
    except KeyError:
        raise ValueError(f"unrecognized probe category: {category!r}") from None


def build_self_distillation_blueprint(
    *,
    example_id: str,
    character: str,
    probe: PersonaProbe,
    alignment_payload: dict[str, Any],
    character_card_path: Path,
    teacher_revision: str,
    seed: int,
) -> DialogueBlueprint:
    """Synthesizes a minimal, valid DialogueBlueprint for one
    self-distillation corpus example, so persona_manifest.py's existing
    character/split/curriculum/timing filtering and reweighting logic
    (which reads blueprint metadata only, never audio or manifest row
    content) picks this example up correctly -- tagged
    TimingKind.INTERRUPTION, since every self-distillation example is by
    construction an interruption-response instance, giving it the same
    interruption-reweighting boost persona_manifest.py already applies.

    example_id must exactly equal the corresponding RenderedExample's id
    (the audio filename's stem) -- persona_manifest.py's row lookup
    matches on this, and a mismatch silently drops the example rather
    than raising.
    """
    words = [word for word, _, _ in alignment_payload["alignments"]]
    truncated = words[:MAX_ASSISTANT_WORDS]
    if len(words) > MAX_ASSISTANT_WORDS:
        # This should be rare -- transcripts are yield-gated to fast
        # responses -- but per the design spec, losing real content to a
        # blueprint-validity constraint must be visible, not silent.
        print(
            f"self-distillation transcript truncated for {example_id}: "
            f"{len(words)} words -> {MAX_ASSISTANT_WORDS}"
        )
    assistant_text = " ".join(truncated)

    character_card_sha256 = hashlib.sha256(character_card_path.read_bytes()).hexdigest()

    return DialogueBlueprint(
        id=example_id,
        character=character,
        scenario=probe.prompt,
        curriculum=map_probe_category_to_curriculum(probe.category),
        timing=TimingKind.INTERRUPTION,
        split=Split.TRAIN,
        utterances=[
            Utterance(speaker="user", text=probe.prompt),
            Utterance(speaker="assistant", text=assistant_text),
        ],
        provenance=Provenance(
            teacher_model=TEACHER_MODEL,
            teacher_revision=teacher_revision,
            character_card_sha256=character_card_sha256,
            prompt_template_revision=PROMPT_TEMPLATE_REVISION,
            seed=seed,
            voice_license=VOICE_LICENSE,
        ),
    )


_REQUIRED_V6_SOURCE_FILES = (
    "dialogues.jsonl",
    "manifests/train.jsonl",
    "manifests/validation.jsonl",
)


def _find_probe(
    manifest: dict[str, list[PersonaProbe]], probe_id: str
) -> tuple[str, PersonaProbe]:
    for character, probes in manifest.items():
        for probe in probes:
            if probe.id == probe_id:
                return character, probe
    raise ValueError(f"probe {probe_id!r} not found in any character's probe list")


def _container_datasets_root(v6_train_rows: list[dict[str, Any]], v6_source_dataset: Path) -> Path:
    """Derives the container-mount path prefix (e.g. "/data") that
    train-mvp-v6-persona.sh's docker run maps onto the real host data_root
    (--volume "$data_root:/data"), by reading it back out of v6's own
    already-established manifest rows -- v6_source_dataset/manifests/train.jsonl
    already stores paths written in this container convention (e.g.
    "/data/datasets/mvp-v6-decoupled/audio/foo.wav"), not real host-filesystem
    absolute paths, since that's what the training container can actually see.
    A path built from Path.resolve() on the host would be invisible inside the
    container and every such row would silently fail to load during training.
    """
    if not v6_train_rows:
        raise ValueError("v6 source train manifest is empty; cannot derive container path root")
    sample_path = Path(v6_train_rows[0]["path"]).as_posix()
    marker = f"/datasets/{v6_source_dataset.name}/"
    index = sample_path.find(marker)
    if index == -1:
        raise ValueError(
            "v6 source train manifest path does not match the expected "
            f".../datasets/{v6_source_dataset.name}/... container-path convention: "
            f"{sample_path!r}"
        )
    return Path(sample_path[:index]) / "datasets"


def build_derived_dataset(
    *,
    v6_source_dataset: Path,
    self_distillation_output: Path,
    probes_manifest_path: Path,
    character_directory: Path,
    teacher_revision: str,
    seed: int,
    output_directory: Path,
) -> dict[str, int | str]:
    """Builds a new, derived training-dataset directory layering the
    self-distillation corpus's examples on top of the existing v6
    corpus, without modifying v6_source_dataset at all.

    output_directory/dialogues.jsonl = v6's dialogues.jsonl content,
    unchanged, plus one synthesized blueprint per self-distillation
    example, appended.

    output_directory/manifests/train.jsonl = v6's train.jsonl rows,
    completely unchanged (still pointing at v6_source_dataset's own
    audio for actual training data loading -- that data isn't moving),
    plus the self-distillation corpus's rows, with paths rewritten to
    point at output_directory/audio -- written using the SAME
    container-mount path convention ("/data/datasets/...") already used
    by v6's own manifest rows (see _container_datasets_root), since
    training reads these paths from inside a Docker container that only
    has $data_root mounted at /data, not the real host filesystem paths
    Path.resolve() would otherwise produce.

    output_directory/audio = a copy of v6_source_dataset's own audio
    contents plus the self-distillation corpus's WAV+JSON pairs, unified
    into one flat directory addressable by blueprint id. This exists
    because tolkien_voice.training.conditioning.annotate_system_prompts
    looks up every blueprint's alignment JSON as
    audio_directory/f"{id}.json" in a single directory, never consulting
    a manifest path -- v6_source_dataset itself is still never written to.

    output_directory/manifests/validation.jsonl = copied byte-for-byte
    from v6_source_dataset -- self-distillation examples are never
    added to validation.
    """
    for relative_path in _REQUIRED_V6_SOURCE_FILES:
        source_file = v6_source_dataset / relative_path
        if not source_file.is_file():
            raise ValueError(f"v6 source dataset is missing required file: {relative_path}")

    output_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    (output_directory / "manifests").mkdir(exist_ok=True, mode=0o700)
    audio_output_directory = output_directory / "audio"
    audio_output_directory.mkdir(exist_ok=True, mode=0o700)

    # tolkien_voice.training.conditioning.annotate_system_prompts looks up
    # every blueprint's alignment JSON as audio_directory / f"{id}.json" in a
    # single flat directory -- it never consults the train/validation
    # manifest's own path field. v6's own audio+json is copied in here too
    # (never mutating v6_source_dataset itself) so the derived dataset has one
    # unified audio/ directory usable the same way train-mvp-v6-persona.sh
    # already uses v6's own "$source_dataset/audio". train.jsonl's v6 rows
    # keep pointing at v6_source_dataset's original audio location below --
    # that's what training itself actually reads; this copy exists only so
    # conditioning has a single directory covering every blueprint id.
    for source_file in (v6_source_dataset / "audio").iterdir():
        if source_file.is_file():
            shutil.copy2(source_file, audio_output_directory / source_file.name)

    probes_manifest = load_persona_manifest(probes_manifest_path)

    v6_train_rows = [
        json.loads(line)
        for line in (v6_source_dataset / "manifests" / "train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    container_datasets_root = _container_datasets_root(v6_train_rows, v6_source_dataset)
    container_audio_directory = container_datasets_root / output_directory.name / "audio"

    self_distillation_train_rows = [
        json.loads(line)
        for line in (self_distillation_output / "manifests" / "train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    new_blueprints: list[DialogueBlueprint] = []
    rewritten_rows: list[dict[str, Any]] = []
    for row in self_distillation_train_rows:
        source_wav_path = Path(row["path"])
        example_id = source_wav_path.stem
        probe_id = example_id.removesuffix("-self-distill")
        character, probe = _find_probe(probes_manifest, probe_id)

        alignment_path = source_wav_path.with_suffix(".json")
        alignment_payload = json.loads(alignment_path.read_text(encoding="utf-8"))
        character_card_path = character_directory / f"{character}.yaml"
        blueprint = build_self_distillation_blueprint(
            example_id=example_id,
            character=character,
            probe=probe,
            alignment_payload=alignment_payload,
            character_card_path=character_card_path,
            teacher_revision=teacher_revision,
            seed=seed,
        )
        new_blueprints.append(blueprint)

        destination_wav = audio_output_directory / source_wav_path.name
        destination_alignment = audio_output_directory / alignment_path.name
        shutil.copy2(source_wav_path, destination_wav)
        # tolkien_voice.training.conditioning.annotate_system_prompts requires
        # each alignment JSON to carry blueprint_hash == the DialogueBlueprint's
        # own canonical_hash. The self-distillation pipeline (self_distillation.py)
        # writes this alignment JSON before any DialogueBlueprint exists for it,
        # so it never has this field -- write it now that the blueprint is built.
        annotated_alignment_payload = {
            **alignment_payload,
            "blueprint_hash": blueprint.canonical_hash,
        }
        destination_alignment.write_text(
            json.dumps(annotated_alignment_payload, indent=2) + "\n", encoding="utf-8"
        )
        destination_alignment.chmod(0o600)
        container_wav_path = container_audio_directory / destination_wav.name
        rewritten_rows.append(
            {"path": container_wav_path.as_posix(), "duration": row["duration"]}
        )

    v6_dialogues_text = (v6_source_dataset / "dialogues.jsonl").read_text(encoding="utf-8")
    combined_dialogues_lines = [v6_dialogues_text.rstrip("\n")] if v6_dialogues_text.strip() else []
    combined_dialogues_lines.extend(
        blueprint.model_dump_json() for blueprint in new_blueprints
    )
    (output_directory / "dialogues.jsonl").write_text(
        "\n".join(combined_dialogues_lines) + "\n", encoding="utf-8"
    )

    combined_train_rows = v6_train_rows + rewritten_rows
    with (output_directory / "manifests" / "train.jsonl").open("w", encoding="utf-8") as handle:
        for row in combined_train_rows:
            handle.write(json.dumps(row) + "\n")

    shutil.copy2(
        v6_source_dataset / "manifests" / "validation.jsonl",
        output_directory / "manifests" / "validation.jsonl",
    )

    return {
        "self_distillation_examples": len(new_blueprints),
        "v6_existing_examples": len(v6_train_rows),
        "total_train_examples": len(combined_train_rows),
        "output_directory": str(output_directory.resolve()),
    }
