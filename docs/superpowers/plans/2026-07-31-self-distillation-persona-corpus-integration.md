# Self-Distillation Persona Corpus Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the self-distillation corpus consumable by `train-mvp-v6-persona.sh` by synthesizing a valid `DialogueBlueprint` per self-distillation example and building a new, derived training-dataset directory that layers the self-distillation data on top of the existing v6 corpus without modifying it. No training is invoked by this plan.

**Architecture:** One new module (`src/tolkien_voice/training/self_distillation_corpus.py`) with two pure/thin pieces: blueprint synthesis (category mapping + one function building a valid `DialogueBlueprint` per example) and derived-dataset assembly (merging dialogues/manifests, copying only the new self-distillation audio, leaving the existing v6 corpus and its validation split completely untouched). One new CLI script (`scripts/integrate-self-distillation-corpus.py`) wiring it together, mirroring this project's existing script conventions.

**Tech Stack:** Python 3.12, pydantic, pytest, existing project conventions (temp-file-then-rename writes at 0o600/0o700 where files are written, small pure functions).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-31-self-distillation-persona-corpus-integration-design.md` -- read this first for the full rationale.
- Every synthesized blueprint's `id` must exactly equal its source `RenderedExample`'s `id` (the audio filename's stem) -- `persona_manifest.py`'s `_rows_by_id` matches on this, and a mismatch silently drops the example with no error, not a crash.
- `timing` is always `TimingKind.INTERRUPTION`; `split` is always `Split.TRAIN`.
- Curriculum mapping is exact, from the spec's table: `lore`->LORE, `uncertainty`->UNCERTAINTY, `break_character`->RESISTANCE, `modern_redirect`->RESISTANCE, `concise`->GENERAL. Any other category string raises `ValueError` -- never silently default to `GENERAL`.
- The existing v6 corpus (`dialogues.jsonl`, `manifests/train.jsonl`, `manifests/validation.jsonl`, its own `audio/`) is never modified or moved -- only read and copied from. `validation.jsonl` in the derived output must be byte-identical to the source.
- `character_card_sha256` = `hashlib.sha256(card_path.read_bytes()).hexdigest()` over `configs/characters/{character}.yaml` -- the same computation `generate.py:319` already uses.
- `voice_license` is always the literal `"self-distillation-v6-continuation"`.
- This plan does not invoke `train-mvp-v6-persona.sh` or any training step -- its only output is a new directory on disk.

---

### Task 1: Blueprint synthesis

**Files:**
- Create: `src/tolkien_voice/training/self_distillation_corpus.py`
- Test: `tests/test_training_self_distillation_corpus.py`

**Interfaces:**
- Consumes: `PersonaProbe`/`ProbeCategory` (`tolkien_voice.evaluation`, unchanged), `DialogueBlueprint`/`Provenance`/`Utterance`/`CurriculumKind`/`TimingKind`/`Split` (`tolkien_voice.training.schema`, unchanged).
- Produces: `map_probe_category_to_curriculum(category: str) -> CurriculumKind` and `build_self_distillation_blueprint(...) -> DialogueBlueprint`, both consumed by Task 2.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_training_self_distillation_corpus.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training_self_distillation_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError` -- the module doesn't exist yet.

- [ ] **Step 3: Implement**

Create `src/tolkien_voice/training/self_distillation_corpus.py`:

```python
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from tolkien_voice.evaluation import PersonaProbe

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_training_self_distillation_corpus.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Run the full suite and lint**

Run: `uv run pytest -q`
Expected: PASS, no regressions.

Run: `uv run ruff check --exclude vendor .`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/tolkien_voice/training/self_distillation_corpus.py tests/test_training_self_distillation_corpus.py
git commit -m "feat: synthesize DialogueBlueprints for self-distillation examples"
```

---

### Task 2: Derived-dataset assembly and CLI

**Files:**
- Modify: `src/tolkien_voice/training/self_distillation_corpus.py`
- Test: `tests/test_training_self_distillation_corpus.py`
- Create: `scripts/integrate-self-distillation-corpus.py`

**Interfaces:**
- Consumes: `build_self_distillation_blueprint`/`map_probe_category_to_curriculum` (Task 1, unchanged), `load_persona_manifest` (`tolkien_voice.evaluation`, unchanged), `DialogueBlueprint` (unchanged).
- Produces: `build_derived_dataset(...) -> dict[str, int | str]`, invoked by the new CLI script.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_training_self_distillation_corpus.py`:

```python
import json
import shutil
import wave
from array import array


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training_self_distillation_corpus.py -v`
Expected: the two new tests FAIL with `ImportError`/`AttributeError` (`build_derived_dataset` doesn't exist yet); the four Task 1 tests still PASS.

- [ ] **Step 3: Implement `build_derived_dataset`**

Append to `src/tolkien_voice/training/self_distillation_corpus.py` (add these imports to the existing import block at the top of the file):

```python
import json
import os
import shutil

from tolkien_voice.evaluation import load_persona_manifest
```

Then append this function:

```python
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
    audio -- that data isn't moving), plus the self-distillation
    corpus's rows, with paths rewritten to point at
    output_directory/self_distillation_audio (the only audio this
    function actually copies -- v6's existing audio is left in place
    and referenced from its original location, not duplicated).

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
    audio_output_directory = output_directory / "self_distillation_audio"
    audio_output_directory.mkdir(exist_ok=True, mode=0o700)

    probes_manifest = load_persona_manifest(probes_manifest_path)

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
        shutil.copy2(alignment_path, destination_alignment)
        rewritten_rows.append({"path": str(destination_wav.resolve()), "duration": row["duration"]})

    v6_dialogues_text = (v6_source_dataset / "dialogues.jsonl").read_text(encoding="utf-8")
    combined_dialogues_lines = [v6_dialogues_text.rstrip("\n")] if v6_dialogues_text.strip() else []
    combined_dialogues_lines.extend(
        blueprint.model_dump_json() for blueprint in new_blueprints
    )
    (output_directory / "dialogues.jsonl").write_text(
        "\n".join(combined_dialogues_lines) + "\n", encoding="utf-8"
    )

    v6_train_rows = [
        json.loads(line)
        for line in (v6_source_dataset / "manifests" / "train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
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
```

Note: `_find_probe`'s helper and the `Path.removesuffix` usage require the
example's `probe_id` (after stripping `-self-distill`) to exist in
`probes_manifest_path` (the same `evals/persona/probes.yaml` the corpus
build itself used) -- this should always hold by construction, since
`build-self-distillation-corpus.py` only ever creates examples from
probes it read from that same file, but `_find_probe` raising a clear
`ValueError` rather than crashing with a confusing `KeyError` matters if
a caller ever points this at a mismatched probes file.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_training_self_distillation_corpus.py -v`
Expected: PASS, all 6 tests.

- [ ] **Step 5: Write the CLI script**

Create `scripts/integrate-self-distillation-corpus.py`:

```python
#!/usr/bin/env python3
"""Builds a new, derived training-dataset directory layering a
self-distillation corpus's examples on top of an existing v6 corpus,
without modifying the v6 corpus. Does not invoke training -- run
scripts/train-mvp-v6-persona.sh (or a script pointed at the derived
output) as a separate, explicit step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tolkien_voice.training.self_distillation_corpus import build_derived_dataset


def main() -> int:
    repository = Path(__file__).parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-source-dataset", type=Path, required=True)
    parser.add_argument("--self-distillation-output", type=Path, required=True)
    parser.add_argument(
        "--probes-manifest", type=Path, default=repository / "evals/persona/probes.yaml"
    )
    parser.add_argument(
        "--character-directory", type=Path, default=repository / "configs/characters"
    )
    parser.add_argument("--teacher-revision", required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()

    summary = build_derived_dataset(
        v6_source_dataset=args.v6_source_dataset,
        self_distillation_output=args.self_distillation_output,
        probes_manifest_path=args.probes_manifest,
        character_directory=args.character_directory,
        teacher_revision=args.teacher_revision,
        seed=args.seed,
        output_directory=args.output_directory,
    )
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Syntax/import check**

Run: `uv run python -m py_compile scripts/integrate-self-distillation-corpus.py`
Expected: no output (valid syntax).

Run: `uv run python -c "import ast; ast.parse(open('scripts/integrate-self-distillation-corpus.py').read())"`
Expected: no output.

- [ ] **Step 7: Run the full suite and lint**

Run: `uv run pytest -q`
Expected: PASS, no regressions.

Run: `uv run ruff check --exclude vendor .`
Expected: `All checks passed!`

- [ ] **Step 8: Commit**

```bash
git add src/tolkien_voice/training/self_distillation_corpus.py tests/test_training_self_distillation_corpus.py scripts/integrate-self-distillation-corpus.py
git commit -m "feat: build a derived training dataset merging self-distillation examples into the v6 corpus"
```

---

## After this task: validation against the real audit/conditioning checks (Spark, no GPU/training needed)

Run this project's existing `tolkien_voice.training.audit` (with
`--minimum-per-character 64`, matching `train-mvp-v6-persona.sh`'s own
invocation) and `tolkien_voice.training.conditioning` against the derived
dataset produced on Spark, using the real self-distillation corpus output
(`/tmp/self-distill-full3/corpus` per the 2026-07-31 live run, if it's
still present) and the real v6 `mvp-v6-decoupled` corpus at
`$data_root/datasets/mvp-v6-decoupled`, e.g.:

```bash
python scripts/integrate-self-distillation-corpus.py \
  --v6-source-dataset "$data_root/datasets/mvp-v6-decoupled" \
  --self-distillation-output /tmp/self-distill-full3/corpus \
  --teacher-revision tolkien-voice/personaplex:3428dfd \
  --output-directory "$data_root/datasets/mvp-v6-decoupled-self-distill-v1"

PYTHONPATH=src python -m tolkien_voice.training.audit \
  --blueprints "$data_root/datasets/mvp-v6-decoupled-self-distill-v1/dialogues.jsonl" \
  --minimum-per-character 64 \
  --characters aragorn sauron
PYTHONPATH=src python -m tolkien_voice.training.conditioning \
  --blueprints "$data_root/datasets/mvp-v6-decoupled-self-distill-v1/dialogues.jsonl" \
  --audio-directory "$data_root/datasets/mvp-v6-decoupled-self-distill-v1" \
  --character-directory configs/characters
```

This is the actual go/no-go check for whether the derived dataset is
trainable -- no GPU or live PersonaPlex instance required. Actually
running `train-mvp-v6-persona.sh` (which needs a small, separate change
to accept a non-default `source_dataset` path, per the spec's
Reproducibility section) remains a distinct, explicit next decision.
