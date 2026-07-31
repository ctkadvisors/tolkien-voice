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
