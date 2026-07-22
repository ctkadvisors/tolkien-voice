from __future__ import annotations

import json
import stat
import re
from pathlib import Path

import pytest
import yaml

from tolkien_voice.evaluation import (
    EXPECTED_SOURCE_REVISION,
    EvaluationSummary,
    UnsafeEvaluationManifest,
    create_eval_run,
    evaluate_acceptance,
    load_duplex_manifest,
    load_persona_manifest,
)


ROOT = Path(__file__).parents[1]
PERSONA = ROOT / "evals/persona/probes.yaml"
DUPLEX = ROOT / "evals/duplex/scenarios.yaml"
CHARACTERS = {
    "gandalf", "sauron", "galadriel", "samwise", "aragorn",
    "frodo", "treebeard", "boromir", "barliman",
}


def test_every_character_has_held_out_persona_coverage() -> None:
    manifest = load_persona_manifest(PERSONA)
    assert set(manifest) == CHARACTERS
    for character, probes in manifest.items():
        categories = [probe.category for probe in probes]
        assert categories.count("lore") >= 2, character
        for required in ("uncertainty", "break_character", "modern_redirect", "concise"):
            assert required in categories, f"{character}: {required}"


def test_duplex_manifest_covers_real_conversation_timing() -> None:
    scenarios = load_duplex_manifest(DUPLEX)
    kinds = {scenario.kind for scenario in scenarios}
    assert {"interruption", "pause", "overlap", "silence", "reconnect"} <= kinds
    interruption_offsets = {
        scenario.offset_ms for scenario in scenarios if scenario.kind == "interruption"
    }
    assert {500, 1000} <= interruption_offsets
    assert any(scenario.kind == "pause" and scenario.duration_ms == 2000 for scenario in scenarios)


def test_manifests_cannot_reference_character_training_inputs(tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe.yaml"
    unsafe.write_text(
        yaml.safe_dump(
            {
                "characters": {
                    "gandalf": [
                        {
                            "id": "leak",
                            "category": "lore",
                            "prompt": "configs/characters/gandalf.yaml",
                            "expected_traits": ["bad"],
                        }
                    ]
                }
            }
        )
    )
    with pytest.raises(UnsafeEvaluationManifest, match="configs/characters"):
        load_persona_manifest(unsafe)


def test_eval_runs_are_private_and_outside_the_repository(tmp_path: Path) -> None:
    run = create_eval_run(tmp_path, characters=("gandalf", "sauron"))
    assert run.parent == tmp_path / "eval-runs"
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", run.name)
    assert (run / "gandalf-summary.json").is_file()
    assert stat.S_IMODE((run / "gandalf-summary.json").stat().st_mode) == 0o600
    summary = json.loads((run / "gandalf-summary.json").read_text())
    assert summary["source_revision"] == EXPECTED_SOURCE_REVISION
    assert summary["interruption_response_ms"] == 0
    assert summary["session_complete"] is False


def measured_summary(character: str, voice: str) -> EvaluationSummary:
    return EvaluationSummary(
        character=character,
        voice=voice,
        source_revision=EXPECTED_SOURCE_REVISION,
        first_audio_ms=520,
        interruption_response_ms=850,
        persona_pass_rate=0.9,
        lore_pass_rate=0.8,
        voice_consistent=True,
        session_duration_seconds=300,
        session_complete=True,
        human_notes="Completed manual acceptance run.",
    )


def test_acceptance_requires_two_complete_measured_sessions() -> None:
    passing = [measured_summary("gandalf", "NATM1.pt"), measured_summary("sauron", "NATM3.pt")]
    result = evaluate_acceptance(passing)
    assert result.passed is True
    assert result.failures == ()

    unmeasured = passing[0].model_copy(update={"interruption_response_ms": 0})
    result = evaluate_acceptance([unmeasured, passing[1]])
    assert result.passed is False
    assert any("unmeasured" in failure for failure in result.failures)

    incomplete = passing[1].model_copy(update={"session_complete": False})
    result = evaluate_acceptance([passing[0], incomplete])
    assert result.passed is False
    assert any("five-minute" in failure for failure in result.failures)


def test_acceptance_rejects_slow_interruption_and_zero_scores() -> None:
    gandalf = measured_summary("gandalf", "NATM1.pt").model_copy(
        update={"interruption_response_ms": 1001, "persona_pass_rate": 0.0}
    )
    sauron = measured_summary("sauron", "NATM3.pt")
    result = evaluate_acceptance([gandalf, sauron])
    assert result.passed is False
    assert any("1,000 ms" in failure for failure in result.failures)
    assert any("persona" in failure and "unmeasured" in failure for failure in result.failures)
