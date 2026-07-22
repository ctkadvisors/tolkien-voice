from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator
import yaml


EXPECTED_SOURCE_REVISION = "3428dfd95309a7f3c84fd93259ded0f810d1ff91"
EXPECTED_ACCEPTANCE_VOICES = {"gandalf": "NATM1.pt", "sauron": "NATM3.pt"}
ProbeCategory = Literal["lore", "uncertainty", "break_character", "modern_redirect", "concise"]
DuplexKind = Literal["interruption", "pause", "overlap", "silence", "reconnect"]


class UnsafeEvaluationManifest(ValueError):
    """An evaluation manifest leaks a training/persona input."""


class PersonaProbe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    category: ProbeCategory
    prompt: str
    expected_traits: list[str] = Field(min_length=1)


class DuplexScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: DuplexKind
    instruction: str
    expected_traits: list[str] = Field(min_length=1)
    offset_ms: int = Field(default=0, ge=0)
    duration_ms: int = Field(default=0, ge=0)


class EvaluationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    character: str
    voice: str
    source_revision: str
    first_audio_ms: int = Field(default=0, ge=0)
    interruption_response_ms: int = Field(default=0, ge=0)
    persona_pass_rate: float = Field(default=0.0, ge=0, le=1)
    lore_pass_rate: float = Field(default=0.0, ge=0, le=1)
    voice_consistent: bool = False
    session_duration_seconds: int = Field(default=0, ge=0)
    session_complete: bool = False
    human_notes: str = ""

    @field_validator("source_revision")
    @classmethod
    def source_must_be_pinned(cls, value: str) -> str:
        if value != EXPECTED_SOURCE_REVISION:
            raise ValueError("evaluation source revision is not the pinned PersonaPlex revision")
        return value


@dataclass(frozen=True)
class AcceptanceResult:
    passed: bool
    failures: tuple[str, ...]


def _load_manifest(path: Path) -> dict:
    source = path.read_text()
    if "configs/characters" in source or "configs\\characters" in source:
        raise UnsafeEvaluationManifest("manifest must not reference configs/characters")
    payload = yaml.safe_load(source)
    if not isinstance(payload, dict):
        raise ValueError(f"evaluation manifest must be a mapping: {path}")
    return payload


def load_persona_manifest(path: Path) -> dict[str, list[PersonaProbe]]:
    payload = _load_manifest(path)
    characters = payload.get("characters")
    if not isinstance(characters, dict):
        raise ValueError("persona manifest requires a characters mapping")
    result = {
        character: [PersonaProbe.model_validate(probe) for probe in probes]
        for character, probes in characters.items()
    }
    ids = [probe.id for probes in result.values() for probe in probes]
    if len(ids) != len(set(ids)):
        raise ValueError("persona probe IDs must be unique")
    return result


def load_duplex_manifest(path: Path) -> list[DuplexScenario]:
    payload = _load_manifest(path)
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("duplex manifest requires a scenarios list")
    result = [DuplexScenario.model_validate(scenario) for scenario in scenarios]
    if len({scenario.id for scenario in result}) != len(result):
        raise ValueError("duplex scenario IDs must be unique")
    return result


def create_eval_run(
    data_root: Path,
    *,
    characters: tuple[str, ...],
    now: datetime | None = None,
    nonce: str | None = None,
) -> Path:
    unknown = set(characters) - set(EXPECTED_ACCEPTANCE_VOICES)
    if unknown:
        raise ValueError(f"unsupported acceptance character(s): {', '.join(sorted(unknown))}")
    repository = Path(__file__).parents[2].resolve()
    resolved_root = data_root.expanduser().resolve()
    if resolved_root == repository or resolved_root.is_relative_to(repository):
        raise ValueError("evaluation outputs must be outside the repository")
    timestamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = nonce or uuid4().hex[:8]
    run = resolved_root / "eval-runs" / f"{timestamp}-{suffix}"
    run.mkdir(parents=True, mode=0o700)
    run.chmod(0o700)
    for character in characters:
        summary = EvaluationSummary(
            character=character,
            voice=EXPECTED_ACCEPTANCE_VOICES[character],
            source_revision=EXPECTED_SOURCE_REVISION,
        )
        summary_path = run / f"{character}-summary.json"
        summary_path.write_text(json.dumps(summary.model_dump(), indent=2) + "\n")
        summary_path.chmod(0o600)
    return run


def evaluate_acceptance(summaries: list[EvaluationSummary]) -> AcceptanceResult:
    failures: list[str] = []
    by_character: dict[str, EvaluationSummary] = {}
    for summary in summaries:
        if summary.character in by_character:
            failures.append(f"{summary.character}: duplicate summary")
        by_character[summary.character] = summary

    for character, expected_voice in EXPECTED_ACCEPTANCE_VOICES.items():
        summary = by_character.get(character)
        if summary is None:
            failures.append(f"{character}: missing five-minute acceptance summary")
            continue
        if summary.voice != expected_voice:
            failures.append(f"{character}: expected voice {expected_voice}")
        if not summary.session_complete or summary.session_duration_seconds < 300:
            failures.append(f"{character}: five-minute session is not marked complete")
        if summary.first_audio_ms <= 0:
            failures.append(f"{character}: first-audio latency is unmeasured")
        if summary.interruption_response_ms <= 0:
            failures.append(f"{character}: interruption response is unmeasured")
        elif summary.interruption_response_ms > 1000:
            failures.append(f"{character}: interruption response exceeds 1,000 ms")
        if summary.persona_pass_rate <= 0:
            failures.append(f"{character}: persona pass rate is unmeasured")
        if summary.lore_pass_rate <= 0:
            failures.append(f"{character}: lore pass rate is unmeasured")
        if not summary.voice_consistent:
            failures.append(f"{character}: voice consistency is not confirmed")
    return AcceptanceResult(passed=not failures, failures=tuple(failures))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a private Tolkien Voice evaluation run")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--characters", nargs="+", default=["gandalf", "sauron"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run = create_eval_run(args.data_root, characters=tuple(args.characters))
    print(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
