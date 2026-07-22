from __future__ import annotations

from typing import Any, Protocol

from .ab_eval import ProbeObservation
from .judge import JudgeScore


class Judge(Protocol):
    def score(
        self,
        *,
        target_character: str,
        probe_prompt: str,
        expected_traits: list[str],
        candidate_transcript: str,
        seed: int,
    ) -> JudgeScore: ...


def order_native_rows(
    rows: list[dict[str, Any]], assignments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("probe_id", "")), str(row.get("blinded_arm", "")))
        if not all(key):
            raise ValueError("native result requires probe_id and blinded_arm")
        if key in indexed:
            raise ValueError(f"duplicate native result: {key[0]} {key[1]}")
        indexed[key] = row

    ordered_assignments = sorted(assignments, key=lambda item: int(item["order"]))
    assignment_keys = [
        (str(item.get("probe_id", "")), str(item.get("arm", "")))
        for item in ordered_assignments
    ]
    if len(assignment_keys) != len(set(assignment_keys)):
        raise ValueError("duplicate blinded assignment")
    if set(assignment_keys) != set(indexed):
        missing = sorted(set(assignment_keys) - set(indexed))
        unexpected = sorted(set(indexed) - set(assignment_keys))
        raise ValueError(
            f"native results do not match blinded assignments; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return [indexed[key] for key in assignment_keys]


def score_native_rows(
    rows: list[dict[str, Any]], *, judge: Judge, seed: int
) -> list[ProbeObservation]:
    observations: list[ProbeObservation] = []
    for index, row in enumerate(rows):
        character = str(row["character"])
        if character not in {"aragorn", "sauron"}:
            raise ValueError(f"unsupported evaluation character: {character}")
        transcript = str(row.get("text", "")).strip()
        if not transcript:
            raise ValueError(f"native probe has no transcript: {row.get('probe_id')}")
        score = judge.score(
            target_character=character,  # type: ignore[arg-type]
            probe_prompt=str(row["prompt"]),
            expected_traits=[str(item) for item in row["expected_traits"]],
            candidate_transcript=transcript,
            seed=seed + index,
        )
        identified = score.identified_character
        observations.append(
            ProbeObservation(
                probe_id=str(row["probe_id"]),
                blinded_arm=str(row["blinded_arm"]),
                character=character,  # type: ignore[arg-type]
                category=str(row["category"]),
                transcript=transcript,
                persona_score=score.persona_score,
                lore_score=score.lore_score,
                confused_as_other_character=(
                    identified in {"aragorn", "sauron"} and identified != character
                ),
                first_audio_ms=int(row["first_audio_ms"]),
                interruption_response_ms=(
                    int(row["interruption_response_ms"])
                    if row.get("interruption_response_ms") is not None
                    else None
                ),
                underrun_count=int(row["underrun_count"]),
                audio_duration_ms=int(row["audio_duration_ms"]),
            )
        )
    return observations
