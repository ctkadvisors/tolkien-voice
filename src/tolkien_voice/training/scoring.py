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
