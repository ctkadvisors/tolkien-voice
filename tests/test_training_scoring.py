from tolkien_voice.training.judge import JudgeScore
from tolkien_voice.training.scoring import score_native_rows


def test_native_rows_are_scored_blindly_and_assembled_for_ab_gate() -> None:
    calls: list[dict] = []

    class FakeJudge:
        def score(self, **kwargs):
            calls.append(kwargs)
            return JudgeScore(
                persona_score=0.9,
                lore_score=0.8,
                identified_character="aragorn",
                rationale="grounded",
            )

    rows = [
        {
            "probe_id": "aragorn-lore-paths",
            "blinded_arm": "arm-c4f72",
            "character": "aragorn",
            "category": "lore",
            "prompt": "Why did you choose the Paths of the Dead?",
            "expected_traits": ["grounded", "resolute"],
            "text": "There was no other road by which I could arrive in time.",
            "first_audio_ms": 420,
            "underrun_count": 1,
            "audio_duration_ms": 5000,
        }
    ]

    observations = score_native_rows(rows, judge=FakeJudge(), seed=90)  # type: ignore[arg-type]

    assert len(observations) == 1
    assert observations[0].blinded_arm == "arm-c4f72"
    assert observations[0].persona_score == 0.9
    assert observations[0].confused_as_other_character is False
    assert calls[0]["seed"] == 90
    assert "blinded_arm" not in calls[0]
