from tolkien_voice.training.judge import JudgeScore
import pytest

from tolkien_voice.training.scoring import order_native_rows, score_native_rows


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


def test_native_rows_follow_precommitted_blinded_assignment_order() -> None:
    rows = [
        {"probe_id": "a", "blinded_arm": "arm-2"},
        {"probe_id": "b", "blinded_arm": "arm-1"},
        {"probe_id": "a", "blinded_arm": "arm-1"},
        {"probe_id": "b", "blinded_arm": "arm-2"},
    ]
    assignments = [
        {"probe_id": "b", "arm": "arm-2", "order": 0},
        {"probe_id": "a", "arm": "arm-1", "order": 1},
        {"probe_id": "a", "arm": "arm-2", "order": 2},
        {"probe_id": "b", "arm": "arm-1", "order": 3},
    ]

    ordered = order_native_rows(rows, assignments)

    assert [(row["probe_id"], row["blinded_arm"]) for row in ordered] == [
        ("b", "arm-2"),
        ("a", "arm-1"),
        ("a", "arm-2"),
        ("b", "arm-1"),
    ]


def test_native_row_order_rejects_incomplete_or_duplicate_results() -> None:
    rows = [
        {"probe_id": "a", "blinded_arm": "arm-1"},
        {"probe_id": "a", "blinded_arm": "arm-1"},
    ]
    assignments = [{"probe_id": "a", "arm": "arm-1", "order": 0}]

    with pytest.raises(ValueError, match="duplicate"):
        order_native_rows(rows, assignments)
