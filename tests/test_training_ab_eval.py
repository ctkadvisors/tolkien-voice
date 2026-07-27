import random

import pytest

from tolkien_voice.training.ab_eval import (
    ProbeObservation,
    build_blinded_assignments,
    evaluate_ab_release,
    paired_confidence_interval,
)


def observation(
    *,
    probe: str,
    arm: str,
    persona: float,
    lore: float,
    confused: bool = False,
    first_audio_ms: int = 500,
    interruption_ms: int = 600,
    underruns: int = 1,
) -> ProbeObservation:
    return ProbeObservation(
        probe_id=probe,
        blinded_arm=arm,
        character="aragorn" if probe.startswith("a") else "sauron",
        category="lore",
        transcript="A short native STS response.",
        persona_score=persona,
        lore_score=lore,
        confused_as_other_character=confused,
        first_audio_ms=first_audio_ms,
        interruption_response_ms=interruption_ms,
        underrun_count=underruns,
        audio_duration_ms=4000,
    )


def test_assignments_hide_variant_names_and_balance_every_probe() -> None:
    assignments, key = build_blinded_assignments(
        ["a-lore", "a-break", "s-lore", "s-break"], seed=20260722
    )

    assert {item.probe_id for item in assignments} == {
        "a-lore",
        "a-break",
        "s-lore",
        "s-break",
    }
    assert len(assignments) == 8
    assert all(item.arm not in {"base", "adapter"} for item in assignments)
    assert set(key.values()) == {"base", "adapter"}
    for probe_id in {item.probe_id for item in assignments}:
        assert len([item for item in assignments if item.probe_id == probe_id]) == 2


def test_paired_interval_is_deterministic_and_rejects_unpaired_results() -> None:
    base = [1.0, 2.0, 3.0, 4.0]
    adapter = [2.0, 3.0, 4.0, 5.0]

    first = paired_confidence_interval(base, adapter, rng=random.Random(7), samples=1000)
    second = paired_confidence_interval(base, adapter, rng=random.Random(7), samples=1000)

    assert first == second == (1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="paired"):
        paired_confidence_interval([1.0], [1.0, 2.0], rng=random.Random(1))


def test_release_passes_only_with_significant_quality_gain_and_duplex_budget() -> None:
    key = {"arm-a": "base", "arm-b": "adapter"}
    results: list[ProbeObservation] = []
    for index in range(20):
        probe = f"a-{index:02d}" if index < 10 else f"s-{index:02d}"
        results.extend(
            [
                observation(probe=probe, arm="arm-a", persona=0.45, lore=0.50),
                observation(probe=probe, arm="arm-b", persona=0.85, lore=0.90),
            ]
        )

    report = evaluate_ab_release(results, key=key, bootstrap_seed=11)

    assert report.passed is True
    assert report.failures == ()
    assert report.adapter.persona_mean > report.base.persona_mean
    assert report.persona_delta_ci95.lower > 0
    assert report.lore_delta_ci95.lower > 0
    assert report.sample_pairs == 20


@pytest.mark.parametrize(
    ("adapter_changes", "failure"),
    [
        ({"persona": 0.45, "lore": 0.50}, "persona improvement"),
        ({"confused": True}, "character confusion"),
        ({"interruption_ms": 1001}, "interruption"),
        ({"first_audio_ms": 551}, "first-audio latency"),
        ({"underruns": 2}, "underrun"),
    ],
)
def test_release_fails_each_required_gate(adapter_changes: dict, failure: str) -> None:
    key = {"arm-a": "base", "arm-b": "adapter"}
    results: list[ProbeObservation] = []
    for index in range(20):
        probe = f"a-{index:02d}" if index < 10 else f"s-{index:02d}"
        base = observation(probe=probe, arm="arm-a", persona=0.45, lore=0.50)
        values = {
            "persona": 0.85,
            "lore": 0.90,
            "confused": False,
            "first_audio_ms": 500,
            "interruption_ms": 600,
            "underruns": 1,
            **adapter_changes,
        }
        adapter = observation(probe=probe, arm="arm-b", **values)
        results.extend([base, adapter])

    report = evaluate_ab_release(results, key=key, bootstrap_seed=11)

    assert report.passed is False
    assert any(failure in item for item in report.failures)
