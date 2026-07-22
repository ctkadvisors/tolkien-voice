from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import random
from statistics import fmean
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field


Variant = Literal["base", "adapter"]


class ProbeObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    probe_id: str = Field(min_length=1)
    blinded_arm: str = Field(min_length=1)
    character: Literal["aragorn", "sauron"]
    category: str = Field(min_length=1)
    transcript: str
    persona_score: float = Field(ge=0, le=1)
    lore_score: float = Field(ge=0, le=1)
    confused_as_other_character: bool
    first_audio_ms: int = Field(ge=0)
    interruption_response_ms: int | None = Field(default=None, gt=0)
    underrun_count: int = Field(ge=0)
    audio_duration_ms: int = Field(gt=0)


@dataclass(frozen=True)
class BlindedAssignment:
    probe_id: str
    arm: str
    order: int


class ConfidenceInterval(NamedTuple):
    lower: float
    estimate: float
    upper: float


@dataclass(frozen=True)
class ArmMetrics:
    persona_mean: float
    lore_mean: float
    confusion_rate: float
    first_audio_mean_ms: float
    interruption_p95_ms: float
    underrun_mean: float


@dataclass(frozen=True)
class ABReleaseReport:
    passed: bool
    failures: tuple[str, ...]
    sample_pairs: int
    base: ArmMetrics
    adapter: ArmMetrics
    persona_delta_ci95: ConfidenceInterval
    lore_delta_ci95: ConfidenceInterval


def build_blinded_assignments(
    probe_ids: list[str], *, seed: int
) -> tuple[list[BlindedAssignment], dict[str, Variant]]:
    if not probe_ids or len(probe_ids) != len(set(probe_ids)):
        raise ValueError("probe IDs must be non-empty and unique")
    rng = random.Random(seed)
    digest = hashlib.sha256(str(seed).encode()).hexdigest()
    arms = (f"arm-{digest[:10]}", f"arm-{digest[10:20]}")
    variants: list[Variant] = ["base", "adapter"]
    rng.shuffle(variants)
    key = dict(zip(arms, variants, strict=True))
    assignments: list[BlindedAssignment] = []
    order = 0
    probes = list(probe_ids)
    rng.shuffle(probes)
    for probe_id in probes:
        probe_arms = list(arms)
        rng.shuffle(probe_arms)
        for arm in probe_arms:
            assignments.append(BlindedAssignment(probe_id=probe_id, arm=arm, order=order))
            order += 1
    return assignments, key


def paired_confidence_interval(
    base: list[float],
    adapter: list[float],
    *,
    rng: random.Random,
    samples: int = 10_000,
) -> ConfidenceInterval:
    if not base or len(base) != len(adapter):
        raise ValueError("paired confidence interval requires equal non-empty samples")
    if samples < 100:
        raise ValueError("bootstrap requires at least 100 samples")
    differences = [right - left for left, right in zip(base, adapter, strict=True)]
    estimate = fmean(differences)
    if len(set(differences)) == 1:
        return ConfidenceInterval(estimate, estimate, estimate)
    bootstrapped = sorted(
        fmean(rng.choice(differences) for _ in differences) for _ in range(samples)
    )
    lower_index = max(0, int(samples * 0.025) - 1)
    upper_index = min(samples - 1, int(samples * 0.975))
    return ConfidenceInterval(bootstrapped[lower_index], estimate, bootstrapped[upper_index])


def evaluate_ab_release(
    observations: list[ProbeObservation],
    *,
    key: dict[str, Variant],
    bootstrap_seed: int,
) -> ABReleaseReport:
    if set(key.values()) != {"base", "adapter"}:
        raise ValueError("blind key must map exactly one arm to base and one to adapter")
    paired: dict[str, dict[Variant, ProbeObservation]] = defaultdict(dict)
    for observation in observations:
        try:
            variant = key[observation.blinded_arm]
        except KeyError as error:
            raise ValueError(f"unknown blinded arm: {observation.blinded_arm}") from error
        if variant in paired[observation.probe_id]:
            raise ValueError(f"duplicate {variant} observation for {observation.probe_id}")
        paired[observation.probe_id][variant] = observation
    incomplete = [probe for probe, values in paired.items() if set(values) != {"base", "adapter"}]
    if incomplete:
        raise ValueError(f"unpaired observations: {', '.join(sorted(incomplete))}")
    if not paired:
        raise ValueError("A/B evaluation requires observations")

    ordered = [paired[probe] for probe in sorted(paired)]
    base_items = [pair["base"] for pair in ordered]
    adapter_items = [pair["adapter"] for pair in ordered]
    base_metrics = _metrics(base_items)
    adapter_metrics = _metrics(adapter_items)
    persona_ci = paired_confidence_interval(
        [item.persona_score for item in base_items],
        [item.persona_score for item in adapter_items],
        rng=random.Random(bootstrap_seed),
    )
    lore_ci = paired_confidence_interval(
        [item.lore_score for item in base_items],
        [item.lore_score for item in adapter_items],
        rng=random.Random(bootstrap_seed + 1),
    )

    failures: list[str] = []
    if persona_ci.lower <= 0:
        failures.append("persona improvement is not positive at 95% paired confidence")
    if lore_ci.lower <= 0:
        failures.append("lore improvement is not positive at 95% paired confidence")
    if adapter_metrics.confusion_rate > 0.05:
        failures.append("adapter character confusion exceeds 5%")
    if adapter_metrics.interruption_p95_ms > 1000:
        failures.append("adapter interruption p95 exceeds 1,000 ms")
    if _regressed_more_than_ten_percent(
        base_metrics.first_audio_mean_ms, adapter_metrics.first_audio_mean_ms
    ):
        failures.append("adapter first-audio latency regressed by more than 10%")
    if _regressed_more_than_ten_percent(
        base_metrics.underrun_mean, adapter_metrics.underrun_mean
    ):
        failures.append("adapter underrun rate regressed by more than 10%")
    return ABReleaseReport(
        passed=not failures,
        failures=tuple(failures),
        sample_pairs=len(ordered),
        base=base_metrics,
        adapter=adapter_metrics,
        persona_delta_ci95=persona_ci,
        lore_delta_ci95=lore_ci,
    )


def _metrics(items: list[ProbeObservation]) -> ArmMetrics:
    interruption_values = [
        float(item.interruption_response_ms)
        for item in items
        if item.interruption_response_ms is not None
    ]
    return ArmMetrics(
        persona_mean=fmean(item.persona_score for item in items),
        lore_mean=fmean(item.lore_score for item in items),
        confusion_rate=fmean(float(item.confused_as_other_character) for item in items),
        first_audio_mean_ms=fmean(item.first_audio_ms for item in items),
        interruption_p95_ms=(
            _percentile(interruption_values, 0.95) if interruption_values else float("inf")
        ),
        underrun_mean=fmean(item.underrun_count for item in items),
    )


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * quantile + 0.999999) - 1))
    return ordered[index]


def _regressed_more_than_ten_percent(base: float, adapter: float) -> bool:
    if base == 0:
        return adapter > 0
    return adapter > base * 1.10
