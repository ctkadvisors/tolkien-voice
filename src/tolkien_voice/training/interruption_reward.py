from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from .judge import JudgeClient
from .native_probe import measure_interruption_response


class InterruptionRewardResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    yield_delay_reward: float
    quality_reward: float
    delay_seconds: float | None


def compute_yield_delay_reward(
    frames: list[tuple[float, float]],
    *,
    barge_in_seconds: float,
    observation_window_seconds: float,
    silence_threshold: float,
    sustained_frames: int,
) -> float:
    """Reward = -d, where d is seconds from barge-in to sustained silence.

    Mirrors Kyutai's R_int = -d (arXiv:2606.11167) but applied to this
    project's own yield-latency definition (time for the assistant to go
    quiet after being barged in on), matching what the release gate
    actually measures -- not the paper's different turn-taking-response
    metric. When the assistant never yields within the observation window,
    clamps to -observation_window_seconds: the worst measurable outcome
    within however long was watched, not an unbounded or undefined penalty.
    """
    if observation_window_seconds <= 0:
        raise ValueError("observation_window_seconds must be positive")
    delay_ms = measure_interruption_response(
        frames,
        barge_in_seconds=barge_in_seconds,
        silence_threshold=silence_threshold,
        sustained_frames=sustained_frames,
    )
    if delay_ms is None:
        return -observation_window_seconds
    return -(delay_ms / 1000)


def compute_quality_reward(
    transcript: str,
    *,
    target_character: Literal["aragorn", "sauron"],
    probe_prompt: str,
    expected_traits: list[str],
    judge: JudgeClient,
    seed: int,
) -> float:
    """The judge's persona_score, not an average with lore_score.

    The failure mode this reward exists to prevent is a policy that learns
    to go silent or generic to trivially maximize the yield-delay reward --
    persona_score penalizes that directly. Lore accuracy is the SFT side's
    concern, already covered by the existing training pipeline.

    Raises JudgeClient's existing JudgeResponseError on judge failure rather
    than substituting a default score -- a caller decides how to handle a
    bad rollout, this function does not silently guess.
    """
    return judge.score(
        target_character=target_character,
        probe_prompt=probe_prompt,
        expected_traits=expected_traits,
        candidate_transcript=transcript,
        seed=seed,
    ).persona_score
