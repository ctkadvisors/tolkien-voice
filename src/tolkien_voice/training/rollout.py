from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .interruption_reward import compute_quality_reward, compute_yield_delay_reward
from .judge import JudgeClient
from .native_probe import probe_native_interruption


class Rollout(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    transcript: str
    yield_delay_reward: float
    quality_reward: float
    delay_seconds: float | None


class RolloutGroup(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    probe_id: str
    character: Literal["aragorn", "sauron"]
    rollouts: list[Rollout]

    @property
    def is_degenerate(self) -> bool:
        """True only if neither reward dimension varies across the group.

        GRPO's advantage is (reward - group_mean); a group with zero
        variance on both dimensions contributes nothing to a policy
        update regardless of how the two rewards eventually get combined
        (sub-project 3's decision, not made yet). Checked independently
        rather than on a combined score so a group with signal on only
        one dimension isn't misclassified as degenerate.
        """
        if len(self.rollouts) <= 1:
            return True
        yield_rewards = {rollout.yield_delay_reward for rollout in self.rollouts}
        quality_rewards = {rollout.quality_reward for rollout in self.rollouts}
        return len(yield_rewards) <= 1 and len(quality_rewards) <= 1


async def collect_rollout_group(
    url: str,
    opening_wav_path: Path,
    interruption_wav_path: Path,
    *,
    probe_id: str,
    character: Literal["aragorn", "sauron"],
    probe_prompt: str,
    expected_traits: list[str],
    judge: JudgeClient,
    group_size: int,
    response_seconds: float = 12.0,
    max_wait_seconds: float = 12.0,
    seed: int,
) -> RolloutGroup:
    """Samples group_size completions for one scenario against an
    already-running PersonaPlex instance and scores each.

    Calls are sequential, not concurrent: PersonaPlex serves one
    full-duplex session at a time. Scores post_barge_text (what the model
    said after being interrupted), not interruption_text (which also
    includes pre-barge speech and would blunt the anti-gaming property --
    a policy that speaks well before barge-in and then goes silent would
    otherwise score well despite exhibiting the exact failure mode the
    quality reward exists to catch).

    Does not start or stop the PersonaPlex instance itself -- that
    lifecycle belongs to the caller, via scripts/_personaplex_ctl.py.
    """
    if group_size < 1:
        raise ValueError("group_size must be at least 1")

    rollouts: list[Rollout] = []
    for index in range(group_size):
        result = await probe_native_interruption(
            url,
            opening_wav_path,
            interruption_wav_path,
            max_wait_seconds=max_wait_seconds,
            response_seconds=response_seconds,
        )
        yield_reward = compute_yield_delay_reward(
            result["output_frames"],
            barge_in_seconds=result["barge_in_ms"] / 1000,
            observation_window_seconds=result["max_wait_seconds"],
            silence_threshold=0.015,
            sustained_frames=12,
        )
        quality_reward = compute_quality_reward(
            result["post_barge_text"],
            target_character=character,
            probe_prompt=probe_prompt,
            expected_traits=expected_traits,
            judge=judge,
            seed=seed + index,
        )
        known_delay_ms = result["interruption_response_ms"]
        rollouts.append(
            Rollout(
                transcript=result["post_barge_text"],
                yield_delay_reward=yield_reward,
                quality_reward=quality_reward,
                delay_seconds=(known_delay_ms / 1000 if known_delay_ms is not None else None),
            )
        )
    return RolloutGroup(probe_id=probe_id, character=character, rollouts=rollouts)
