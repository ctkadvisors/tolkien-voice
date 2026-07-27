from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


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
