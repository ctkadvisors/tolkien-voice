from tolkien_voice.training.rollout import Rollout, RolloutGroup


def _rollout(*, yield_delay_reward: float, quality_reward: float) -> Rollout:
    return Rollout(
        transcript="text",
        yield_delay_reward=yield_delay_reward,
        quality_reward=quality_reward,
        delay_seconds=0.5,
    )


def test_single_rollout_group_is_degenerate() -> None:
    group = RolloutGroup(
        probe_id="aragorn-lore-rangers",
        character="aragorn",
        rollouts=[_rollout(yield_delay_reward=-0.5, quality_reward=0.7)],
    )

    assert group.is_degenerate is True


def test_group_is_degenerate_when_both_rewards_are_constant() -> None:
    group = RolloutGroup(
        probe_id="aragorn-lore-rangers",
        character="aragorn",
        rollouts=[
            _rollout(yield_delay_reward=-0.5, quality_reward=0.7),
            _rollout(yield_delay_reward=-0.5, quality_reward=0.7),
        ],
    )

    assert group.is_degenerate is True


def test_group_is_not_degenerate_when_yield_reward_varies() -> None:
    group = RolloutGroup(
        probe_id="aragorn-lore-rangers",
        character="aragorn",
        rollouts=[
            _rollout(yield_delay_reward=-0.5, quality_reward=0.7),
            _rollout(yield_delay_reward=-1.2, quality_reward=0.7),
        ],
    )

    assert group.is_degenerate is False


def test_group_is_not_degenerate_when_quality_reward_varies() -> None:
    group = RolloutGroup(
        probe_id="aragorn-lore-rangers",
        character="aragorn",
        rollouts=[
            _rollout(yield_delay_reward=-0.5, quality_reward=0.7),
            _rollout(yield_delay_reward=-0.5, quality_reward=0.2),
        ],
    )

    assert group.is_degenerate is False
