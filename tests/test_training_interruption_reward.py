from tolkien_voice.training.interruption_reward import compute_yield_delay_reward


def test_yield_delay_reward_is_negative_seconds_to_silence() -> None:
    frames = [
        (1.00, 0.08),
        (1.04, 0.07),
        (1.08, 0.06),
        (1.12, 0.01),
        (1.16, 0.06),
        (1.20, 0.01),
        (1.24, 0.008),
        (1.28, 0.006),
        (1.32, 0.005),
        (1.36, 0.004),
    ]

    reward = compute_yield_delay_reward(
        frames,
        barge_in_seconds=1.10,
        observation_window_seconds=12.0,
        silence_threshold=0.015,
        sustained_frames=4,
    )

    assert reward == -0.1


def test_yield_delay_reward_clamps_to_negative_window_when_never_yielded() -> None:
    frames = [(1.0 + index * 0.04, 0.08) for index in range(10)]

    reward = compute_yield_delay_reward(
        frames,
        barge_in_seconds=1.10,
        observation_window_seconds=12.0,
        silence_threshold=0.015,
        sustained_frames=4,
    )

    assert reward == -12.0
