import json

import httpx
import pytest

from tolkien_voice.training.interruption_reward import (
    compute_quality_reward,
    compute_yield_delay_reward,
)
from tolkien_voice.training.judge import JudgeClient, JudgeResponseError


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


def test_quality_reward_is_the_judges_persona_score() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        content = json.dumps(
            {
                "persona_score": 0.7,
                "lore_score": 0.95,
                "identified_character": "aragorn",
                "rationale": "Stayed in character despite the interruption.",
            }
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    judge = JudgeClient(base_url="http://teacher.test/v1", transport=httpx.MockTransport(handler))

    reward = compute_quality_reward(
        "I took that road because no other force could reach the river in time.",
        target_character="aragorn",
        probe_prompt="Why did you choose the Paths of the Dead?",
        expected_traits=["grounded", "resolute", "first-person"],
        judge=judge,
        seed=44,
    )

    assert reward == 0.7


def test_quality_reward_propagates_judge_failure_instead_of_guessing() -> None:
    judge = JudgeClient(
        base_url="http://teacher.test/v1",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": []})),
        max_attempts=1,
    )

    with pytest.raises(JudgeResponseError):
        compute_quality_reward(
            "...",
            target_character="aragorn",
            probe_prompt="Why did you choose the Paths of the Dead?",
            expected_traits=["grounded"],
            judge=judge,
            seed=44,
        )
