import json

import httpx
import pytest

from tolkien_voice.training.judge import JudgeClient, JudgeResponseError


def test_judge_scores_transcript_with_deterministic_blinded_request() -> None:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        content = json.dumps(
            {
                "persona_score": 0.9,
                "lore_score": 0.8,
                "identified_character": "aragorn",
                "rationale": "Restrained first-person answer grounded in the character's history.",
            }
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    client = JudgeClient(
        base_url="http://teacher.test/v1",
        transport=httpx.MockTransport(handler),
    )
    result = client.score(
        target_character="aragorn",
        probe_prompt="Why did you choose the Paths of the Dead?",
        expected_traits=["grounded", "resolute", "first-person"],
        candidate_transcript="I took that road because no other force could reach the river in time.",
        seed=44,
    )

    assert result.persona_score == 0.9
    assert result.identified_character == "aragorn"
    assert captured[0]["temperature"] == 0
    assert captured[0]["seed"] == 44
    serialized = json.dumps(captured[0]).casefold()
    assert "adapter" not in serialized
    assert "baseline" not in serialized
    assert "control arm" not in serialized


def test_judge_rejects_out_of_contract_scores() -> None:
    content = json.dumps(
        {
            "persona_score": 7,
            "lore_score": 0.8,
            "identified_character": "aragorn",
            "rationale": "invalid range",
        }
    )
    client = JudgeClient(
        base_url="http://teacher.test/v1",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"choices": [{"message": {"content": content}}]}
            )
        ),
        max_attempts=1,
    )

    with pytest.raises(JudgeResponseError, match="invalid judge score"):
        client.score(
            target_character="aragorn",
            probe_prompt="A probe",
            expected_traits=["grounded"],
            candidate_transcript="A candidate",
            seed=1,
        )
