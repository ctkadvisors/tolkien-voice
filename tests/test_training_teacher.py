import json

import httpx
import pytest

from tolkien_voice.training.schema import Provenance, Split, TimingKind
from tolkien_voice.training.teacher import (
    TeacherClient,
    TeacherResponseError,
    generate_blueprint,
)


def response_payload(
    *, character: str = "aragorn", assistant_text: str = "The road remains open." 
) -> dict:
    return {
        "character": character,
        "utterances": [
            {"speaker": "user", "text": "Is the road open?"},
            {"speaker": "assistant", "text": assistant_text},
        ],
    }


def provenance(seed: int = 7) -> Provenance:
    return Provenance(
        teacher_model="tolkien-27b-v4-q8",
        teacher_revision="v4",
        character_card_sha256="a" * 64,
        prompt_template_revision="sts-dialogue-v1",
        seed=seed,
        voice_license="private-apple-system-voice",
    )


def test_teacher_requests_strict_json_with_fixed_seed() -> None:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(response_payload())}}]},
        )

    client = TeacherClient(
        base_url="http://teacher.test/v1",
        model="tolkien",
        transport=httpx.MockTransport(handler),
    )

    result = client.generate(
        character="aragorn",
        character_prompt="Identity: Aragorn, a ranger and king.",
        scenario="A traveler asks about the road.",
        timing=TimingKind.NORMAL,
        seed=7,
    )

    assert result.character == "aragorn"
    assert captured[0]["model"] == "tolkien"
    assert captured[0]["seed"] == 7
    assert captured[0]["response_format"] == {"type": "json_object"}
    assert captured[0]["temperature"] == 0.7


def test_teacher_retry_preserves_seed() -> None:
    seeds: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seeds.append(body["seed"])
        if len(seeds) == 1:
            raise httpx.ReadTimeout("slow teacher", request=request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(response_payload())}}]},
        )

    client = TeacherClient(
        base_url="http://teacher.test/v1",
        model="tolkien",
        transport=httpx.MockTransport(handler),
        max_attempts=2,
    )

    client.generate(
        character="aragorn",
        character_prompt="Identity: Aragorn.",
        scenario="Road counsel.",
        timing=TimingKind.PAUSE,
        seed=99,
    )

    assert seeds == [99, 99]


def test_teacher_rejects_malformed_json() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, json={"choices": [{"message": {"content": "```json\n{}\n```"}}]}
        )
    )
    client = TeacherClient(base_url="http://teacher.test/v1", transport=transport)

    with pytest.raises(TeacherResponseError, match="strict JSON"):
        client.generate(
            character="aragorn",
            character_prompt="Identity: Aragorn.",
            scenario="Road counsel.",
            timing=TimingKind.NORMAL,
            seed=1,
        )


def test_teacher_rejects_wrong_character() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps(response_payload(character="gandalf"))}}
                ]
            },
        )
    )
    client = TeacherClient(base_url="http://teacher.test/v1", transport=transport)

    with pytest.raises(TeacherResponseError, match="requested character"):
        client.generate(
            character="aragorn",
            character_prompt="Identity: Aragorn.",
            scenario="Road counsel.",
            timing=TimingKind.NORMAL,
            seed=1,
        )


def test_generate_blueprint_rejects_long_teacher_monologue() -> None:
    reply = " ".join(f"word{index}" for index in range(46))
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(response_payload(assistant_text=reply))
                        }
                    }
                ]
            },
        )
    )
    client = TeacherClient(base_url="http://teacher.test/v1", transport=transport)

    with pytest.raises(TeacherResponseError, match="45 words"):
        generate_blueprint(
            client,
            dialogue_id="aragorn-road-0001",
            character="aragorn",
            character_prompt="Identity: Aragorn.",
            scenario="Road counsel.",
            timing=TimingKind.NORMAL,
            split=Split.TRAIN,
            provenance=provenance(),
        )
