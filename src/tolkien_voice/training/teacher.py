from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from .schema import (
    DialogueBlueprint,
    Provenance,
    Split,
    TimingKind,
    Utterance,
)


class TeacherResponseError(ValueError):
    """The teacher returned output that is unsafe to add to training data."""


class TeacherDialogue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    character: str
    utterances: list[Utterance]


class TeacherClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str = "tolkien",
        transport: httpx.BaseTransport | None = None,
        timeout_seconds: float = 180,
        max_attempts: int = 3,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        self.model = model
        self.max_attempts = max_attempts
        self._client = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            transport=transport,
            timeout=timeout_seconds,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> TeacherClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def generate(
        self,
        *,
        character: str,
        character_prompt: str,
        scenario: str,
        timing: TimingKind,
        seed: int,
    ) -> TeacherDialogue:
        body = {
            "model": self.model,
            "messages": _build_messages(character, character_prompt, scenario, timing),
            "temperature": 0.7,
            "max_tokens": 700,
            "seed": seed,
            "response_format": {"type": "json_object"},
        }
        response: httpx.Response | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self._client.post("chat/completions", json=body)
                response.raise_for_status()
                break
            except (httpx.TimeoutException, httpx.HTTPStatusError):
                if attempt == self.max_attempts:
                    raise
        assert response is not None
        content = _extract_content(response.json())
        try:
            raw = json.loads(content)
        except (json.JSONDecodeError, TypeError) as error:
            raise TeacherResponseError("teacher response must be strict JSON without fences") from error
        try:
            dialogue = TeacherDialogue.model_validate(raw)
        except ValidationError as error:
            raise TeacherResponseError(f"invalid teacher dialogue: {error}") from error
        if dialogue.character != character:
            raise TeacherResponseError(
                f"teacher answered as {dialogue.character!r}, not the requested character {character!r}"
            )
        return dialogue


def generate_blueprint(
    client: TeacherClient,
    *,
    dialogue_id: str,
    character: str,
    character_prompt: str,
    scenario: str,
    timing: TimingKind,
    split: Split,
    provenance: Provenance,
) -> DialogueBlueprint:
    dialogue = client.generate(
        character=character,
        character_prompt=character_prompt,
        scenario=scenario,
        timing=timing,
        seed=provenance.seed,
    )
    try:
        return DialogueBlueprint(
            id=dialogue_id,
            character=character,
            scenario=scenario,
            timing=timing,
            split=split,
            utterances=dialogue.utterances,
            provenance=provenance,
        )
    except ValidationError as error:
        raise TeacherResponseError(f"teacher dialogue violates dataset contract: {error}") from error


def _extract_content(payload: Any) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise TeacherResponseError("teacher response is missing choices[0].message.content") from error
    if not isinstance(content, str):
        raise TeacherResponseError("teacher response content must be a string")
    return content


def _build_messages(
    character: str,
    character_prompt: str,
    scenario: str,
    timing: TimingKind,
) -> list[dict[str, str]]:
    system = (
        "Create one short, casual two-speaker dialogue for native speech-model training. "
        "Return exactly one JSON object with keys character and utterances. Each utterance must "
        "have speaker (user or assistant) and text. Start with user, alternate speakers, use 2-8 "
        "utterances, keep every assistant reply at 45 words or fewer, preserve the requested "
        "character, and never include stage directions, markdown, commentary, or quoted source "
        "passages. The dialogue must sound speakable and allow natural interruption."
    )
    user = (
        f"Character ID: {character}\n"
        f"Character card:\n{character_prompt}\n\n"
        f"Scenario: {scenario}\n"
        f"Conversation timing: {timing.value}\n"
        "Generate the JSON dialogue now."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
