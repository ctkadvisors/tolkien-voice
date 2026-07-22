from __future__ import annotations

import json
import re
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from .schema import (
    CurriculumKind,
    DialogueBlueprint,
    Provenance,
    Split,
    TimingKind,
    Utterance,
)


class TeacherResponseError(ValueError):
    """The teacher returned output that is unsafe to add to training data."""


CHARACTER_ALIASES = {
    "aragorn": frozenset(
        {"aragorn", "aragorn son of arathorn", "aragorn strider", "strider", "king elessar"}
    ),
    "sauron": frozenset({"sauron", "sauron the dark lord", "dark lord sauron"}),
}


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
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self._client.post("chat/completions", json=body)
                response.raise_for_status()
            except (httpx.TimeoutException, httpx.HTTPStatusError):
                if attempt == self.max_attempts:
                    raise
                continue
            try:
                return _parse_teacher_response(response, character)
            except TeacherResponseError:
                if attempt == self.max_attempts:
                    raise
        raise AssertionError("teacher retry loop ended unexpectedly")


def _parse_teacher_response(response: httpx.Response, character: str) -> TeacherDialogue:
    try:
        payload = response.json()
    except (json.JSONDecodeError, TypeError) as error:
        raise TeacherResponseError("teacher HTTP response must contain JSON") from error
    content = _extract_content(payload)
    try:
        raw = json.loads(content)
    except (json.JSONDecodeError, TypeError) as error:
        raise TeacherResponseError("teacher response must be strict JSON without fences") from error
    raw = _normalize_labels(raw, character)
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
    curriculum: CurriculumKind = CurriculumKind.GENERAL,
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
            curriculum=curriculum,
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


def _normalize_labels(payload: Any, requested_character: str) -> Any:
    if not isinstance(payload, dict):
        return payload
    normalized = dict(payload)
    character = normalized.get("character")
    aliases = CHARACTER_ALIASES.get(requested_character, frozenset({requested_character.casefold()}))
    if isinstance(character, str) and _normalize_character_label(character) in aliases:
        normalized["character"] = requested_character
    utterances = normalized.get("utterances")
    if isinstance(utterances, list):
        normalized_utterances = []
        for utterance in utterances:
            if not isinstance(utterance, dict):
                normalized_utterances.append(utterance)
                continue
            normalized_utterance = dict(utterance)
            speaker = normalized_utterance.get("speaker")
            if isinstance(speaker, str) and _normalize_character_label(speaker) in aliases:
                normalized_utterance["speaker"] = "assistant"
            normalized_utterances.append(normalized_utterance)
        normalized["utterances"] = normalized_utterances
    return normalized


def _normalize_character_label(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _build_messages(
    character: str,
    character_prompt: str,
    scenario: str,
    timing: TimingKind,
) -> list[dict[str, str]]:
    system = (
        "Create one short, casual two-speaker dialogue for native speech-model training. "
        "Return exactly one JSON object with keys character and utterances. Each utterance must "
        "have speaker (user or assistant) and text. Start with user, alternate speakers, use 4-8 "
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
