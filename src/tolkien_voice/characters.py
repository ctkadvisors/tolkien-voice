from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import yaml


VOICE_PATTERN = re.compile(r"^(NATF|NATM)[0-3]\.pt$")


class UnknownCharacterError(KeyError):
    """Raised when a character ID is not registered."""


class VoiceNotAllowedError(ValueError):
    """Raised when a voice is not allowed for the selected character."""


class CharacterCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    display_name: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    role: str = Field(min_length=1)
    manner: list[str] = Field(min_length=1)
    boundaries: list[str] = Field(min_length=1)
    lore_scope: str = Field(min_length=1)
    default_voice: str
    allowed_voices: list[str] = Field(min_length=1)
    eval_probes: list[str] = Field(min_length=3)

    @field_validator("default_voice")
    @classmethod
    def default_voice_must_be_included(cls, value: str) -> str:
        if not VOICE_PATTERN.fullmatch(value):
            raise ValueError("voice must be an included PersonaPlex voice")
        return value

    @field_validator("allowed_voices")
    @classmethod
    def allowed_voices_must_be_included(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values):
            raise ValueError("allowed voice values must be unique")
        if any(not VOICE_PATTERN.fullmatch(value) for value in values):
            raise ValueError("voice must be an included PersonaPlex voice")
        return values

    @model_validator(mode="after")
    def default_must_be_allowed(self) -> CharacterCard:
        if self.default_voice not in self.allowed_voices:
            raise ValueError("default voice must appear in allowed voices")
        return self


@dataclass(frozen=True)
class ResolvedCharacter:
    id: str
    display_name: str
    summary: str
    text_prompt: str
    voice_prompt: str
    allowed_voices: tuple[str, ...]


class CharacterRegistry:
    def __init__(self, cards: dict[str, CharacterCard]) -> None:
        self._cards = dict(cards)

    @classmethod
    def from_directory(cls, directory: str | Path) -> CharacterRegistry:
        root = Path(directory).expanduser().resolve()
        cards: dict[str, CharacterCard] = {}
        for path in sorted(root.glob("*.yaml")):
            with path.open(encoding="utf-8") as handle:
                raw: Any = yaml.safe_load(handle)
            card = CharacterCard.model_validate(raw)
            if path.stem != card.id:
                raise ValueError(f"character filename {path.name!r} must match id {card.id!r}")
            if card.id in cards:
                raise ValueError(f"duplicate character id: {card.id}")
            cards[card.id] = card
        if not cards:
            raise ValueError(f"no character cards found in {root}")
        return cls(cards)

    def ids(self) -> tuple[str, ...]:
        return tuple(self._cards)

    def cards(self) -> tuple[CharacterCard, ...]:
        return tuple(self._cards.values())

    def resolve(self, character_id: str, voice_id: str | None = None) -> ResolvedCharacter:
        try:
            card = self._cards[character_id]
        except KeyError as error:
            raise UnknownCharacterError(character_id) from error
        selected_voice = voice_id or card.default_voice
        if selected_voice not in card.allowed_voices:
            raise VoiceNotAllowedError(
                f"voice {selected_voice!r} is not allowed for character {character_id!r}"
            )
        return ResolvedCharacter(
            id=card.id,
            display_name=card.display_name,
            summary=card.summary,
            text_prompt=_build_prompt(card),
            voice_prompt=selected_voice,
            allowed_voices=tuple(card.allowed_voices),
        )


def _build_prompt(card: CharacterCard) -> str:
    sections = [
        f"Identity: {card.role}",
        "Manner: " + "; ".join(card.manner),
        f"Lore behavior: {card.lore_scope}",
        "Boundaries: " + "; ".join(card.boundaries),
    ]
    return "<system> " + "\n".join(sections) + " <system>"

