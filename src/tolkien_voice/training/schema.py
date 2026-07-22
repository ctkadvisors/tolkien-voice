from __future__ import annotations

from enum import StrEnum
import hashlib
from pathlib import Path
import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


CHARACTER_IDS = frozenset(
    {
        "aragorn",
        "barliman",
        "boromir",
        "frodo",
        "galadriel",
        "gandalf",
        "samwise",
        "sauron",
        "treebeard",
    }
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_ASSISTANT_WORDS = 45


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    HELD_OUT = "held_out"


class TimingKind(StrEnum):
    NORMAL = "normal"
    PAUSE = "pause"
    OVERLAP = "overlap"
    INTERRUPTION = "interruption"


class CurriculumKind(StrEnum):
    GENERAL = "general"
    COUNSEL = "counsel"
    LORE = "lore"
    UNCERTAINTY = "uncertainty"
    RESISTANCE = "resistance"
    EMOTION = "emotion"
    DEBATE = "debate"
    RELATIONSHIP = "relationship"
    REPAIR = "repair"


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    teacher_model: str = Field(min_length=1)
    teacher_revision: str = Field(min_length=1)
    character_card_sha256: str
    prompt_template_revision: str = Field(min_length=1)
    seed: int = Field(ge=0)
    voice_license: str = Field(min_length=1)

    @field_validator("teacher_model", "teacher_revision", "prompt_template_revision", "voice_license")
    @classmethod
    def nonblank_strings(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("character_card_sha256")
    @classmethod
    def character_hash_is_sha256(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("character_card_sha256 must be a lowercase SHA-256 digest")
        return value


class Utterance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    speaker: Literal["user", "assistant"]
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("utterance text must not be blank")
        return value


class DialogueBlueprint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,127}$")
    character: str
    scenario: str = Field(min_length=1)
    curriculum: CurriculumKind = CurriculumKind.GENERAL
    timing: TimingKind
    split: Split
    utterances: list[Utterance] = Field(min_length=2, max_length=12)
    provenance: Provenance

    @field_validator("character")
    @classmethod
    def character_is_registered(cls, value: str) -> str:
        if value not in CHARACTER_IDS:
            raise ValueError(f"character must be one of {', '.join(sorted(CHARACTER_IDS))}")
        return value

    @field_validator("scenario")
    @classmethod
    def scenario_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("scenario must not be blank")
        return value

    @model_validator(mode="after")
    def conversation_is_short_and_alternating(self) -> DialogueBlueprint:
        if self.utterances[0].speaker != "user":
            raise ValueError("dialogue must start with the user")
        if not any(item.speaker == "assistant" for item in self.utterances):
            raise ValueError("dialogue requires an assistant utterance")
        for previous, current in zip(self.utterances, self.utterances[1:], strict=False):
            if previous.speaker == current.speaker:
                raise ValueError("dialogue speakers must alternate")
        for item in self.utterances:
            if item.speaker == "assistant" and len(item.text.split()) > MAX_ASSISTANT_WORDS:
                raise ValueError(
                    f"assistant utterances must contain at most {MAX_ASSISTANT_WORDS} words"
                )
        return self

    @property
    def canonical_hash(self) -> str:
        transcript = "\n".join(
            f"{utterance.speaker}:{normalize_text(utterance.text)}"
            for utterance in self.utterances
        )
        return hashlib.sha256(transcript.encode("utf-8")).hexdigest()


class RenderedExample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,127}$")
    blueprint_hash: str
    wav_path: Path
    alignment_path: Path
    duration_sec: float = Field(gt=0)
    sample_rate: Literal[24000] = 24000
    channels: Literal[2] = 2
    split: Split
    voice_license: str = Field(min_length=1)

    @field_validator("blueprint_hash")
    @classmethod
    def blueprint_hash_is_sha256(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("blueprint_hash must be a lowercase SHA-256 digest")
        return value

    @field_validator("voice_license")
    @classmethod
    def voice_license_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("voice_license must not be blank")
        return value


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def assert_no_leakage(
    training: list[DialogueBlueprint],
    held_out: list[DialogueBlueprint],
) -> None:
    training_hashes = {item.canonical_hash for item in training}
    held_out_hashes = {item.canonical_hash for item in held_out}
    overlap = training_hashes & held_out_hashes
    if overlap:
        raise ValueError(f"training/held-out leakage detected for {len(overlap)} dialogue(s)")
