from pathlib import Path

import pytest
import yaml

from tolkien_voice.characters import CharacterRegistry, VoiceNotAllowedError


CARDS = Path(__file__).parents[1] / "configs" / "characters"


@pytest.fixture
def registry() -> CharacterRegistry:
    return CharacterRegistry.from_directory(CARDS)


def test_registry_contains_initial_roster(registry: CharacterRegistry) -> None:
    assert set(registry.ids()) == {
        "gandalf",
        "sauron",
        "galadriel",
        "samwise",
        "aragorn",
        "frodo",
        "treebeard",
        "boromir",
        "barliman",
    }


def test_registry_rejects_voice_not_allowed_for_character(
    registry: CharacterRegistry,
) -> None:
    with pytest.raises(VoiceNotAllowedError):
        registry.resolve("sauron", "NATF0.pt")


def test_prompt_is_wrapped_once(registry: CharacterRegistry) -> None:
    prompt = registry.resolve("gandalf", "NATM1.pt").text_prompt

    assert prompt.startswith("<system>")
    assert prompt.endswith("<system>")
    assert prompt.count("<system>") == 2


def test_default_voice_is_used_when_voice_is_omitted(registry: CharacterRegistry) -> None:
    resolved = registry.resolve("galadriel")

    assert resolved.voice_prompt == "NATF2.pt"


def test_filename_must_match_character_id(tmp_path: Path) -> None:
    card = {
        "id": "gandalf",
        "display_name": "Gandalf",
        "summary": "A wandering wizard.",
        "role": "Fictional roleplay as Gandalf.",
        "manner": ["Grave"],
        "boundaries": ["Admit uncertainty"],
        "lore_scope": "Tolkien's legendarium",
        "default_voice": "NATM1.pt",
        "allowed_voices": ["NATM1.pt"],
        "eval_probes": ["one", "two", "three"],
    }
    (tmp_path / "wrong.yaml").write_text(yaml.safe_dump(card), encoding="utf-8")

    with pytest.raises(ValueError, match="filename"):
        CharacterRegistry.from_directory(tmp_path)


def test_voice_name_must_be_an_included_personaplex_voice(tmp_path: Path) -> None:
    card = {
        "id": "gandalf",
        "display_name": "Gandalf",
        "summary": "A wandering wizard.",
        "role": "Fictional roleplay as Gandalf.",
        "manner": ["Grave"],
        "boundaries": ["Admit uncertainty"],
        "lore_scope": "Tolkien's legendarium",
        "default_voice": "actor-clone.wav",
        "allowed_voices": ["actor-clone.wav"],
        "eval_probes": ["one", "two", "three"],
    }
    (tmp_path / "gandalf.yaml").write_text(yaml.safe_dump(card), encoding="utf-8")

    with pytest.raises(ValueError, match="voice"):
        CharacterRegistry.from_directory(tmp_path)
