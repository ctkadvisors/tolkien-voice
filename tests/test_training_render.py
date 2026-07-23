from __future__ import annotations

from array import array
import json
from pathlib import Path
import wave

import pytest

from tolkien_voice.training.render import MacOSSaySynthesizer, render_blueprint, render_corpus
from tolkien_voice.training.schema import (
    DialogueBlueprint,
    Provenance,
    Split,
    TimingKind,
    Utterance,
)


def write_mono_tone(
    path: Path,
    *,
    sample_rate: int = 24_000,
    amplitude: int = 1000,
    duration_seconds: float = 0.2,
) -> None:
    samples = array("h", [amplitude] * round(sample_rate * duration_seconds))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())


def blueprint(timing: TimingKind = TimingKind.NORMAL) -> DialogueBlueprint:
    return DialogueBlueprint(
        id=f"aragorn-{timing.value}-0001",
        character="aragorn",
        scenario="A traveler asks for counsel.",
        timing=timing,
        split=Split.TRAIN,
        utterances=[
            Utterance(speaker="user", text="Is the road safe?"),
            Utterance(speaker="assistant", text="Walk carefully, and keep watch."),
            Utterance(speaker="user", text="Wait. What was that sound?"),
            Utterance(speaker="assistant", text="Only wind among the stones."),
        ],
        provenance=Provenance(
            teacher_model="tolkien-27b-v4-q8",
            teacher_revision="v4",
            character_card_sha256="a" * 64,
            prompt_template_revision="sts-dialogue-v1",
            seed=1,
            voice_license="private-apple-system-voice",
        ),
    )


def read_stereo(path: Path) -> tuple[wave._wave_params, array]:
    with wave.open(str(path), "rb") as source:
        params = source.getparams()
        samples = array("h")
        samples.frombytes(source.readframes(source.getnframes()))
    return params, samples


def test_render_places_assistant_left_and_user_right(tmp_path: Path) -> None:
    def synthesize(utterance: Utterance, destination: Path) -> None:
        write_mono_tone(destination, amplitude=1000 if utterance.speaker == "user" else 2000)

    rendered = render_blueprint(
        blueprint(),
        tmp_path,
        synthesize=synthesize,
        system_prompt="<system> Aragorn prompt <system>",
    )

    params, samples = read_stereo(rendered.wav_path)
    assert params.nchannels == 2
    assert params.framerate == 24_000
    left = samples[0::2]
    right = samples[1::2]
    assert max(left) == 2000
    assert max(right) == 1000

    alignment = json.loads(rendered.alignment_path.read_text())
    assert alignment["alignments"][0][0] == "Walk"
    assert {entry[2] for entry in alignment["alignments"]} == {"SPEAKER_MAIN"}
    assert alignment["system_prompt"] == "<system> Aragorn prompt <system>"


def test_render_resamples_mono_input_to_24khz(tmp_path: Path) -> None:
    def synthesize(utterance: Utterance, destination: Path) -> None:
        write_mono_tone(destination, sample_rate=16_000)

    rendered = render_blueprint(blueprint(), tmp_path, synthesize=synthesize)

    params, _ = read_stereo(rendered.wav_path)
    assert params.framerate == 24_000


def test_overlap_has_simultaneous_user_and_assistant_audio(tmp_path: Path) -> None:
    def synthesize(utterance: Utterance, destination: Path) -> None:
        write_mono_tone(destination)

    rendered = render_blueprint(blueprint(TimingKind.OVERLAP), tmp_path, synthesize=synthesize)
    _, samples = read_stereo(rendered.wav_path)

    simultaneous_frames = sum(
        1 for left, right in zip(samples[0::2], samples[1::2], strict=True) if left and right
    )
    assert simultaneous_frames > 0


def test_interruption_truncates_assistant_when_user_barges_in(tmp_path: Path) -> None:
    def synthesize(utterance: Utterance, destination: Path) -> None:
        write_mono_tone(destination)

    rendered = render_blueprint(
        blueprint(TimingKind.INTERRUPTION), tmp_path, synthesize=synthesize
    )
    alignment = json.loads(rendered.alignment_path.read_text())
    first_assistant = alignment["timeline"][1]
    interrupting_user = alignment["timeline"][2]

    assert interrupting_user["start_sec"] < first_assistant["natural_end_sec"]
    assert first_assistant["end_sec"] <= interrupting_user["start_sec"] + 0.100001


def test_interruption_occurs_early_in_assistant_utterance(tmp_path: Path) -> None:
    def synthesize(utterance: Utterance, destination: Path) -> None:
        write_mono_tone(destination, duration_seconds=3.0)

    rendered = render_blueprint(
        blueprint(TimingKind.INTERRUPTION), tmp_path, synthesize=synthesize
    )
    timeline = json.loads(rendered.alignment_path.read_text())["timeline"]
    first_assistant = timeline[1]
    interrupting_user = timeline[2]
    natural_duration = (
        first_assistant["natural_end_sec"] - first_assistant["start_sec"]
    )
    elapsed = interrupting_user["start_sec"] - first_assistant["start_sec"]

    assert 0.5 <= elapsed <= 1.0
    assert elapsed / natural_duration <= 0.34


def test_render_rejects_clipped_source_audio(tmp_path: Path) -> None:
    def synthesize(utterance: Utterance, destination: Path) -> None:
        write_mono_tone(destination, amplitude=32_767)

    with pytest.raises(ValueError, match="clipped"):
        render_blueprint(blueprint(), tmp_path, synthesize=synthesize)


def test_macos_say_requests_24khz_pcm_wave(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def capture(command: list[str], *, check: bool) -> None:
        assert check is True
        commands.append(command)
        write_mono_tone(Path(command[command.index("-o") + 1]))

    monkeypatch.setattr("tolkien_voice.training.render.subprocess.run", capture)
    synthesizer = MacOSSaySynthesizer(user_voice="Samantha", assistant_voice="Daniel")

    synthesizer(Utterance(speaker="assistant", text="Keep watch."), tmp_path / "speech.wav")

    assert commands == [
        [
            "/usr/bin/say",
            "-v",
            "Daniel",
            "--file-format=WAVE",
            "--data-format=LEI16@24000",
            "-o",
            str(tmp_path / "speech.wav"),
            "Keep watch.",
        ]
    ]


def test_macos_say_rejects_zero_frame_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def empty_output(command: list[str], *, check: bool) -> None:
        destination = Path(command[command.index("-o") + 1])
        with wave.open(str(destination), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(24_000)

    monkeypatch.setattr("tolkien_voice.training.render.subprocess.run", empty_output)

    with pytest.raises(RuntimeError, match="zero audio frames"):
        MacOSSaySynthesizer(user_voice="Samantha", assistant_voice="Daniel")(
            Utterance(speaker="user", text="Can you hear me?"),
            tmp_path / "empty.wav",
        )


def test_parallel_corpus_render_preserves_blueprint_order(tmp_path: Path) -> None:
    blueprints = [
        blueprint().model_copy(update={"id": f"aragorn-normal-{index:04d}"})
        for index in range(8)
    ]

    def synthesize(_: Utterance, destination: Path) -> None:
        write_mono_tone(destination)

    rendered = render_corpus(
        blueprints,
        tmp_path,
        synthesizer_for=lambda _: synthesize,
        workers=4,
    )

    assert [item.id for item in rendered] == [item.id for item in blueprints]
