from __future__ import annotations

from array import array
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import re
import string
import subprocess
import tempfile
from typing import Callable
import wave

from tolkien_voice.characters import CharacterRegistry

from .manifest import write_moshi_manifests
from .schema import DialogueBlueprint, RenderedExample, TimingKind, Utterance


SAMPLE_RATE = 24_000
MAX_SAFE_SAMPLE = int(32_767 * 0.99)
Synthesizer = Callable[[Utterance, Path], None]
SynthesizerFactory = Callable[[DialogueBlueprint], Synthesizer]


def render_blueprint(
    blueprint: DialogueBlueprint,
    output_directory: Path,
    *,
    synthesize: Synthesizer,
    system_prompt: str | None = None,
) -> RenderedExample:
    output_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix=f".{blueprint.id}-", dir=output_directory) as temp:
        temp_directory = Path(temp)
        clips: list[array] = []
        for index, utterance in enumerate(blueprint.utterances):
            clip_path = temp_directory / f"{index:02d}.wav"
            synthesize(utterance, clip_path)
            samples, source_rate = _read_mono_pcm16(clip_path)
            if max((abs(sample) for sample in samples), default=0) >= MAX_SAFE_SAMPLE:
                raise ValueError(f"synthesized source audio is clipped: {clip_path}")
            clips.append(_resample(samples, source_rate, SAMPLE_RATE))

    timeline = _build_timeline(blueprint, clips)
    total_frames = max(math.ceil(item["end_sec"] * SAMPLE_RATE) for item in timeline)
    left = array("i", [0] * total_frames)
    right = array("i", [0] * total_frames)
    for item, clip in zip(timeline, clips, strict=True):
        channel = left if item["speaker"] == "assistant" else right
        start_frame = round(item["start_sec"] * SAMPLE_RATE)
        end_frame = min(round(item["end_sec"] * SAMPLE_RATE), start_frame + len(clip))
        for target, sample in enumerate(clip[: end_frame - start_frame], start=start_frame):
            mixed = channel[target] + sample
            if abs(mixed) >= MAX_SAFE_SAMPLE:
                raise ValueError(f"mixed audio is clipped in {blueprint.id}")
            channel[target] = mixed

    interleaved = array("h")
    for left_sample, right_sample in zip(left, right, strict=True):
        interleaved.extend((left_sample, right_sample))

    wav_path = output_directory / f"{blueprint.id}.wav"
    temporary_wav = wav_path.with_suffix(".wav.tmp")
    with wave.open(str(temporary_wav), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(interleaved.tobytes())
    os.replace(temporary_wav, wav_path)
    wav_path.chmod(0o600)

    alignment_path = output_directory / f"{blueprint.id}.json"
    temporary_alignment = alignment_path.with_suffix(".json.tmp")
    alignment_payload = {
        "alignments": _assistant_alignments(timeline),
        "timeline": timeline,
        "blueprint_hash": blueprint.canonical_hash,
        "text_conditions": None,
        "system_prompt": system_prompt,
    }
    temporary_alignment.write_text(json.dumps(alignment_payload, indent=2) + "\n")
    temporary_alignment.chmod(0o600)
    os.replace(temporary_alignment, alignment_path)

    return RenderedExample(
        id=blueprint.id,
        blueprint_hash=blueprint.canonical_hash,
        wav_path=wav_path,
        alignment_path=alignment_path,
        duration_sec=total_frames / SAMPLE_RATE,
        split=blueprint.split,
        voice_license=blueprint.provenance.voice_license,
    )


def _read_mono_pcm16(path: Path) -> tuple[array, int]:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1:
            raise ValueError(f"TTS output must be mono: {path}")
        if source.getsampwidth() != 2:
            raise ValueError(f"TTS output must be PCM16: {path}")
        sample_rate = source.getframerate()
        samples = array("h")
        samples.frombytes(source.readframes(source.getnframes()))
    return samples, sample_rate


def _resample(samples: array, source_rate: int, target_rate: int) -> array:
    if source_rate <= 0:
        raise ValueError("source sample rate must be positive")
    if source_rate == target_rate or not samples:
        return array("h", samples)
    target_length = max(1, round(len(samples) * target_rate / source_rate))
    result = array("h")
    scale = source_rate / target_rate
    for target_index in range(target_length):
        source_position = min(target_index * scale, len(samples) - 1)
        lower = int(source_position)
        upper = min(lower + 1, len(samples) - 1)
        fraction = source_position - lower
        value = round(samples[lower] * (1 - fraction) + samples[upper] * fraction)
        result.append(value)
    return result


def _build_timeline(blueprint: DialogueBlueprint, clips: list[array]) -> list[dict]:
    timeline: list[dict] = []
    for index, (utterance, clip) in enumerate(zip(blueprint.utterances, clips, strict=True)):
        duration = len(clip) / SAMPLE_RATE
        if not timeline:
            start = 0.25
        else:
            previous = timeline[-1]
            if blueprint.timing == TimingKind.OVERLAP:
                start = max(previous["start_sec"], previous["end_sec"] - 0.1)
            elif (
                blueprint.timing == TimingKind.INTERRUPTION
                and utterance.speaker == "user"
                and previous["speaker"] == "assistant"
            ):
                start = max(previous["start_sec"] + 0.05, previous["natural_end_sec"] - 0.15)
                previous["end_sec"] = round(min(previous["end_sec"], start + 0.1), 6)
            else:
                gap = 0.9 if blueprint.timing == TimingKind.PAUSE and index == 1 else 0.35
                start = previous["end_sec"] + gap
        natural_end = start + duration
        timeline.append(
            {
                "speaker": utterance.speaker,
                "text": utterance.text,
                "start_sec": round(start, 6),
                "end_sec": round(natural_end, 6),
                "natural_end_sec": round(natural_end, 6),
            }
        )
    return timeline


def _assistant_alignments(timeline: list[dict]) -> list[list]:
    alignments: list[list] = []
    punctuation = string.punctuation + "“”‘’"
    for item in timeline:
        if item["speaker"] != "assistant":
            continue
        words = [word.strip(punctuation) for word in re.findall(r"\S+", item["text"])]
        words = [word for word in words if word]
        if not words:
            continue
        duration = item["natural_end_sec"] - item["start_sec"]
        for index, word in enumerate(words):
            start = item["start_sec"] + duration * index / len(words)
            end = item["start_sec"] + duration * (index + 1) / len(words)
            if start >= item["end_sec"]:
                break
            alignments.append(
                [word, [round(start, 6), round(min(end, item["end_sec"]), 6)], "SPEAKER_MAIN"]
            )
    return alignments


class MacOSSaySynthesizer:
    def __init__(self, *, user_voice: str, assistant_voice: str) -> None:
        self.user_voice = user_voice
        self.assistant_voice = assistant_voice

    def __call__(self, utterance: Utterance, destination: Path) -> None:
        voice = self.assistant_voice if utterance.speaker == "assistant" else self.user_voice
        subprocess.run(
            [
                "/usr/bin/say",
                "-v",
                voice,
                "--file-format=WAVE",
                "--data-format=LEI16@24000",
                "-o",
                str(destination),
                utterance.text,
            ],
            check=True,
        )
        try:
            with wave.open(str(destination), "rb") as rendered:
                frames = rendered.getnframes()
        except (FileNotFoundError, wave.Error) as error:
            raise RuntimeError(f"speech synthesizer did not produce a PCM WAV: {destination}") from error
        if frames == 0:
            raise RuntimeError(f"speech synthesizer produced zero audio frames: {destination}")


def render_corpus(
    blueprints: list[DialogueBlueprint],
    output_directory: Path,
    *,
    synthesizer_for: SynthesizerFactory,
    system_prompt_for: Callable[[DialogueBlueprint], str] | None = None,
    workers: int = 1,
) -> list[RenderedExample]:
    if not blueprints:
        raise ValueError("blueprint corpus is empty")
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")

    def render(blueprint: DialogueBlueprint) -> RenderedExample:
        return render_blueprint(
            blueprint,
            output_directory / "audio",
            synthesize=synthesizer_for(blueprint),
            system_prompt=(system_prompt_for(blueprint) if system_prompt_for else None),
        )

    if workers == 1:
        return [render(blueprint) for blueprint in blueprints]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tolkien-render") as pool:
        return list(pool.map(render, blueprints))


def _load_blueprints(path: Path) -> list[DialogueBlueprint]:
    result: list[DialogueBlueprint] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                result.append(DialogueBlueprint.model_validate_json(line))
            except ValueError as error:
                raise ValueError(f"invalid blueprint on line {line_number}: {error}") from error
    if not result:
        raise ValueError("blueprint file is empty")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render Tolkien dialogue blueprints as duplex audio")
    parser.add_argument("--blueprints", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--user-voice", default="Samantha")
    parser.add_argument("--aragorn-voice", default="Daniel")
    parser.add_argument("--sauron-voice", default="Ralph")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--manifest-audio-root", type=Path)
    parser.add_argument(
        "--character-directory",
        type=Path,
        default=Path(__file__).parents[3] / "configs" / "characters",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    blueprints = _load_blueprints(args.blueprints)
    registry = CharacterRegistry.from_directory(args.character_directory)

    def synthesizer_for(blueprint: DialogueBlueprint) -> Synthesizer:
        voice = args.aragorn_voice if blueprint.character == "aragorn" else args.sauron_voice
        return MacOSSaySynthesizer(user_voice=args.user_voice, assistant_voice=voice)

    rendered = render_corpus(
        blueprints,
        args.output_directory,
        synthesizer_for=synthesizer_for,
        system_prompt_for=lambda blueprint: registry.resolve(blueprint.character).text_prompt,
        workers=args.workers,
    )
    manifests = write_moshi_manifests(
        args.output_directory / "manifests",
        rendered,
        manifest_audio_root=args.manifest_audio_root,
    )
    print(
        json.dumps(
            {
                "examples": len(rendered),
                "manifests": {split.value: str(path) for split, path in manifests.items()},
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
