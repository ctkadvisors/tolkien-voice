from __future__ import annotations

from array import array
import json
import os
from pathlib import Path
import wave

from .schema import RenderedExample, Split


SAMPLE_RATE = 24_000
MAX_SAFE_SAMPLE = int(32_767 * 0.99)


def validate_rendered_examples(examples: list[RenderedExample]) -> None:
    if not examples:
        raise ValueError("rendered dataset must not be empty")
    ids = [item.id for item in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("rendered example IDs must be unique")
    hashes = [item.blueprint_hash for item in examples]
    if len(hashes) != len(set(hashes)):
        raise ValueError("rendered blueprint hashes must be unique")
    for item in examples:
        if not item.wav_path.is_file():
            raise ValueError(f"missing WAV: {item.wav_path}")
        if not item.alignment_path.is_file():
            raise ValueError(f"missing alignment: {item.alignment_path}")
        if item.wav_path.stem != item.alignment_path.stem:
            raise ValueError(f"WAV and alignment stems differ for {item.id}")
        with wave.open(str(item.wav_path), "rb") as source:
            if source.getnchannels() != 2:
                raise ValueError(f"rendered WAV must have two channels: {item.id}")
            if source.getsampwidth() != 2:
                raise ValueError(f"rendered WAV must be PCM16: {item.id}")
            if source.getframerate() != SAMPLE_RATE:
                raise ValueError(f"rendered WAV must be 24,000 Hz: {item.id}")
            duration = source.getnframes() / source.getframerate()
            samples = array("h")
            samples.frombytes(source.readframes(source.getnframes()))
        if abs(duration - item.duration_sec) > 0.05:
            raise ValueError(f"duration metadata does not match WAV: {item.id}")
        left = samples[0::2]
        right = samples[1::2]
        if not any(left) or not any(right):
            raise ValueError(f"rendered WAV contains a silent channel: {item.id}")
        if max(abs(sample) for sample in samples) >= MAX_SAFE_SAMPLE:
            raise ValueError(f"rendered WAV is clipped: {item.id}")
        payload = json.loads(item.alignment_path.read_text())
        alignments = payload.get("alignments") if isinstance(payload, dict) else None
        if not isinstance(alignments, list) or not alignments:
            raise ValueError(f"alignment list is empty: {item.id}")


def write_moshi_manifests(
    output_directory: Path,
    examples: list[RenderedExample],
    *,
    manifest_audio_root: Path | None = None,
) -> dict[Split, Path]:
    validate_rendered_examples(examples)
    output_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    result: dict[Split, Path] = {}
    for split in Split:
        selected = [item for item in examples if item.split == split]
        if not selected:
            continue
        destination = output_directory / f"{split.value}.jsonl"
        temporary = destination.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as output:
            for item in selected:
                audio_path = (
                    item.wav_path.resolve()
                    if manifest_audio_root is None
                    else manifest_audio_root / item.wav_path.name
                )
                output.write(
                    json.dumps(
                        {
                            "path": str(audio_path),
                            "duration": item.duration_sec,
                        }
                    )
                    + "\n"
                )
        temporary.chmod(0o600)
        os.replace(temporary, destination)
        result[split] = destination
    return result
