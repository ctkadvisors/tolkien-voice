from __future__ import annotations

from array import array
import json
from pathlib import Path
import wave

import pytest

from tolkien_voice.training.manifest import validate_rendered_examples, write_moshi_manifests
from tolkien_voice.training.schema import RenderedExample, Split


def write_stereo(path: Path, *, sample_rate: int = 24_000, silent_right: bool = False) -> None:
    samples = array("h")
    for _ in range(sample_rate):
        samples.extend((1000, 0 if silent_right else 1000))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())


def example(tmp_path: Path, *, item_id: str = "aragorn-normal-0001") -> RenderedExample:
    wav_path = tmp_path / f"{item_id}.wav"
    alignment_path = tmp_path / f"{item_id}.json"
    write_stereo(wav_path)
    alignment_path.write_text(
        json.dumps(
            {
                "alignments": [["Walk", [0.1, 0.4], "SPEAKER_MAIN"]],
                "system_prompt": "<system> Aragorn prompt <system>",
            }
        )
    )
    return RenderedExample(
        id=item_id,
        blueprint_hash="a" * 64,
        wav_path=wav_path,
        alignment_path=alignment_path,
        duration_sec=1.0,
        split=Split.TRAIN,
        voice_license="private-apple-system-voice",
    )


def test_manifest_validation_and_moshi_jsonl(tmp_path: Path) -> None:
    item = example(tmp_path)

    validate_rendered_examples([item])
    paths = write_moshi_manifests(tmp_path / "manifests", [item])

    row = json.loads(paths[Split.TRAIN].read_text())
    assert row == {"path": str(item.wav_path.resolve()), "duration": 1.0}


def test_manifest_can_target_the_training_container_mount(tmp_path: Path) -> None:
    item = example(tmp_path)

    paths = write_moshi_manifests(
        tmp_path / "manifests",
        [item],
        manifest_audio_root=Path("/data/datasets/mvp-v1/audio"),
    )

    row = json.loads(paths[Split.TRAIN].read_text())
    assert row["path"] == f"/data/datasets/mvp-v1/audio/{item.wav_path.name}"


def test_manifest_rejects_duplicate_ids(tmp_path: Path) -> None:
    item = example(tmp_path)

    with pytest.raises(ValueError, match="unique"):
        validate_rendered_examples([item, item])


def test_manifest_rejects_silent_channel(tmp_path: Path) -> None:
    item = example(tmp_path)
    write_stereo(item.wav_path, silent_right=True)

    with pytest.raises(ValueError, match="silent"):
        validate_rendered_examples([item])


def test_manifest_rejects_wrong_sample_rate(tmp_path: Path) -> None:
    item = example(tmp_path)
    write_stereo(item.wav_path, sample_rate=16_000)

    with pytest.raises(ValueError, match="24,000"):
        validate_rendered_examples([item])


def test_manifest_rejects_unconditioned_training_example(tmp_path: Path) -> None:
    item = example(tmp_path)
    item.alignment_path.write_text(
        json.dumps({"alignments": [["Walk", [0.1, 0.4], "SPEAKER_MAIN"]]})
    )

    with pytest.raises(ValueError, match="system prompt"):
        validate_rendered_examples([item])
