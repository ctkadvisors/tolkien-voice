from __future__ import annotations

from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import string
import wave

from .manifest import SAMPLE_RATE
from .native_probe import FRAME_SECONDS, load_user_channel
from .rollout import Rollout, RolloutGroup
from .schema import RenderedExample, Split

MAX_SAFE_SAMPLE = int(32_767 * 0.99)
VOICE_LICENSE = "self-distillation-v6-continuation"


def select_best_rollout(
    group: RolloutGroup,
    *,
    yield_delay_reward_gate: float,
) -> Rollout | None:
    """Among rollouts that yielded within yield_delay_reward_gate seconds
    (yield_delay_reward is negative seconds, so clearing the gate means
    yield_delay_reward >= yield_delay_reward_gate), returns the one with
    the highest quality_reward. Returns None if no rollout clears the
    gate -- mirrors the project's release gate structure: interruption
    timing is a hard pass/fail threshold, persona quality is what's
    compared once timing passes.
    """
    survivors = [
        rollout
        for rollout in group.rollouts
        if rollout.yield_delay_reward >= yield_delay_reward_gate
    ]
    if not survivors:
        return None
    return max(survivors, key=lambda rollout: rollout.quality_reward)


def leading_silence_seconds(wav_path: Path, *, activity_threshold: float = 0.01) -> float:
    """Seconds of near-silence before the first audible frame in wav_path.

    Uses the same absolute-amplitude threshold probe_native_interruption
    applies inline to find barge_in_seconds, so a corpus builder can place
    this same clip on a reconstructed timeline with its audible onset
    landing exactly where the live probe measured it.
    """
    sample_rate, samples, _ = load_user_channel(wav_path)
    frame_samples = round(sample_rate * FRAME_SECONDS)
    for offset in range(0, len(samples), frame_samples):
        frame = samples[offset : offset + frame_samples]
        if max((abs(sample) for sample in frame), default=0.0) >= activity_threshold:
            return offset / sample_rate
    return len(samples) / sample_rate


def _decode_opus_pcm(opus_path: Path) -> array:
    import time

    import numpy as np
    import sphn

    # OpusStreamReader decodes on a background worker thread: append_bytes()
    # only enqueues the data, it does not block until it's decoded. A single
    # immediate read_pcm() call races the worker and returns an empty array.
    # close() lets the worker finish, after which read_pcm() keeps returning
    # decoded chunks until fully drained, then None -- poll until that point.
    reader = sphn.OpusStreamReader(SAMPLE_RATE)
    reader.append_bytes(opus_path.read_bytes())
    reader.close()
    chunks: list[np.ndarray] = []
    while True:
        chunk = reader.read_pcm()
        if chunk is None:
            break
        if chunk.size:
            chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
        else:
            time.sleep(0.005)
    decoded = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    clipped = np.clip(decoded, -1.0, 1.0)
    return array("h", (clipped * 32_767).astype(np.int16).tolist())


def _mono_pcm16(wav_path: Path) -> array:
    with wave.open(str(wav_path), "rb") as source:
        if source.getsampwidth() != 2:
            raise ValueError(f"expected PCM16: {wav_path}")
        channels = source.getnchannels()
        interleaved = array("h")
        interleaved.frombytes(source.readframes(source.getnframes()))
    return interleaved[1::2] if channels == 2 else interleaved


def _word_alignments(transcript: str, *, start_sec: float, end_sec: float) -> list[list]:
    punctuation = string.punctuation + "“”‘’"
    words = [word.strip(punctuation) for word in re.findall(r"\S+", transcript)]
    words = [word for word in words if word]
    if not words:
        return []
    duration = end_sec - start_sec
    alignments: list[list] = []
    for index, word in enumerate(words):
        word_start = start_sec + duration * index / len(words)
        word_end = start_sec + duration * (index + 1) / len(words)
        alignments.append(
            [word, [round(word_start, 6), round(word_end, 6)], "SPEAKER_MAIN"]
        )
    return alignments


def _word_alignments_from_frames(
    frames: list[tuple[float, str]], *, real_speech_end_sec: float
) -> list[list]:
    """Builds word-level alignments from real per-chunk timestamps rather
    than uniformly spacing a transcript across an entire captured turn.
    Each chunk's words are spaced across [this chunk's timestamp, the next
    chunk's timestamp) -- or, for the last chunk, [this chunk's timestamp,
    real_speech_end_sec) -- reusing _word_alignments per chunk so a chunk's
    own words still get sensible internal spacing without assuming the
    whole transcript arrived at a single instant.

    real_speech_end_sec must be the real end of audible assistant speech
    (the caller's own natural end for whatever assistant audio it placed),
    not the end of the full captured audio span -- that span can include a
    trailing silence window well past where the assistant actually stopped
    talking, and bounding the last chunk by it would spread that chunk's
    words across seconds of silence it was never spoken over.
    """
    alignments: list[list] = []
    for index, (chunk_start, chunk_text) in enumerate(frames):
        chunk_end = frames[index + 1][0] if index + 1 < len(frames) else real_speech_end_sec
        chunk_end = max(chunk_end, chunk_start)
        alignments.extend(_word_alignments(chunk_text, start_sec=chunk_start, end_sec=chunk_end))
    return alignments


def build_self_distillation_example(
    rollout: Rollout,
    *,
    scenario_id: str,
    character: str,
    opening_wav_path: Path,
    interruption_wav_path: Path,
    system_prompt: str,
    split: Split,
    output_directory: Path,
) -> RenderedExample:
    """Reconstructs a training example from only the post-barge portion of
    the model's real session: opening_wav_path at t=0 on the user (right)
    channel; interruption_wav_path positioned so its audible onset lands
    at max(opening_end_sec, rollout.barge_in_seconds) -- never before the
    opening finishes, by construction; the model's real post-barge audio
    (extracted from rollout.raw_audio_path by content position -- counting
    decoded output_frames chunks before barge-in, not converting wall-clock
    time -- for exactly rollout.delay_seconds) placed starting at that same
    onset point on the assistant (left) channel.

    Deliberately does not attempt to reconstruct the assistant's pre-barge
    turn (backchannel activity, near-silent listening) at all -- the
    corpus's semantic content (rollout.transcript) has always been
    post-barge-only, and reconstructing the full session's audio requires
    correctly modeling genuine full-duplex overlap between the assistant's
    pre-barge activity and the user's opening, which is not needed for
    this corpus's actual purpose (teaching correct post-interruption yield
    behavior). See docs/superpowers/specs/2026-07-30-self-distillation-post-barge-audio-design.md.
    """
    del character  # not needed for construction; kept for caller symmetry with scenario metadata
    output_directory.mkdir(parents=True, exist_ok=True, mode=0o700)

    opening_samples = _mono_pcm16(opening_wav_path)
    interruption_samples = _mono_pcm16(interruption_wav_path)
    assistant_samples = _decode_opus_pcm(rollout.raw_audio_path)

    frame_samples = round(SAMPLE_RATE * FRAME_SECONDS)
    barge_in_frame_index = sum(
        1 for received_seconds, _ in rollout.output_frames
        if received_seconds < rollout.barge_in_seconds
    )
    extraction_start = barge_in_frame_index * frame_samples
    if extraction_start >= len(assistant_samples):
        raise ValueError(
            f"decoded assistant audio for {scenario_id} does not extend to "
            f"barge-in (decoded {len(assistant_samples)} samples, barge-in "
            f"at sample {extraction_start})"
        )
    if rollout.delay_seconds is None:
        raise ValueError(
            f"cannot extract post-barge audio for {scenario_id}: "
            f"rollout.delay_seconds is None (this rollout never yielded)"
        )
    extraction_end = min(
        len(assistant_samples), extraction_start + round(rollout.delay_seconds * SAMPLE_RATE)
    )
    post_barge_samples = assistant_samples[extraction_start:extraction_end]

    onset_offset = leading_silence_seconds(interruption_wav_path)
    opening_end_sec = len(opening_samples) / SAMPLE_RATE
    interruption_onset_sec = max(opening_end_sec, rollout.barge_in_seconds)
    interruption_start_sec = max(0.0, interruption_onset_sec - onset_offset)
    assistant_start_sec = interruption_start_sec + onset_offset

    interruption_end_sec = interruption_start_sec + len(interruption_samples) / SAMPLE_RATE
    assistant_end_sec = assistant_start_sec + len(post_barge_samples) / SAMPLE_RATE
    total_seconds = max(opening_end_sec, interruption_end_sec, assistant_end_sec)
    total_frames = math.ceil(total_seconds * SAMPLE_RATE)

    left = array("i", [0] * total_frames)   # assistant
    right = array("i", [0] * total_frames)  # user

    def _mix(channel: array, samples: array, start_sec: float) -> None:
        start_frame = round(start_sec * SAMPLE_RATE)
        for offset, sample in enumerate(samples):
            target = start_frame + offset
            if target >= len(channel):
                break
            mixed = channel[target] + sample
            if abs(mixed) >= MAX_SAFE_SAMPLE:
                raise ValueError(f"mixed audio is clipped in {scenario_id}")
            channel[target] = mixed

    _mix(right, opening_samples, 0.0)
    _mix(right, interruption_samples, interruption_start_sec)
    _mix(left, post_barge_samples, assistant_start_sec)

    interleaved = array("h")
    for left_sample, right_sample in zip(left, right, strict=True):
        interleaved.extend((left_sample, right_sample))

    example_id = f"{scenario_id}-self-distill"
    wav_path = output_directory / f"{example_id}.wav"
    temporary_wav = wav_path.with_suffix(".wav.tmp")
    with wave.open(str(temporary_wav), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(interleaved.tobytes())
    os.replace(temporary_wav, wav_path)
    wav_path.chmod(0o600)

    segment_end_sec = assistant_start_sec + len(post_barge_samples) / SAMPLE_RATE

    def _content_position_sec(wall_clock_seconds: float) -> float:
        frame_index = sum(
            1 for received_seconds, _ in rollout.output_frames
            if received_seconds < wall_clock_seconds
        )
        return assistant_start_sec + (frame_index - barge_in_frame_index) * FRAME_SECONDS

    translated_frames = [
        (min(segment_end_sec, _content_position_sec(chunk_start)), text)
        for chunk_start, text in rollout.post_barge_text_frames
    ]
    alignments = _word_alignments_from_frames(
        translated_frames, real_speech_end_sec=segment_end_sec
    )
    alignment_path = output_directory / f"{example_id}.json"
    temporary_alignment = alignment_path.with_suffix(".json.tmp")
    temporary_alignment.write_text(
        json.dumps(
            {
                "alignments": alignments,
                "system_prompt": system_prompt,
            },
            indent=2,
        )
        + "\n"
    )
    temporary_alignment.chmod(0o600)
    os.replace(temporary_alignment, alignment_path)

    blueprint_hash = hashlib.sha256(
        f"{scenario_id}:{rollout.transcript}".encode("utf-8")
    ).hexdigest()

    return RenderedExample(
        id=example_id,
        blueprint_hash=blueprint_hash,
        wav_path=wav_path,
        alignment_path=alignment_path,
        duration_sec=total_frames / SAMPLE_RATE,
        split=split,
        voice_license=VOICE_LICENSE,
    )
