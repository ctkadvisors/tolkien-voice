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


def _last_active_frame_seconds(
    frames: list[tuple[float, float]], *, activity_threshold: float = 0.02
) -> float | None:
    """Returns the received_seconds of the last frame whose rms is at or
    above activity_threshold -- the last moment there was real audible
    assistant output -- or None if no frame in frames is ever active.

    frames is the same (received_seconds, rms) shape as
    probe_native_interruption's output_frames / Rollout.output_frames.
    activity_threshold mirrors probe_native_interruption's own
    activity_threshold default (0.02), which is the threshold that function
    already uses to decide the assistant *started* speaking; using the same
    value here keeps "did it speak" and "when did it stop" symmetric rather
    than inventing a second magic number.
    """
    last_active: float | None = None
    for received_seconds, rms in frames:
        if rms >= activity_threshold:
            last_active = received_seconds
    return last_active


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
    (see _last_active_frame_seconds), not the end of the full captured
    audio span -- that span includes a trailing response_seconds window of
    mostly silence, and bounding the last chunk by it would spread that
    chunk's words across seconds of silence it was never spoken over.
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
    """Reconstructs the real recorded session timeline: opening_wav_path
    at t=0 and interruption_wav_path positioned so its audible onset lands
    at rollout.barge_in_seconds on the user (right) channel;
    rollout.raw_audio_path decoded from Opus and positioned starting at
    rollout.first_output_received_seconds on the assistant (left) channel.
    """
    del character  # not needed for construction; kept for caller symmetry with scenario metadata
    output_directory.mkdir(parents=True, exist_ok=True, mode=0o700)

    opening_samples = _mono_pcm16(opening_wav_path)
    interruption_samples = _mono_pcm16(interruption_wav_path)
    assistant_samples = _decode_opus_pcm(rollout.raw_audio_path)

    if (
        rollout.first_output_received_seconds is not None
        and rollout.last_output_received_seconds is not None
    ):
        # first_output_received_seconds and last_output_received_seconds are
        # both wall-clock receive timestamps from probe_native_interruption,
        # but they are captured at very different points in that function's
        # sender loop (which sleeps FRAME_SECONDS after every send, so
        # wall-clock time drifts ahead of nominal stream time as the session
        # runs). If PersonaPlex's server-side model free-runs on its own
        # clock rather than staying in lockstep with received frames, that
        # drift is harmless; if it runs in lockstep, the drift is real and
        # would misplace assistant audio relative to the interruption on the
        # reconstructed timeline -- corrupting exactly the yield-timing
        # signal this corpus exists to teach. Nobody can currently verify
        # which case holds (PersonaPlex's server source isn't available), so
        # rather than guess-correct the timing, cross-check the wall-clock
        # span against the independently-measured decoded-audio duration and
        # fail loudly here, before any partial artifact is written, rather
        # than silently emitting a corpus example with corrupted timing.
        wall_clock_span = rollout.last_output_received_seconds - rollout.first_output_received_seconds
        decoded_duration = len(assistant_samples) / SAMPLE_RATE
        drift = abs(wall_clock_span - decoded_duration)
        if drift > 0.15:
            raise ValueError(
                f"assistant audio timing is inconsistent for {scenario_id}: "
                f"wall-clock span {wall_clock_span:.3f}s vs decoded duration "
                f"{decoded_duration:.3f}s (drift {drift:.3f}s)"
            )

    onset_offset = leading_silence_seconds(interruption_wav_path)
    interruption_start_sec = max(0.0, rollout.barge_in_seconds - onset_offset)
    assistant_start_sec = rollout.first_output_received_seconds or 0.0

    opening_end_sec = len(opening_samples) / SAMPLE_RATE
    interruption_end_sec = interruption_start_sec + len(interruption_samples) / SAMPLE_RATE
    assistant_end_sec = assistant_start_sec + len(assistant_samples) / SAMPLE_RATE
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
    _mix(left, assistant_samples, assistant_start_sec)

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

    # The last post_barge_text_frames chunk must be bounded by the real end
    # of audible assistant speech, not assistant_end_sec -- assistant_end_sec
    # is the end of the entire captured assistant audio, which includes a
    # trailing response_seconds window (default 12.0s) of mostly silence
    # after the assistant actually stops talking. Falling back to
    # assistant_end_sec when no frame is ever active is a safe degenerate
    # default (matches the old behavior) rather than crashing on audio that
    # apparently never had any real output.
    real_speech_end_sec = _last_active_frame_seconds(rollout.output_frames)
    if real_speech_end_sec is None:
        real_speech_end_sec = assistant_end_sec
    alignments = _word_alignments_from_frames(
        rollout.post_barge_text_frames, real_speech_end_sec=real_speech_end_sec
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
