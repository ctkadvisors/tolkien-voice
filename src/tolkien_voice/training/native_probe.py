from __future__ import annotations

from array import array
import asyncio
from contextlib import suppress
from pathlib import Path
from typing import Any
import wave


FRAME_SECONDS = 0.04


class ActivityDetector:
    def __init__(self, *, threshold: float, sustained_frames: int) -> None:
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        if sustained_frames <= 0:
            raise ValueError("sustained_frames must be positive")
        self.threshold = threshold
        self.sustained_frames = sustained_frames
        self._active_frames = 0

    def append(self, rms: float) -> bool:
        self._active_frames = self._active_frames + 1 if rms >= self.threshold else 0
        return self._active_frames >= self.sustained_frames


def pad_pcm_frame(samples: list[float], *, frame_samples: int) -> list[float]:
    if frame_samples <= 0:
        raise ValueError("frame_samples must be positive")
    if len(samples) > frame_samples:
        raise ValueError("PCM chunk exceeds codec frame size")
    return [*samples, *([0.0] * (frame_samples - len(samples)))]


def load_user_channel(path: Path) -> tuple[int, list[float], float]:
    with wave.open(str(path), "rb") as source:
        if source.getsampwidth() != 2:
            raise ValueError("probe WAV must be PCM16")
        sample_rate = source.getframerate()
        channels = source.getnchannels()
        if channels not in {1, 2}:
            raise ValueError("probe WAV must be mono or stereo")
        interleaved = array("h")
        interleaved.frombytes(source.readframes(source.getnframes()))
    selected = interleaved[1::2] if channels == 2 else interleaved
    samples = [sample / 32768.0 for sample in selected]
    active = [index for index, sample in enumerate(selected) if abs(sample) >= 200]
    speech_end = (active[-1] + 1) / sample_rate if active else 0.0
    return sample_rate, samples, speech_end


def summarize_audio_packets(
    received_seconds: list[float], *, gap_threshold: float = 0.12
) -> dict[str, int]:
    gaps = [right - left for left, right in zip(received_seconds, received_seconds[1:])]
    return {
        "audio_messages": len(received_seconds),
        "underrun_count": sum(gap > gap_threshold for gap in gaps),
        "max_packet_gap_ms": round(max(gaps, default=0) * 1000),
    }


def detect_speech_start(
    frames: list[tuple[float, float]], *, threshold: float, sustained_frames: int
) -> float | None:
    detector = ActivityDetector(threshold=threshold, sustained_frames=sustained_frames)
    active_times: list[float] = []
    for received_seconds, rms in frames:
        if rms >= threshold:
            active_times.append(received_seconds)
            if len(active_times) > sustained_frames:
                active_times.pop(0)
        else:
            active_times.clear()
        if detector.append(rms):
            return active_times[0]
    return None


def measure_interruption_response(
    frames: list[tuple[float, float]],
    *,
    barge_in_seconds: float,
    silence_threshold: float,
    sustained_frames: int,
) -> int | None:
    if sustained_frames <= 0:
        raise ValueError("sustained_frames must be positive")
    quiet_started: float | None = None
    quiet_count = 0
    for received_seconds, rms in frames:
        if received_seconds < barge_in_seconds:
            continue
        if rms <= silence_threshold:
            if quiet_started is None:
                quiet_started = received_seconds
            quiet_count += 1
            if quiet_count >= sustained_frames:
                return round(max(0.0, quiet_started - barge_in_seconds) * 1000)
        else:
            quiet_started = None
            quiet_count = 0
    return None


def summarize_interruption_text(
    frames: list[tuple[float, str]], *, barge_in_seconds: float
) -> dict[str, str | int | None]:
    post_barge = [(received, text) for received, text in frames if received >= barge_in_seconds]
    return {
        "interruption_text": "".join(text for _, text in frames).strip(),
        "post_barge_text": "".join(text for _, text in post_barge).strip(),
        "first_post_barge_text_ms": (
            round(max(0.0, post_barge[0][0] - barge_in_seconds) * 1000)
            if post_barge
            else None
        ),
    }


async def probe_native_session(
    url: str,
    wav_path: Path,
    *,
    response_seconds: float = 12.0,
    raw_audio_path: Path | None = None,
) -> dict[str, Any]:
    try:
        import aiohttp
        import numpy as np
        import sphn
    except ImportError as error:
        raise RuntimeError("native probe requires aiohttp, numpy, and sphn") from error

    sample_rate, samples, speech_end_seconds = load_user_channel(wav_path)
    if sample_rate != 24_000:
        raise ValueError("probe WAV must be 24 kHz")
    if response_seconds <= 0:
        raise ValueError("response_seconds must be positive")

    text_parts: list[str] = []
    audio_times: list[float] = []
    output_frames: list[tuple[float, float]] = []
    audio_bytes = bytearray()
    loop = asyncio.get_running_loop()
    total_stream_seconds = len(samples) / sample_rate + response_seconds
    timeout_seconds = total_stream_seconds + 30
    frame_samples = round(sample_rate * FRAME_SECONDS)

    async with aiohttp.ClientSession() as client:
        async with client.ws_connect(url, compress=0, max_msg_size=1024 * 1024) as websocket:
            handshake = await asyncio.wait_for(websocket.receive(), timeout=timeout_seconds)
            if handshake.type != aiohttp.WSMsgType.BINARY or handshake.data != b"\x00":
                raise RuntimeError("native STS handshake was not received")
            started = loop.time()
            sender_done = asyncio.Event()
            reader = sphn.OpusStreamReader(sample_rate)

            async def receive_output() -> None:
                while True:
                    try:
                        message = await asyncio.wait_for(
                            websocket.receive(), timeout=FRAME_SECONDS * 3
                        )
                    except TimeoutError:
                        if sender_done.is_set():
                            return
                        continue
                    if message.type != aiohttp.WSMsgType.BINARY:
                        if message.type in {
                            aiohttp.WSMsgType.CLOSE,
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        }:
                            return
                        continue
                    payload = message.data
                    if not payload:
                        continue
                    if payload[0] == 1:
                        received_seconds = loop.time() - started
                        audio_times.append(received_seconds)
                        audio_bytes.extend(payload[1:])
                        reader.append_bytes(payload[1:])
                        decoded = np.asarray(reader.read_pcm(), dtype=np.float32).reshape(-1)
                        for offset in range(0, decoded.size, frame_samples):
                            frame = decoded[offset : offset + frame_samples]
                            if frame.size == 0:
                                continue
                            rms = float(np.sqrt(np.mean(np.square(frame))))
                            output_frames.append((received_seconds, rms))
                    elif payload[0] == 2:
                        text_parts.append(payload[1:].decode("utf-8", errors="replace"))

            receiver = asyncio.create_task(receive_output())
            writer = sphn.OpusStreamWriter(sample_rate)
            stream_samples = [*samples, *([0.0] * round(response_seconds * sample_rate))]
            try:
                for offset in range(0, len(stream_samples), frame_samples):
                    frame = np.asarray(
                        pad_pcm_frame(
                            stream_samples[offset : offset + frame_samples],
                            frame_samples=frame_samples,
                        ),
                        dtype=np.float32,
                    )
                    writer.append_pcm(frame)
                    payload = writer.read_bytes()
                    if payload:
                        await websocket.send_bytes(b"\x01" + payload)
                    await asyncio.sleep(FRAME_SECONDS)
            finally:
                sender_done.set()
                await receiver

    if not audio_times:
        raise RuntimeError("native STS probe returned no audio")
    if raw_audio_path is not None:
        raw_audio_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        raw_audio_path.write_bytes(audio_bytes)
        raw_audio_path.chmod(0o600)
    packet_summary = summarize_audio_packets(audio_times)
    first_speech_seconds = detect_speech_start(
        output_frames, threshold=0.02, sustained_frames=3
    )
    return {
        **packet_summary,
        "text_messages": len(text_parts),
        "audio_bytes": len(audio_bytes),
        "first_audio_ms": round(
            max(
                0.0,
                (first_speech_seconds if first_speech_seconds is not None else audio_times[0])
                - speech_end_seconds,
            )
            * 1000
        ),
        "audio_duration_ms": round(len(audio_times) * FRAME_SECONDS * 1000),
        "text": "".join(text_parts).strip(),
    }


async def probe_native_interruption(
    url: str,
    opening_wav_path: Path,
    interruption_wav_path: Path,
    *,
    max_wait_seconds: float = 12.0,
    response_seconds: float = 4.0,
    activity_threshold: float = 0.02,
    silence_threshold: float = 0.015,
    raw_audio_path: Path | None = None,
) -> dict[str, Any]:
    """Measure how quickly native STS yields after real speech arrives mid-response."""
    try:
        import aiohttp
        import numpy as np
        import sphn
    except ImportError as error:
        raise RuntimeError("native probe requires aiohttp, numpy, and sphn") from error

    opening_rate, opening_samples, _ = load_user_channel(opening_wav_path)
    interruption_rate, interruption_samples, _ = load_user_channel(interruption_wav_path)
    if opening_rate != 24_000 or interruption_rate != 24_000:
        raise ValueError("interruption probe WAVs must be 24 kHz")
    if max_wait_seconds <= 0 or response_seconds <= 0:
        raise ValueError("interruption probe timing must be positive")

    sample_rate = opening_rate
    frame_samples = round(sample_rate * FRAME_SECONDS)
    loop = asyncio.get_running_loop()
    output_started = asyncio.Event()
    output_detector = ActivityDetector(threshold=activity_threshold, sustained_frames=3)
    output_frames: list[tuple[float, float]] = []
    audio_times: list[float] = []
    text_frames: list[tuple[float, str]] = []
    audio_bytes = bytearray()
    barge_in_seconds: float | None = None
    timeout_seconds = (
        len(opening_samples) / sample_rate
        + max_wait_seconds
        + len(interruption_samples) / sample_rate
        + response_seconds
        + 30
    )

    async with aiohttp.ClientSession() as client:
        async with client.ws_connect(url, compress=0, max_msg_size=1024 * 1024) as websocket:
            handshake = await asyncio.wait_for(websocket.receive(), timeout=timeout_seconds)
            if handshake.type != aiohttp.WSMsgType.BINARY or handshake.data != b"\x00":
                raise RuntimeError("native STS handshake was not received")
            started = loop.time()
            reader = sphn.OpusStreamReader(sample_rate)

            async def receive_output() -> None:
                while True:
                    message = await websocket.receive()
                    if message.type != aiohttp.WSMsgType.BINARY:
                        if message.type in {
                            aiohttp.WSMsgType.CLOSE,
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        }:
                            return
                        continue
                    payload = message.data
                    if not payload:
                        continue
                    received_seconds = loop.time() - started
                    if payload[0] == 2:
                        text_frames.append(
                            (received_seconds, payload[1:].decode("utf-8", errors="replace"))
                        )
                        continue
                    if payload[0] != 1:
                        continue
                    audio_times.append(received_seconds)
                    audio_bytes.extend(payload[1:])
                    reader.append_bytes(payload[1:])
                    decoded = np.asarray(reader.read_pcm(), dtype=np.float32).reshape(-1)
                    for offset in range(0, decoded.size, frame_samples):
                        frame = decoded[offset : offset + frame_samples]
                        if frame.size == 0:
                            continue
                        rms = float(np.sqrt(np.mean(np.square(frame))))
                        output_frames.append((received_seconds, rms))
                        if output_detector.append(rms):
                            output_started.set()

            receiver = asyncio.create_task(receive_output())
            writer = sphn.OpusStreamWriter(sample_rate)

            async def send_frame(frame: Any) -> None:
                values = pad_pcm_frame(list(frame), frame_samples=frame_samples)
                writer.append_pcm(np.asarray(values, dtype=np.float32))
                payload = writer.read_bytes()
                if payload:
                    await websocket.send_bytes(b"\x01" + payload)
                await asyncio.sleep(FRAME_SECONDS)

            try:
                for offset in range(0, len(opening_samples), frame_samples):
                    await send_frame(opening_samples[offset : offset + frame_samples])

                wait_deadline = loop.time() + max_wait_seconds
                silence = np.zeros(frame_samples, dtype=np.float32)
                while not output_started.is_set() and loop.time() < wait_deadline:
                    await send_frame(silence)
                if not output_started.is_set():
                    raise RuntimeError("native STS did not speak before interruption timeout")

                for offset in range(0, len(interruption_samples), frame_samples):
                    frame = interruption_samples[offset : offset + frame_samples]
                    if barge_in_seconds is None and max((abs(item) for item in frame), default=0) >= 0.01:
                        barge_in_seconds = loop.time() - started
                    await send_frame(frame)
                if barge_in_seconds is None:
                    raise ValueError("interruption probe contains no audible speech")

                for _ in range(round(response_seconds / FRAME_SECONDS)):
                    await send_frame(silence)
            finally:
                receiver.cancel()
                with suppress(asyncio.CancelledError):
                    await receiver

    if raw_audio_path is not None:
        raw_audio_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        raw_audio_path.write_bytes(audio_bytes)
        raw_audio_path.chmod(0o600)
    assert barge_in_seconds is not None
    packet_summary = summarize_audio_packets(audio_times)
    return {
        "interruption_response_ms": measure_interruption_response(
            output_frames,
            barge_in_seconds=barge_in_seconds,
            silence_threshold=silence_threshold,
            # Require roughly half a second of quiet so a normal pause is not
            # mistaken for the assistant yielding to a barge-in.
            sustained_frames=12,
        ),
        "barge_in_ms": round(barge_in_seconds * 1000),
        "output_activity_frames": len(output_frames),
        "interruption_audio_bytes": len(audio_bytes),
        "interruption_underrun_count": packet_summary["underrun_count"],
        "interruption_max_packet_gap_ms": packet_summary["max_packet_gap_ms"],
        **summarize_interruption_text(text_frames, barge_in_seconds=barge_in_seconds),
    }
