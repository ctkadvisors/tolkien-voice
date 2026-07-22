from __future__ import annotations

from array import array
import asyncio
from pathlib import Path
from typing import Any
import wave


FRAME_SECONDS = 0.04


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
    audio_bytes = bytearray()
    loop = asyncio.get_running_loop()
    total_stream_seconds = len(samples) / sample_rate + response_seconds
    timeout_seconds = total_stream_seconds + 30

    async with aiohttp.ClientSession() as client:
        async with client.ws_connect(url, compress=0, max_msg_size=1024 * 1024) as websocket:
            handshake = await asyncio.wait_for(websocket.receive(), timeout=timeout_seconds)
            if handshake.type != aiohttp.WSMsgType.BINARY or handshake.data != b"\x00":
                raise RuntimeError("native STS handshake was not received")
            started = loop.time()
            deadline = started + total_stream_seconds

            async def receive_output() -> None:
                while loop.time() < deadline:
                    remaining = deadline - loop.time()
                    try:
                        message = await asyncio.wait_for(websocket.receive(), timeout=remaining)
                    except TimeoutError:
                        return
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
                        audio_times.append(loop.time() - started)
                        audio_bytes.extend(payload[1:])
                    elif payload[0] == 2:
                        text_parts.append(payload[1:].decode("utf-8", errors="replace"))

            receiver = asyncio.create_task(receive_output())
            writer = sphn.OpusStreamWriter(sample_rate)
            frame_samples = round(sample_rate * FRAME_SECONDS)
            stream_samples = [*samples, *([0.0] * round(response_seconds * sample_rate))]
            for offset in range(0, len(stream_samples), frame_samples):
                frame = np.asarray(stream_samples[offset : offset + frame_samples], dtype=np.float32)
                writer.append_pcm(frame)
                payload = writer.read_bytes()
                if payload:
                    await websocket.send_bytes(b"\x01" + payload)
                await asyncio.sleep(FRAME_SECONDS)
            await receiver

    if not audio_times:
        raise RuntimeError("native STS probe returned no audio")
    if raw_audio_path is not None:
        raw_audio_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        raw_audio_path.write_bytes(audio_bytes)
        raw_audio_path.chmod(0o600)
    packet_summary = summarize_audio_packets(audio_times)
    return {
        **packet_summary,
        "text_messages": len(text_parts),
        "audio_bytes": len(audio_bytes),
        "first_audio_ms": round(max(0.0, audio_times[0] - speech_end_seconds) * 1000),
        "audio_duration_ms": round(len(audio_times) * FRAME_SECONDS * 1000),
        "text": "".join(text_parts).strip(),
    }
