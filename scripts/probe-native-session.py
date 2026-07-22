#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import wave

import aiohttp
import numpy as np
import sphn


def load_user_channel(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as source:
        if source.getsampwidth() != 2:
            raise ValueError("probe WAV must be PCM16")
        sample_rate = source.getframerate()
        channels = source.getnchannels()
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    samples = samples.reshape(-1, channels)
    user_channel = samples[:, 1 if channels > 1 else 0]
    return sample_rate, user_channel.astype(np.float32) / 32768.0


async def probe(url: str, wav_path: Path, *, timeout_seconds: float) -> dict[str, object]:
    sample_rate, pcm = load_user_channel(wav_path)
    if sample_rate != 24_000:
        raise ValueError("probe WAV must be 24 kHz")
    counts = {"audio_messages": 0, "text_messages": 0, "audio_bytes": 0}
    text_parts: list[str] = []
    async with aiohttp.ClientSession() as client:
        async with client.ws_connect(url, compress=0, max_msg_size=1024 * 1024) as websocket:
            handshake = await asyncio.wait_for(websocket.receive(), timeout=timeout_seconds)
            if handshake.type != aiohttp.WSMsgType.BINARY or handshake.data != b"\x00":
                raise RuntimeError("native STS handshake was not received")

            async def receive_output() -> None:
                deadline = asyncio.get_running_loop().time() + timeout_seconds
                while asyncio.get_running_loop().time() < deadline:
                    remaining = deadline - asyncio.get_running_loop().time()
                    message = await asyncio.wait_for(websocket.receive(), timeout=remaining)
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
                        counts["audio_messages"] += 1
                        counts["audio_bytes"] += len(payload) - 1
                        if counts["audio_messages"] >= 2:
                            return
                    elif payload[0] == 2:
                        counts["text_messages"] += 1
                        text_parts.append(payload[1:].decode("utf-8", errors="replace"))

            receiver = asyncio.create_task(receive_output())
            writer = sphn.OpusStreamWriter(sample_rate)
            frame_samples = 1_920
            for offset in range(0, len(pcm), frame_samples):
                writer.append_pcm(pcm[offset : offset + frame_samples])
                payload = writer.read_bytes()
                if payload:
                    await websocket.send_bytes(b"\x01" + payload)
                if receiver.done():
                    break
                await asyncio.sleep(0.04)
            try:
                await receiver
            finally:
                if not receiver.done():
                    receiver.cancel()
                    await asyncio.gather(receiver, return_exceptions=True)
    return {**counts, "text": "".join(text_parts).strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe a ready native STS session")
    parser.add_argument("--url", required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30)
    args = parser.parse_args()
    result = asyncio.run(probe(args.url, args.wav, timeout_seconds=args.timeout_seconds))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["audio_messages"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
