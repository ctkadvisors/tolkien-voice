import asyncio
from array import array
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import wave

import pytest

import tolkien_voice.training.native_probe as native_probe
from tolkien_voice.training.native_probe import (
    ActivityDetector,
    load_user_channel,
    measure_interruption_response,
    pad_pcm_frame,
    probe_native_session,
    summarize_interruption_text,
    summarize_audio_packets,
)


def test_probe_loads_right_user_channel_and_finds_speech_end(tmp_path: Path) -> None:
    path = tmp_path / "duplex.wav"
    frames = array("h")
    for index in range(2400):
        frames.extend((2000, 1000 if 400 <= index < 1600 else 0))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(frames.tobytes())

    sample_rate, samples, speech_end_seconds = load_user_channel(path)

    assert sample_rate == 24_000
    assert len(samples) == 2400
    assert 0.066 <= speech_end_seconds <= 0.067
    assert samples[0] == 0
    assert samples[500] > 0


def test_packet_summary_counts_only_playout_sized_gaps() -> None:
    summary = summarize_audio_packets([0.10, 0.14, 0.18, 0.36, 0.40], gap_threshold=0.12)

    assert summary["audio_messages"] == 5
    assert summary["underrun_count"] == 1
    assert summary["max_packet_gap_ms"] == 180


def test_interruption_response_requires_sustained_agent_silence() -> None:
    frames = [
        (1.00, 0.08),
        (1.04, 0.07),
        (1.08, 0.06),
        (1.12, 0.01),
        (1.16, 0.06),
        (1.20, 0.01),
        (1.24, 0.008),
        (1.28, 0.006),
        (1.32, 0.005),
        (1.36, 0.004),
    ]

    response_ms = measure_interruption_response(
        frames,
        barge_in_seconds=1.10,
        silence_threshold=0.015,
        sustained_frames=4,
    )

    assert response_ms == 100


def test_interruption_response_is_unmeasured_when_agent_never_yields() -> None:
    frames = [(1.0 + index * 0.04, 0.08) for index in range(10)]

    response_ms = measure_interruption_response(
        frames,
        barge_in_seconds=1.10,
        silence_threshold=0.015,
        sustained_frames=4,
    )

    assert response_ms is None


def test_interruption_response_ignores_brief_pause_when_agent_resumes() -> None:
    frames = [(1.00 + index * 0.04, 0.08) for index in range(3)]
    frames.extend((1.12 + index * 0.04, 0.01) for index in range(10))
    frames.extend((1.52 + index * 0.04, 0.08) for index in range(2))

    response_ms = measure_interruption_response(
        frames,
        barge_in_seconds=1.10,
        silence_threshold=0.015,
        sustained_frames=12,
    )

    assert response_ms is None


def test_interruption_text_is_split_at_barge_in_with_response_timestamp() -> None:
    summary = summarize_interruption_text(
        [(0.90, "The road"), (1.08, " ends."), (1.24, " I will stop.")],
        barge_in_seconds=1.10,
    )

    assert summary == {
        "interruption_text": "The road ends. I will stop.",
        "post_barge_text": "I will stop.",
        "first_post_barge_text_ms": 140,
    }


def test_interruption_text_response_is_unmeasured_without_post_barge_tokens() -> None:
    summary = summarize_interruption_text([(0.90, "Before.")], barge_in_seconds=1.10)

    assert summary["post_barge_text"] == ""
    assert summary["first_post_barge_text_ms"] is None


def test_activity_detector_requires_consecutive_active_frames() -> None:
    detector = ActivityDetector(threshold=0.03, sustained_frames=3)

    assert detector.append(0.04) is False
    assert detector.append(0.01) is False
    assert detector.append(0.04) is False
    assert detector.append(0.05) is False
    assert detector.append(0.06) is True


def test_pcm_frame_is_zero_padded_to_codec_frame_size() -> None:
    padded = pad_pcm_frame([0.5, -0.25], frame_samples=4)

    assert padded == [0.5, -0.25, 0.0, 0.0]


@pytest.mark.asyncio
async def test_probe_native_session_waits_for_sender_and_measures_real_speech(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav_path = tmp_path / "probe.wav"
    frames = array("h", [1000] * 960)
    with wave.open(str(wav_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(frames.tobytes())

    ws_msg_type = SimpleNamespace(BINARY="binary", CLOSE="close", CLOSED="closed", ERROR="error")

    class FakeMessage:
        def __init__(self, message_type: str, data: bytes) -> None:
            self.type = message_type
            self.data = data

    class FakeWebSocket:
        def __init__(self) -> None:
            self._queue: asyncio.Queue[FakeMessage] = asyncio.Queue()
            self._handshake_sent = False
            self._send_count = 0

        async def __aenter__(self) -> "FakeWebSocket":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def receive(self) -> FakeMessage:
            if not self._handshake_sent:
                self._handshake_sent = True
                return FakeMessage(ws_msg_type.BINARY, b"\x00")
            return await self._queue.get()

        async def send_bytes(self, _data: bytes) -> None:
            self._send_count += 1
            await asyncio.sleep(0.02)
            if self._send_count == 1:
                await self._queue.put(FakeMessage(ws_msg_type.BINARY, b"\x01silence"))
            elif self._send_count == 2:
                await self._queue.put(FakeMessage(ws_msg_type.BINARY, b"\x01silence"))
            elif self._send_count == 3:
                await self._queue.put(FakeMessage(ws_msg_type.BINARY, b"\x01speech-1"))
                await self._queue.put(FakeMessage(ws_msg_type.BINARY, b"\x02tail"))
                await self._queue.put(FakeMessage(ws_msg_type.BINARY, b"\x01speech-2"))
                await self._queue.put(FakeMessage(ws_msg_type.BINARY, b"\x01speech-3"))

    websocket = FakeWebSocket()

    class FakeClientSession:
        async def __aenter__(self) -> "FakeClientSession":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        def ws_connect(self, *_: object, **__: object) -> FakeWebSocket:
            return websocket

    class FakeWriter:
        def __init__(self, sample_rate: int) -> None:
            self.sample_rate = sample_rate

        def append_pcm(self, _frame: object) -> None:
            return None

        def read_bytes(self) -> bytes:
            return b"frame"

    class FakeReader:
        def __init__(self, sample_rate: int) -> None:
            self.sample_rate = sample_rate
            self._payload = b""

        def append_bytes(self, payload: bytes) -> None:
            self._payload = payload

        def read_pcm(self) -> list[float]:
            frame_samples = round(self.sample_rate * native_probe.FRAME_SECONDS)
            value = 0.05 if self._payload.startswith(b"speech") else 0.0
            return [value] * frame_samples

    fake_aiohttp = ModuleType("aiohttp")
    fake_aiohttp.ClientSession = FakeClientSession
    fake_aiohttp.WSMsgType = ws_msg_type
    fake_numpy = ModuleType("numpy")

    class FakeArray:
        def __init__(self, values: list[float]) -> None:
            self._values = values

        @property
        def size(self) -> int:
            return len(self._values)

        def reshape(self, _shape: int) -> "FakeArray":
            return self

        def __getitem__(self, item: slice | int) -> "FakeArray | float":
            if isinstance(item, slice):
                return FakeArray(self._values[item])
            return self._values[item]

        def __iter__(self):
            return iter(self._values)

    def asarray(values: object, dtype: object | None = None) -> FakeArray:
        del dtype
        return FakeArray(list(values))

    def square(values: FakeArray) -> FakeArray:
        return FakeArray([value * value for value in values])

    def mean(values: FakeArray) -> float:
        squared = list(values)
        return sum(squared) / len(squared)

    fake_numpy.asarray = asarray
    fake_numpy.float32 = "float32"
    fake_numpy.mean = mean
    fake_numpy.sqrt = math.sqrt
    fake_numpy.square = square
    fake_sphn = ModuleType("sphn")
    fake_sphn.OpusStreamReader = FakeReader
    fake_sphn.OpusStreamWriter = FakeWriter
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "sphn", fake_sphn)

    summary = await probe_native_session("ws://probe", wav_path, response_seconds=0.08)

    assert summary["audio_messages"] == 5
    assert summary["text"] == "tail"
    assert summary["first_audio_ms"] >= 100


@pytest.mark.asyncio
async def test_probe_native_interruption_exposes_output_frames_for_reward_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tolkien_voice.training.native_probe import probe_native_interruption

    def write_wav(path: Path, amplitude: int) -> None:
        frames = array("h", [amplitude] * 960)
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(24_000)
            output.writeframes(frames.tobytes())

    opening_wav = tmp_path / "opening.wav"
    interruption_wav = tmp_path / "interruption.wav"
    write_wav(opening_wav, amplitude=1000)
    write_wav(interruption_wav, amplitude=2000)

    ws_msg_type = SimpleNamespace(BINARY="binary", CLOSE="close", CLOSED="closed", ERROR="error")

    class FakeMessage:
        def __init__(self, message_type: str, data: bytes) -> None:
            self.type = message_type
            self.data = data

    class FakeWebSocket:
        def __init__(self) -> None:
            self._queue: asyncio.Queue[FakeMessage] = asyncio.Queue()
            self._handshake_sent = False
            self._send_count = 0

        async def __aenter__(self) -> "FakeWebSocket":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def receive(self) -> FakeMessage:
            if not self._handshake_sent:
                self._handshake_sent = True
                return FakeMessage(ws_msg_type.BINARY, b"\x00")
            return await self._queue.get()

        async def send_bytes(self, _data: bytes) -> None:
            self._send_count += 1
            await asyncio.sleep(0.02)
            if self._send_count == 1:
                for _ in range(3):
                    await self._queue.put(FakeMessage(ws_msg_type.BINARY, b"\x01speech"))

    websocket = FakeWebSocket()

    class FakeClientSession:
        async def __aenter__(self) -> "FakeClientSession":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        def ws_connect(self, *_: object, **__: object) -> FakeWebSocket:
            return websocket

    class FakeWriter:
        def __init__(self, sample_rate: int) -> None:
            self.sample_rate = sample_rate

        def append_pcm(self, _frame: object) -> None:
            return None

        def read_bytes(self) -> bytes:
            return b"frame"

    class FakeReader:
        def __init__(self, sample_rate: int) -> None:
            self.sample_rate = sample_rate
            self._payload = b""

        def append_bytes(self, payload: bytes) -> None:
            self._payload = payload

        def read_pcm(self) -> list[float]:
            frame_samples = round(self.sample_rate * native_probe.FRAME_SECONDS)
            value = 0.05 if self._payload.startswith(b"speech") else 0.0
            return [value] * frame_samples

    fake_aiohttp = ModuleType("aiohttp")
    fake_aiohttp.ClientSession = FakeClientSession
    fake_aiohttp.WSMsgType = ws_msg_type
    fake_numpy = ModuleType("numpy")

    class FakeArray:
        def __init__(self, values: list[float]) -> None:
            self._values = values

        @property
        def size(self) -> int:
            return len(self._values)

        def reshape(self, _shape: int) -> "FakeArray":
            return self

        def __getitem__(self, item: slice | int) -> "FakeArray | float":
            if isinstance(item, slice):
                return FakeArray(self._values[item])
            return self._values[item]

        def __iter__(self):
            return iter(self._values)

    def asarray(values: object, dtype: object | None = None) -> FakeArray:
        del dtype
        return FakeArray(list(values))

    def square(values: FakeArray) -> FakeArray:
        return FakeArray([value * value for value in values])

    def mean(values: FakeArray) -> float:
        squared = list(values)
        return sum(squared) / len(squared)

    def zeros(shape: int, dtype: object | None = None) -> FakeArray:
        del dtype
        return FakeArray([0.0] * shape)

    fake_numpy.asarray = asarray
    fake_numpy.float32 = "float32"
    fake_numpy.mean = mean
    fake_numpy.sqrt = math.sqrt
    fake_numpy.square = square
    fake_numpy.zeros = zeros
    fake_sphn = ModuleType("sphn")
    fake_sphn.OpusStreamReader = FakeReader
    fake_sphn.OpusStreamWriter = FakeWriter
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "sphn", fake_sphn)

    result = await probe_native_interruption(
        "ws://probe",
        opening_wav,
        interruption_wav,
        max_wait_seconds=1.0,
        response_seconds=0.04,
    )

    assert "output_frames" in result
    assert len(result["output_frames"]) > 0
    assert all(isinstance(frame, tuple) and len(frame) == 2 for frame in result["output_frames"])
    assert result["max_wait_seconds"] == 1.0
    assert result["first_output_received_seconds"] == result["output_frames"][0][0]
