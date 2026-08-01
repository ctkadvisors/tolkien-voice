from __future__ import annotations

import pytest

from tolkien_voice.barge_in import BARGE_IN_CONTROL_BYTE, BargeInDetector, BargeInGate, frame_rms


def test_frame_rms_of_silence_is_zero() -> None:
    assert frame_rms([0.0, 0.0, 0.0]) == 0.0


def test_frame_rms_of_constant_signal_equals_its_magnitude() -> None:
    assert frame_rms([0.1, -0.1, 0.1, -0.1]) == pytest.approx(0.1)


def test_frame_rms_of_empty_frame_is_zero() -> None:
    assert frame_rms([]) == 0.0


def _detector() -> BargeInDetector:
    return BargeInDetector(
        activation_threshold=0.02,
        activation_sustained_frames=3,
        deactivation_threshold=0.015,
        deactivation_sustained_frames=12,
    )


def test_detector_starts_idle() -> None:
    assert _detector().active is False


def test_detector_activates_after_sustained_loud_frames() -> None:
    detector = _detector()

    assert detector.append(0.03) is False
    assert detector.append(0.03) is False
    assert detector.append(0.03) is True
    assert detector.active is True


def test_detector_ignores_a_brief_dip_below_activation_threshold() -> None:
    detector = _detector()

    assert detector.append(0.03) is False
    assert detector.append(0.01) is False
    assert detector.append(0.03) is False
    assert detector.append(0.03) is False
    assert detector.append(0.03) is True


def test_detector_requires_sustained_quiet_to_deactivate() -> None:
    detector = _detector()
    for _ in range(3):
        detector.append(0.03)
    assert detector.active is True

    for _ in range(11):
        assert detector.append(0.01) is False
    assert detector.active is True
    assert detector.append(0.01) is True
    assert detector.active is False


def test_detector_ignores_a_brief_loud_blip_while_deactivating() -> None:
    detector = _detector()
    for _ in range(3):
        detector.append(0.03)
    assert detector.active is True

    for _ in range(5):
        detector.append(0.01)
    detector.append(0.03)  # blip resets the deactivation run
    for _ in range(11):
        assert detector.append(0.01) is False
    assert detector.append(0.01) is True
    assert detector.active is False


def test_detector_rejects_non_positive_construction_values() -> None:
    with pytest.raises(ValueError):
        BargeInDetector(
            activation_threshold=-0.01,
            activation_sustained_frames=3,
            deactivation_threshold=0.015,
            deactivation_sustained_frames=12,
        )
    with pytest.raises(ValueError):
        BargeInDetector(
            activation_threshold=0.02,
            activation_sustained_frames=0,
            deactivation_threshold=0.015,
            deactivation_sustained_frames=12,
        )


class _FakeReader:
    def __init__(self) -> None:
        self.appended: list[bytes] = []
        self.next_pcm: list[float] = []

    def append_bytes(self, payload: bytes) -> None:
        self.appended.append(payload)

    def read_pcm(self) -> list[float]:
        return self.next_pcm


def _gate(reader: "_FakeReader") -> BargeInGate:
    return BargeInGate(
        reader=reader,
        detector=BargeInDetector(
            activation_threshold=0.02,
            activation_sustained_frames=3,
            deactivation_threshold=0.015,
            deactivation_sustained_frames=12,
        ),
        sample_rate=24_000,
    )


def test_gate_reports_activation_on_the_frame_that_crosses_the_threshold() -> None:
    reader = _FakeReader()
    gate = _gate(reader)
    frame_samples = round(24_000 * 0.04)
    reader.next_pcm = [0.03] * frame_samples

    assert gate.observe_user_audio(b"opus-1") is False
    assert gate.observe_user_audio(b"opus-2") is False
    assert gate.observe_user_audio(b"opus-3") is True
    assert gate.muting is True
    assert reader.appended == [b"opus-1", b"opus-2", b"opus-3"]


def test_gate_processes_multiple_frames_decoded_from_one_payload() -> None:
    reader = _FakeReader()
    gate = _gate(reader)
    frame_samples = round(24_000 * 0.04)
    reader.next_pcm = [0.03] * (frame_samples * 3)

    assert gate.observe_user_audio(b"opus") is True
    assert gate.muting is True


def test_gate_stays_quiet_below_the_activation_threshold() -> None:
    reader = _FakeReader()
    gate = _gate(reader)
    frame_samples = round(24_000 * 0.04)
    reader.next_pcm = [0.01] * frame_samples

    assert gate.observe_user_audio(b"opus-1") is False
    assert gate.observe_user_audio(b"opus-2") is False
    assert gate.muting is False


def test_barge_in_control_byte_matches_the_web_protocol_value() -> None:
    assert BARGE_IN_CONTROL_BYTE == 4
