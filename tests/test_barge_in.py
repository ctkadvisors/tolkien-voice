from __future__ import annotations

import pytest

from tolkien_voice.barge_in import BargeInDetector, frame_rms


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
