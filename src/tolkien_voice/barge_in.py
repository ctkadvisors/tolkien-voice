from __future__ import annotations

import math
from typing import Protocol, Sequence

FRAME_SECONDS = 0.04


def frame_rms(samples: Sequence[float]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


class BargeInDetector:
    """Edge-triggered activity state machine with asymmetric thresholds:
    fast to activate (a few sustained loud frames means the user has
    started talking), slow to deactivate (many sustained quiet frames
    means the user has genuinely gone quiet, not just paused between
    words) -- deliberately different sensitivity in each direction, not
    a single reusable threshold applied twice.
    """

    def __init__(
        self,
        *,
        activation_threshold: float,
        activation_sustained_frames: int,
        deactivation_threshold: float,
        deactivation_sustained_frames: int,
    ) -> None:
        if activation_threshold < 0 or deactivation_threshold < 0:
            raise ValueError("thresholds must be non-negative")
        if activation_sustained_frames <= 0 or deactivation_sustained_frames <= 0:
            raise ValueError("sustained-frame counts must be positive")
        self.activation_threshold = activation_threshold
        self.activation_sustained_frames = activation_sustained_frames
        self.deactivation_threshold = deactivation_threshold
        self.deactivation_sustained_frames = deactivation_sustained_frames
        self.active = False
        self._run_length = 0

    def append(self, rms: float) -> bool:
        was_active = self.active
        if self.active:
            self._run_length = (
                self._run_length + 1 if rms <= self.deactivation_threshold else 0
            )
            if self._run_length >= self.deactivation_sustained_frames:
                self.active = False
                self._run_length = 0
        else:
            self._run_length = (
                self._run_length + 1 if rms >= self.activation_threshold else 0
            )
            if self._run_length >= self.activation_sustained_frames:
                self.active = True
                self._run_length = 0
        return self.active != was_active


BARGE_IN_CONTROL_BYTE = 4


class OpusPcmReader(Protocol):
    def append_bytes(self, payload: bytes) -> None: ...

    def read_pcm(self) -> Sequence[float]: ...


class BargeInGate:
    """Owns one connection's Opus-decode state and BargeInDetector,
    translating incoming Opus payloads from the user's channel into
    activation/deactivation decisions for that connection.
    """

    def __init__(
        self,
        *,
        reader: OpusPcmReader,
        detector: BargeInDetector,
        sample_rate: int,
    ) -> None:
        self._reader = reader
        self._detector = detector
        self._frame_samples = round(sample_rate * FRAME_SECONDS)

    @property
    def muting(self) -> bool:
        return self._detector.active

    def observe_user_audio(self, opus_payload: bytes) -> bool:
        """Decodes one Opus payload from the user's channel and feeds its
        RMS-per-frame content into the detector. Returns True exactly when
        this call causes the detector to activate.
        """
        self._reader.append_bytes(opus_payload)
        decoded = list(self._reader.read_pcm())
        activated = False
        for offset in range(0, len(decoded), self._frame_samples):
            chunk = decoded[offset : offset + self._frame_samples]
            if not chunk:
                continue
            if self._detector.append(frame_rms(chunk)) and self._detector.active:
                activated = True
        return activated
