from __future__ import annotations

import math
from typing import Sequence

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
