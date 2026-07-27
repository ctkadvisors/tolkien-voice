#!/usr/bin/env python3
"""Validate the interruption reward against real collected recordings.

Go/no-go check per docs/superpowers/specs/2026-07-27-interruption-reward-module-design.md:
does the reward's ranking of "handled interruption well" agree with what a
sealed native A/B eval already established, before this is wired to
anything that spends GPU-hours on rollouts scored by it. Requires
observations collected AFTER src/tolkien_voice/training/native_probe.py's
output_frames change (Task 1) -- rows collected before that change will be
skipped with a warning, not silently scored as zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tolkien_voice.training.interruption_reward import (
    InterruptionRewardResult,
    compute_quality_reward,
    compute_yield_delay_reward,
)
from tolkien_voice.training.judge import JudgeClient, JudgeResponseError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:9090/v1")
    parser.add_argument("--model", default="tolkien")
    parser.add_argument("--observation-window-seconds", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()

    rows: list[dict] = []
    for path in args.input:
        with path.open(encoding="utf-8") as source:
            rows.extend(json.loads(line) for line in source if line.strip())

    skipped = [row for row in rows if "output_frames" not in row]
    scoreable = [row for row in rows if "output_frames" in row]
    if skipped:
        print(
            f"WARNING: skipping {len(skipped)} row(s) collected before "
            "output_frames was persisted (Task 1) -- these cannot be scored."
        )
    if not scoreable:
        print("No scoreable rows (all collected before output_frames was added).")
        return 1

    with JudgeClient(base_url=args.base_url, model=args.model) as judge:
        for row in scoreable:
            frames = [tuple(frame) for frame in row["output_frames"]]
            yield_reward = compute_yield_delay_reward(
                frames,
                barge_in_seconds=row["barge_in_ms"] / 1000,
                observation_window_seconds=args.observation_window_seconds,
                silence_threshold=0.015,
                sustained_frames=12,
            )
            known_delay_ms = row.get("interruption_response_ms")
            try:
                quality_reward = compute_quality_reward(
                    row["interruption_text"],
                    target_character=row["character"],
                    probe_prompt=row["prompt"],
                    expected_traits=row["expected_traits"],
                    judge=judge,
                    seed=args.seed,
                )
            except JudgeResponseError as error:
                print(f"WARNING: judge failed for {row['probe_id']} ({row['blinded_arm']}): {error}")
                continue
            result = InterruptionRewardResult(
                yield_delay_reward=yield_reward,
                quality_reward=quality_reward,
                delay_seconds=(known_delay_ms / 1000 if known_delay_ms is not None else None),
            )
            print(
                json.dumps(
                    {
                        "probe_id": row["probe_id"],
                        "blinded_arm": row["blinded_arm"],
                        "known_interruption_response_ms": known_delay_ms,
                        **result.model_dump(),
                    }
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
