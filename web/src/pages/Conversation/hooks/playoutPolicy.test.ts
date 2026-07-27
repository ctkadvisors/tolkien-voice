import { expect, it } from "vitest";

import { createPlayoutPolicy } from "./playoutPolicy";


it("keeps enough audio to absorb normal packet jitter without unbounded latency", () => {
  const policy = createPlayoutPolicy(48_000);

  expect(policy.initialBufferSamples).toBe(5_760);
  expect(policy.partialBufferSamples).toBe(960);
  expect(policy.maxBufferSamples).toBe(7_680);
  expect(policy.totalMaximumSamples).toBe(14_400);
});


it("preserves the same millisecond bounds at a different browser sample rate", () => {
  const policy = createPlayoutPolicy(44_100);

  expect(policy.initialBufferSamples / 44_100).toBeCloseTo(0.12);
  expect(policy.totalMaximumSamples / 44_100).toBeCloseTo(0.3);
});
