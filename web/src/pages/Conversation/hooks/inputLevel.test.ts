import { expect, it } from "vitest";

import { measureInputLevel } from "./inputLevel";


it("reports RMS and peak from microphone time-domain samples", () => {
  const level = measureInputLevel(new Float32Array([0.5, -0.5, 0.25, -0.25]));

  expect(level.peak).toBeCloseTo(0.5);
  expect(level.rms).toBeCloseTo(Math.sqrt(0.15625));
});


it("clamps invalid microphone samples instead of poisoning the meter", () => {
  const level = measureInputLevel(new Float32Array([Number.NaN, 2, -2, 0]));

  expect(level).toEqual({ rms: Math.sqrt(0.5), peak: 1 });
});
