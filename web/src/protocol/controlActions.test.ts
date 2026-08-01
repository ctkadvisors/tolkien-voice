import { describe, expect, it } from "vitest";
import { shouldFlushPlayback } from "./controlActions";

describe("shouldFlushPlayback", () => {
  it("is true for a bargeIn control message", () => {
    expect(shouldFlushPlayback({ type: "control", action: "bargeIn" })).toBe(true);
  });

  it("is false for other control actions", () => {
    expect(shouldFlushPlayback({ type: "control", action: "pause" })).toBe(false);
  });

  it("is false for non-control message types", () => {
    expect(shouldFlushPlayback({ type: "audio", data: new Uint8Array() })).toBe(false);
    expect(shouldFlushPlayback({ type: "ping" })).toBe(false);
  });
});
