import { describe, expect, it, vi } from "vitest";
import { decodeMessage, encodeMessage } from "./encoder";

describe("control message round-trip", () => {
  it("encodes and decodes the bargeIn action", () => {
    const encoded = encodeMessage({ type: "control", action: "bargeIn" });
    expect(encoded).toEqual(new Uint8Array([0x03, 0b00000100]));

    const decoded = decodeMessage(encoded);
    expect(decoded).toEqual({ type: "control", action: "bargeIn" });
  });

  it("logs and returns a ping instead of throwing on an unrecognized control action", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);

    const decoded = decodeMessage(new Uint8Array([0x03, 0b11111111]));

    expect(decoded).toEqual({ type: "ping" });
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });
});
