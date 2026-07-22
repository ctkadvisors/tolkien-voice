import { describe, expect, it } from "vitest";

import { DECODER_WORKER_URL } from "./decoderWorker";


describe("decoder worker asset", () => {
  it("loads the untouched public decoder beside its WASM", () => {
    expect(DECODER_WORKER_URL).toBe("/assets/decoderWorker.min.js");
  });
});
