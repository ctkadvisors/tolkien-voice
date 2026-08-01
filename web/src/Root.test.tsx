import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { App } from "./Root";

const api = vi.hoisted(() => ({
  getCharacters: vi.fn(),
  getSession: vi.fn(),
  startSession: vi.fn(),
  endSession: vi.fn(),
}));

vi.mock("./api", () => api);

const GANDALF = {
  id: "gandalf",
  display_name: "Gandalf",
  summary: "A wizard",
  default_voice: "NATM1.pt",
  allowed_voices: ["NATM1.pt"],
};

describe("backend session failures", () => {
  it("surfaces the recorded activation failure instead of the default blurb", async () => {
    api.getCharacters.mockResolvedValue([GANDALF]);
    api.getSession.mockResolvedValue({
      state: "idle",
      error: "PersonaPlex failed to start",
    });

    render(<App />);

    expect(await screen.findByText("PersonaPlex failed to start")).toBeTruthy();
  });

  it("names the blocking owner when the model lease timed out", async () => {
    api.getCharacters.mockResolvedValue([GANDALF]);
    api.getSession.mockResolvedValue({
      state: "idle",
      error: "large-model lease timed out",
      blocking_owner: { kind: "llm", name: "qwen-coder" },
    });

    render(<App />);

    expect(
      await screen.findByText("large-model lease timed out (in use by qwen-coder)")
    ).toBeTruthy();
  });
});
