import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ConversationPage } from "./ConversationPage";
import type { Session } from "../types";


vi.mock("./Conversation/Conversation", () => ({
  Conversation: ({ onConversationEnd }: { onConversationEnd: () => void }) => (
    <button onClick={onConversationEnd}>End native conversation</button>
  ),
}));

vi.mock("../decoder/decoderWorker", () => ({
  prewarmDecoderWorker: vi.fn(),
}));


const baseSession: Session = {
  state: "waiting_for_lease",
  session_id: "76697582-935e-4912-a24c-bdca71e00fae",
  character_id: "gandalf",
  character_name: "Gandalf",
  voice_id: "NATM1.pt",
};


describe("ConversationPage", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.stubGlobal("AudioContext", class {
      sampleRate = 48000;
      destination = {};
      audioWorklet = { addModule: vi.fn().mockResolvedValue(undefined) };
      resume = vi.fn().mockResolvedValue(undefined);
      close = vi.fn().mockResolvedValue(undefined);
    });
    vi.stubGlobal("AudioWorkletNode", class {
      connect = vi.fn();
      disconnect = vi.fn();
    });
  });

  it("shows lease ownership and never offers eviction", () => {
    const getUserMedia = vi.fn();
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia },
    });
    render(
      <ConversationPage
        session={{ ...baseSession, blocking_owner: { kind: "text", name: "coder" } }}
        onEnd={vi.fn()}
      />,
    );

    expect(screen.getByRole("heading", { name: /Waiting for coder/ })).toBeVisible();
    expect(screen.queryByRole("button", { name: /evict|kill|stop coder/i })).not.toBeInTheDocument();
    expect(getUserMedia).not.toHaveBeenCalled();
  });

  it("does not request microphone access while the model loads", () => {
    const getUserMedia = vi.fn();
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia },
    });
    render(
      <ConversationPage session={{ ...baseSession, state: "loading" }} onEnd={vi.fn()} />,
    );

    expect(screen.getByRole("heading", { name: /Waking Gandalf/ })).toBeVisible();
    expect(getUserMedia).not.toHaveBeenCalled();
  });

  it("keeps microphone denial recoverable", async () => {
    const getUserMedia = vi.fn().mockRejectedValue(new DOMException("Denied", "NotAllowedError"));
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia },
    });
    render(
      <ConversationPage session={{ ...baseSession, state: "ready" }} onEnd={vi.fn()} />,
    );

    expect(await screen.findByRole("heading", { name: /Microphone access is blocked/ })).toBeVisible();
    expect(screen.getByRole("button", { name: "Try microphone again" })).toBeEnabled();
  });

  it("ends cleanly and exposes no recording controls", async () => {
    const stop = vi.fn();
    const getUserMedia = vi.fn().mockResolvedValue({ getTracks: () => [{ stop }] });
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia },
    });
    const onEnd = vi.fn();
    const user = userEvent.setup();
    render(
      <ConversationPage session={{ ...baseSession, state: "ready" }} onEnd={onEnd} />,
    );

    await waitFor(() => expect(getUserMedia).toHaveBeenCalledTimes(1));
    await user.click(await screen.findByRole("button", { name: "End native conversation" }));

    expect(onEnd).toHaveBeenCalledTimes(1);
    expect(screen.queryByText(/record|download/i)).not.toBeInTheDocument();
  });
});
