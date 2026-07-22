import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ConversationPage } from "./ConversationPage";
import type { Session } from "../types";
import { prewarmDecoderWorker } from "../decoder/decoderWorker";


vi.mock("./Conversation/Conversation", () => ({
  Conversation: ({
    onConversationEnd,
    inputDevices = [],
    selectedInputDeviceId = "",
    onInputDeviceChange,
  }: {
    onConversationEnd: () => void;
    inputDevices?: MediaDeviceInfo[];
    selectedInputDeviceId?: string;
    onInputDeviceChange?: (deviceId: string) => void;
  }) => (
    <>
      <label>
        Microphone
        <select
          aria-label="Microphone input"
          value={selectedInputDeviceId}
          onChange={(event) => onInputDeviceChange?.(event.target.value)}
        >
          {inputDevices.map((device) => (
            <option key={device.deviceId} value={device.deviceId}>{device.label}</option>
          ))}
        </select>
      </label>
      <button onClick={onConversationEnd}>End native conversation</button>
    </>
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
    vi.clearAllMocks();
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
    const user = userEvent.setup();
    render(
      <ConversationPage session={{ ...baseSession, state: "ready" }} onEnd={vi.fn()} />,
    );

    await user.click(screen.getByRole("button", { name: "Enter conversation" }));
    expect(await screen.findByRole("heading", { name: /Microphone access is blocked/ })).toBeVisible();
    expect(screen.getByRole("button", { name: "Try microphone again" })).toBeEnabled();
  });

  it("requires a fresh user gesture before starting browser audio", async () => {
    const getUserMedia = vi.fn().mockResolvedValue({ getTracks: () => [] });
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia },
    });
    const user = userEvent.setup();
    render(
      <ConversationPage session={{ ...baseSession, state: "ready" }} onEnd={vi.fn()} />,
    );

    expect(getUserMedia).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "Enter conversation" }));

    await waitFor(() => expect(getUserMedia).toHaveBeenCalledTimes(1));
    expect(await screen.findByRole("button", { name: "End native conversation" })).toBeVisible();
  });

  it("resumes audio before awaiting microphone permission", async () => {
    const calls: string[] = [];
    vi.stubGlobal("AudioContext", class {
      sampleRate = 48000;
      destination = {};
      audioWorklet = { addModule: vi.fn().mockResolvedValue(undefined) };
      constructor() {
        calls.push("context");
      }
      resume = vi.fn(() => {
        calls.push("resume");
        return Promise.resolve();
      });
      close = vi.fn().mockResolvedValue(undefined);
    });
    const getUserMedia = vi.fn(() => {
      calls.push("permission");
      return Promise.resolve({ getTracks: () => [] });
    });
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia },
    });
    const user = userEvent.setup();
    render(
      <ConversationPage session={{ ...baseSession, state: "ready" }} onEnd={vi.fn()} />,
    );

    await user.click(screen.getByRole("button", { name: "Enter conversation" }));
    await waitFor(() => expect(getUserMedia).toHaveBeenCalledTimes(1));

    expect(calls.slice(0, 3)).toEqual(["context", "resume", "permission"]);
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

    await user.click(screen.getByRole("button", { name: "Enter conversation" }));
    await waitFor(() => expect(getUserMedia).toHaveBeenCalledTimes(1));
    await user.click(await screen.findByRole("button", { name: "End native conversation" }));

    expect(onEnd).toHaveBeenCalledTimes(1);
    expect(screen.queryByText(/record|download/i)).not.toBeInTheDocument();
  });

  it("keeps the granted stream and switches to the selected input device", async () => {
    const firstStop = vi.fn();
    const secondStop = vi.fn();
    const firstStream = {
      getTracks: () => [{ stop: firstStop, getSettings: () => ({ deviceId: "built-in" }) }],
    } as unknown as MediaStream;
    const secondStream = {
      getTracks: () => [{ stop: secondStop, getSettings: () => ({ deviceId: "usb" }) }],
    } as unknown as MediaStream;
    const getUserMedia = vi.fn()
      .mockResolvedValueOnce(firstStream)
      .mockResolvedValueOnce(secondStream);
    const enumerateDevices = vi.fn().mockResolvedValue([
      { kind: "audioinput", deviceId: "built-in", label: "MacBook Microphone" },
      { kind: "audioinput", deviceId: "usb", label: "USB Microphone" },
    ]);
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia, enumerateDevices },
    });
    const user = userEvent.setup();
    render(
      <ConversationPage session={{ ...baseSession, state: "ready" }} onEnd={vi.fn()} />,
    );

    await user.click(screen.getByRole("button", { name: "Enter conversation" }));
    const selector = await screen.findByRole("combobox", { name: "Microphone input" });
    await user.selectOptions(selector, "usb");

    await waitFor(() => expect(getUserMedia).toHaveBeenCalledTimes(2));
    expect(getUserMedia.mock.calls[1]?.[0]).toMatchObject({
      audio: { deviceId: { exact: "usb" } },
    });
    expect(firstStop).toHaveBeenCalledTimes(1);
    expect(secondStop).not.toHaveBeenCalled();
    expect(prewarmDecoderWorker).toHaveBeenCalledTimes(1);
  });

  it("keeps a granted microphone live when device enumeration is unavailable", async () => {
    const stop = vi.fn();
    const stream = {
      getTracks: () => [{ stop, getSettings: () => ({ deviceId: "default" }) }],
    } as unknown as MediaStream;
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: {
        getUserMedia: vi.fn().mockResolvedValue(stream),
        enumerateDevices: vi.fn().mockRejectedValue(new Error("enumeration unavailable")),
      },
    });
    const user = userEvent.setup();
    render(
      <ConversationPage session={{ ...baseSession, state: "ready" }} onEnd={vi.fn()} />,
    );

    await user.click(screen.getByRole("button", { name: "Enter conversation" }));

    expect(await screen.findByRole("button", { name: "End native conversation" })).toBeVisible();
    expect(stop).not.toHaveBeenCalled();
  });
});
