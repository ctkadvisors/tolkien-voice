import { render, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";

import { Conversation } from "./Conversation";


vi.mock("./components/ServerAudio/ServerAudio", () => ({
  ServerAudio: () => null,
}));
vi.mock("./components/TextDisplay/TextDisplay", () => ({
  TextDisplay: () => null,
}));
vi.mock("./components/UserAudio/UserAudio", () => ({
  UserAudio: () => null,
}));


class FakeWebSocket {
  static OPEN = 1;
  static instances: FakeWebSocket[] = [];

  binaryType = "";
  readyState = 0;
  close = vi.fn();
  send = vi.fn();
  addEventListener = vi.fn();

  constructor(public readonly url: string) {
    FakeWebSocket.instances.push(this);
  }
}


beforeEach(() => {
  FakeWebSocket.instances = [];
  vi.stubGlobal("WebSocket", FakeWebSocket);
});


it("keeps one WebSocket while connection state rerenders", async () => {
  render(
    <Conversation
      sessionId="d2399de0-ae35-4064-8e94-deaa64a5b300"
      characterName="Aragorn"
      audioContext={{ current: {} as AudioContext }}
      worklet={{ current: {} as AudioWorkletNode }}
      onConversationEnd={vi.fn()}
    />,
  );

  await waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));
  await new Promise((resolve) => window.setTimeout(resolve, 20));

  expect(FakeWebSocket.instances).toHaveLength(1);
  expect(FakeWebSocket.instances[0]?.url).toContain(
    "/api/session/d2399de0-ae35-4064-8e94-deaa64a5b300/chat",
  );
});
