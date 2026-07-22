import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { beforeEach, expect, it, vi } from "vitest";

import { MediaContext } from "../MediaContext";
import { UserMediaStatuses, useUserAudio } from "./useUserAudio";


const recorderState = vi.hoisted(() => ({
  configurations: [] as Array<Record<string, unknown>>,
  instances: [] as Array<{
    onstop: () => void;
  }>,
  rejectStart: false,
  deferStop: false,
}));

vi.mock("opus-recorder", () => ({
  default: class FakeRecorder {
    encodedSamplePosition = 0;
    ondataavailable = (_data: Uint8Array) => undefined;
    onstart = () => undefined;
    onstop = () => undefined;

    constructor(configuration: Record<string, unknown>) {
      recorderState.configurations.push(configuration);
      recorderState.instances.push(this);
    }

    async start() {
      if (recorderState.rejectStart) throw new Error("encoder failed");
      this.onstart();
    }

    async stop() {
      if (!recorderState.deferStop) this.onstop();
    }
  },
}));


beforeEach(() => {
  recorderState.configurations = [];
  recorderState.instances = [];
  recorderState.rejectStart = false;
  recorderState.deferStop = false;
});


it("feeds the same selected microphone source to the meter and encoder", async () => {
  const source = { connect: vi.fn(), disconnect: vi.fn() } as unknown as MediaStreamAudioSourceNode;
  const analyser = { getFloatTimeDomainData: vi.fn() } as unknown as AnalyserNode;
  const stream = { getTracks: vi.fn(() => []) } as unknown as MediaStream;
  const audioContext = {
    sampleRate: 48_000,
    createAnalyser: vi.fn(() => analyser),
    createMediaStreamSource: vi.fn(() => source),
  } as unknown as AudioContext;
  const wrapper = ({ children }: { children: ReactNode }) => (
    <MediaContext.Provider
      value={{
        audioContext: { current: audioContext },
        worklet: { current: {} as AudioWorkletNode },
        micDuration: { current: 0 },
        actualAudioPlayed: { current: 0 },
      }}
    >
      {children}
    </MediaContext.Provider>
  );
  const { result } = renderHook(
    () => useUserAudio({ mediaStream: stream }),
    { wrapper },
  );

  await act(async () => result.current.startRecordingUser());

  expect(audioContext.createMediaStreamSource).toHaveBeenCalledWith(stream);
  expect(source.connect).toHaveBeenCalledWith(analyser);
  expect(recorderState.configurations[0]?.sourceNode).toBe(source);
  expect(result.current.status).toBe(UserMediaStatuses.RECORDING);
});


it("surfaces recorder startup failures instead of pretending to record", async () => {
  recorderState.rejectStart = true;
  const source = { connect: vi.fn(), disconnect: vi.fn() } as unknown as MediaStreamAudioSourceNode;
  const stream = { getTracks: vi.fn(() => []) } as unknown as MediaStream;
  const audioContext = {
    sampleRate: 48_000,
    createAnalyser: vi.fn(() => ({ getFloatTimeDomainData: vi.fn() })),
    createMediaStreamSource: vi.fn(() => source),
  } as unknown as AudioContext;
  const wrapper = ({ children }: { children: ReactNode }) => (
    <MediaContext.Provider
      value={{
        audioContext: { current: audioContext },
        worklet: { current: {} as AudioWorkletNode },
        micDuration: { current: 0 },
        actualAudioPlayed: { current: 0 },
      }}
    >
      {children}
    </MediaContext.Provider>
  );
  const { result } = renderHook(
    () => useUserAudio({ mediaStream: stream }),
    { wrapper },
  );

  await act(async () => result.current.startRecordingUser());

  await waitFor(() => expect(result.current.status).toBe(UserMediaStatuses.ERROR));
  expect(result.current.error).toBe("encoder failed");
  expect(source.disconnect).toHaveBeenCalled();
});


it("ignores a stale recorder stop after an input-device switch", async () => {
  const firstSource = { connect: vi.fn(), disconnect: vi.fn() } as unknown as MediaStreamAudioSourceNode;
  const secondSource = { connect: vi.fn(), disconnect: vi.fn() } as unknown as MediaStreamAudioSourceNode;
  const firstStream = {} as MediaStream;
  const secondStream = {} as MediaStream;
  const audioContext = {
    sampleRate: 48_000,
    createAnalyser: vi.fn(() => ({ getFloatTimeDomainData: vi.fn() })),
    createMediaStreamSource: vi.fn()
      .mockReturnValueOnce(firstSource)
      .mockReturnValueOnce(secondSource),
  } as unknown as AudioContext;
  const wrapper = ({ children }: { children: ReactNode }) => (
    <MediaContext.Provider
      value={{
        audioContext: { current: audioContext },
        worklet: { current: {} as AudioWorkletNode },
        micDuration: { current: 0 },
        actualAudioPlayed: { current: 0 },
      }}
    >
      {children}
    </MediaContext.Provider>
  );
  const { result, rerender } = renderHook(
    ({ stream }) => useUserAudio({ mediaStream: stream }),
    { wrapper, initialProps: { stream: firstStream } },
  );
  await act(async () => result.current.startRecordingUser());
  recorderState.deferStop = true;
  act(() => result.current.stopRecording());
  rerender({ stream: secondStream });
  await act(async () => result.current.startRecordingUser());

  act(() => recorderState.instances[0]?.onstop());

  expect(result.current.status).toBe(UserMediaStatuses.RECORDING);
  expect(secondSource.disconnect).not.toHaveBeenCalled();
});
