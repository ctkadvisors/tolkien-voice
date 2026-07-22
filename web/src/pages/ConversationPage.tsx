import { useCallback, useEffect, useRef, useState } from "react";
import type { MutableRefObject } from "react";
import moshiProcessorUrl from "../audio-processor.ts?worker&url";

import type { Session } from "../types";
import { prewarmDecoderWorker } from "../decoder/decoderWorker";
import { Conversation } from "./Conversation/Conversation";


type Props = {
  session: Session;
  onEnd: () => void | Promise<void>;
};


type AudioState = "idle" | "requesting" | "ready" | "denied" | "error";


export function ConversationPage({ session, onEnd }: Props) {
  const audioContext = useRef<AudioContext | null>(null);
  const worklet = useRef<AudioWorkletNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const [mediaStream, setMediaStream] = useState<MediaStream | null>(null);
  const [inputDevices, setInputDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedInputDeviceId, setSelectedInputDeviceId] = useState("");
  const [inputError, setInputError] = useState<string | null>(null);
  const [audioState, setAudioState] = useState<AudioState>("idle");

  const prepareAudio = useCallback(async (deviceId?: string) => {
    const initialRequest = mediaStreamRef.current === null;
    if (initialRequest) setAudioState("requesting");
    setInputError(null);
    let context = audioContext.current;
    let nextStream: MediaStream | null = null;
    try {
      if (context === null) {
        context = new AudioContext();
        audioContext.current = context;
      }
      const resumePromise = context.resume();
      const permissionPromise = navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
          ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
        },
        video: false,
      });
      [, nextStream] = await Promise.all([resumePromise, permissionPromise]);
      if (worklet.current === null) {
        prewarmDecoderWorker(context.sampleRate);
        try {
          worklet.current = new AudioWorkletNode(context, "moshi-processor");
        } catch {
          await context.audioWorklet.addModule(moshiProcessorUrl);
          worklet.current = new AudioWorkletNode(context, "moshi-processor");
        }
        worklet.current.connect(context.destination);
      }
      let devices: MediaDeviceInfo[] = [];
      try {
        devices = await navigator.mediaDevices.enumerateDevices?.() ?? [];
      } catch {
        setInputError("Microphone is live; the browser did not expose its device list");
      }
      const previous = mediaStreamRef.current;
      mediaStreamRef.current = nextStream;
      setMediaStream(nextStream);
      previous?.getTracks().forEach((track) => track.stop());
      const track = nextStream.getTracks()[0];
      setSelectedInputDeviceId(track?.getSettings?.().deviceId || deviceId || "");
      setInputDevices(devices.filter((device) => device.kind === "audioinput"));
      setAudioState("ready");
    } catch (error) {
      nextStream?.getTracks().forEach((track) => track.stop());
      if (!initialRequest) {
        setInputError(error instanceof Error ? error.message : "Microphone switch failed");
        return;
      }
      if (context !== null) {
        await context.close().catch(() => undefined);
      }
      if (audioContext.current === context) {
        audioContext.current = null;
      }
      const denied = error instanceof DOMException && error.name === "NotAllowedError";
      setAudioState(denied ? "denied" : "error");
    }
  }, []);

  const mayConnect = session.state === "ready" || session.state === "disconnected";

  useEffect(() => () => {
    mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    worklet.current?.disconnect();
    void audioContext.current?.close();
  }, []);

  const endConversation = useCallback(async () => {
    mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    mediaStreamRef.current = null;
    setMediaStream(null);
    await onEnd();
  }, [onEnd]);

  if (session.state === "waiting_for_lease") {
    const owner = session.blocking_owner?.name;
    return (
      <WaitingRoom
        title={owner ? `Waiting for ${owner}` : "Waiting for the Spark"}
        detail="The active model will finish normally. This conversation will begin when memory is free."
        onEnd={onEnd}
      />
    );
  }

  if (session.state === "loading") {
    return (
      <WaitingRoom
        title={`Waking ${session.character_name ?? "the character"}`}
        detail="PersonaPlex is loading and warming its duplex audio streams."
        onEnd={onEnd}
      />
    );
  }

  if (mayConnect && audioState === "idle") {
    return (
      <WaitingRoom
        title={`${session.character_name ?? "Your character"} is ready`}
        detail="Enter when you are ready to start the private microphone and duplex audio stream."
        onEnd={onEnd}
      >
        <button className="secondary-button" type="button" onClick={() => void prepareAudio()}>
          Enter conversation
        </button>
      </WaitingRoom>
    );
  }

  if (audioState === "denied" || audioState === "error") {
    return (
      <WaitingRoom
        title="Microphone access is blocked"
        detail="Allow microphone access for this private Tailscale site, then try again."
        onEnd={onEnd}
      >
        <button className="secondary-button" type="button" onClick={() => void prepareAudio()}>
          Try microphone again
        </button>
      </WaitingRoom>
    );
  }

  if (audioState !== "ready" || !audioContext.current || !worklet.current || !mediaStream) {
    return (
      <WaitingRoom
        title="Opening the listening room"
        detail="Preparing your microphone and local audio decoder."
        onEnd={onEnd}
      />
    );
  }

  return (
    <Conversation
      sessionId={session.session_id ?? ""}
      characterName={session.character_name ?? session.character_id ?? "Character"}
      audioContext={audioContext as MutableRefObject<AudioContext>}
      worklet={worklet as MutableRefObject<AudioWorkletNode>}
      mediaStream={mediaStream}
      inputDevices={inputDevices}
      selectedInputDeviceId={selectedInputDeviceId}
      onInputDeviceChange={(deviceId) => void prepareAudio(deviceId)}
      inputError={inputError}
      onConversationEnd={endConversation}
    />
  );
}


function WaitingRoom({
  title,
  detail,
  onEnd,
  children,
}: {
  title: string;
  detail: string;
  onEnd: () => void | Promise<void>;
  children?: React.ReactNode;
}) {
  return (
    <main className="waiting-shell">
      <div className="pulse-seal" aria-hidden="true"><span>✦</span></div>
      <p className="eyebrow">The listening room</p>
      <h1>{title}</h1>
      <p>{detail}</p>
      <div className="waiting-actions">
        {children}
        <button className="text-button" type="button" onClick={() => void onEnd()}>Cancel</button>
      </div>
      <span className="sr-only" role="status" aria-live="polite">{title}. {detail}</span>
    </main>
  );
}
