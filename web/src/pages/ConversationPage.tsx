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
  const [audioState, setAudioState] = useState<AudioState>("idle");

  const prepareAudio = useCallback(async () => {
    setAudioState("requesting");
    let context: AudioContext | null = null;
    let permissionStream: MediaStream | null = null;
    try {
      context = new AudioContext();
      audioContext.current = context;
      const resumePromise = context.resume();
      const permissionPromise = navigator.mediaDevices.getUserMedia({ audio: true });
      [, permissionStream] = await Promise.all([resumePromise, permissionPromise]);
      permissionStream.getTracks().forEach((track) => track.stop());
      prewarmDecoderWorker(context.sampleRate);
      try {
        worklet.current = new AudioWorkletNode(context, "moshi-processor");
      } catch {
        await context.audioWorklet.addModule(moshiProcessorUrl);
        worklet.current = new AudioWorkletNode(context, "moshi-processor");
      }
      worklet.current.connect(context.destination);
      setAudioState("ready");
    } catch (error) {
      permissionStream?.getTracks().forEach((track) => track.stop());
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
    worklet.current?.disconnect();
    void audioContext.current?.close();
  }, []);

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

  if (audioState !== "ready" || !audioContext.current || !worklet.current) {
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
      onConversationEnd={onEnd}
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
