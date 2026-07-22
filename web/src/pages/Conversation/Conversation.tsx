import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { MutableRefObject } from "react";

import { MediaContext } from "./MediaContext";
import { SocketContext } from "./SocketContext";
import { ServerAudio } from "./components/ServerAudio/ServerAudio";
import { TextDisplay } from "./components/TextDisplay/TextDisplay";
import { UserAudio } from "./components/UserAudio/UserAudio";
import { useSocket } from "./hooks/useSocket";


type Props = {
  sessionId: string;
  characterName: string;
  audioContext: MutableRefObject<AudioContext>;
  worklet: MutableRefObject<AudioWorkletNode>;
  onConversationEnd: () => void | Promise<void>;
};


export function Conversation({
  sessionId,
  characterName,
  audioContext,
  worklet,
  onConversationEnd,
}: Props) {
  const transcriptRef = useRef<HTMLDivElement>(null);
  const micDuration = useRef(0);
  const actualAudioPlayed = useRef(0);
  const [showTranscript, setShowTranscript] = useState(false);
  const [connectionEnded, setConnectionEnded] = useState(false);
  const socketUrl = useMemo(() => {
    const url = new URL(window.location.origin);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    url.pathname = `/api/session/${encodeURIComponent(sessionId)}/chat`;
    url.search = "";
    return url.toString();
  }, [sessionId]);
  const { socketStatus, socket, sendMessage, start, stop } = useSocket({
    uri: socketUrl,
    onDisconnect: () => setConnectionEnded(true),
  });

  useEffect(() => {
    start();
    return stop;
  }, [start, stop]);

  const end = useCallback(async () => {
    stop();
    await onConversationEnd();
  }, [onConversationEnd, stop]);

  const statusLabel = socketStatus === "connected"
    ? `Speaking with ${characterName}`
    : socketStatus === "connecting"
      ? `Entering ${characterName}'s listening room`
      : "Connection paused";

  return (
    <SocketContext.Provider value={{ socketStatus, sendMessage, socket }}>
      <MediaContext.Provider value={{ audioContext, worklet, micDuration, actualAudioPlayed }}>
        <main className="conversation-shell">
          <header className="conversation-header">
            <div>
              <p className="eyebrow">Live audience</p>
              <h1>{characterName}</h1>
            </div>
            <div className={`live-state is-${socketStatus}`} role="status" aria-live="polite">
              <span aria-hidden="true" />
              {statusLabel}
            </div>
            <button className="end-button" type="button" onClick={() => void end()}>
              End conversation
            </button>
          </header>

          <section className="duplex-stage" aria-label="Live full-duplex audio">
            <div className="voice-orbit character-orbit">
              <span>{characterName}</span>
              <ServerAudio setGetAudioStats={() => undefined} theme="dark" />
            </div>
            <div className="between-mark" aria-hidden="true">↔</div>
            <div className="voice-orbit listener-orbit">
              <span>You</span>
              <UserAudio theme="dark" />
            </div>
          </section>

          {connectionEnded ? (
            <div className="connection-card">
              <p>The audio path closed after three reconnect attempts.</p>
              <button className="secondary-button" type="button" onClick={() => {
                setConnectionEnded(false);
                start();
              }}>
                Reconnect
              </button>
            </div>
          ) : null}

          <section className="transcript-drawer">
            <button
              className="text-button"
              type="button"
              aria-expanded={showTranscript}
              onClick={() => setShowTranscript((visible) => !visible)}
            >
              {showTranscript ? "Hide live text" : "Show live text"}
            </button>
            {showTranscript ? (
              <div className="transcript" ref={transcriptRef}>
                <TextDisplay containerRef={transcriptRef} />
              </div>
            ) : null}
          </section>
          <p className="privacy-note">Ephemeral audio · private tailnet · no recording</p>
        </main>
      </MediaContext.Provider>
    </SocketContext.Provider>
  );
}
