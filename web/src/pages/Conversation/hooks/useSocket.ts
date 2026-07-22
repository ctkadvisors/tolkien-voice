import { useCallback, useEffect, useRef, useState } from "react";

import { decodeMessage, encodeMessage } from "../../../protocol/encoder";
import type { SocketStatus, WSMessage } from "../../../protocol/types";


const RETRY_DELAYS_MS = [500, 1000, 2000] as const;


export function useSocket({
  uri,
  onDisconnect,
}: {
  uri: string;
  onDisconnect?: () => void;
}) {
  const socketRef = useRef<WebSocket | null>(null);
  const retryTimer = useRef<number | null>(null);
  const retryAttempt = useRef(0);
  const manuallyStopped = useRef(false);
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [socketStatus, setSocketStatus] = useState<SocketStatus>("disconnected");

  const clearRetry = useCallback(() => {
    if (retryTimer.current !== null) {
      window.clearTimeout(retryTimer.current);
      retryTimer.current = null;
    }
  }, []);

  const connect = useCallback(() => {
    clearRetry();
    manuallyStopped.current = false;
    socketRef.current?.close();
    const nextSocket = new WebSocket(uri);
    nextSocket.binaryType = "arraybuffer";
    socketRef.current = nextSocket;
    setSocket(nextSocket);
    setSocketStatus("connecting");

    nextSocket.addEventListener("message", (event) => {
      const message = decodeMessage(new Uint8Array(event.data));
      if (message.type === "handshake") {
        retryAttempt.current = 0;
        setSocketStatus("connected");
      }
    });
    nextSocket.addEventListener("close", () => {
      if (socketRef.current !== nextSocket) return;
      socketRef.current = null;
      setSocket(null);
      setSocketStatus("disconnected");
      if (manuallyStopped.current) return;
      const delay = RETRY_DELAYS_MS[retryAttempt.current];
      if (delay === undefined) {
        onDisconnect?.();
        return;
      }
      retryAttempt.current += 1;
      retryTimer.current = window.setTimeout(connect, delay);
    });
  }, [clearRetry, onDisconnect, uri]);

  const stop = useCallback(() => {
    manuallyStopped.current = true;
    clearRetry();
    socketRef.current?.close();
    socketRef.current = null;
    setSocket(null);
    setSocketStatus("disconnected");
  }, [clearRetry]);

  const sendMessage = useCallback((message: WSMessage) => {
    const current = socketRef.current;
    if (!current || current.readyState !== WebSocket.OPEN) return;
    current.send(encodeMessage(message));
  }, []);

  useEffect(() => stop, [stop]);

  return { socketStatus, socket, sendMessage, start: connect, stop };
}

