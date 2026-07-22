import { createContext, useContext } from "react";
import type { MutableRefObject } from "react";


type MediaContextType = {
  audioContext: MutableRefObject<AudioContext>;
  worklet: MutableRefObject<AudioWorkletNode>;
  micDuration: MutableRefObject<number>;
  actualAudioPlayed: MutableRefObject<number>;
};


export const MediaContext = createContext<MediaContextType | null>(null);


export function useMediaContext(): MediaContextType {
  const context = useContext(MediaContext);
  if (!context) {
    throw new Error("useMediaContext must be used within a MediaContext provider");
  }
  return context;
}

