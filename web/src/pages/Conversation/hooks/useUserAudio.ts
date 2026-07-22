import { useCallback, useRef, useState } from "react";
import Recorder from "opus-recorder";
import encoderPath from "opus-recorder/dist/encoderWorker.min.js?url";

import { useMediaContext } from "../MediaContext";


export enum UserMediaStatuses {
  IDLE = "IDLE",
  WAITING_FOR_PERMISSION = "WAITING_FOR_PERMISSION",
  ERROR = "ERROR",
  RECORDING = "RECORDING",
  STOPPED = "STOPPED",
  STOPPING = "STOPPING",
}


type Args = {
  mediaStream: MediaStream;
  onDataChunk?: (chunk: Uint8Array) => void;
  onRecordingStart?: () => void;
  onRecordingStop?: () => void;
};

type OpusRecorder = {
  encodedSamplePosition: number;
  ondataavailable: (data: Uint8Array) => void;
  onstart: () => void;
  onstop: () => void;
  start: () => Promise<void>;
  stop: () => Promise<void>;
};


export function useUserAudio({
  mediaStream,
  onDataChunk,
  onRecordingStart = () => undefined,
  onRecordingStop = () => undefined,
}: Args) {
  const { audioContext, micDuration } = useMediaContext();
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState(UserMediaStatuses.IDLE);
  const recorder = useRef<OpusRecorder | null>(null);
  const activeSource = useRef<MediaStreamAudioSourceNode | null>(null);

  const startRecordingUser = useCallback(async () => {
    setStatus(UserMediaStatuses.WAITING_FOR_PERMISSION);
    setError(null);
    let source: MediaStreamAudioSourceNode | null = null;
    try {
      const analyser = audioContext.current.createAnalyser();
      const createdSource = audioContext.current.createMediaStreamSource(mediaStream);
      source = createdSource;
      activeSource.current = createdSource;
      createdSource.connect(analyser);
      const nextRecorder = new Recorder({
        sourceNode: createdSource,
        encoderPath,
        bufferLength: Math.round((960 * audioContext.current.sampleRate) / 24000),
        encoderFrameSize: 20,
        encoderSampleRate: 24000,
        maxFramesPerPage: 2,
        numberOfChannels: 1,
        recordingGain: 1,
        resampleQuality: 3,
        encoderComplexity: 0,
        encoderApplication: 2049,
        streamPages: true,
      }) as OpusRecorder;
      recorder.current = nextRecorder;
      nextRecorder.ondataavailable = (data: Uint8Array) => {
        micDuration.current = nextRecorder.encodedSamplePosition / 48000;
        onDataChunk?.(data);
      };
      nextRecorder.onstart = () => {
        setStatus(UserMediaStatuses.RECORDING);
        onRecordingStart();
      };
      nextRecorder.onstop = () => {
        createdSource.disconnect();
        if (recorder.current !== nextRecorder) return;
        activeSource.current = null;
        recorder.current = null;
        setStatus(UserMediaStatuses.STOPPED);
        onRecordingStop();
      };
      await nextRecorder.start();
      return { analyser, mediaStream, source: createdSource };
    } catch (cause) {
      source?.disconnect();
      activeSource.current = null;
      recorder.current = null;
      setError(cause instanceof Error ? cause.message || cause.name : "MicrophoneError");
      setStatus(UserMediaStatuses.ERROR);
      return null;
    }
  }, [audioContext, mediaStream, micDuration, onDataChunk, onRecordingStart, onRecordingStop]);

  const stopRecording = useCallback(() => {
    setStatus(UserMediaStatuses.STOPPING);
    if (recorder.current) {
      void recorder.current.stop().catch((cause: unknown) => {
        setError(cause instanceof Error ? cause.message : "RecorderStopError");
        setStatus(UserMediaStatuses.ERROR);
      });
    } else {
      activeSource.current?.disconnect();
      activeSource.current = null;
      setStatus(UserMediaStatuses.STOPPED);
    }
  }, []);

  return { status, error, startRecordingUser, stopRecording };
}
