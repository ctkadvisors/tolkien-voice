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
  constraints: MediaStreamConstraints;
  onDataChunk?: (chunk: Uint8Array) => void;
  onRecordingStart?: () => void;
  onRecordingStop?: () => void;
};

type OpusRecorder = {
  encodedSamplePosition: number;
  ondataavailable: (data: Uint8Array) => void;
  onstart: () => void;
  onstop: () => void;
  start: () => void;
  stop: () => void;
};


export function useUserAudio({
  constraints,
  onDataChunk,
  onRecordingStart = () => undefined,
  onRecordingStop = () => undefined,
}: Args) {
  const { audioContext, micDuration } = useMediaContext();
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState(UserMediaStatuses.IDLE);
  const recorder = useRef<OpusRecorder | null>(null);
  const mediaStream = useRef<MediaStream | null>(null);

  const startRecordingUser = useCallback(async () => {
    setStatus(UserMediaStatuses.WAITING_FOR_PERMISSION);
    try {
      mediaStream.current = await navigator.mediaDevices.getUserMedia(constraints);
      const analyser = audioContext.current.createAnalyser();
      const source = audioContext.current.createMediaStreamSource(mediaStream.current);
      source.connect(analyser);
      const nextRecorder = new Recorder({
        mediaTrackConstraints: constraints,
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
        source.disconnect();
        mediaStream.current?.getTracks().forEach((track) => track.stop());
        mediaStream.current = null;
        recorder.current = null;
        setStatus(UserMediaStatuses.STOPPED);
        onRecordingStop();
      };
      nextRecorder.start();
      return { analyser, mediaStream: mediaStream.current, source };
    } catch (cause) {
      setError(cause instanceof Error ? cause.name : "MicrophoneError");
      setStatus(UserMediaStatuses.ERROR);
      return null;
    }
  }, [audioContext, constraints, micDuration, onDataChunk, onRecordingStart, onRecordingStop]);

  const stopRecording = useCallback(() => {
    setStatus(UserMediaStatuses.STOPPING);
    if (recorder.current) {
      recorder.current.stop();
    } else {
      mediaStream.current?.getTracks().forEach((track) => track.stop());
      mediaStream.current = null;
      setStatus(UserMediaStatuses.STOPPED);
    }
  }, []);

  return { status, error, startRecordingUser, stopRecording };
}
