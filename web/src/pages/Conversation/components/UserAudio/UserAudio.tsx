import { FC, useCallback, useEffect, useRef, useState } from "react";
import { useSocketContext } from "../../SocketContext";
import { useUserAudio } from "../../hooks/useUserAudio";
import { ClientVisualizer } from "../AudioVisualizer/ClientVisualizer";
import { type ThemeType } from "../../hooks/useSystemTheme";
import { useInputLevel } from "../../hooks/inputLevel";

type UserAudioProps = {
  theme: ThemeType;
  mediaStream: MediaStream;
  inputDevices: MediaDeviceInfo[];
  selectedInputDeviceId: string;
  onInputDeviceChange: (deviceId: string) => void;
  inputError: string | null;
};
export const UserAudio: FC<UserAudioProps> = ({
  theme,
  mediaStream,
  inputDevices,
  selectedInputDeviceId,
  onInputDeviceChange,
  inputError,
}) => {
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
  const { sendMessage, socketStatus } = useSocketContext();
  const containerRef = useRef<HTMLDivElement>(null);
  const onRecordingStart = useCallback(() => {
    console.log("Recording started");
  }, []);

  const onRecordingStop = useCallback(() => {
    console.log("Recording stopped");
  }, []);

  const onRecordingChunk = useCallback(
    (chunk: Uint8Array) => {
      if (socketStatus !== "connected") {
        return;
      }
      sendMessage({
        type: "audio",
        data: chunk,
      });
    },
    [sendMessage, socketStatus],
  );

  const { status, error, startRecordingUser, stopRecording } = useUserAudio({
    mediaStream,
    onDataChunk: onRecordingChunk,
    onRecordingStart,
    onRecordingStop,
  });
  const inputLevel = useInputLevel(analyser);

  useEffect(() => {
    let res: Awaited<ReturnType<typeof startRecordingUser>>;
    if (socketStatus === "connected") {
      startRecordingUser().then(result => {
        if (result) {
          res = result;
          setAnalyser(result.analyser);
        }
      });
    }
    return () => {
      console.log("Stop recording called from somewhere else.");
      stopRecording();
      res?.source?.disconnect();
    };
  }, [startRecordingUser, stopRecording, socketStatus]);

  return (
    <div className="listener-console">
      <div className="user-audio h-5/6 aspect-square" ref={containerRef}>
        <ClientVisualizer theme={theme} analyser={analyser} parent={containerRef}/>
      </div>
      <label className="input-device-field">
        <span>Microphone input</span>
        <select
          aria-label="Microphone input"
          value={selectedInputDeviceId}
          onChange={(event) => onInputDeviceChange(event.target.value)}
        >
          {inputDevices.map((device, index) => (
            <option key={device.deviceId} value={device.deviceId}>
              {device.label || `Microphone ${index + 1}`}
            </option>
          ))}
        </select>
      </label>
      <div className="input-meter">
        <div>
          <span>Input signal</span>
          <output aria-label="Microphone peak level">
            {Math.round(inputLevel.peak * 100)}%
          </output>
        </div>
        <meter
          aria-label="Live microphone level"
          min={0}
          max={0.25}
          low={0.015}
          high={0.18}
          optimum={0.08}
          value={Math.min(0.25, inputLevel.rms)}
        />
      </div>
      <p className={`input-state is-${status.toLowerCase()}`} role="status">
        {error || inputError || (status === "RECORDING" ? "Microphone is live" : "Opening microphone")}
      </p>
    </div>
  );
};
