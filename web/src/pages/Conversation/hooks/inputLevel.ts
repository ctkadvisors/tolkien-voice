import { useEffect, useState } from "react";


export type InputLevel = { rms: number; peak: number };


export function measureInputLevel(samples: Float32Array): InputLevel {
  if (samples.length === 0) return { rms: 0, peak: 0 };
  let peak = 0;
  let squares = 0;
  for (const rawSample of samples) {
    const sample = Number.isFinite(rawSample) ? Math.max(-1, Math.min(1, rawSample)) : 0;
    peak = Math.max(peak, Math.abs(sample));
    squares += sample * sample;
  }
  return { rms: Math.sqrt(squares / samples.length), peak };
}


export function useInputLevel(analyser: AnalyserNode | null): InputLevel {
  const [level, setLevel] = useState<InputLevel>({ rms: 0, peak: 0 });

  useEffect(() => {
    if (analyser === null) {
      setLevel({ rms: 0, peak: 0 });
      return;
    }
    analyser.fftSize = 512;
    const samples = new Float32Array(analyser.fftSize);
    const interval = window.setInterval(() => {
      analyser.getFloatTimeDomainData(samples);
      setLevel(measureInputLevel(samples));
    }, 100);
    return () => window.clearInterval(interval);
  }, [analyser]);

  return level;
}
