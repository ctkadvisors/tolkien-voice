export type PlayoutPolicy = {
  initialBufferSamples: number;
  partialBufferSamples: number;
  maxBufferSamples: number;
  totalMaximumSamples: number;
  partialBufferIncrement: number;
  maxPartialWithIncrements: number;
  maxBufferSamplesIncrement: number;
  maxMaxBufferWithIncrements: number;
};


export function createPlayoutPolicy(sampleRate: number): PlayoutPolicy {
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
    throw new RangeError("sample rate must be positive");
  }
  const samples = (milliseconds: number) => Math.round(milliseconds * sampleRate / 1000);
  const initialBufferSamples = samples(120);
  const partialBufferSamples = samples(20);
  const maxBufferSamples = samples(160);
  return {
    initialBufferSamples,
    partialBufferSamples,
    maxBufferSamples,
    totalMaximumSamples: initialBufferSamples + partialBufferSamples + maxBufferSamples,
    partialBufferIncrement: samples(10),
    maxPartialWithIncrements: samples(60),
    maxBufferSamplesIncrement: samples(20),
    maxMaxBufferWithIncrements: samples(160),
  };
}
