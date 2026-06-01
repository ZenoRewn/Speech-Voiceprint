// AudioWorklet processor — resamples whatever the input AudioContext is
// running at (typically 48 kHz on Mac/Chrome, 44.1 kHz on some setups) down
// to 16 kHz mono int16 and posts the bytes back to the main thread.
//
// Linear interpolation is fine for speech and avoids pulling in a real
// resampling library. The api accepts any chunk size, so we batch ~100 ms
// of output (1600 samples = 3200 bytes) per message to keep WS framing
// manageable.

const TARGET_SR = 16000;
const TARGET_CHUNK = 1600;  // 100 ms

class PcmDownsamplerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._inSr = sampleRate;            // provided by the AudioWorkletGlobalScope
    this._ratio = this._inSr / TARGET_SR;
    this._phase = 0;                     // virtual read position into input buffer
    this._inTail = new Float32Array(0);  // leftover samples not yet consumed
    this._out = new Int16Array(TARGET_CHUNK);
    this._outFill = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const ch0 = input[0];
    if (!ch0 || ch0.length === 0) return true;

    // Concatenate leftover with this render quantum.
    const merged = new Float32Array(this._inTail.length + ch0.length);
    merged.set(this._inTail);
    merged.set(ch0, this._inTail.length);

    // Walk the virtual read pointer; emit one sample each time it crosses.
    while (this._phase + 1 < merged.length) {
      const i0 = Math.floor(this._phase);
      const i1 = i0 + 1;
      const frac = this._phase - i0;
      const sample = merged[i0] * (1 - frac) + merged[i1] * frac;
      // Clamp + scale to int16
      let s = Math.max(-1, Math.min(1, sample));
      this._out[this._outFill++] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      if (this._outFill === TARGET_CHUNK) {
        // Send a copy — the buffer is reused next round.
        this.port.postMessage(this._out.buffer.slice(0));
        this._outFill = 0;
      }
      this._phase += this._ratio;
    }

    // Carry samples that weren't fully consumed yet (we keep the last `i0+1`
    // sample so the next round can interpolate from it).
    const consumedThrough = Math.floor(this._phase);
    if (consumedThrough > 0) {
      this._inTail = merged.slice(consumedThrough);
      this._phase -= consumedThrough;
    } else {
      this._inTail = merged;
    }
    return true;
  }
}

registerProcessor("pcm-downsampler", PcmDownsamplerProcessor);
