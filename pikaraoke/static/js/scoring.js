/*
 * Microphone-based karaoke scoring (voice-quality, no reference track).
 *
 * Ported from the frank_karaoke Flutter app's voice-only scoring path
 * (lib/features/audio/pitch_detector.dart and
 *  lib/features/scoring/scoring_session.dart). Runs entirely in the browser:
 * captures the singer's mic, bandpass-filters to the vocal range, detects pitch
 * with YIN, and rewards clean, confident, musical singing. No per-song data and
 * no audio leaves the device.
 *
 * Requires a secure context (https or localhost) for getUserMedia.
 */

const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

function hzToMidi(hz) {
  return 69 + 12 * Math.log2(hz / 440);
}

function midiToNoteName(midi) {
  const n = Math.round(midi);
  return `${NOTE_NAMES[((n % 12) + 12) % 12]}${Math.floor(n / 12) - 1}`;
}

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

/**
 * YIN fundamental-frequency estimator with a confidence output.
 * Threshold 0.70 is relaxed (vs the textbook 0.10-0.15) because the mic also
 * picks up the backing track, so the vocal CMNDF minimum is shallower.
 */
class PitchDetector {
  constructor(sampleRate, threshold = 0.7) {
    this.sampleRate = sampleRate;
    this.threshold = threshold;
    this._diff = null;
    this._cmndf = null;
  }

  /** Returns {pitchHz, confidence}; pitchHz is 0 when no pitch is found. */
  detect(samples) {
    const halfLen = samples.length >> 1;
    if (halfLen < 2) return { pitchHz: 0, confidence: 0 };

    if (!this._diff || this._diff.length !== halfLen) {
      this._diff = new Float64Array(halfLen);
      this._cmndf = new Float64Array(halfLen);
    }
    const diff = this._diff;
    const cmndf = this._cmndf;

    // Difference function
    for (let tau = 1; tau < halfLen; tau++) {
      let sum = 0;
      for (let i = 0; i < halfLen; i++) {
        const delta = samples[i] - samples[i + tau];
        sum += delta * delta;
      }
      diff[tau] = sum;
    }

    // Cumulative mean normalized difference
    cmndf[0] = 1;
    let runningSum = 0;
    for (let tau = 1; tau < halfLen; tau++) {
      runningSum += diff[tau];
      cmndf[tau] = runningSum > 0 ? (diff[tau] * tau) / runningSum : 1;
    }

    // Absolute threshold: first dip below threshold, refined to local min
    const minTau = Math.floor(this.sampleRate / 1000); // cap ~1000 Hz fundamental
    let tauEstimate = -1;
    for (let tau = minTau; tau < halfLen; tau++) {
      if (cmndf[tau] < this.threshold) {
        while (tau + 1 < halfLen && cmndf[tau + 1] < cmndf[tau]) tau++;
        tauEstimate = tau;
        break;
      }
    }
    if (tauEstimate === -1) return { pitchHz: 0, confidence: 0 };

    const cmndfMin = cmndf[tauEstimate];
    const betterTau = this._parabolicInterpolation(cmndf, tauEstimate, halfLen);
    if (betterTau <= 0) return { pitchHz: 0, confidence: 0 };

    return {
      pitchHz: this.sampleRate / betterTau,
      confidence: clamp01(1 - cmndfMin / this.threshold),
    };
  }

  _parabolicInterpolation(cmndf, tau, halfLen) {
    if (tau <= 0 || tau >= halfLen - 1) return tau;
    const s0 = cmndf[tau - 1];
    const s1 = cmndf[tau];
    const s2 = cmndf[tau + 1];
    const denominator = 2 * s1 - s2 - s0;
    if (Math.abs(denominator) < 1e-10) return tau;
    return tau + (s2 - s0) / (2 * denominator);
  }
}

/**
 * Captures the microphone and produces a live 0-100 score plus a cumulative
 * final score. Bandpass + voice gate + YIN + voice-quality metric, smoothed
 * with an EMA for the live readout.
 */
class ScoringSession {
  constructor(options = {}) {
    this.onUpdate = options.onUpdate || (() => {});

    // Voice-quality tuning (mirrors frank_karaoke defaults).
    this._pitchTolerance = 2.5; // semitones; snap window for cleanliness
    this._noiseGate = 0.008; // RMS below this = silence
    this._singingThreshold = 0.02; // fallback voice floor before baseline exists
    this._warmupMs = 5000; // ignore startup clicks/applause
    this._emaAlpha = 0.15; // ~1s live-score response
    this._frameMs = 46; // ~2048 samples at 44.1 kHz

    this._reset();

    this._audioContext = null;
    this._stream = null;
    this._analyser = null;
    this._timer = null;
    this._isActive = false;
    this._isPaused = false;
  }

  _reset() {
    this._emaScore = 0;
    this._emaInitialized = false;
    this._scoreSum = 0;
    this._voicedFrames = 0;
    this._processedFrames = 0;
    this._warmupDone = false;
    this._silentFrames = 0;
    this._recentPitches = [];
    this._prevSingerPitch = 0;
    this._rmsHistory = [];
    this._baselineRms = 0;
    this._startTime = 0;
  }

  get live() {
    return this._emaInitialized ? clamp(Math.round(this._emaScore * 100), 0, 100) : 0;
  }

  get overall() {
    if (this._voicedFrames === 0) return 0;
    return clamp(Math.round((this._scoreSum / this._voicedFrames) * 100), 0, 100);
  }

  /**
   * Request the mic and begin scoring. Resolves false (without throwing) when
   * the mic is unavailable/denied or the context is insecure, so callers can
   * fall back gracefully.
   */
  async start() {
    if (this._isActive) return true;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      console.warn("Scoring: getUserMedia unavailable (insecure context?)");
      return false;
    }
    try {
      this._stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          // Disable processing that destroys the pitch signal (matches Frank).
          autoGainControl: false,
          echoCancellation: false,
          noiseSuppression: false,
        },
      });
    } catch (e) {
      console.warn("Scoring: mic access denied/unavailable:", e.name);
      return false;
    }

    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    this._audioContext = new AudioCtx();
    if (this._audioContext.state === "suspended") {
      await this._audioContext.resume();
    }
    this._detector = new PitchDetector(this._audioContext.sampleRate);

    const source = this._audioContext.createMediaStreamSource(this._stream);

    // Bandpass 200-3500 Hz via cascaded biquads (Butterworth Q).
    const highpass = this._audioContext.createBiquadFilter();
    highpass.type = "highpass";
    highpass.frequency.value = 200;
    highpass.Q.value = 0.707;

    const lowpass = this._audioContext.createBiquadFilter();
    lowpass.type = "lowpass";
    lowpass.frequency.value = 3500;
    lowpass.Q.value = 0.707;

    this._analyser = this._audioContext.createAnalyser();
    this._analyser.fftSize = 2048;
    this._frameBuf = new Float32Array(this._analyser.fftSize);

    source.connect(highpass);
    highpass.connect(lowpass);
    lowpass.connect(this._analyser);
    // Not connected to destination: we analyze, never play the mic back.

    this._reset();
    this._isActive = true;
    this._isPaused = false;
    this._startTime = performance.now();
    this._timer = setInterval(() => this._processFrame(), this._frameMs);
    return true;
  }

  pause() {
    this._isPaused = true;
  }

  resume() {
    this._isPaused = false;
  }

  /** Stop capture and release the mic. Returns the final 0-100 score. */
  stop() {
    const finalScore = this.overall;
    this._isActive = false;
    if (this._timer) {
      clearInterval(this._timer);
      this._timer = null;
    }
    if (this._stream) {
      this._stream.getTracks().forEach((t) => t.stop());
      this._stream = null;
    }
    if (this._audioContext) {
      this._audioContext.close().catch(() => {});
      this._audioContext = null;
    }
    this._analyser = null;
    return finalScore;
  }

  _processFrame() {
    if (!this._isActive || this._isPaused || !this._analyser) return;

    if (!this._warmupDone && performance.now() - this._startTime >= this._warmupMs) {
      this._warmupDone = true;
    }

    this._analyser.getFloatTimeDomainData(this._frameBuf);
    const samples = this._frameBuf;

    // RMS energy of the (bandpassed) frame.
    let sumSq = 0;
    for (let i = 0; i < samples.length; i++) sumSq += samples[i] * samples[i];
    const rms = Math.sqrt(sumSq / samples.length);
    this._processedFrames++;

    // During warmup, learn the ambient/speaker floor but don't score.
    if (!this._warmupDone) {
      this._rmsHistory.push(rms);
      return;
    }

    // Adaptive baseline = 25th percentile of recent RMS (the "speaker bleed"
    // floor); the singer's voice rides above it.
    this._rmsHistory.push(rms);
    if (this._rmsHistory.length > 100) this._rmsHistory.shift();
    if (this._rmsHistory.length >= 20 && this._processedFrames % 25 === 0) {
      const sorted = [...this._rmsHistory].sort((a, b) => a - b);
      this._baselineRms = sorted[Math.floor(sorted.length / 4)];
    }

    // Noise gate: nobody singing.
    if (rms < this._noiseGate) {
      this._silentFrames++;
      if (this._silentFrames > 12) {
        this._recentPitches = [];
        this._prevSingerPitch = 0;
      }
      this._emit(0, "--", 0, false);
      return;
    }
    this._silentFrames = 0;

    // Voice detection: must be clearly above the speaker-bleed baseline.
    const isVoice =
      this._baselineRms > 0.001 ? rms > this._baselineRms * 1.5 : rms > this._singingThreshold;
    if (!isVoice) {
      this._emit(0, "--", 0, false);
      return;
    }

    const { pitchHz, confidence } = this._detector.detect(samples);
    if (pitchHz < 60 || confidence < 0.3) {
      this._emit(0, "--", confidence, false);
      return;
    }

    const singerMidi = hzToMidi(pitchHz);
    this._recentPitches.push(singerMidi);
    if (this._recentPitches.length > 15) this._recentPitches.shift();

    const frameScore = this._scoreVoiceOnly(singerMidi, confidence);
    this._pushScore(frameScore);
    this._prevSingerPitch = pitchHz;

    this._emit(pitchHz, midiToNoteName(singerMidi), confidence, true);
  }

  /**
   * Combined voice-quality score for one frame (0..1):
   *   confidence 40% + pitch cleanliness 30% + musicality 30%.
   */
  _scoreVoiceOnly(singerMidi, confidence) {
    // 1. Confidence: clear tonal singing scores high, speech/noise low.
    const confScore = clamp01((confidence - 0.3) / 0.6);

    // 2. Cleanliness: closeness to the nearest semitone, gated by confidence.
    const deviationCents = Math.abs(singerMidi - Math.round(singerMidi)) * 100;
    const snapTolerance = (this._pitchTolerance * 100) / 3;
    const cleanScore = clamp01(1 - deviationCents / snapTolerance) * confScore;

    // 3. Musicality: pitch range + interval quality over recent history.
    let musicalScore = 0.3;
    if (this._recentPitches.length >= 5) {
      const maxP = Math.max(...this._recentPitches);
      const minP = Math.min(...this._recentPitches);
      const range = maxP - minP;

      let rangeScore;
      if (range < 0.5) {
        rangeScore = 0; // monotone
      } else if (range <= 6) {
        rangeScore = clamp01(range / 6); // sweet spot
      } else {
        rangeScore = Math.max(0.3, 1 - (range - 6) / 10); // overly wild
      }

      let intervalScore = 0.5;
      if (this._prevSingerPitch > 0) {
        const interval = Math.abs(singerMidi - hzToMidi(this._prevSingerPitch));
        if (interval < 0.3) intervalScore = 0.6; // holding a note
        else if (interval <= 5) intervalScore = 1.0; // musical step/third
        else if (interval <= 8) intervalScore = 0.4; // wide jump
        else intervalScore = 0.1; // wild
      }

      musicalScore = rangeScore * 0.5 + intervalScore * 0.5;
    }

    return clamp01(confScore * 0.4 + cleanScore * 0.3 + musicalScore * 0.3);
  }

  _pushScore(score) {
    if (!this._emaInitialized) {
      this._emaScore = score;
      this._emaInitialized = true;
    } else {
      this._emaScore = this._emaAlpha * score + (1 - this._emaAlpha) * this._emaScore;
    }
    this._voicedFrames++;
    this._scoreSum += score;
  }

  _emit(pitchHz, note, confidence, voiced) {
    // Cents from the nearest semitone (-50..+50), for the tuner needle.
    let centsOff = 0;
    if (voiced && pitchHz > 0) {
      const midi = hzToMidi(pitchHz);
      centsOff = (midi - Math.round(midi)) * 100;
    }
    this.onUpdate({
      live: this.live,
      overall: this.overall,
      pitchHz,
      note,
      confidence,
      voiced,
      centsOff,
    });
  }
}

function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

// Exposed globally for the non-module splash scripts, and via module exports
// for the standalone Node test harness.
if (typeof window !== "undefined") {
  window.ScoringSession = ScoringSession;
  window.PitchDetector = PitchDetector;
}
if (typeof module !== "undefined" && module.exports) {
  module.exports = { ScoringSession, PitchDetector, hzToMidi, midiToNoteName };
}
