/*
 * Real-time pitch visualization: a scrolling, full-width "note highway" plus a
 * current-note tuner. Driven by ScoringSession.onUpdate (see scoring.js).
 *
 * Time runs left->right. A fixed "now" marker sits at NOW_FRAC across the width;
 * the singer's sung pitch trails to its left, and the expected-note reference
 * scrolls toward it from the right:
 *   - UltraStar mode: note bars from a parsed .txt chart (known ahead of time).
 *   - Live mode: the backing track's detected pitch (history only, no look-ahead).
 *
 * Pitch is plotted on a log-frequency axis (100-800 Hz) shared by the gridlines,
 * the reference, and the singer trail. The trail dot coloring is ported from
 * frank_karaoke (lib/features/overlay/webview_overlay.dart).
 */

const VIZ_MIN_HZ = 100;
const VIZ_MAX_HZ = 800;
const VIZ_LOG_MIN = Math.log(VIZ_MIN_HZ);
const VIZ_LOG_RANGE = Math.log(VIZ_MAX_HZ) - VIZ_LOG_MIN;

// Time window shown on screen (seconds) and where "now" sits across the width.
// PAST/WINDOW sets the "now" marker position (6.5/10 = 65% from the left).
const PAST_SECONDS = 6.5;
const FUTURE_SECONDS = 3.5;
const WINDOW_SECONDS = PAST_SECONDS + FUTURE_SECONDS;
const NOW_FRAC = PAST_SECONDS / WINDOW_SECONDS;

const VIZ_GRID = [
  [130.81, "C3"],
  [196.0, "G3"],
  [261.63, "C4"],
  [392.0, "G4"],
  [523.25, "C5"],
  [783.99, "G5"],
];

/** Map a frequency to 0..1 on the log axis. Returns 0 for non-positive input. */
function normalizePitch(hz) {
  if (!(hz > 0)) return 0;
  const n = (Math.log(hz) - VIZ_LOG_MIN) / VIZ_LOG_RANGE;
  return Math.max(0, Math.min(1, n));
}

function midiToFreq(midi) {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

class PitchViz {
  constructor(canvasEl, tunerNoteEl, tunerNeedleEl) {
    this._canvas = canvasEl;
    this._ctx = canvasEl ? canvasEl.getContext("2d") : null;
    this._tunerNote = tunerNoteEl;
    this._tunerNeedle = tunerNeedleEl;
    this._raf = null;
    this._onResize = () => this._resizeCanvas();
    this._clearState();
  }

  _clearState() {
    this._singer = []; // {t, p, q} sung pitch samples (absolute playback seconds)
    this._refNotes = []; // UltraStar note bars {t0, t1, midi}
    this._refTrail = []; // live-detected reference {t, p}
    this._mode = "none"; // "ultrastar" | "live" | "none"
    this._lastTime = 0; // last known playback time (s)
    this._lastPerf = 0; // performance.now() when _lastTime was set
  }

  /** Provide the expected-note timeline from a parsed UltraStar chart. */
  setReferenceNotes(notes) {
    this._refNotes = Array.isArray(notes) ? notes : [];
    this._mode = "ultrastar";
  }

  /** Switch to live backing-track detection (no look-ahead notes). */
  setLiveReference() {
    this._mode = "live";
  }

  /** Append a live-detected backing-track pitch sample at playback time t. */
  pushLiveReference(t, pitchHz) {
    if (this._mode !== "live") return;
    this._refTrail.push({ t, p: normalizePitch(pitchHz) });
  }

  /** Begin rendering. Clears prior song state and sizes the canvas. */
  start() {
    this._clearState();
    this._resizeCanvas();
    window.addEventListener("resize", this._onResize);
    if (this._tunerNote) this._tunerNote.textContent = "--";
    if (!this._raf) this._loop();
  }

  /** Stop rendering and clear the canvas. */
  stop() {
    if (this._raf) {
      cancelAnimationFrame(this._raf);
      this._raf = null;
    }
    window.removeEventListener("resize", this._onResize);
    if (this._ctx) this._ctx.clearRect(0, 0, this._canvas.width, this._canvas.height);
    this._clearState();
    if (this._tunerNeedle) {
      this._tunerNeedle.style.left = "50%";
      this._tunerNeedle.style.opacity = "0.25";
    }
  }

  /** Feed one scoring frame plus the current playback time (seconds). */
  update({ pitchHz, confidence, voiced, note, centsOff }, videoTime) {
    if (typeof videoTime === "number" && isFinite(videoTime)) {
      this._lastTime = videoTime;
      this._lastPerf = perfNow();
    }
    this._singer.push({
      t: this._lastTime,
      p: voiced ? normalizePitch(pitchHz) : 0,
      q: voiced ? confidence : 0,
    });
    this._trim();
    this._updateTuner(voiced, note, centsOff);
  }

  _trim() {
    const cutoff = this._lastTime - PAST_SECONDS - 1;
    while (this._singer.length && this._singer[0].t < cutoff) this._singer.shift();
    while (this._refTrail.length && this._refTrail[0].t < cutoff) this._refTrail.shift();
  }

  _updateTuner(voiced, note, centsOff) {
    if (this._tunerNote) this._tunerNote.textContent = voiced ? note : "--";
    if (this._tunerNeedle) {
      const pct = Math.max(0, Math.min(100, (centsOff || 0) + 50));
      this._tunerNeedle.style.left = `${pct}%`;
      this._tunerNeedle.style.opacity = voiced ? "1" : "0.25";
    }
  }

  _resizeCanvas() {
    if (!this._canvas) return;
    const w = this._canvas.clientWidth || window.innerWidth;
    const h = this._canvas.clientHeight || this._canvas.height;
    this._canvas.width = w;
    this._canvas.height = h;
  }

  // Estimate the current playback time between data frames for smooth scrolling.
  _estimateNow() {
    if (!this._lastPerf) return this._lastTime;
    return this._lastTime + (perfNow() - this._lastPerf) / 1000;
  }

  _loop() {
    this._render(this._estimateNow());
    this._raf = requestAnimationFrame(() => this._loop());
  }

  _xForTime(t, now) {
    return ((t - now + PAST_SECONDS) / WINDOW_SECONDS) * this._w;
  }

  _yForNorm(p) {
    const pad = this._pad;
    return this._h - p * (this._h - 2 * pad) - pad;
  }

  _render(now) {
    const ctx = this._ctx;
    if (!ctx) return;
    const w = (this._w = this._canvas.width);
    const h = (this._h = this._canvas.height);
    this._pad = 10;
    ctx.clearRect(0, 0, w, h);

    this._drawGrid(ctx, w, h);

    if (this._mode === "ultrastar") this._drawNoteBars(ctx, now);
    else if (this._mode === "live") this._drawLiveReference(ctx, now);

    this._drawNowMarker(ctx, w, h);
    this._drawSinger(ctx, now);
  }

  _drawGrid(ctx, w, h) {
    ctx.font = "10px system-ui, sans-serif";
    for (const [freq, label] of VIZ_GRID) {
      const ny = normalizePitch(freq);
      if (ny <= 0 || ny >= 1) continue;
      const y = this._yForNorm(ny);
      ctx.strokeStyle = "rgba(255,255,255,0.06)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(26, y);
      ctx.lineTo(w, y);
      ctx.stroke();
      ctx.fillStyle = "rgba(255,255,255,0.25)";
      ctx.fillText(label, 4, y + 3);
    }
  }

  _drawNowMarker(ctx, w, h) {
    const x = NOW_FRAC * w;
    ctx.strokeStyle = "rgba(255,255,255,0.45)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }

  _drawNoteBars(ctx, now) {
    const from = now - PAST_SECONDS;
    const to = now + FUTURE_SECONDS;
    const barH = Math.max(6, (this._h - 2 * this._pad) / 28);
    for (const n of this._refNotes) {
      if (n.t1 < from || n.t0 > to) continue;
      const x0 = this._xForTime(n.t0, now);
      const x1 = this._xForTime(n.t1, now);
      const y = this._yForNorm(normalizePitch(midiToFreq(n.midi)));
      const active = n.t0 <= now && now <= n.t1; // being sung right now
      ctx.fillStyle = active ? "rgba(0,210,255,0.95)" : "rgba(108,92,231,0.65)";
      this._roundRect(ctx, x0, y - barH / 2, Math.max(3, x1 - x0), barH, barH / 2);
      ctx.fill();
    }
  }

  _drawLiveReference(ctx, now) {
    ctx.strokeStyle = "rgba(108,92,231,0.7)";
    ctx.lineWidth = 2;
    let started = false;
    for (const pt of this._refTrail) {
      if (pt.p <= 0) {
        started = false;
        continue;
      }
      const x = this._xForTime(pt.t, now);
      const y = this._yForNorm(pt.p);
      if (!started) {
        ctx.beginPath();
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    }
    if (started) ctx.stroke();
  }

  _drawSinger(ctx, now) {
    let pvx = -1;
    let pvy = -1;
    for (const s of this._singer) {
      if (s.p <= 0) {
        pvx = -1;
        continue;
      }
      const x = this._xForTime(s.t, now);
      const y = this._yForNorm(s.p);
      const r = Math.round(255 * (1 - s.q));
      const g = Math.round(220 * s.q + 35);
      const col = `rgb(${r},${g},80)`;
      if (pvx >= 0 && Math.abs(pvy - y) < this._h * 0.4) {
        ctx.strokeStyle = col;
        ctx.lineWidth = 2.5;
        ctx.globalAlpha = 0.6;
        ctx.beginPath();
        ctx.moveTo(pvx, pvy);
        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
      ctx.shadowColor = col;
      ctx.shadowBlur = s.q > 0.5 ? 10 : 4;
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.arc(x, y, s.q > 0.5 ? 4 : 2.5, 0, 6.283);
      ctx.fill();
      ctx.shadowBlur = 0;
      pvx = x;
      pvy = y;
    }
  }

  _roundRect(ctx, x, y, w, h, r) {
    const rr = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.arcTo(x + w, y, x + w, y + h, rr);
    ctx.arcTo(x + w, y + h, x, y + h, rr);
    ctx.arcTo(x, y + h, x, y, rr);
    ctx.arcTo(x, y, x + w, y, rr);
    ctx.closePath();
  }
}

// performance.now() is unavailable in some headless test contexts; fall back to 0
// (only affects sub-frame interpolation, not the test-covered pure functions).
function perfNow() {
  return typeof performance !== "undefined" && performance.now ? performance.now() : 0;
}

if (typeof window !== "undefined") {
  window.PitchViz = PitchViz;
}
if (typeof module !== "undefined" && module.exports) {
  module.exports = { PitchViz, normalizePitch, midiToFreq };
}
