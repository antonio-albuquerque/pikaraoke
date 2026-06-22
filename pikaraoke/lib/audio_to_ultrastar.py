"""Convert an audio file into an UltraStar-like .txt note chart.

Pipeline: decode audio to mono PCM (ffmpeg) -> frame-wise pitch track (librosa pYIN)
-> segment frames into discrete notes -> serialize to UltraStar text via
pikaraoke.lib.ultrastar.notes_to_ultrastar.

This extracts a single dominant pitch line (monophonic), so results are best on a
melody-forward source such as a vocal stem; dense mixes yield whatever pitch dominates.

librosa and numpy are heavy and optional (the `notes` dependency group), so they are
imported lazily inside the functions that need them. segment_notes is pure Python and
needs neither, which keeps it unit-testable without the optional install.
"""

import logging
import math
import os

# Default pitch-tracking range: ~C2 to ~C6 covers typical sung melodies.
DEFAULT_FMIN = 65.0
DEFAULT_FMAX = 1047.0
DEFAULT_SR = 22050
DEFAULT_HOP = 256


def _hz_to_midi(hz: float) -> float:
    return 69 + 12 * math.log2(hz / 440.0)


def decode_pcm(audio_path: str, sr: int = DEFAULT_SR) -> tuple:
    """Decode any audio/video file to a mono float32 numpy array at `sr` via ffmpeg.

    Returns (samples, sr). ffmpeg handles mp4/webm/cdg/mp3 robustly, matching how the
    rest of PiKaraoke shells out to ffmpeg.
    """
    import ffmpeg  # lazy: part of the optional toolchain
    import numpy as np

    try:
        out, _ = (
            ffmpeg.input(audio_path)
            .output("pipe:", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        stderr = e.stderr.decode("utf-8", "ignore") if e.stderr else ""
        raise RuntimeError(f"ffmpeg failed to decode {audio_path}: {stderr}") from e

    return np.frombuffer(out, dtype=np.float32), sr


def track_pitch(
    samples,
    sr: int,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
    hop_length: int = DEFAULT_HOP,
) -> tuple:
    """Run pYIN over the samples. Returns (f0_hz, voiced_prob) frame arrays.

    f0_hz holds NaN on unvoiced frames (librosa's convention).
    """
    import librosa  # lazy: optional, heavy

    f0_hz, _voiced_flag, voiced_prob = librosa.pyin(
        samples, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length
    )
    return f0_hz, voiced_prob


def segment_notes(
    f0_hz,
    voiced_prob,
    hop_seconds: float,
    *,
    min_note_ms: float = 100.0,
    min_confidence: float = 0.5,
    max_gap_ms: float = 120.0,
) -> list[dict]:
    """Group consecutive frames into notes. Pure Python (no numpy/librosa).

    A note is held while the rounded semitone stays stable and frames are voiced and
    confident. Short unvoiced/low-confidence dips (up to max_gap_ms) are bridged; longer
    gaps or a pitch change close the note. Notes shorter than min_note_ms are dropped.

    Returns a list of {"t0", "t1", "midi"} in seconds, ordered by start time.
    """
    notes: list[dict] = []
    max_gap_frames = max_gap_ms / 1000.0 / hop_seconds if hop_seconds > 0 else 0

    start_i = None  # first frame index of the open note
    last_voiced_i = None  # last frame that confirmed the open note's pitch
    cur_midi = None
    gap = 0  # consecutive non-matching frames since last_voiced_i

    def close():
        nonlocal start_i, last_voiced_i, cur_midi, gap
        if start_i is not None and last_voiced_i is not None:
            t0 = start_i * hop_seconds
            t1 = (last_voiced_i + 1) * hop_seconds
            if (t1 - t0) * 1000 >= min_note_ms:
                notes.append({"t0": t0, "t1": t1, "midi": cur_midi})
        start_i = last_voiced_i = cur_midi = None
        gap = 0

    for i in range(len(f0_hz)):
        hz = f0_hz[i]
        prob = voiced_prob[i] if i < len(voiced_prob) else 0
        voiced = prob >= min_confidence and hz is not None and hz > 0 and math.isfinite(hz)
        midi = round(_hz_to_midi(hz)) if voiced else None

        if not voiced:
            if start_i is not None:
                gap += 1
                if gap > max_gap_frames:
                    close()
            continue

        if start_i is None:
            start_i, last_voiced_i, cur_midi, gap = i, i, midi, 0
        elif midi == cur_midi:
            last_voiced_i, gap = i, 0
        else:
            close()
            start_i, last_voiced_i, cur_midi, gap = i, i, midi, 0

    close()
    return notes


def convert(
    audio_path: str,
    out_path: str | None = None,
    *,
    title: str | None = None,
    artist: str | None = None,
    bpm: float = 400.0,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
    hop_length: int = DEFAULT_HOP,
    min_note_ms: float = 100.0,
    min_confidence: float = 0.5,
    max_gap_ms: float = 120.0,
) -> str:
    """Convert an audio file to an UltraStar .txt and write it.

    The default output path is the input path with a .txt extension — exactly what
    pikaraoke.lib.ultrastar.ultrastar_path_for looks for, so the pitch highway picks it
    up automatically. Returns the output path.
    """
    from pikaraoke.lib.ultrastar import notes_to_ultrastar

    samples, sr = decode_pcm(audio_path)
    f0_hz, voiced_prob = track_pitch(samples, sr, fmin=fmin, fmax=fmax, hop_length=hop_length)
    notes = segment_notes(
        f0_hz,
        voiced_prob,
        hop_length / sr,
        min_note_ms=min_note_ms,
        min_confidence=min_confidence,
        max_gap_ms=max_gap_ms,
    )

    base = os.path.splitext(os.path.basename(audio_path))[0]
    gap_ms = notes[0]["t0"] * 1000 if notes else 0.0
    text = notes_to_ultrastar(
        notes,
        title=title or base,
        artist=artist or "Unknown",
        bpm=bpm,
        gap_ms=gap_ms,
        audio_filename=os.path.basename(audio_path),
    )

    if out_path is None:
        out_path = os.path.splitext(audio_path)[0] + ".txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    logging.info(f"Wrote {len(notes)} notes to {out_path}")
    return out_path
