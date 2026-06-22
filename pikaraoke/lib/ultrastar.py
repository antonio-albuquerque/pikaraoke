"""Parse UltraStar .txt song files into timed notes for pitch visualization.

UltraStar format (the relevant subset):
    #BPM:300            beats per minute (UltraStar counts 4 beats per quarter note)
    #GAP:1200           milliseconds before the first beat
    : 0 4 12 Some       note line: <type> <startBeat> <lengthBeats> <pitch> <text>
    * 4 2 14 word       golden note (scored higher in UltraStar; same shape here)
    F 6 2 12 free       freestyle note
    - 8                 line/phrase break
    E                   end of song

Pitch is a MIDI note relative to C4=0, so the absolute MIDI number is pitch + 60.
"""

import logging
import os

# UltraStar treats #BPM as quad-beats: one beat = 60000 / (BPM * 4) ms.
_BEAT_MS_NUMERATOR = 60000 / 4
# UltraStar pitch 0 == middle C (MIDI 60).
_MIDI_OFFSET = 60
_NOTE_TYPES = (":", "*", "F")


def ultrastar_path_for(song_file: str) -> str | None:
    """Return the sibling .txt note file for a song path, or None if absent."""
    if not song_file:
        return None
    candidate = os.path.splitext(song_file)[0] + ".txt"
    return candidate if os.path.isfile(candidate) else None


def _read_text(path: str) -> str:
    """Read a note file, tolerating the latin-1 encoding UltraStar files often use."""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, encoding="latin-1") as f:
            return f.read()


def parse_ultrastar(path: str) -> list[dict]:
    """Parse an UltraStar .txt into timed notes.

    Returns a list of {"t0": seconds, "t1": seconds, "midi": int} ordered by start
    time. Returns [] when the file has no usable BPM or no notes, rather than raising,
    so a malformed chart simply falls back to live detection upstream.
    """
    try:
        text = _read_text(path)
    except OSError as e:
        logging.warning(f"Could not read UltraStar file {path}: {e}")
        return []

    bpm: float | None = None
    gap_ms = 0.0
    for line in text.splitlines():
        if not line.startswith("#"):
            break  # headers precede notes; stop at the first non-header
        key, _, value = line[1:].partition(":")
        key = key.strip().upper()
        value = value.strip().replace(",", ".")
        if key == "BPM":
            try:
                bpm = float(value)
            except ValueError:
                pass
        elif key == "GAP":
            try:
                gap_ms = float(value)
            except ValueError:
                pass

    if not bpm or bpm <= 0:
        logging.warning(f"UltraStar file {path} has no valid #BPM; skipping notes")
        return []

    beat_ms = _BEAT_MS_NUMERATOR / bpm

    def beat_to_seconds(beat: float) -> float:
        return (gap_ms + beat * beat_ms) / 1000.0

    notes: list[dict] = []
    for line in text.splitlines():
        if not line or line[0] not in _NOTE_TYPES:
            continue
        parts = line.split(maxsplit=4)
        # <type> <startBeat> <length> <pitch> [text]
        if len(parts) < 4:
            continue
        try:
            start_beat = int(parts[1])
            length = int(parts[2])
            pitch = int(parts[3])
        except ValueError:
            continue
        if length <= 0:
            continue
        notes.append(
            {
                "t0": beat_to_seconds(start_beat),
                "t1": beat_to_seconds(start_beat + length),
                "midi": pitch + _MIDI_OFFSET,
            }
        )

    notes.sort(key=lambda n: n["t0"])
    return notes


def notes_to_ultrastar(
    notes: list[dict],
    *,
    title: str = "Unknown",
    artist: str = "Unknown",
    bpm: float = 400.0,
    gap_ms: float = 0.0,
    audio_filename: str = "",
) -> str:
    """Serialize timed notes into UltraStar .txt text.

    Inverts parse_ultrastar using the same conventions: beat length = 15000/BPM ms,
    UltraStar pitch = MIDI - 60. BPM is only a quantization grid (a higher BPM gives
    finer timing), not the song's musical tempo. Notes get a placeholder syllable, as
    there is no lyrics source. Input notes are {"t0", "t1", "midi"} in seconds.
    """
    if bpm <= 0:
        raise ValueError("bpm must be positive")
    beat_ms = _BEAT_MS_NUMERATOR / bpm

    def seconds_to_beat(seconds: float) -> int:
        return round((seconds * 1000 - gap_ms) / beat_ms)

    lines = [
        f"#TITLE:{title}",
        f"#ARTIST:{artist}",
    ]
    if audio_filename:
        lines.append(f"#MP3:{audio_filename}")
    lines += [f"#BPM:{bpm:g}", f"#GAP:{gap_ms:g}"]

    for note in sorted(notes, key=lambda n: n["t0"]):
        start_beat = seconds_to_beat(note["t0"])
        length = max(1, seconds_to_beat(note["t1"]) - start_beat)
        pitch = round(note["midi"]) - _MIDI_OFFSET
        lines.append(f": {start_beat} {length} {pitch} ~")

    lines.append("E")
    return "\n".join(lines) + "\n"
