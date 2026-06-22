"""Convert an audio file into an UltraStar-like .txt note chart.

Detects the dominant melody with pYIN and writes a chart the PiKaraoke pitch highway
can display. Best results come from a melody-forward source (a vocal stem or solo
melody); dense mixes yield whatever pitch dominates.

Requires the optional `notes` dependency group:
    uv run --group notes python build_scripts/audio_to_ultrastar.py song.mp3
    uv run --group notes python build_scripts/audio_to_ultrastar.py song.mp4 \\
        -o charts/song.txt --title "My Song" --artist "Someone" --bpm 400

By default the .txt is written next to the input (same name, .txt extension), which is
exactly where PiKaraoke looks for a song's chart.
"""

import argparse
import logging

from pikaraoke.lib.audio_to_ultrastar import DEFAULT_FMAX, DEFAULT_FMIN, convert


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert audio to an UltraStar-like .txt chart")
    parser.add_argument("audio", help="Path to the audio or video file to analyze")
    parser.add_argument(
        "-o", "--output", help="Output .txt path (default: input path with .txt extension)"
    )
    parser.add_argument("--title", help="Song title (default: input filename)")
    parser.add_argument("--artist", help="Song artist (default: Unknown)")
    parser.add_argument(
        "--bpm",
        type=float,
        default=400.0,
        help="Quantization grid in beats/min, not the musical tempo; higher = finer timing "
        "(default: 400)",
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=DEFAULT_FMIN,
        help=f"Min pitch in Hz (default: {DEFAULT_FMIN})",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=DEFAULT_FMAX,
        help=f"Max pitch in Hz (default: {DEFAULT_FMAX})",
    )
    parser.add_argument(
        "--min-note-ms",
        type=float,
        default=100.0,
        help="Drop notes shorter than this many ms (default: 100)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum pYIN voiced probability to count a frame (default: 0.5)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_path = convert(
        args.audio,
        args.output,
        title=args.title,
        artist=args.artist,
        bpm=args.bpm,
        fmin=args.fmin,
        fmax=args.fmax,
        min_note_ms=args.min_note_ms,
        min_confidence=args.min_confidence,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
