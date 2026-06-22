"""Remove the lead vocal from an audio/video file to make a karaoke backing track.

Uses Demucs to split the input into an instrumental and an isolated vocal stem, written
next to the input (the original is never modified). With --chart, the clean vocal stem is
fed to the UltraStar converter to also emit a note chart for the pitch highway. With --video,
the original video is kept and only its audio is swapped for the instrumental — a karaoke
video with on-screen lyrics and the lead vocal removed.

Requires the optional `separation` dependency group (Demucs + torch):
    uv run --group separation python build_scripts/remove_vocals.py song.mp3
    uv run --group separation python build_scripts/remove_vocals.py video.mp4 --video
    uv run --group separation --group notes python build_scripts/remove_vocals.py song.mp3 --chart

First run downloads the Demucs model (~hundreds of MB); CPU works but takes minutes per song.
By default stems are written as .wav (not a song-library extension). --mp3 emits .mp3, which
WOULD appear as songs if written into the PiKaraoke song directory.
"""

import argparse
import logging

from pikaraoke.lib.vocal_remover import DEFAULT_MODEL, remove_vocals


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove the lead vocal to make a backing track")
    parser.add_argument("audio", help="Path to the audio or video file")
    parser.add_argument(
        "-o", "--output-dir", help="Where to write the stems (default: input's directory)"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"Demucs model (default: {DEFAULT_MODEL})"
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Force a device (default: auto)")
    parser.add_argument("--mp3", action="store_true", help="Write .mp3 stems instead of .wav")
    parser.add_argument(
        "--video",
        action="store_true",
        help="Keep the original video and swap in the instrumental audio, writing "
        "'<name> (Karaoke).<ext>' (input must have a video stream). Best of both worlds: "
        "on-screen lyrics with the lead vocal removed.",
    )
    parser.add_argument(
        "--chart",
        action="store_true",
        help="Also generate an UltraStar .txt from the vocal stem (needs the 'notes' group)",
    )
    args = parser.parse_args()

    if args.video and args.chart:
        parser.error("--chart is not available with --video (no vocal stem is kept in video mode)")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    outputs = remove_vocals(
        args.audio,
        args.output_dir,
        model=args.model,
        device=args.device,
        mp3=args.mp3,
        video=args.video,
    )

    if args.video:
        print(f"Karaoke video: {outputs['karaoke_video']}")
        return

    print(f"Instrumental: {outputs['instrumental']}")
    print(f"Vocal stem:   {outputs['vocals']}")

    if args.chart:
        from pikaraoke.lib.audio_to_ultrastar import convert

        # convert() imports librosa/numpy lazily, so a missing 'notes' group surfaces here.
        try:
            chart_path = convert(outputs["vocals"])
        except ImportError:
            parser.error(
                "--chart needs the 'notes' dependency group: add `--group notes` to your command"
            )
        print(f"Chart:        {chart_path}")


if __name__ == "__main__":
    main()
