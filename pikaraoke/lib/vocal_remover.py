"""Remove the lead vocal from an audio/video file to make a karaoke backing track.

Uses Demucs (htdemucs --two-stems=vocals) to split the input into an instrumental
("no_vocals") and an isolated vocal stem. The vocal stem is a clean input for the
UltraStar chart converter (pikaraoke.lib.audio_to_ultrastar).

Demucs and its heavy torch stack are invoked as a subprocess and live in the optional
`separation` dependency group, so importing this module never loads torch and the app
gains no runtime weight.
"""

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile

DEFAULT_MODEL = "htdemucs"

# Demucs names its output subfolder after each stem it keeps.
_INSTRUMENTAL_STEM = "no_vocals"
_VOCALS_STEM = "vocals"

# Trailing YouTube-ID suffixes PiKaraoke uses (triple-dash or bracketed, 11 chars).
_ID_SUFFIX_RE = re.compile(r"(?:---[A-Za-z0-9_-]{11}| \[[A-Za-z0-9_-]{11}\])$")

# WebM can't hold AAC audio, so karaoke videos from .webm sources go to .mkv.
_VIDEO_OUT_FALLBACK = {".webm": ".mkv"}


class VocalRemovalError(RuntimeError):
    """Raised when Demucs separation fails or Demucs is unavailable."""


def _build_command(
    input_path: str, work_dir: str, model: str, device: str | None, mp3: bool
) -> list[str]:
    """Assemble the `demucs --two-stems=vocals` command."""
    cmd = [sys.executable, "-m", "demucs", "--two-stems", "vocals", "-n", model, "-o", work_dir]
    if device:
        cmd += ["-d", device]
    if mp3:
        cmd.append("--mp3")
    cmd.append(input_path)
    return cmd


def _run_demucs(cmd: list[str]) -> None:
    """Run the Demucs subprocess, raising VocalRemovalError on failure or missing install."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise VocalRemovalError(
            "Demucs is not installed. Install the optional dependency group, e.g. "
            "`uv run --group separation python build_scripts/remove_vocals.py ...`"
        ) from e
    if result.returncode != 0:
        raise VocalRemovalError(
            f"Demucs failed (exit {result.returncode}): {result.stderr.strip()}"
        )


def _strip_trailing_id(base: str) -> str:
    """Drop a trailing YouTube-ID suffix so the derived name displays cleanly."""
    return _ID_SUFFIX_RE.sub("", base)


def _has_video_stream(path: str) -> bool:
    """True if the file contains a video stream (ffprobe)."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        raise VocalRemovalError("ffprobe not found (FFmpeg is required)") from e
    return "video" in result.stdout


def _mux_instrumental_video(video_path: str, instrumental_path: str, out_path: str) -> None:
    """Combine the original video stream with the instrumental audio into out_path.

    Copies the video stream untouched (keeps any on-screen lyrics) and encodes the
    instrumental as AAC.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-i",
        instrumental_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "256k",
        "-shortest",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise VocalRemovalError(
            f"ffmpeg mux failed (exit {result.returncode}): {result.stderr.strip()}"
        )


def remove_vocals(
    input_path: str,
    out_dir: str | None = None,
    *,
    model: str = DEFAULT_MODEL,
    device: str | None = None,
    mp3: bool = False,
    video: bool = False,
) -> dict[str, str]:
    """Remove the lead vocal from input.

    Default: writes `<basename> (Instrumental).<ext>` and `<basename> (Vocals).<ext>` into
    out_dir, returning {"instrumental": path, "vocals": path}.

    With video=True (input must have a video stream): keeps the original video (e.g. on-screen
    lyrics) and swaps in the instrumental audio, writing `<name> (Karaoke).<ext>` and returning
    {"karaoke_video": path}. The original file is never modified.
    """
    if not os.path.isfile(input_path):
        raise VocalRemovalError(f"Input file not found: {input_path}")
    if video and not _has_video_stream(input_path):
        raise VocalRemovalError(f"--video requires a file with a video stream: {input_path}")

    base = os.path.splitext(os.path.basename(input_path))[0]
    # Video mode re-encodes audio to AAC anyway, so always separate to wav there.
    ext = "wav" if video else ("mp3" if mp3 else "wav")
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(input_path))

    with tempfile.TemporaryDirectory() as work_dir:
        logging.info("Running Demucs separation (this can take a while on CPU)...")
        _run_demucs(_build_command(input_path, work_dir, model, device, mp3=ext == "mp3"))

        stem_dir = os.path.join(work_dir, model, base)
        instrumental_src = os.path.join(stem_dir, f"{_INSTRUMENTAL_STEM}.{ext}")
        vocals_src = os.path.join(stem_dir, f"{_VOCALS_STEM}.{ext}")
        for src in (instrumental_src, vocals_src):
            if not os.path.isfile(src):
                raise VocalRemovalError(f"Expected Demucs output missing: {src}")

        if video:
            in_ext = os.path.splitext(input_path)[1].lower()
            out_ext = _VIDEO_OUT_FALLBACK.get(in_ext, in_ext)
            dest = os.path.join(out_dir, f"{_strip_trailing_id(base)} (Karaoke){out_ext}")
            _mux_instrumental_video(input_path, instrumental_src, dest)
            logging.info(f"Wrote karaoke video: {dest}")
            return {"karaoke_video": dest}

        outputs = {}
        for key, src, label in (
            ("instrumental", instrumental_src, "Instrumental"),
            ("vocals", vocals_src, "Vocals"),
        ):
            dest = os.path.join(out_dir, f"{base} ({label}).{ext}")
            shutil.move(src, dest)
            outputs[key] = dest

    logging.info(f"Wrote instrumental: {outputs['instrumental']}")
    logging.info(f"Wrote vocal stem: {outputs['vocals']}")
    return outputs
