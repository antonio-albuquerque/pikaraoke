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
import shutil
import subprocess
import sys
import tempfile

DEFAULT_MODEL = "htdemucs"

# Demucs names its output subfolder after each stem it keeps.
_INSTRUMENTAL_STEM = "no_vocals"
_VOCALS_STEM = "vocals"


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


def remove_vocals(
    input_path: str,
    out_dir: str | None = None,
    *,
    model: str = DEFAULT_MODEL,
    device: str | None = None,
    mp3: bool = False,
) -> dict[str, str]:
    """Split input into an instrumental and a vocal stem.

    Writes `<basename> (Instrumental).<ext>` and `<basename> (Vocals).<ext>` into out_dir
    (default: the input file's directory). Returns {"instrumental": path, "vocals": path}.
    The original file is never modified.
    """
    if not os.path.isfile(input_path):
        raise VocalRemovalError(f"Input file not found: {input_path}")

    base = os.path.splitext(os.path.basename(input_path))[0]
    ext = "mp3" if mp3 else "wav"
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(input_path))

    with tempfile.TemporaryDirectory() as work_dir:
        logging.info("Running Demucs separation (this can take a while on CPU)...")
        _run_demucs(_build_command(input_path, work_dir, model, device, mp3))

        stem_dir = os.path.join(work_dir, model, base)
        sources = {
            "instrumental": (os.path.join(stem_dir, f"{_INSTRUMENTAL_STEM}.{ext}"), "Instrumental"),
            "vocals": (os.path.join(stem_dir, f"{_VOCALS_STEM}.{ext}"), "Vocals"),
        }
        outputs: dict[str, str] = {}
        for key, (src, label) in sources.items():
            if not os.path.isfile(src):
                raise VocalRemovalError(f"Expected Demucs output missing: {src}")
            dest = os.path.join(out_dir, f"{base} ({label}).{ext}")
            shutil.move(src, dest)
            outputs[key] = dest

    logging.info(f"Wrote instrumental: {outputs['instrumental']}")
    logging.info(f"Wrote vocal stem: {outputs['vocals']}")
    return outputs
