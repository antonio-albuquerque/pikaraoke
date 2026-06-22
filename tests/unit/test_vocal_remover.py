"""Tests for the vocal-removal module. Demucs itself is mocked (no torch needed)."""

import os
from unittest.mock import patch

import pytest

from pikaraoke.lib import vocal_remover
from pikaraoke.lib.vocal_remover import (
    VocalRemovalError,
    _build_command,
    _mux_instrumental_video,
    _strip_trailing_id,
    remove_vocals,
)


def _arg_after(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def _fake_demucs(ext="wav"):
    """Side effect for _run_demucs: create the stem files Demucs would produce."""

    def run(cmd):
        work_dir = _arg_after(cmd, "-o")
        model = _arg_after(cmd, "-n")
        base = os.path.splitext(os.path.basename(cmd[-1]))[0]
        stem_dir = os.path.join(work_dir, model, base)
        os.makedirs(stem_dir, exist_ok=True)
        for stem in ("no_vocals", "vocals"):
            with open(os.path.join(stem_dir, f"{stem}.{ext}"), "w") as f:
                f.write("audio")

    return run


class TestBuildCommand:
    def test_includes_two_stems_and_model(self):
        cmd = _build_command("song.mp3", "/work", "htdemucs", None, False)
        assert "--two-stems" in cmd and cmd[cmd.index("--two-stems") + 1] == "vocals"
        assert _arg_after(cmd, "-n") == "htdemucs"
        assert _arg_after(cmd, "-o") == "/work"
        assert cmd[-1] == "song.mp3"
        assert "-d" not in cmd and "--mp3" not in cmd

    def test_device_and_mp3_flags(self):
        cmd = _build_command("song.mp3", "/work", "htdemucs", "cpu", True)
        assert _arg_after(cmd, "-d") == "cpu"
        assert "--mp3" in cmd


class TestRemoveVocals:
    def test_produces_renamed_stems_and_keeps_original(self, tmp_path):
        song = tmp_path / "Title---abcdefghijk.mp4"
        song.write_text("original")

        with patch.object(vocal_remover, "_run_demucs", side_effect=_fake_demucs()):
            out = remove_vocals(str(song))

        assert out["instrumental"] == str(tmp_path / "Title---abcdefghijk (Instrumental).wav")
        assert out["vocals"] == str(tmp_path / "Title---abcdefghijk (Vocals).wav")
        assert os.path.isfile(out["instrumental"])
        assert os.path.isfile(out["vocals"])
        assert song.read_text() == "original"  # original untouched

    def test_respects_out_dir_and_mp3(self, tmp_path):
        song = tmp_path / "song.mp3"
        song.write_text("x")
        out_dir = tmp_path / "stems"
        out_dir.mkdir()

        with patch.object(vocal_remover, "_run_demucs", side_effect=_fake_demucs("mp3")):
            out = remove_vocals(str(song), str(out_dir), mp3=True)

        assert out["instrumental"] == str(out_dir / "song (Instrumental).mp3")
        assert os.path.isfile(out["vocals"])

    def test_missing_input_raises(self):
        with pytest.raises(VocalRemovalError, match="not found"):
            remove_vocals("/nope/missing.mp3")

    def test_missing_output_raises(self, tmp_path):
        song = tmp_path / "song.mp3"
        song.write_text("x")
        # _run_demucs "succeeds" but produces nothing.
        with patch.object(vocal_remover, "_run_demucs", side_effect=lambda cmd: None):
            with pytest.raises(VocalRemovalError, match="output missing"):
                remove_vocals(str(song))


class TestRunDemucs:
    def test_missing_demucs_raises_clear_error(self):
        with patch("pikaraoke.lib.vocal_remover.subprocess.run", side_effect=FileNotFoundError()):
            with pytest.raises(VocalRemovalError, match="not installed"):
                vocal_remover._run_demucs(["demucs"])

    def test_nonzero_exit_raises(self):
        class Result:
            returncode = 1
            stderr = "boom"

        with patch("pikaraoke.lib.vocal_remover.subprocess.run", return_value=Result()):
            with pytest.raises(VocalRemovalError, match="exit 1"):
                vocal_remover._run_demucs(["demucs"])


class TestStripTrailingId:
    def test_strips_triple_dash_id(self):
        assert _strip_trailing_id("My Song---IS7ESPZn7do") == "My Song"

    def test_strips_bracketed_id(self):
        assert _strip_trailing_id("My Song [IS7ESPZn7do]") == "My Song"

    def test_leaves_plain_name(self):
        assert _strip_trailing_id("My Song") == "My Song"


class TestMuxCommand:
    def test_maps_video_copy_and_aac_audio(self):
        captured = {}

        class Result:
            returncode = 0
            stderr = ""

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return Result()

        with patch("pikaraoke.lib.vocal_remover.subprocess.run", side_effect=fake_run):
            _mux_instrumental_video("in.mp4", "inst.wav", "out.mp4")

        cmd = captured["cmd"]
        assert cmd[:2] == ["ffmpeg", "-y"]
        assert "0:v:0" in cmd and "1:a:0" in cmd
        assert cmd[cmd.index("-c:v") + 1] == "copy"
        assert cmd[cmd.index("-c:a") + 1] == "aac"
        assert cmd[-1] == "out.mp4"


class TestVideoMode:
    def test_produces_karaoke_video_with_clean_name(self, tmp_path):
        song = tmp_path / "My Song---IS7ESPZn7do.mp4"
        song.write_text("original")

        def fake_mux(video_path, instrumental_path, out_path):
            with open(out_path, "w") as f:
                f.write("karaoke video")

        with (
            patch.object(vocal_remover, "_has_video_stream", return_value=True),
            patch.object(vocal_remover, "_run_demucs", side_effect=_fake_demucs()),
            patch.object(vocal_remover, "_mux_instrumental_video", side_effect=fake_mux),
        ):
            out = remove_vocals(str(song), video=True)

        # ID stripped, (Karaoke) appended, original extension kept.
        assert out["karaoke_video"] == str(tmp_path / "My Song (Karaoke).mp4")
        assert os.path.isfile(out["karaoke_video"])
        assert "instrumental" not in out and "vocals" not in out
        assert song.read_text() == "original"

    def test_webm_falls_back_to_mkv(self, tmp_path):
        song = tmp_path / "clip.webm"
        song.write_text("x")

        with (
            patch.object(vocal_remover, "_has_video_stream", return_value=True),
            patch.object(vocal_remover, "_run_demucs", side_effect=_fake_demucs()),
            patch.object(
                vocal_remover,
                "_mux_instrumental_video",
                side_effect=lambda v, i, o: open(o, "w").close(),
            ),
        ):
            out = remove_vocals(str(song), video=True)

        assert out["karaoke_video"].endswith("clip (Karaoke).mkv")

    def test_audio_only_input_rejected(self, tmp_path):
        song = tmp_path / "audio.mp3"
        song.write_text("x")
        with patch.object(vocal_remover, "_has_video_stream", return_value=False):
            with pytest.raises(VocalRemovalError, match="video stream"):
                remove_vocals(str(song), video=True)
