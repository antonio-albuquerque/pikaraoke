"""Tests for the UltraStar .txt note parser."""

from pikaraoke.lib.ultrastar import parse_ultrastar, ultrastar_path_for

# BPM 60 (quad-beat) -> beat = 60000/4/60 = 250 ms. GAP 1000 ms.
SAMPLE = """#TITLE:Test
#ARTIST:Nobody
#BPM:60
#GAP:1000
: 0 4 0 Hel
: 4 4 12 lo
- 8
* 8 2 7 world
F 12 4 -2 free
E
"""


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


def test_parses_notes_with_timing_and_pitch(tmp_path):
    path = _write(tmp_path, "song.txt", SAMPLE)
    notes = parse_ultrastar(path)

    assert len(notes) == 4  # three note types parsed, line break/end ignored

    first = notes[0]
    # beat 0 -> GAP only = 1.0s; length 4 beats * 250ms = 1.0s -> ends at 2.0s
    assert first["t0"] == 1.0
    assert first["t1"] == 2.0
    assert first["midi"] == 60  # pitch 0 -> middle C

    # Golden (*) and freestyle (F) notes are treated like normal notes.
    assert notes[2]["midi"] == 7 + 60
    assert notes[3]["midi"] == -2 + 60


def test_notes_sorted_by_start_time(tmp_path):
    out_of_order = "#BPM:60\n#GAP:0\n: 8 2 5 b\n: 0 2 3 a\n: 4 2 4 c\nE\n"
    notes = parse_ultrastar(_write(tmp_path, "x.txt", out_of_order))
    assert [n["t0"] for n in notes] == sorted(n["t0"] for n in notes)


def test_missing_bpm_returns_empty(tmp_path):
    path = _write(tmp_path, "nobpm.txt", "#TITLE:x\n: 0 4 0 a\nE\n")
    assert parse_ultrastar(path) == []


def test_decimal_comma_bpm(tmp_path):
    # European locale BPM with a comma decimal separator.
    path = _write(tmp_path, "comma.txt", "#BPM:120,5\n#GAP:0\n: 0 4 0 a\nE\n")
    notes = parse_ultrastar(path)
    assert len(notes) == 1
    assert notes[0]["t0"] == 0.0


def test_malformed_note_lines_skipped(tmp_path):
    content = "#BPM:60\n#GAP:0\n: 0 4 0 ok\n: bad line\n: 4 0 5 zerolen\nE\n"
    notes = parse_ultrastar(_write(tmp_path, "bad.txt", content))
    assert len(notes) == 1  # zero-length and unparseable lines dropped


def test_ultrastar_path_for(tmp_path):
    song = tmp_path / "MySong---abcdefghijk.mp4"
    song.write_text("x")
    assert ultrastar_path_for(str(song)) is None

    (tmp_path / "MySong---abcdefghijk.txt").write_text("#BPM:60\n")
    assert ultrastar_path_for(str(song)) == str(tmp_path / "MySong---abcdefghijk.txt")


def test_ultrastar_path_for_none_input():
    assert ultrastar_path_for("") is None
