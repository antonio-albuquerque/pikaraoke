"""Tests for the audio->UltraStar converter's pure-Python pieces.

segment_notes and notes_to_ultrastar need neither librosa nor numpy, so they are tested
directly without the optional `notes` dependency group.
"""

from pikaraoke.lib.audio_to_ultrastar import segment_notes
from pikaraoke.lib.ultrastar import notes_to_ultrastar, parse_ultrastar

HOP = 0.02  # 20 ms per frame
A4 = 440.0  # MIDI 69
B4 = 493.883  # MIDI 71


def _frames(*runs):
    """Build (f0, prob) from (hz_or_None, count) runs. None = unvoiced frame."""
    f0, prob = [], []
    for hz, count in runs:
        for _ in range(count):
            f0.append(hz)
            prob.append(0.0 if hz is None else 0.9)
    return f0, prob


def test_held_pitch_becomes_one_note():
    f0, prob = _frames((A4, 20))  # 400 ms
    notes = segment_notes(f0, prob, HOP)
    assert len(notes) == 1
    assert notes[0]["midi"] == 69
    assert notes[0]["t0"] == 0.0
    assert abs(notes[0]["t1"] - 0.4) < 1e-9


def test_short_blip_dropped():
    f0, prob = _frames((A4, 3), (None, 5))  # 60 ms < default min_note_ms 100
    assert segment_notes(f0, prob, HOP) == []


def test_brief_gap_is_bridged():
    # 200ms + 40ms gap + 200ms, same pitch -> single note (gap < max_gap_ms 120... )
    f0, prob = _frames((A4, 10), (None, 2), (A4, 10))
    notes = segment_notes(f0, prob, HOP)
    assert len(notes) == 1
    assert notes[0]["midi"] == 69


def test_long_gap_splits_note():
    f0, prob = _frames((A4, 10), (None, 10), (A4, 10))  # 200 ms gap > max_gap_ms
    notes = segment_notes(f0, prob, HOP)
    assert len(notes) == 2
    assert all(n["midi"] == 69 for n in notes)


def test_pitch_change_splits_note():
    f0, prob = _frames((A4, 10), (B4, 10))
    notes = segment_notes(f0, prob, HOP)
    assert [n["midi"] for n in notes] == [69, 71]


def test_low_confidence_treated_as_unvoiced():
    f0 = [A4] * 10
    prob = [0.1] * 10  # below default min_confidence 0.5
    assert segment_notes(f0, prob, HOP) == []


def test_nan_frames_ignored():
    f0, prob = _frames((float("nan"), 5), (A4, 20))
    notes = segment_notes(f0, prob, HOP)
    assert len(notes) == 1
    assert notes[0]["midi"] == 69


def test_serializer_round_trip(tmp_path):
    notes = [
        {"t0": 1.0, "t1": 2.0, "midi": 60},
        {"t0": 2.0, "t1": 2.5, "midi": 67},
    ]
    bpm = 400
    gap_ms = notes[0]["t0"] * 1000  # mirrors convert()
    text = notes_to_ultrastar(notes, title="T", artist="A", bpm=bpm, gap_ms=gap_ms)

    assert "#BPM:400" in text
    assert "#TITLE:T" in text
    assert text.rstrip().endswith("E")

    path = tmp_path / "rt.txt"
    path.write_text(text, encoding="utf-8")
    parsed = parse_ultrastar(str(path))

    assert len(parsed) == len(notes)
    one_beat = 15000 / bpm / 1000  # seconds
    for original, got in zip(notes, parsed):
        assert got["midi"] == original["midi"]
        assert abs(got["t0"] - original["t0"]) <= one_beat
        assert abs(got["t1"] - original["t1"]) <= one_beat
