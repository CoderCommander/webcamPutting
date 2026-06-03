"""Tests for the GSPro window watcher's club-name parsing."""

from __future__ import annotations

from birdman_putting.gspro_watcher import parse_club


class TestParseClub:
    """Verify OCR text -> GSPro club code mapping is robust to noise."""

    def test_clean_putter(self) -> None:
        assert parse_club("Putter") == "PT"

    def test_lowercase(self) -> None:
        assert parse_club("putter") == "PT"

    def test_uppercase(self) -> None:
        assert parse_club("PUTTER") == "PT"

    def test_with_punctuation(self) -> None:
        # Tesseract often sprinkles punctuation noise around real text
        assert parse_club("|. PUTTER ;") == "PT"

    def test_with_garbage_prefix(self) -> None:
        assert parse_club("xyz Putter abc") == "PT"

    def test_driver(self) -> None:
        assert parse_club("Driver") == "DR"

    def test_irons(self) -> None:
        assert parse_club("9 Iron") == "9I"
        assert parse_club("7 Iron") == "7I"
        assert parse_club("3 Iron") == "3I"

    def test_iron_variants(self) -> None:
        # Tesseract regularly collapses spaces — the parser must tolerate it.
        assert parse_club("9iron") == "9I"
        assert parse_club("9 iron") == "9I"
        assert parse_club("8Iron") == "8I"  # observed live: "8Iron"
        assert parse_club("8-Iron") == "8I"
        assert parse_club("5IRON") == "5I"

    def test_iron_ocr_confusables(self) -> None:
        """Tesseract often misreads 'i' as 'l' or '1' (identical glyphs)."""
        assert parse_club("5lron") == "5I"  # observed live: "5lron"
        assert parse_club("9lron") == "9I"
        assert parse_club("8 lron") == "8I"
        assert parse_club("71RON") == "7I"
        assert parse_club("3 1ron") == "3I"

    def test_wood_variants(self) -> None:
        assert parse_club("3wood") == "3W"
        assert parse_club("3 Wood") == "3W"
        assert parse_club("5-wood") == "5W"

    def test_wood_ocr_confusables(self) -> None:
        """Tesseract may misread 'o' as '0' in wood."""
        assert parse_club("3w0od") == "3W"
        assert parse_club("3wo0d") == "3W"
        assert parse_club("3w00d") == "3W"

    def test_pitching_wedge_via_pitch_substring(self) -> None:
        # 'Pitching Wedge' must be recognised — pitch-substring matches
        assert parse_club("Pitching Wedge") == "PW"

    def test_wedge_without_qualifier_returns_none(self) -> None:
        # Bare 'Wedge' is ambiguous — don't guess
        assert parse_club("Wedge") is None

    def test_iron_only_with_anchor(self) -> None:
        # A bare digit with no anchor should not match
        assert parse_club("9") is None
        assert parse_club("245 yds") is None

    def test_wedges(self) -> None:
        assert parse_club("Lob Wedge") == "LW"
        assert parse_club("Sand Wedge") == "SW"
        assert parse_club("Approach Wedge") == "AW"
        assert parse_club("Gap Wedge") == "GW"
        assert parse_club("Pitching Wedge") == "PW"

    def test_wedge_short_form_with_space(self) -> None:
        """GSPro compact labels: 'L Wedge', 'G Wedge', 'P Wedge', etc."""
        assert parse_club("L Wedge") == "LW"
        assert parse_club("S Wedge") == "SW"
        assert parse_club("A Wedge") == "AW"
        assert parse_club("G Wedge") == "GW"
        assert parse_club("P Wedge") == "PW"

    def test_wedge_short_form_collapsed(self) -> None:
        """Tesseract drops the space — 'GWedge' is what we see live."""
        assert parse_club("LWedge") == "LW"
        assert parse_club("SWedge") == "SW"
        assert parse_club("AWedge") == "AW"
        assert parse_club("GWedge") == "GW"  # observed live: "GWedge"
        assert parse_club("PWedge") == "PW"

    def test_woods(self) -> None:
        assert parse_club("3 Wood") == "3W"
        assert parse_club("5 Wood") == "5W"

    def test_hybrid(self) -> None:
        assert parse_club("Hybrid") == "HY"
        assert parse_club("4 Rescue") == "HY"

    def test_unknown_returns_none(self) -> None:
        assert parse_club("Tee Box") is None
        assert parse_club("Hole 1") is None
        assert parse_club("") is None
        assert parse_club("   ") is None

    def test_putter_wins_over_iron(self) -> None:
        """Mixed nonsense — putter takes priority because it's a longer
        unique match."""
        assert parse_club("Putter | Hole 1") == "PT"

    def test_long_wedge_beats_substring_match(self) -> None:
        """Lob Wedge must match before plain 'Wedge' would (no plain
        'Wedge' pattern, so this just confirms order-independence)."""
        assert parse_club("Currently selected: Lob Wedge") == "LW"
        assert parse_club("Lob wedge selected") == "LW"

    def test_handles_none_safely(self) -> None:
        # Defensive: function expects str, but should not crash on empty
        assert parse_club("") is None
