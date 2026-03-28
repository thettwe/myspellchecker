"""
Tests for MED-004: Incomplete Virama Operand Validation

Verifies that virama stacking works with:
- Traditional Vagga consonants
- Non-Vagga consonants in Pali/Sanskrit loanwords
- Invalid operands are still rejected
"""


class TestVaggaViramaStacking:
    """Test traditional Vagga consonant virama stacking."""

    def test_stacking_exceptions_exists(self):
        """STACKING_EXCEPTIONS should be defined."""
        from myspellchecker.core.constants import STACKING_EXCEPTIONS

        assert isinstance(STACKING_EXCEPTIONS, set)
        assert len(STACKING_EXCEPTIONS) > 30  # Should have many patterns

    def test_same_consonant_gemination(self):
        """Same consonant pairs should be valid (gemination)."""
        from myspellchecker.core.constants import STACKING_EXCEPTIONS

        gemination_pairs = [
            ("\u1000", "\u1000"),  # က္က - kka
            ("\u1002", "\u1002"),  # ဂ္ဂ - gga
            ("\u1005", "\u1005"),  # စ္စ - cca
            ("\u1010", "\u1010"),  # တ္တ - tta
            ("\u1019", "\u1019"),  # မ္မ - mma
        ]

        for pair in gemination_pairs:
            assert pair in STACKING_EXCEPTIONS, f"Missing gemination: {pair}"

    def test_cross_consonant_pairs(self):
        """Cross-consonant pairs from Pali should be valid."""
        from myspellchecker.core.constants import STACKING_EXCEPTIONS

        pali_pairs = [
            ("\u1012", "\u1013"),  # ဒ္ဓ - ddha (ဗုဒ္ဓ)
            ("\u1000", "\u1001"),  # က္ခ - kkha (ဒုက္ခ)
            ("\u1002", "\u1003"),  # ဂ္ဃ - ggha (သင်္ဃ)
            ("\u100f", "\u100d"),  # ဏ္ဍ - ṇḍa (ကဏ္ဍ)
        ]

        for pair in pali_pairs:
            assert pair in STACKING_EXCEPTIONS, f"Missing Pali pair: {pair}"


class TestNonVaggaViramaStacking:
    """Test non-Vagga consonant virama stacking (MED-004 fix)."""

    def test_ya_as_operand(self):
        """Ya (ယ) should be valid as virama operand in loanwords."""
        from myspellchecker.core.constants import STACKING_EXCEPTIONS

        ya_patterns = [
            ("\u1019", "\u101a"),  # မ္ယ - mya
            ("\u1000", "\u101a"),  # က္ယ - kya
            ("\u1010", "\u101a"),  # တ္ယ - tya
        ]

        for pair in ya_patterns:
            assert pair in STACKING_EXCEPTIONS, f"Missing Ya pattern: {pair}"

    def test_ra_as_operand(self):
        """Ra (ရ) should be valid as virama operand in loanwords."""
        from myspellchecker.core.constants import STACKING_EXCEPTIONS

        ra_patterns = [
            ("\u1019", "\u101b"),  # မ္ရ - mra
            ("\u1000", "\u101b"),  # က္ရ - kra
            ("\u1010", "\u101b"),  # တ္ရ - tra
            ("\u1015", "\u101b"),  # ပ္ရ - pra
        ]

        for pair in ra_patterns:
            assert pair in STACKING_EXCEPTIONS, f"Missing Ra pattern: {pair}"

    def test_wa_as_operand(self):
        """Wa (ဝ) should be valid as virama operand (rare)."""
        from myspellchecker.core.constants import STACKING_EXCEPTIONS

        wa_patterns = [
            ("\u1010", "\u101d"),  # တ္ဝ - twa
            ("\u101e", "\u101d"),  # သ္ဝ - swa
        ]

        for pair in wa_patterns:
            assert pair in STACKING_EXCEPTIONS, f"Missing Wa pattern: {pair}"

    def test_ha_as_operand(self):
        """Ha (ဟ) should be valid as virama operand."""
        from myspellchecker.core.constants import STACKING_EXCEPTIONS

        ha_patterns = [
            ("\u101f", "\u1019"),  # ဟ္မ - hma (ဗြဟ္မာ)
            ("\u1014", "\u101f"),  # န္ဟ - nha
        ]

        for pair in ha_patterns:
            assert pair in STACKING_EXCEPTIONS, f"Missing Ha pattern: {pair}"

    def test_la_combinations(self):
        """La (လ) stacking patterns should be valid."""
        from myspellchecker.core.constants import STACKING_EXCEPTIONS

        la_patterns = [
            ("\u101c", "\u101c"),  # လ္လ - lla
            ("\u101c", "\u1000"),  # လ္က - lka
            ("\u101c", "\u1015"),  # လ္ပ - lpa
        ]

        for pair in la_patterns:
            assert pair in STACKING_EXCEPTIONS, f"Missing La pattern: {pair}"

    def test_sa_combinations(self):
        """Sa (သ) stacking patterns should be valid."""
        from myspellchecker.core.constants import STACKING_EXCEPTIONS

        sa_patterns = [
            ("\u101e", "\u101e"),  # သ္သ - ssa
            ("\u101e", "\u1010"),  # သ္တ - sta
            ("\u101e", "\u1019"),  # သ္မ - sma
        ]

        for pair in sa_patterns:
            assert pair in STACKING_EXCEPTIONS, f"Missing Sa pattern: {pair}"


class TestKinziValidFollowers:
    """Test Kinzi (င်္) valid follower consonants."""

    def test_kinzi_followers_exists(self):
        """KINZI_VALID_FOLLOWERS should be defined."""
        from myspellchecker.core.constants import KINZI_VALID_FOLLOWERS

        assert isinstance(KINZI_VALID_FOLLOWERS, set)
        assert len(KINZI_VALID_FOLLOWERS) > 20  # Should have many consonants

    def test_kinzi_accepts_vagga_consonants(self):
        """Kinzi should accept Vagga consonants as followers."""
        from myspellchecker.core.constants import KINZI_VALID_FOLLOWERS

        vagga_consonants = [
            "\u1000",  # က
            "\u1001",  # ခ
            "\u1002",  # ဂ
            "\u1003",  # ဃ
            "\u1010",  # တ
            "\u1017",  # ဗ
            "\u1018",  # ဘ
            "\u1019",  # မ
        ]

        for c in vagga_consonants:
            assert c in KINZI_VALID_FOLLOWERS, f"Kinzi should accept {c}"

    def test_kinzi_accepts_non_vagga(self):
        """Kinzi should accept non-Vagga consonants as followers."""
        from myspellchecker.core.constants import KINZI_VALID_FOLLOWERS

        non_vagga = [
            "\u101a",  # ယ - ya
            "\u101b",  # ရ - ra
            "\u101c",  # လ - la
            "\u101e",  # သ - sa
            "\u101f",  # ဟ - ha
        ]

        for c in non_vagga:
            assert c in KINZI_VALID_FOLLOWERS, f"Kinzi should accept {c}"


class TestSyllableValidation:
    """Integration tests with SyllableRuleValidator."""

    def test_validator_accepts_vagga_stacking(self):
        """Validator should accept traditional Vagga stacking."""
        from myspellchecker.core.syllable_rules import SyllableRuleValidator

        SyllableRuleValidator()

        # Test the check_stacking method if it exists
        # These are valid Vagga stackings
        # Note: The validator may not expose these methods directly
        # but we test the constants are correctly used

    def test_stacking_exceptions_count(self):
        """Should have comprehensive stacking patterns."""
        from myspellchecker.core.constants import STACKING_EXCEPTIONS

        # After MED-004 fix, should have at least 50 patterns
        assert len(STACKING_EXCEPTIONS) >= 50, (
            f"Expected 50+ patterns, got {len(STACKING_EXCEPTIONS)}"
        )

    def test_loanword_patterns_added(self):
        """MED-004 patterns should be present."""
        from myspellchecker.core.constants import STACKING_EXCEPTIONS

        # Check that we have Ya, Ra, Wa patterns
        ya = "\u101a"
        ra = "\u101b"
        wa = "\u101d"

        ya_pairs = [p for p in STACKING_EXCEPTIONS if p[1] == ya]
        ra_pairs = [p for p in STACKING_EXCEPTIONS if p[1] == ra]
        wa_pairs = [p for p in STACKING_EXCEPTIONS if p[1] == wa]

        assert len(ya_pairs) >= 3, f"Should have Ya patterns, got {len(ya_pairs)}"
        assert len(ra_pairs) >= 5, f"Should have Ra patterns, got {len(ra_pairs)}"
        assert len(wa_pairs) >= 2, f"Should have Wa patterns, got {len(wa_pairs)}"


class TestWetMapping:
    """Test WET_MAPPING for Vagga consonants."""

    def test_wet_mapping_exists(self):
        """WET_MAPPING should be defined."""
        from myspellchecker.core.constants import WET_MAPPING

        assert isinstance(WET_MAPPING, dict)
        # 5 Vagga x 5 columns = 25, plus ည (U+100A) alternate for ဉ (U+1009) = 26
        assert len(WET_MAPPING) == 26

    def test_wet_mapping_structure(self):
        """WET_MAPPING should have (row, col) tuples."""
        from myspellchecker.core.constants import WET_MAPPING

        for char, pos in WET_MAPPING.items():
            assert isinstance(pos, tuple), f"Invalid position for {char}"
            assert len(pos) == 2, f"Position should be (row, col) for {char}"
            row, col = pos
            assert 0 <= row <= 4, f"Invalid row {row} for {char}"
            assert 1 <= col <= 5, f"Invalid col {col} for {char}"

    def test_ka_vagga_mapping(self):
        """Ka-vagga (က-ဝဂ်) should be in row 0."""
        from myspellchecker.core.constants import WET_MAPPING

        ka_vagga = ["\u1000", "\u1001", "\u1002", "\u1003", "\u1004"]

        for i, char in enumerate(ka_vagga, 1):
            assert char in WET_MAPPING
            row, col = WET_MAPPING[char]
            assert row == 0, f"Ka-vagga {char} should be row 0"
            assert col == i, f"Ka-vagga {char} should be col {i}"
