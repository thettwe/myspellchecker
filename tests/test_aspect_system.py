"""
Tests for Myanmar Aspect Marker System.

Tests the AspectChecker functionality:
- Aspect marker detection
- Aspect marker typo correction
- Valid/invalid aspect combinations
- Sequence validation
- Aspect pattern detection
"""

from myspellchecker.grammar.checkers.aspect import (
    AspectChecker,
    AspectError,
    AspectInfo,
    AspectPattern,
)


class TestAspectCheckerInitialization:
    """Tests for AspectChecker initialization."""

    def test_initialization(self):
        """AspectChecker initializes with proper data structures."""
        checker = AspectChecker()
        assert checker.markers is not None
        assert checker.marker_set is not None
        assert checker.typo_map is not None
        assert checker.valid_combinations is not None
        assert checker.invalid_sequences is not None

    def test_data_loaded_correctly(self):
        """Data constants are loaded correctly."""
        checker = AspectChecker()
        # Should have aspect markers loaded
        assert len(checker.markers) > 0
        assert len(checker.marker_set) > 0


class TestAspectMarkerDetection:
    """Tests for aspect marker detection."""

    def test_completion_markers(self):
        """Completion aspect markers are detected."""
        checker = AspectChecker()

        # ပြီ - completion
        assert checker.is_aspect_marker("ပြီ")
        info = checker.get_aspect_info("ပြီ")
        assert info is not None
        assert info.category == "completion"
        assert info.is_final  # Can be sentence-final

        # ပြီး - sequential completion
        assert checker.is_aspect_marker("ပြီး")
        info = checker.get_aspect_info("ပြီး")
        assert info is not None
        assert info.category == "completion"

    def test_progressive_marker(self):
        """Progressive aspect marker is detected."""
        checker = AspectChecker()

        assert checker.is_aspect_marker("နေ")
        info = checker.get_aspect_info("နေ")
        assert info is not None
        assert info.category == "progressive"
        assert not info.is_final  # Usually needs following element

    def test_habitual_marker(self):
        """Habitual aspect marker is detected."""
        checker = AspectChecker()

        assert checker.is_aspect_marker("တတ်")
        info = checker.get_aspect_info("တတ်")
        assert info is not None
        assert info.category == "habitual"

    def test_resultative_marker(self):
        """Resultative aspect marker is detected."""
        checker = AspectChecker()

        assert checker.is_aspect_marker("ထား")
        info = checker.get_aspect_info("ထား")
        assert info is not None
        assert info.category == "resultative"

    def test_directional_markers(self):
        """Directional aspect markers are detected."""
        checker = AspectChecker()

        # လာ - coming
        assert checker.is_aspect_marker("လာ")
        info = checker.get_aspect_info("လာ")
        assert info is not None
        assert info.category == "direction"

        # သွား - going
        assert checker.is_aspect_marker("သွား")
        info = checker.get_aspect_info("သွား")
        assert info is not None
        assert info.category == "direction"

    def test_desiderative_marker(self):
        """Desiderative aspect marker is detected."""
        checker = AspectChecker()

        assert checker.is_aspect_marker("ချင်")
        info = checker.get_aspect_info("ချင်")
        assert info is not None
        assert info.category == "desiderative"

    def test_potential_markers(self):
        """Potential aspect markers are detected."""
        checker = AspectChecker()

        # နိုင် - can/able
        assert checker.is_aspect_marker("နိုင်")
        info = checker.get_aspect_info("နိုင်")
        assert info is not None
        assert info.category == "potential"

        # ရ - permitted/possible
        assert checker.is_aspect_marker("ရ")
        info = checker.get_aspect_info("ရ")
        assert info is not None
        assert info.category == "potential"

    def test_immediate_marker(self):
        """Immediate aspect marker is detected."""
        checker = AspectChecker()

        assert checker.is_aspect_marker("လိုက်")
        info = checker.get_aspect_info("လိုက်")
        assert info is not None
        assert info.category == "immediate"

    def test_experiential_marker(self):
        """Experiential aspect marker is detected."""
        checker = AspectChecker()

        assert checker.is_aspect_marker("ဖူး")
        info = checker.get_aspect_info("ဖူး")
        assert info is not None
        assert info.category == "experiential"
        assert info.is_final  # Can be sentence-final

    def test_directional_as_aspect(self):
        """Directional words like သွား and လာ are aspect markers."""
        checker = AspectChecker()

        # သွား and လာ are aspect markers (directional aspect)
        # They indicate motion direction in verb phrases
        assert checker.is_aspect_marker("သွား")  # directional: motion away
        assert checker.is_aspect_marker("လာ")  # directional: motion toward

    def test_non_aspect_words(self):
        """Non-aspect words are not detected as markers."""
        checker = AspectChecker()

        # Regular words that are not aspect markers
        assert not checker.is_aspect_marker("စား")  # "eat" - regular verb
        assert not checker.is_aspect_marker("တယ်")  # sentence ending particle
        assert not checker.is_aspect_marker("သည်")  # formal ending
        assert not checker.is_aspect_marker("ကို")  # object marker
        assert not checker.is_aspect_marker("ဖတ်")  # "read" - regular verb


class TestAspectTypoDetection:
    """Tests for aspect marker typo detection."""

    def test_completion_typos(self):
        """Completion marker typos are detected."""
        checker = AspectChecker()

        # ပြိ → ပြီ
        assert checker.is_aspect_typo("ပြိ")
        assert checker.get_typo_correction("ပြိ") == "ပြီ"

        # Note: "ပရီ" (Paris) was removed from typo map — it is a valid word,
        # not a plausible typo for "ပြီ" (completion marker).
        assert not checker.is_aspect_typo("ပရီ")

    def test_progressive_typos_removed_false_positives(self):
        """Valid words are not flagged as progressive marker typos."""
        checker = AspectChecker()

        # "နဲ" is a valid particle meaning "with" / "a little" (colloquial form of "နှင့်"),
        # not a typo for "နေ" (progressive marker). It was removed from the typo map.
        assert not checker.is_aspect_typo("နဲ")

    def test_habitual_typos(self):
        """Habitual marker typos are detected."""
        checker = AspectChecker()

        # တတ → တတ်
        assert checker.is_aspect_typo("တတ")
        assert checker.get_typo_correction("တတ") == "တတ်"

        # Note: "တက်" (climb/attend) was removed from typo map — it's a
        # different word from "တတ်" (habitual aspect marker), not a typo.

    def test_resultative_typos(self):
        """Resultative marker typos are detected."""
        checker = AspectChecker()

        # ထာ → ထား
        assert checker.is_aspect_typo("ထာ")
        assert checker.get_typo_correction("ထာ") == "ထား"

    def test_directional_typos(self):
        """Directional marker typos are detected."""
        checker = AspectChecker()

        # သွာ → သွား
        assert checker.is_aspect_typo("သွာ")
        assert checker.get_typo_correction("သွာ") == "သွား"

    def test_desiderative_typos(self):
        """Desiderative marker typos are detected."""
        checker = AspectChecker()

        # ချင → ချင်
        assert checker.is_aspect_typo("ချင")
        assert checker.get_typo_correction("ချင") == "ချင်"

    def test_potential_typos(self):
        """Potential marker typos are detected."""
        checker = AspectChecker()

        # နိင် → နိုင်
        assert checker.is_aspect_typo("နိင်")
        assert checker.get_typo_correction("နိင်") == "နိုင်"

    def test_immediate_typos(self):
        """Immediate marker typos are detected."""
        checker = AspectChecker()

        # လိက် → လိုက်
        assert checker.is_aspect_typo("လိက်")
        assert checker.get_typo_correction("လိက်") == "လိုက်"

    def test_experiential_typos(self):
        """Experiential marker typos are detected."""
        checker = AspectChecker()

        # ဖူ → ဖူး
        assert checker.is_aspect_typo("ဖူ")
        assert checker.get_typo_correction("ဖူ") == "ဖူး"

    def test_valid_markers_not_typos(self):
        """Valid aspect markers are not flagged as typos."""
        checker = AspectChecker()

        for marker in checker.marker_set:
            assert not checker.is_aspect_typo(marker)


class TestSequenceValidation:
    """Tests for sequence validation."""

    def test_valid_sequence_no_errors(self):
        """Valid sequences have no errors."""
        checker = AspectChecker()

        # Simple completion: verb + ပြီ
        errors = checker.validate_sequence(["သွား", "ပြီ"])
        assert errors == []

        # Progressive: verb + နေ + tense
        errors = checker.validate_sequence(["စား", "နေ", "တယ်"])
        assert errors == []

        # Combined: verb + ချင် + ending
        errors = checker.validate_sequence(["လာ", "ချင်", "တယ်"])
        assert errors == []

    def test_typo_detection_in_sequence(self):
        """Typos in sequences are detected."""
        checker = AspectChecker()

        # ပြိ is typo for ပြီ
        errors = checker.validate_sequence(["သွား", "ပြိ"])
        assert len(errors) == 1
        assert errors[0].error_type == "aspect_typo"
        assert errors[0].suggestion == "ပြီ"

        # တတ is typo for တတ်
        errors = checker.validate_sequence(["သွား", "တတ"])
        assert len(errors) == 1
        assert errors[0].error_type == "aspect_typo"
        assert errors[0].suggestion == "တတ်"

    def test_invalid_sequence_detection(self):
        """Invalid aspect sequences are detected."""
        checker = AspectChecker()

        # ပြီ + နေ is invalid (completion before progressive)
        errors = checker.validate_sequence(["သွား", "ပြီ", "နေ"])
        assert len(errors) >= 1
        assert any(e.error_type == "invalid_sequence" for e in errors)

    def test_duplicate_marker_detection(self):
        """Duplicate markers are detected as invalid."""
        checker = AspectChecker()

        # ပြီ + ပြီ is invalid
        errors = checker.validate_sequence(["သွား", "ပြီ", "ပြီ"])
        assert len(errors) >= 1
        assert any(e.error_type == "invalid_sequence" for e in errors)

    def test_multiple_errors(self):
        """Multiple errors in one sequence are all detected."""
        checker = AspectChecker()

        # ပြိ (typo) + ပြီ (duplicate after correction)
        errors = checker.validate_sequence(["သွား", "ပြိ", "ပြီ"])
        assert len(errors) >= 1  # At least the typo should be detected


class TestAspectInfo:
    """Tests for AspectInfo dataclass."""

    def test_aspect_info_creation(self):
        """AspectInfo is created correctly."""
        info = AspectInfo(
            marker="ပြီ",
            category="completion",
            description="Completion - action is done",
            can_combine=True,
            register="neutral",
            is_final=True,
        )
        assert info.marker == "ပြီ"
        assert info.category == "completion"
        assert info.is_final

    def test_aspect_info_str(self):
        """AspectInfo has a string representation."""
        info = AspectInfo(
            marker="နေ",
            category="progressive",
            description="Progressive",
            can_combine=True,
            register="neutral",
            is_final=False,
        )
        s = str(info)
        assert "နေ" in s
        assert "progressive" in s


class TestAspectError:
    """Tests for AspectError dataclass."""

    def test_aspect_error_creation(self):
        """AspectError is created correctly."""
        error = AspectError(
            text="ပြိ",
            position=1,
            suggestions=["ပြီ"],
            error_type="aspect_typo",
            confidence=0.9,
            reason="Typo for completion marker",
        )
        assert error.position == 1
        assert error.word == "ပြိ"  # Backward compatibility property
        assert error.text == "ပြိ"
        assert error.error_type == "aspect_typo"
        assert error.suggestion == "ပြီ"  # Backward compatibility property
        assert error.suggestions == ["ပြီ"]

    def test_aspect_error_str(self):
        """AspectError has a string representation."""
        error = AspectError(
            text="ပြိ",
            position=1,
            suggestions=["ပြီ"],
            error_type="aspect_typo",
            confidence=0.9,
            reason="Typo",
        )
        s = str(error)
        assert "ပြိ" in s
        assert "ပြီ" in s


class TestAspectPattern:
    """Tests for AspectPattern dataclass."""

    def test_aspect_pattern_creation(self):
        """AspectPattern is created correctly."""
        pattern = AspectPattern(
            start_index=0,
            end_index=1,
            markers=["နေ", "ပြီ"],
            categories=["progressive", "completion"],
            is_valid=True,
            confidence=0.9,
        )
        assert pattern.markers == ["နေ", "ပြီ"]
        assert pattern.is_valid

    def test_aspect_pattern_str(self):
        """AspectPattern has a string representation."""
        pattern = AspectPattern(
            start_index=0,
            end_index=1,
            markers=["နေ", "ပြီ"],
            categories=["progressive", "completion"],
            is_valid=True,
            confidence=0.9,
        )
        s = str(pattern)
        assert "valid" in s


class TestFinalMarkers:
    """Tests for final vs non-final marker classification."""

    def test_final_markers(self):
        """Final markers are correctly identified."""
        checker = AspectChecker()

        # ပြီ can be sentence-final
        info = checker.get_aspect_info("ပြီ")
        assert info is not None
        assert info.is_final

        # ဖူး can be sentence-final
        info = checker.get_aspect_info("ဖူး")
        assert info is not None
        assert info.is_final

    def test_non_final_markers(self):
        """Non-final markers are correctly identified."""
        checker = AspectChecker()

        # ပြီး is typically not sentence-final
        info = checker.get_aspect_info("ပြီး")
        assert info is not None
        assert not info.is_final

        # နေ typically needs following element
        info = checker.get_aspect_info("နေ")
        assert info is not None
        assert not info.is_final


class TestConstantsIntegrity:
    """Tests for constants integrity."""

    def test_aspect_markers_have_all_fields(self):
        """All aspect markers have required fields."""
        checker = AspectChecker()
        for _marker, data in checker.markers.items():
            assert isinstance(data, dict)
            assert "category" in data
            assert "description" in data
            # defaults handled in code

    def test_typo_map_corrections_valid(self):
        """All typo corrections map to valid markers."""
        checker = AspectChecker()
        for typo, correction in checker.typo_map.items():
            # Correction should be a valid marker
            assert correction in checker.marker_set, (
                f"Typo '{typo}' maps to '{correction}' which is not a valid marker"
            )

    def test_valid_combinations_use_valid_markers(self):
        """Valid combinations use valid markers."""
        checker = AspectChecker()
        for (first, second), _desc in checker.valid_combinations.items():
            assert first in checker.marker_set, f"Combination uses invalid marker: {first}"
            assert second in checker.marker_set, f"Combination uses invalid marker: {second}"

    def test_invalid_sequences_use_valid_markers(self):
        """Invalid sequences use valid markers."""
        checker = AspectChecker()
        for (first, second), _reason in checker.invalid_sequences.items():
            assert first in checker.marker_set, f"Invalid sequence uses invalid marker: {first}"
            assert second in checker.marker_set, f"Invalid sequence uses invalid marker: {second}"

    def test_final_markers_are_valid(self):
        """Final markers are valid markers."""
        checker = AspectChecker()
        for marker in checker.final_markers:
            assert marker in checker.marker_set

    def test_non_final_markers_are_valid(self):
        """Non-final markers are valid markers."""
        checker = AspectChecker()
        for marker in checker.non_final_markers:
            assert marker in checker.marker_set
