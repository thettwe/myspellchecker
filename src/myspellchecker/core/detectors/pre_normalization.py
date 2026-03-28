"""Pre-normalization text-level detectors.

These detectors run on the raw (un-normalized) text to catch errors that
normalization would silently fix, destroying the evidence.  They all
return ``list[SyllableError]`` — the caller merges them into the main
error list after normalization.

Extracted from ``spellchecker.py`` to reduce file size while preserving
the exact same method signatures and behaviour.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from myspellchecker.core.constants import (
    ET_BROKEN_VIRAMA,
    ET_INCOMPLETE_STACKING,
    ET_LEADING_VOWEL_E,
    ET_MEDIAL_ORDER_ERROR,
    ET_SYLLABLE,
    ET_VOWEL_AFTER_ASAT,
    ET_ZAWGYI_ENCODING,
    MEDIALS,
    VOWEL_SIGNS,
)

if TYPE_CHECKING:
    from myspellchecker.core.config import SpellCheckerConfig
    from myspellchecker.providers.base import DictionaryProvider
from myspellchecker.core.detector_data import TEXT_DETECTOR_CONFIDENCES
from myspellchecker.core.detectors.utils import iter_occurrences
from myspellchecker.core.response import Error, SyllableError
from myspellchecker.text.zawgyi_support import convert_zawgyi_to_unicode, get_zawgyi_detector


class PreNormalizationDetectorsMixin:
    """Mixin providing pre-normalization detector methods and their data constants.

    These detectors run BEFORE Unicode normalization to catch issues that
    the normalizer would silently fix (medial reordering, virama→asat, etc.).
    """

    # --- Type stubs for attributes provided by SpellChecker ---
    config: "SpellCheckerConfig"
    provider: "DictionaryProvider"
    logger: Any

    # ------------------------------------------------------------------ #
    # Data constants used by pre-normalization detectors                  #
    # ------------------------------------------------------------------ #

    # Zero-width characters that are invisible but may break word boundaries.
    # Superset of myanmar_constants.ZERO_WIDTH_CHARS — includes directional
    # marks (U+200E/200F) and invisible operators (U+2060-2063).
    _ZERO_WIDTH_CHARS = frozenset("\u200b\u200c\u200d\ufeff\u200e\u200f\u2060\u2061\u2062\u2063")

    # Myanmar medial signs that cannot appear before a consonant in standard Unicode
    _MYANMAR_MEDIALS: frozenset[str] = frozenset(MEDIALS)  # ျ ြ ွ ှ
    # Myanmar consonant range
    _CONSONANT_MIN = "\u1000"
    _CONSONANT_MAX = "\u1021"
    # Zawgyi medial → Unicode medial mapping (when moved from before to after consonant)
    _ZAWGYI_MEDIAL_MAP = {"\u103b": "\u103c"}  # ျ → ြ

    # Word boundary characters for ZWS word-span expansion
    _WORD_BOUNDARIES: frozenset[str] = frozenset(" ။၊\n\r\t")

    # Myanmar vowel signs that cannot follow virama (U+1039).
    _VOWEL_SIGNS_FOR_VIRAMA: frozenset[str] = frozenset(
        {"\u102b", "\u102c", "\u102d", "\u102e", "\u102f", "\u1030", "\u1031", "\u1032"}
    )

    # Known missing-consonant stacking patterns: when virama is followed by
    # a vowel, the user likely forgot the stacked consonant.
    _BROKEN_VIRAMA_STACKING: dict[tuple[str, str], str] = {
        ("\u1019", "\u102c"): "\u1018",  # မ္ + ာ → မ္ဘ + ာ (ကမ္ဘာ = world)
    }

    # All wrong medial orderings per UTN #11 canonical: Ya < Ra < Wa < Ha
    _MEDIAL_ORDER_WRONG: dict[str, str] = {
        "\u103e\u103d": "\u103d\u103e",  # Ha+Wa → Wa+Ha
        "\u103c\u103b": "\u103b\u103c",  # Ra+Ya → Ya+Ra
        "\u103d\u103b": "\u103b\u103d",  # Wa+Ya → Ya+Wa
        "\u103e\u103b": "\u103b\u103e",  # Ha+Ya → Ya+Ha
        "\u103d\u103c": "\u103c\u103d",  # Wa+Ra → Ra+Wa
        "\u103e\u103c": "\u103c\u103e",  # Ha+Ra → Ra+Ha
    }

    # Myanmar diacritics checked for accidental duplication within a syllable.
    _DIACRITICS_CHECKED_FOR_DUPLICATES: frozenset[str] = frozenset(
        {
            "\u1031",  # ေ  vowel sign e
            "\u102f",  # ု  vowel sign u
            "\u1030",  # ူ  vowel sign uu
            "\u102d",  # ိ  vowel sign i
            "\u102e",  # ီ  vowel sign ii
            "\u1036",  # ံ  anusvara
            "\u103a",  # ်  asat
            "\u1037",  # ့  aukmyit (dot below)
            "\u1038",  # း  visarga
        }
    )

    # Pre-normalization vowel reordering errors.
    # key = wrong form (raw text), value = list of correct forms
    _VOWEL_REORDER_ERRORS: dict[str, list[str]] = {
        "ကေါင်း": ["ကောင်း", "ခေါင်း"],
        "ေကာင်း": ["ကောင်း"],
    }

    # Vowel signs that may appear wrongly before a medial sign.
    # Canonical Myanmar order: consonant + medials + vowels, so vowel-before-medial
    # is always wrong and normalization silently fixes it.
    _VOWEL_SIGNS_BEFORE_MEDIAL: frozenset[str] = frozenset(
        {
            "\u102d",  # ိ  vowel sign i
            "\u102e",  # ီ  vowel sign ii
            "\u102f",  # ု  vowel sign u
            "\u1030",  # ူ  vowel sign uu
        }
    )

    # Known incomplete Pali/Sanskrit stacking forms.
    _STACKING_COMPLETIONS: dict[str, str] = {
        "\u1018\u100f\u1039\u102c": "\u1018\u100f\u1039\u100d\u102c",  # ဘဏ္ာ → ဘဏ္ဍာ
    }

    # Vowel signs that should not follow asat (U+103A) or dot-below (U+1037).
    # Identical to myanmar_constants.VOWEL_SIGNS (U+102B-U+1032).
    _VOWEL_SIGNS: frozenset[str] = frozenset(VOWEL_SIGNS)

    # ------------------------------------------------------------------ #
    # Detector methods                                                    #
    # ------------------------------------------------------------------ #

    def _detect_zawgyi(self, text: str) -> tuple[Any, Any]:
        """Detect Zawgyi encoding in text and return config and warning."""
        from myspellchecker.core.config.text_configs import ZawgyiConfig
        from myspellchecker.text.normalize import check_zawgyi_and_warn

        zawgyi_config = ZawgyiConfig(
            warning_threshold=self.config.validation.zawgyi_confidence_threshold,
            detection_threshold=self.config.validation.zawgyi_confidence_threshold,
        )

        zawgyi_warning = None
        if self.config.validation.use_zawgyi_detection:
            zawgyi_warning = check_zawgyi_and_warn(text, config=zawgyi_config)
            if zawgyi_warning:
                self.logger.warning(
                    f"Zawgyi encoding detected (confidence: {zawgyi_warning.confidence:.2f})"
                )

        return zawgyi_config, zawgyi_warning

    def _detect_zawgyi_words(self, text: str) -> list[SyllableError]:
        """Detect Zawgyi-encoded words in the original text before normalization.

        Checks each word using two methods:
        1. ZawgyiDetector probability (for fully Zawgyi-encoded words)
        2. Structural check: a Myanmar medial sign appears where no consonant
           precedes it (e.g. at word start, or after asat/non-consonant),
           which is invalid in standard Unicode and indicates Zawgyi encoding
        """
        if not self.config.validation.use_zawgyi_detection:
            return []

        errors: list[SyllableError] = []
        detector = get_zawgyi_detector()
        threshold = self.config.validation.zawgyi_confidence_threshold

        current_idx = 0
        for word in text.split():
            if not word:
                continue

            idx = text.find(word, current_idx)
            if idx == -1:
                continue
            current_idx = idx + len(word)

            if not any("\u1000" <= ch <= "\u109f" for ch in word):
                continue

            is_zawgyi = False
            suggestion = None

            has_medial = any(ch in self._MYANMAR_MEDIALS for ch in word)
            has_zawgyi_cp = any(ord(ch) >= 0x1060 for ch in word)
            if detector and (has_medial or has_zawgyi_cp):
                prob = detector.get_zawgyi_probability(word)
                if prob >= threshold:
                    is_zawgyi = True
                    suggestion = convert_zawgyi_to_unicode(word, threshold=0.1)

            if not is_zawgyi:
                for i, ch in enumerate(word):
                    if ch not in self._MYANMAR_MEDIALS:
                        continue
                    if i == 0 or not (
                        self._CONSONANT_MIN <= word[i - 1] <= self._CONSONANT_MAX
                        or word[i - 1] in self._MYANMAR_MEDIALS
                    ):
                        is_zawgyi = True
                        medial_pos = i
                        if medial_pos + 1 < len(word) and (
                            self._CONSONANT_MIN <= word[medial_pos + 1] <= self._CONSONANT_MAX
                        ):
                            prefix = word[:medial_pos]
                            consonant = word[medial_pos + 1]
                            medial = word[medial_pos]
                            rest = word[medial_pos + 2 :]
                            unicode_medial = self._ZAWGYI_MEDIAL_MAP.get(medial, medial)
                            suggestion = prefix + consonant + unicode_medial + rest
                        break

            if is_zawgyi and suggestion and suggestion != word:
                if not self.provider.is_valid_word(suggestion):
                    is_zawgyi = False
                    suggestion = None

            if is_zawgyi:
                suggestions = [suggestion] if suggestion and suggestion != word else []
                errors.append(
                    SyllableError(
                        text=word,
                        position=idx,
                        suggestions=suggestions,
                        confidence=TEXT_DETECTOR_CONFIDENCES["zawgyi_detected"],
                        error_type=ET_ZAWGYI_ENCODING,
                    )
                )

        return errors

    def _detect_zero_width_chars(self, text: str) -> list[SyllableError]:
        """Detect invisible zero-width characters in the original text.

        These characters (U+200B zero-width space, U+200C/D joiners, U+FEFF BOM, etc.)
        are invisible but can break word boundaries and confuse text processing.
        Detection runs on the original text BEFORE normalization removes them.
        """
        errors: list[SyllableError] = []
        reported_positions: set[int] = set()
        for i, ch in enumerate(text):
            if ch in self._ZERO_WIDTH_CHARS and i not in reported_positions:
                start = i
                while start > 0 and text[start - 1] not in self._WORD_BOUNDARIES:
                    start -= 1
                end = i + 1
                while end < len(text) and text[end] not in self._WORD_BOUNDARIES:
                    end += 1
                word_span = text[start:end]
                cleaned = "".join(c for c in word_span if c not in self._ZERO_WIDTH_CHARS)
                for j in range(start, end):
                    if text[j] in self._ZERO_WIDTH_CHARS:
                        reported_positions.add(j)
                errors.append(
                    SyllableError(
                        text=word_span,
                        position=start,
                        suggestions=[cleaned],
                        confidence=TEXT_DETECTOR_CONFIDENCES["zero_width_chars"],
                        error_type=ET_SYLLABLE,
                    )
                )
        return errors

    def _detect_broken_virama(self, text: str) -> list[SyllableError]:
        """Detect virama (U+1039) followed by a vowel sign in the original text.

        Virama should only precede a consonant (to form a stacked consonant pair).
        When followed by a vowel sign, the virama is likely a typo and should be
        removed.  This must run BEFORE normalization because the normalizer
        converts virama to asat (U+103A), destroying the evidence.
        """
        virama = "\u1039"
        errors: list[SyllableError] = []
        i = 0
        while i < len(text) - 1:
            if text[i] == virama and text[i + 1] in self._VOWEL_SIGNS_FOR_VIRAMA:
                start = i
                preceding = ""
                if i > 0 and "\u1000" <= text[i - 1] <= "\u1021":
                    start = i - 1
                    preceding = text[i - 1]
                end = i + 2
                while end < len(text) and text[end] in self._VOWEL_SIGNS_FOR_VIRAMA:
                    end += 1
                if end < len(text) and text[end] == "\u1037":
                    end += 1

                suggestions: list[str] = []

                following_vowel = text[i + 1]
                stacking_key = (preceding, following_vowel)
                if stacking_key in self._BROKEN_VIRAMA_STACKING:
                    missing = self._BROKEN_VIRAMA_STACKING[stacking_key]
                    word_start = start
                    while word_start > 0 and text[word_start - 1] != " ":
                        word_start -= 1
                    start = word_start
                    error_text = text[start:end]
                    fixed = text[start:i] + virama + missing + text[i + 1 : end]
                    suggestions.append(fixed)
                    if fixed.endswith("\u1037"):
                        suggestions.append(fixed[:-1])
                    fallback = text[start:i] + text[i + 1 : end]
                    if fallback != error_text:
                        suggestions.append(fallback)
                else:
                    error_text = text[start:end]
                    corrected = text[start:i] + text[i + 1 : end]
                    if corrected != error_text:
                        suggestions.append(corrected)

                if suggestions:
                    errors.append(
                        SyllableError(
                            text=error_text,
                            position=start,
                            suggestions=suggestions,
                            confidence=TEXT_DETECTOR_CONFIDENCES["broken_virama"],
                            error_type=ET_BROKEN_VIRAMA,
                        )
                    )
                i = end
            else:
                i += 1
        return errors

    def _detect_medial_order_errors(self, text: str) -> list[SyllableError]:
        """Detect medial ordering errors BEFORE normalization reorders them.

        UTN #11 canonical order: Ya (103B) < Ra (103C) < Wa (103D) < Ha (103E).
        """
        errors: list[SyllableError] = []
        found_positions: set[int] = set()

        for wrong_pair, correct_pair in self._MEDIAL_ORDER_WRONG.items():
            for idx, _end in iter_occurrences(text, wrong_pair):
                syl_start = idx
                while syl_start > 0:
                    prev_cp = ord(text[syl_start - 1])
                    if prev_cp < 0x1000 or prev_cp > 0x109F:
                        break
                    if 0x1000 <= prev_cp <= 0x1021:
                        syl_start -= 1
                        break
                    syl_start -= 1

                syl_end = idx + len(wrong_pair)
                while syl_end < len(text):
                    cp = ord(text[syl_end])
                    if 0x102B <= cp <= 0x103E or cp in (0x1039, 0x103A, 0x1036, 0x1037, 0x1038):
                        syl_end += 1
                    elif (
                        0x1000 <= cp <= 0x1021
                        and syl_end + 1 < len(text)
                        and ord(text[syl_end + 1]) == 0x103A
                    ):
                        syl_end += 2
                    else:
                        break

                if syl_start not in found_positions:
                    found_positions.add(syl_start)
                    error_text = text[syl_start:syl_end]
                    corrected = error_text.replace(wrong_pair, correct_pair)

                    if corrected != error_text:
                        suggestions = [corrected]
                        _medials = {"\u103b", "\u103c", "\u103d", "\u103e"}
                        for ch in wrong_pair:
                            if ch in _medials:
                                stripped = error_text.replace(ch, "", 1)
                                if (
                                    stripped
                                    and stripped != error_text
                                    and stripped not in suggestions
                                ):
                                    suggestions.append(stripped)

                        if self.provider and len(suggestions) > 1:
                            indexed = [(s, idx_s) for idx_s, s in enumerate(suggestions)]
                            indexed.sort(
                                key=lambda t: (
                                    not self.provider.is_valid_word(t[0]),
                                    t[1],
                                )
                            )
                            suggestions = [t[0] for t in indexed]

                        errors.append(
                            SyllableError(
                                text=error_text,
                                position=syl_start,
                                suggestions=suggestions,
                                confidence=TEXT_DETECTOR_CONFIDENCES["medial_order"],
                                error_type=ET_MEDIAL_ORDER_ERROR,
                            )
                        )

        return errors

    def _detect_leading_vowel_e(self, text: str) -> list[SyllableError]:
        """Detect Zawgyi-style leading vowel-e (U+1031 before consonant).

        In Unicode, the e-vowel ေ comes AFTER the consonant it modifies
        even though it renders to the left.
        """
        errors: list[SyllableError] = []
        vowel_e = "\u1031"
        current_idx = 0

        for word in text.split():
            idx = text.find(word, current_idx)
            if idx >= 0:
                current_idx = idx + len(word)
            if not word or word[0] != vowel_e or len(word) < 2:
                continue
            next_ch = ord(word[1])
            if not (0x1000 <= next_ch <= 0x1021):
                continue
            j = 2
            while j < len(word) and "\u103b" <= word[j] <= "\u103e":
                j += 1
            reordered = word[1:j] + vowel_e + word[j:]
            if idx >= 0:
                errors.append(
                    SyllableError(
                        text=word,
                        position=idx,
                        suggestions=[reordered],
                        confidence=TEXT_DETECTOR_CONFIDENCES["leading_vowel_e"],
                        error_type=ET_LEADING_VOWEL_E,
                    )
                )
        return errors

    def _detect_vowel_reorder_errors(self, text: str) -> list[SyllableError]:
        """Detect vowel ordering errors that normalization silently fixes.

        Must run BEFORE normalization to catch the wrong form.
        """
        errors: list[SyllableError] = []
        for wrong, corrections in self._VOWEL_REORDER_ERRORS.items():
            for idx, _end in iter_occurrences(text, wrong):
                errors.append(
                    SyllableError(
                        text=wrong,
                        position=idx,
                        suggestions=list(corrections),
                        confidence=TEXT_DETECTOR_CONFIDENCES.get("vowel_reorder", 0.95),
                        error_type=ET_SYLLABLE,
                    )
                )
        return errors

    def _detect_vowel_medial_reorder(self, text: str) -> list[SyllableError]:
        """Detect vowel sign appearing before a medial sign (wrong order).

        In canonical Myanmar encoding, medial signs (U+103B-U+103E) always
        precede vowel signs (U+102D-U+1030).  When a vowel appears before
        a medial, it is a keyboard-input error that normalization silently
        fixes.  This detector must run BEFORE normalization to flag the error.

        Example: ရိှ (vowel-i + ha-htoe) → ရှိ (ha-htoe + vowel-i)
        """
        errors: list[SyllableError] = []
        found_positions: set[int] = set()

        i = 0
        while i < len(text) - 1:
            ch = text[i]
            next_ch = text[i + 1]
            if ch in self._VOWEL_SIGNS_BEFORE_MEDIAL and next_ch in self._MYANMAR_MEDIALS:
                # Found a vowel-before-medial error.  Expand to full syllable.
                syl_start = i
                while syl_start > 0:
                    prev_cp = ord(text[syl_start - 1])
                    if prev_cp < 0x1000 or prev_cp > 0x109F:
                        break
                    if 0x1000 <= prev_cp <= 0x1021:
                        syl_start -= 1
                        break
                    syl_start -= 1

                syl_end = i + 2
                while syl_end < len(text):
                    cp = ord(text[syl_end])
                    if 0x102B <= cp <= 0x103E or cp in (0x1039, 0x103A, 0x1036, 0x1037, 0x1038):
                        syl_end += 1
                    elif (
                        0x1000 <= cp <= 0x1021
                        and syl_end + 1 < len(text)
                        and ord(text[syl_end + 1]) == 0x103A
                    ):
                        syl_end += 2
                    else:
                        break

                if syl_start not in found_positions:
                    found_positions.add(syl_start)
                    error_text = text[syl_start:syl_end]
                    # Swap the vowel and medial to produce the corrected form.
                    corrected = error_text.replace(ch + next_ch, next_ch + ch, 1)

                    if corrected != error_text:
                        errors.append(
                            SyllableError(
                                text=error_text,
                                position=syl_start,
                                suggestions=[corrected],
                                confidence=TEXT_DETECTOR_CONFIDENCES.get(
                                    "vowel_medial_reorder", 0.95
                                ),
                                error_type=ET_MEDIAL_ORDER_ERROR,
                            )
                        )
                i = syl_end
            else:
                i += 1

        return errors

    def _detect_incomplete_stacking(self, text: str) -> list[SyllableError]:
        """Detect incomplete Pali/Sanskrit stacking (virama present, stacked consonant missing).

        Must run BEFORE normalization because normalization reorders virama+vowel
        sequences, making the original pattern undetectable.
        """
        errors: list[SyllableError] = []
        for wrong, correct in self._STACKING_COMPLETIONS.items():
            for idx, _end in iter_occurrences(text, wrong):
                errors.append(
                    SyllableError(
                        text=wrong,
                        position=idx,
                        suggestions=[correct],
                        confidence=TEXT_DETECTOR_CONFIDENCES["incomplete_stacking"],
                        error_type=ET_INCOMPLETE_STACKING,
                    )
                )
        return errors

    def _detect_duplicate_diacritics(self, text: str) -> list[SyllableError]:
        """Detect consecutive duplicate diacritics BEFORE normalization.

        Example: ကိုု (U+102F appears twice) → ကို.
        """
        errors: list[SyllableError] = []
        found_positions: set[int] = set()

        i = 0
        while i < len(text) - 1:
            ch = text[i]
            if ch in self._DIACRITICS_CHECKED_FOR_DUPLICATES and text[i + 1] == ch:
                syl_start = i
                while syl_start > 0:
                    prev_cp = ord(text[syl_start - 1])
                    if prev_cp < 0x1000 or prev_cp > 0x109F:
                        break
                    if 0x1000 <= prev_cp <= 0x1021:
                        syl_start -= 1
                        break
                    syl_start -= 1

                syl_end = i + 2
                while syl_end < len(text) and text[syl_end] == ch:
                    syl_end += 1
                while syl_end < len(text):
                    cp = ord(text[syl_end])
                    if 0x102B <= cp <= 0x103E or cp in (0x1039, 0x103A, 0x1036, 0x1037, 0x1038):
                        syl_end += 1
                    elif (
                        0x1000 <= cp <= 0x1021
                        and syl_end + 1 < len(text)
                        and ord(text[syl_end + 1]) == 0x103A
                    ):
                        syl_end += 2
                    else:
                        break

                if syl_start not in found_positions:
                    found_positions.add(syl_start)
                    error_text = text[syl_start:syl_end]
                    corrected = error_text.replace(ch + ch, ch)
                    if corrected != error_text:
                        errors.append(
                            SyllableError(
                                text=error_text,
                                position=syl_start,
                                suggestions=[corrected],
                                confidence=TEXT_DETECTOR_CONFIDENCES["duplicate_diacritic"],
                                error_type=ET_SYLLABLE,
                            )
                        )
                i = syl_end
            else:
                i += 1

        return errors

    def _detect_vowel_after_dotbelow(self, text: str) -> list[Error]:
        """Detect dot-below (U+1037) followed by a vowel sign — pre-normalization.

        Must run BEFORE normalization because normalization reorders vowel signs,
        destroying the evidence that the vowel was spurious.
        """
        dot_below = "\u1037"
        errors: list[Error] = []
        reported_positions: set[int] = set()

        i = 0
        while i < len(text) - 1:
            if text[i] == dot_below and text[i + 1] in self._VOWEL_SIGNS:
                start = i
                while start > 0 and text[start - 1] not in (" ", "\u104a", "\u104b"):
                    start -= 1
                end = i + 2
                while end < len(text) and text[end] not in (" ", "\u104a", "\u104b"):
                    end += 1

                if start in reported_positions:
                    i = end
                    continue

                word = text[start:end]
                vowel_offset = (i + 1) - start
                corrected = word[:vowel_offset] + word[vowel_offset + 1 :]

                if corrected != word:
                    errors.append(
                        SyllableError(
                            text=word,
                            position=start,
                            suggestions=[corrected],
                            confidence=TEXT_DETECTOR_CONFIDENCES["vowel_after_asat"],
                            error_type=ET_VOWEL_AFTER_ASAT,
                        )
                    )
                    reported_positions.add(start)
                i = end
            else:
                i += 1

        return errors
