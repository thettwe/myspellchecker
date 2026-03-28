"""SyllableValidator - Layer 1 validation for individual syllables."""

from __future__ import annotations

from myspellchecker.algorithms import SymSpell
from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.constants import (
    ET_COLLOQUIAL_INFO,
    ET_COLLOQUIAL_VARIANT,
    ET_MEDIAL_CONFUSION,
    ET_PARTICLE_TYPO,
    ET_SYLLABLE,
    KINZI_SEQUENCE,
    ValidationLevel,
)
from myspellchecker.core.detector_data import TEXT_DETECTOR_CONFIDENCES
from myspellchecker.core.response import Error, SyllableError
from myspellchecker.core.syllable_rules import SyllableRuleValidator
from myspellchecker.core.validators.base import Validator
from myspellchecker.grammar.patterns import (
    get_medial_confusion_correction,
    get_particle_typo_correction,
)
from myspellchecker.providers.interfaces import SyllableRepository
from myspellchecker.segmenters import Segmenter
from myspellchecker.text.phonetic_data import get_standard_forms, is_colloquial_variant
from myspellchecker.utils.logging_utils import get_logger

# Module logger
logger = get_logger(__name__)

# Informal pronouns that are register-critical — colloquial_info should still
# be emitted for these even when high-frequency, since they signal informal register.
_REGISTER_CRITICAL_PRONOUNS = frozenset({"ငါ"})


class SyllableValidator(Validator):
    """
    Validator for syllable-level errors (Layer 1).

    This validator now includes:
    - Particle typo detection (high-confidence rule-based patterns)
    - Structural syllable validation
    - Dictionary validation

    **Interface Segregation**:
        This validator depends on SyllableRepository instead of the full
        DictionaryProvider. It only uses:
        - is_valid_syllable(syllable: str) -> bool
        - get_syllable_frequency(syllable: str) -> int

        This makes the validator more testable and reduces coupling.
    """

    def __init__(
        self,
        config: SpellCheckerConfig,
        segmenter: Segmenter,
        repository: SyllableRepository,
        symspell: SymSpell | None,
        syllable_rule_validator: SyllableRuleValidator | None = None,
    ):
        """Initialize with validation components.

        ``symspell`` may be None (skip_init=True) — disables suggestion generation.
        ``syllable_rule_validator`` is optional — without it, only dictionary lookup runs.
        """
        super().__init__(config)
        self.segmenter = segmenter
        self.repository = repository
        self.symspell = symspell  # Can be None when symspell.skip_init is True
        self.syllable_rule_validator = syllable_rule_validator

        # Bare-consonant merge guard threshold from config
        self._BARE_CONSONANT_MERGE_MIN_FREQ = config.validation.bare_consonant_merge_min_freq

    def _check_particle_typo(self, syllable: str) -> tuple[str, str, float] | None:
        """
        Check if a syllable is a known particle typo.

        Uses the PARTICLE_TYPO_PATTERNS from grammar.patterns which contains
        50+ high-confidence patterns for common particle typos like:
        - Missing asat: တယ → တယ်, သည → သည်
        - Missing tone: ပြီ → ပြီး, ခဲ → ခဲ့

        Args:
            syllable: The syllable to check.

        Returns:
            Tuple of (correction, description, confidence) if typo found, None otherwise.
        """
        return get_particle_typo_correction(syllable)

    def _check_medial_confusion(self, syllable: str) -> tuple[str, str, str] | None:
        """
        Check if a syllable has ya-pin (ျ) vs ya-yit (ြ) confusion.

        Uses the MEDIAL_CONFUSION_PATTERNS from grammar.patterns which contains
        patterns for common confusions like:
        - ကျောင်း vs ကြောင်း (school vs because)
        - ကြေးဇူး vs ကျေးဇူး (thanks)

        Args:
            syllable: The syllable to check.

        Returns:
            Tuple of (correction, description, context_hint) if confusion found, None otherwise.
        """
        return get_medial_confusion_correction(syllable)

    # Myanmar aspirated/unaspirated consonant pairs (unaspirated → aspirated)
    _ASPIRATION_PAIRS: dict[str, str] = {
        "\u1000": "\u1001",  # က → ခ (ka → kha)
        "\u1005": "\u1006",  # စ → ဆ (sa → hsa)
        "\u1010": "\u1011",  # တ → ထ (ta → hta)
        "\u1015": "\u1016",  # ပ → ဖ (pa → pha)
    }

    def _check_aspiration_confusion(self, syllable: str) -> str | None:
        """Check if substituting the initial consonant's aspiration yields a valid syllable.

        Common Myanmar typing error: unaspirated consonant used where
        aspirated was intended (e.g., ကေါင်း → ခေါင်း, head).
        Only checks the first character (initial consonant).
        Requires the candidate to have significant frequency to avoid
        suggesting rare/noise entries.

        Returns the corrected syllable if valid, None otherwise.
        """
        if not syllable:
            return None
        initial = syllable[0]
        aspirated = self._ASPIRATION_PAIRS.get(initial)
        if aspirated:
            candidate = aspirated + syllable[1:]
            if self.repository.is_valid_syllable(candidate):
                freq = self.repository.get_syllable_frequency(candidate)
                if freq >= self.config.symspell.count_threshold:
                    return candidate
        return None

    def _check_colloquial_variant(self, text: str, position: int) -> SyllableError | None:
        """
        Check if text is a colloquial spelling variant.

        Handles colloquial variants based on the `colloquial_strictness` config:
        - 'strict': Flag as error with standard forms as suggestions
        - 'lenient': Return info note with low confidence
        - 'off': No handling, return None

        Args:
            text: The syllable or word to check.
            position: Position in the original text.

        Returns:
            SyllableError if colloquial variant detected (based on strictness),
            None otherwise.
        """
        strictness = self.config.validation.colloquial_strictness

        if strictness == "off":
            return None

        if not is_colloquial_variant(text):
            return None

        standard_forms = sorted(get_standard_forms(text))
        if not standard_forms:
            return None

        if strictness == "strict":
            return SyllableError(
                text=text,
                position=position,
                suggestions=standard_forms,
                confidence=self.config.validation.syllable_error_confidence,
                error_type=ET_COLLOQUIAL_VARIANT,
            )
        elif strictness == "lenient":
            # Suppress informational notes for very high-frequency syllables.
            # Exception: informal pronouns like ငါ are register-critical —
            # they signal informal register even when high-frequency, so
            # colloquial_info should still be emitted for them.
            if hasattr(self.repository, "get_word_frequency"):
                syl_freq = self.repository.get_word_frequency(text)
                threshold = self.config.frequency_guards.colloquial_high_freq_suppression
                if (
                    isinstance(syl_freq, (int, float))
                    and syl_freq >= threshold
                    and text not in _REGISTER_CRITICAL_PRONOUNS
                ):
                    return None
            return SyllableError(
                text=text,
                position=position,
                suggestions=standard_forms,
                confidence=self.config.validation.colloquial_info_confidence,
                error_type=ET_COLLOQUIAL_INFO,
            )

        # Explicit return for type safety - should not reach here due to Literal type
        # constraint, but provides defensive programming and clear control flow
        return None

    def validate(self, text: str) -> list[Error]:
        """
        Validate text at syllable level and return errors found.

        This method segments the input text into syllables and validates
        each one against:
            1. Structural rules (Myanmar orthographic patterns)
            2. Dictionary lookup (known valid syllables)
            3. Frequency threshold (filters out rare/typo syllables)
            4. Pattern matching (particle typos, medial confusion)
            5. Colloquial variant detection

        Args:
            text: Normalized Myanmar text to validate. Should be preprocessed
                with normalization (e.g., via ``normalize_for_lookup()``).

        Returns:
            List of SyllableError objects, each containing:
                - text: The invalid syllable
                - position: Character position in input text
                - suggestions: Possible corrections (may be empty)
                - confidence: Error confidence score (0.0-1.0)
                - error_type: Type of error detected

        Example:
            >>> validator = SyllableValidator.create(
            ...     repository=provider,
            ...     segmenter=segmenter,
            ...     symspell=symspell,
            ... )
            >>> errors = validator.validate("မြန်မာစာ")
            >>> for error in errors:
            ...     print(f"{error.text}: {error.suggestions}")
        """
        errors: list[Error] = []
        syllables = self.segmenter.segment_syllables(text)
        current_idx = 0

        for syllable in syllables:
            position, current_idx = self._find_syllable_position(text, syllable, current_idx)
            if position is None:
                continue

            error = self._validate_single_syllable(syllable, position, text)
            if error:
                errors.append(error)

        # Post-process: merge adjacent invalid syllables that form valid
        # syllables when combined with asat (e.g., 'တ'+'ယ' → 'တယ်')
        errors = self._try_merge_adjacent_errors(errors)

        # Post-process: suppress bare-consonant errors that are segmenter
        # artifacts (the consonant + adjacent word forms a valid word)
        errors = self._suppress_bare_consonant_artifacts(errors, text)

        return errors

    def _find_syllable_position(
        self, text: str, syllable: str, current_idx: int
    ) -> tuple[int | None, int]:
        """Find syllable position in text, returning (position, next_idx)."""
        return self._find_token_position(text, syllable, current_idx)

    def _validate_single_syllable(
        self, syllable: str, position: int, text: str = ""
    ) -> SyllableError | None:
        """Validate a single syllable and return error if invalid."""
        if not self._should_validate(syllable):
            return None

        is_valid, is_in_dictionary = self._check_validity(syllable, position, text)

        # Check for known patterns if not in dictionary
        if not is_in_dictionary:
            pattern_error = self._check_known_patterns(syllable, position)
            if pattern_error:
                return pattern_error

        if not is_valid:
            return self._create_syllable_error(syllable, position)

        # Check for colloquial variants
        # Even valid syllables may be colloquial variants that should be flagged
        colloquial_error = self._check_colloquial_variant(syllable, position)
        if colloquial_error:
            return colloquial_error

        return None

    def _should_validate(self, syllable: str) -> bool:
        """Check if syllable should be validated."""
        is_not_empty = bool(syllable.strip())
        is_not_punctuation = not self.is_punctuation(syllable)
        is_myanmar = self._is_myanmar_with_config(syllable)
        return is_not_empty and is_not_punctuation and is_myanmar

    def _check_validity(
        self, syllable: str, position: int = 0, text: str = ""
    ) -> tuple[bool, bool]:
        """Check structural and dictionary validity. Returns (is_valid, is_in_dictionary)."""
        # Step 1: Rule-based structural check
        if self.syllable_rule_validator and not self.syllable_rule_validator.validate(syllable):
            # Tokens that fail rule validation may still be valid words:
            # - Multi-syllable segmenter artifacts (2+ asats, e.g. အက်ပ် "app")
            # - Loanwords with unusual structure (e.g. ဘေ့စ် "base", မေးလ် "mail")
            # Accept via word fallback if the token is in dictionary.
            if self._try_word_fallback(syllable, position, text):
                return True, True
            return False, False

        # Step 2: Dictionary check
        is_in_dictionary = self.repository.is_valid_syllable(syllable)
        if not is_in_dictionary:
            # Kinzi syllables (containing င်္) are valid Pali/Sanskrit loanword
            # components that may not appear in the syllable dictionary.
            if KINZI_SEQUENCE in syllable:
                return True, False
            # Fallback: accept if the token is a valid word at a word boundary.
            if self._try_word_fallback(syllable, position, text):
                return True, True
            return False, False

        # Step 3: Frequency threshold check
        freq = self.repository.get_syllable_frequency(syllable)
        if freq < self.config.symspell.count_threshold:
            # Kinzi syllables may have zero syllable frequency.
            if KINZI_SEQUENCE in syllable:
                return True, False
            # Fallback: zero-freq syllables that are valid words at word boundaries.
            if self._try_word_fallback(syllable, position, text):
                return True, True
            return False, False

        return True, True

    def _try_word_fallback(self, syllable: str, position: int, text: str) -> bool:
        """Check if a syllable is a valid word at a word boundary.

        For single bare consonants (e.g., က, မ), enforces standalone particle
        check to prevent fragment false-negatives. For multi-char syllables,
        accepts directly if the word exists in the dictionary.

        Returns True if the syllable should be accepted as a valid word.
        """
        if not hasattr(self.repository, "is_valid_word"):
            return False
        if not self.repository.is_valid_word(syllable):
            return False
        # Single bare consonant? Enforce standalone particle check to reject fragments.
        if len(syllable) == 1 and 0x1000 <= ord(syllable[0]) <= 0x1021:
            return self._is_standalone_particle(text, position, len(syllable))
        return True

    @staticmethod
    def _is_standalone_particle(text: str, position: int, length: int) -> bool:
        """Check if a bare-consonant token is a standalone particle, not a fragment.

        A single bare consonant (like 'က', 'ယ') might be:
          (a) a standalone particle (e.g., 'က' in 'သူက လာတယ်')
          (b) a fragment of a misspelled syllable (e.g., 'တ'+'ယ' in 'ပါတယ')

        Fragments appear as two adjacent bare consonants (no medial/vowel/asat
        between them).  If the immediately preceding or following character is
        also a Myanmar consonant (U+1000-U+1021), treat this token as a likely
        fragment and deny the word-level fallback so that the merge step
        can combine them.

        When no text is provided, defaults to True for backward compatibility.
        """
        if not text:
            return True
        end = position + length

        if position > 0:
            prev = ord(text[position - 1])
            if 0x1000 <= prev <= 0x1021:
                return False

        if end < len(text):
            nxt = ord(text[end])
            if 0x1000 <= nxt <= 0x1021:
                return False

        return True

    # Myanmar digit zero ↔ letter wa (visually identical, different codepoints)
    _DIGIT_LETTER_MAP: dict[str, str] = {"\u1040": "\u101d"}  # ၀ → ဝ

    @staticmethod
    def _check_broken_virama(syllable: str) -> str | None:
        """Check for virama (U+1039) followed by vowel instead of consonant.

        A virama should stack a following consonant (e.g. က္က). When it appears
        before a vowel sign, the virama is spurious and removing it produces the
        correct syllable.
        """
        virama = "\u1039"
        if virama not in syllable:
            return None
        idx = syllable.index(virama)
        if idx + 1 < len(syllable):
            next_char = syllable[idx + 1]
            # Vowel signs: U+102B-U+1032, plus e-vowel U+1031
            if ("\u102b" <= next_char <= "\u1032") or next_char == "\u1031":
                return syllable[:idx] + syllable[idx + 1 :]
        return None

    def _check_known_patterns(self, syllable: str, position: int) -> SyllableError | None:
        """Check for particle typos, digit-letter confusion, and medial confusion."""
        # Check broken virama (virama + vowel → remove virama)
        virama_fix = self._check_broken_virama(syllable)
        if virama_fix and self.repository.is_valid_syllable(virama_fix):
            return SyllableError(
                text=syllable,
                position=position,
                suggestions=[virama_fix],
                confidence=TEXT_DETECTOR_CONFIDENCES["broken_virama_fix"],
                error_type=ET_SYLLABLE,
            )

        # Check digit-letter confusion (၀ → ဝ) — flag just the digit character
        for digit, letter in self._DIGIT_LETTER_MAP.items():
            if digit in syllable:
                corrected = syllable.replace(digit, letter)
                if self.repository.is_valid_syllable(corrected):
                    digit_idx = syllable.index(digit)
                    return SyllableError(
                        text=digit,
                        position=position + digit_idx,
                        suggestions=[letter],
                        confidence=TEXT_DETECTOR_CONFIDENCES["digit_letter_confusion"],
                        error_type=ET_SYLLABLE,
                    )

        # Check particle typos (high-confidence patterns)
        particle_typo = self._check_particle_typo(syllable)
        if particle_typo:
            correction, _, confidence = particle_typo
            return SyllableError(
                text=syllable,
                position=position,
                suggestions=[correction],
                confidence=confidence,
                error_type=ET_PARTICLE_TYPO,
            )

        # Check medial confusion (ျ vs ြ patterns)
        medial_confusion = self._check_medial_confusion(syllable)
        if medial_confusion:
            correction, _, _ = medial_confusion
            return SyllableError(
                text=syllable,
                position=position,
                suggestions=[correction],
                confidence=self.config.validation.medial_confusion_confidence,
                error_type=ET_MEDIAL_CONFUSION,
            )

        # Check aspirated/unaspirated consonant confusion (က↔ခ, စ↔ဆ, etc.)
        aspiration_correction = self._check_aspiration_confusion(syllable)
        if aspiration_correction:
            return SyllableError(
                text=syllable,
                position=position,
                suggestions=[aspiration_correction],
                confidence=TEXT_DETECTOR_CONFIDENCES["aspiration_confusion"],
                error_type=ET_SYLLABLE,
            )

        return None

    def _create_syllable_error(self, syllable: str, position: int) -> SyllableError:
        """Create a syllable error with suggestions from SymSpell."""
        if self.symspell is not None:
            symspell_suggestions = self.symspell.lookup(
                syllable,
                level=ValidationLevel.SYLLABLE.value,
                max_suggestions=self.config.max_suggestions,
                use_phonetic=self.config.use_phonetic,
            )
            suggestions = self._filter_suggestions([s.term for s in symspell_suggestions])
        else:
            suggestions = []

        return SyllableError(
            text=syllable,
            position=position,
            suggestions=suggestions,
            confidence=self.config.validation.syllable_error_confidence,
        )

    def _try_merge_adjacent_errors(self, errors: list[Error]) -> list[Error]:
        """Merge adjacent invalid syllables that form a valid syllable with asat.

        Myanmar text like 'တယ' (missing asat) gets segmented into two bare
        consonants ['တ', 'ယ']. Neither syllable alone can suggest 'တယ်'.
        This pass tries merging consecutive error syllables and checks if
        the combination + asat is a known valid syllable.

        Returns a new error list with merged errors where applicable.
        """
        if len(errors) < 2:
            return errors

        asat = "\u103a"  # ်
        result: list[Error] = []
        i = 0

        while i < len(errors):
            if i + 1 < len(errors):
                e1 = errors[i]
                e2 = errors[i + 1]

                # Both must be invalid_syllable errors at adjacent positions
                if (
                    e1.error_type == "invalid_syllable"
                    and e2.error_type == "invalid_syllable"
                    and e2.position == e1.position + len(e1.text)
                ):
                    merged_text = e1.text + e2.text
                    # If merged text (without asat) is already a valid word,
                    # suppress both errors — it's a valid compound, not an error.
                    # e.g., မ+ရ = မရ ("can't get"), freq may be 0 but still valid.
                    if hasattr(self.repository, "is_valid_word") and self.repository.is_valid_word(
                        merged_text
                    ):
                        i += 2
                        continue

                    merged = merged_text + asat
                    if self.repository.is_valid_syllable(merged):
                        merged_suggestions = [merged]
                        # Keep original suggestions as fallback
                        for s in e1.suggestions:
                            if s not in merged_suggestions:
                                merged_suggestions.append(s)
                        result.append(
                            SyllableError(
                                text=e1.text + e2.text,
                                position=e1.position,
                                suggestions=merged_suggestions[: self.config.max_suggestions],
                                confidence=self.config.validation.syllable_error_confidence,
                                error_type=ET_SYLLABLE,
                            )
                        )
                        i += 2
                        continue

            result.append(errors[i])
            i += 1

        return result

    def _suppress_bare_consonant_artifacts(self, errors: list[Error], text: str) -> list[Error]:
        """Suppress single bare-consonant invalid_syllable errors from segmenter splits.

        When the segmenter incorrectly splits a word, it can produce a single
        Myanmar consonant (U+1000-U+1021) as a standalone token.  If
        concatenating that consonant with the adjacent word in the original
        text produces a valid high-frequency word, the consonant is a
        segmenter artifact, not a user error.

        Examples:
            - ``ရ`` from ``ရတယ်`` (split of ရတယ်, freq=1.7M)
            - ``ထ`` from ``ထမင်း`` (split of ထမင်း, freq=40K)
            - ``အ`` from ``အစီအစဉ်`` (split of the compound)
        """
        if not errors:
            return errors
        if not hasattr(self.repository, "is_valid_word"):
            return errors
        if not hasattr(self.repository, "get_word_frequency"):
            return errors

        tokens = text.split()
        if not tokens:
            return errors

        error_positions = {e.position for e in errors}
        error_end_positions = {e.position + len(e.text) for e in errors}

        # Map error positions to token context
        result: list[Error] = []
        for error in errors:
            if self._is_bare_consonant_artifact(
                error, text, tokens, error_positions, error_end_positions
            ):
                continue
            result.append(error)

        return result

    def _is_bare_consonant_artifact(
        self,
        error: Error,
        text: str,
        tokens: list[str],
        error_positions: set[int],
        error_end_positions: set[int],
    ) -> bool:
        """Check if an invalid_syllable error is a segmenter artifact."""
        if error.error_type != ET_SYLLABLE:
            return False

        # Check multi-char merged errors (e.g., နက from ဌာန+က compound boundary)
        # If the merged error can be split into chars where each is a valid word
        # AND the preceding text + first char forms a valid word, it's a compound
        # boundary, not a real error.
        if len(error.text) >= 2 and all(0x1000 <= ord(c) <= 0x1021 for c in error.text):
            if self._is_compound_boundary_artifact(error, text):
                return True

        if len(error.text) != 1:
            return False
        cp = ord(error.text[0])
        if not (0x1000 <= cp <= 0x1021):
            return False

        pos = error.position

        # Check 1: Adjacent error suppression — if another error immediately
        # follows or precedes this bare consonant, both are likely fragments
        # of the same broken word (e.g., ဘ + ဏ္ာ from broken ဘဏ္ဍာ).
        bare_end = pos + len(error.text)
        if bare_end in error_positions:
            return True
        if pos in error_end_positions:
            return True

        # Check 2: consonant + next_word → valid high-freq word?
        # Try the full next token first, then progressively shorter prefixes
        # to handle compounds like မ+သွားချင်ဘူ where only မသွား is in dict.
        end = pos + 1
        # Skip spaces after the consonant
        while end < len(text) and text[end] == " ":
            end += 1
        if end < len(text):
            # Grab the next word (up to next space or end)
            next_end = end
            while next_end < len(text) and text[next_end] != " ":
                next_end += 1
            next_word = text[end:next_end]
            if next_word:
                # Try full next word, then progressively shorter prefixes
                for trim in range(len(next_word)):
                    prefix = next_word[: len(next_word) - trim] if trim else next_word
                    merged = error.text + prefix
                    if self.repository.is_valid_word(merged):  # type: ignore[attr-defined]
                        freq = self.repository.get_word_frequency(merged)  # type: ignore[attr-defined]
                        if (
                            isinstance(freq, (int, float))
                            and freq >= self._BARE_CONSONANT_MERGE_MIN_FREQ
                        ):
                            return True

        # Check 3: prev_word + consonant → valid high-freq word?
        # Try the full prev token, then progressively shorter suffixes
        # to handle compounds like အစည်း+အ where only a suffix+consonant is in dict.
        start = pos - 1
        # Skip spaces before the consonant
        while start >= 0 and text[start] == " ":
            start -= 1
        if start >= 0:
            prev_start = start
            while prev_start > 0 and text[prev_start - 1] != " ":
                prev_start -= 1
            prev_word = text[prev_start : start + 1]
            if prev_word:
                # Try full prev word, then progressively shorter suffixes
                for trim in range(len(prev_word)):
                    suffix = prev_word[trim:] if trim else prev_word
                    merged = suffix + error.text
                    if self.repository.is_valid_word(merged):  # type: ignore[attr-defined]
                        freq = self.repository.get_word_frequency(merged)  # type: ignore[attr-defined]
                        if (
                            isinstance(freq, (int, float))
                            and freq >= self._BARE_CONSONANT_MERGE_MIN_FREQ
                        ):
                            return True

        return False

    def _is_compound_boundary_artifact(self, error: Error, text: str) -> bool:
        """Check if a merged bare-consonant error spans a compound boundary.

        For merged errors like 'နက' (from ဌာန+က), checks whether:
        1. Each character is a valid word (န, က)
        2. Preceding text + first char forms a valid high-freq word (ဌာ+န=ဌာန)

        This detects compound word + particle boundaries where the syllable
        tokenizer split a compound's final consonant from the body, then the
        merge step recombined it with the following particle.
        """
        pos = error.position
        err_text = error.text

        # All chars must be valid words
        if not all(self.repository.is_valid_word(c) for c in err_text):  # type: ignore[attr-defined]
            return False

        # Walk backward from error position to find the preceding syllable
        # (skip non-Myanmar chars like spaces)
        start = pos - 1
        while start >= 0 and text[start] == " ":
            start -= 1
        if start < 0:
            return False

        # Find the start of the preceding syllable/word
        prev_start = start
        while prev_start > 0 and text[prev_start - 1] != " ":
            prev_start -= 1
        preceding = text[prev_start : start + 1]
        if not preceding:
            return False

        # Check if preceding (or suffix of it) + first char = valid word
        # with freq >= threshold.  In long compounds like
        # ကျန်းမာရေးနှင့်အားကစားဝန်ကြီးဌာ, the full string isn't in the
        # dictionary, but a suffix like ဌာ + န = ဌာန is.
        first_char = err_text[0]
        for trim in range(len(preceding)):
            suffix = preceding[trim:] if trim else preceding
            merged = suffix + first_char
            if self.repository.is_valid_word(merged):  # type: ignore[attr-defined]
                freq = self.repository.get_word_frequency(merged)  # type: ignore[attr-defined]
                if isinstance(freq, (int, float)) and freq >= self._BARE_CONSONANT_MERGE_MIN_FREQ:
                    return True

        return False

    @classmethod
    def create(
        cls,
        repository: SyllableRepository,
        segmenter: Segmenter,
        symspell: SymSpell | None,
        config: SpellCheckerConfig | None = None,
        syllable_rule_validator: SyllableRuleValidator | None = None,
    ) -> "SyllableValidator":
        """
        Factory method for creating SyllableValidator instances.

        Provides a convenient way to create validators with sensible defaults.

        Args:
            repository: SyllableRepository for syllable lookup.
            segmenter: Segmenter for text tokenization.
            symspell: SymSpell instance for suggestions.
            config: Configuration (uses defaults if None).
            syllable_rule_validator: Optional rule validator for orthographic rules.

        Returns:
            Configured SyllableValidator instance.

        Example:
            >>> from myspellchecker.providers import MemoryProvider
            >>> from myspellchecker.segmenters import DefaultSegmenter
            >>> from myspellchecker.algorithms import SymSpell
            >>>
            >>> provider = MemoryProvider(syllables={"မြန်": 100})
            >>> segmenter = DefaultSegmenter()
            >>> symspell = SymSpell()
            >>> validator = SyllableValidator.create(
            ...     repository=provider,
            ...     segmenter=segmenter,
            ...     symspell=symspell,
            ... )
            >>> errors = validator.validate("မြန်မာ")
        """
        if config is None:
            config = SpellCheckerConfig()
        return cls(
            config=config,
            segmenter=segmenter,
            repository=repository,
            symspell=symspell,
            syllable_rule_validator=syllable_rule_validator,
        )
