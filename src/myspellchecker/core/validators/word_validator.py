"""WordValidator - Layer 2 validation for multi-syllable words."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from myspellchecker.text.compound_resolver import CompoundResolver
    from myspellchecker.text.reduplication import ReduplicationEngine

from myspellchecker.algorithms import (
    NgramContextChecker,
    SymSpell,
)
from myspellchecker.algorithms.suggestion_strategy import (
    SuggestionContext,
    SuggestionStrategy,
)
from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.constants import ET_COLLOQUIAL_INFO, ET_COLLOQUIAL_VARIANT
from myspellchecker.core.loan_word_variants import get_loan_word_standard
from myspellchecker.core.response import Error, WordError
from myspellchecker.core.token_refinement import build_validation_token_paths
from myspellchecker.core.validators.base import Validator
from myspellchecker.providers.interfaces import SyllableRepository, WordRepository
from myspellchecker.segmenters import Segmenter
from myspellchecker.text.morphology import WordAnalysis, analyze_word
from myspellchecker.text.phonetic_data import get_standard_forms, is_colloquial_variant
from myspellchecker.utils.logging_utils import get_logger

# Module logger
logger = get_logger(__name__)

# Regex for stripping non-Myanmar punctuation attached to token boundaries.
# Matches leading/trailing quotes, brackets, colons, periods, etc.
_BOUNDARY_PUNCT_RE = re.compile(
    r'^["\'\u201c\u201d\u2018\u2019"()\[\]{},;:.\u2026\-\u2013\u2014/\\]+|'
    r'["\'\u201c\u201d\u2018\u2019"()\[\]{},;:.\u2026\-\u2013\u2014/\\]+$'
)

# Informal pronouns that are register-critical -- colloquial_info should still
# be emitted for these even when high-frequency, since they signal informal register.
_REGISTER_CRITICAL_PRONOUNS: frozenset[str] = frozenset({"\u1004\u102b"})  # ငါ


class WordValidator(Validator):
    """
    Validator for word-level errors (Layer 2).

    This validator includes OOV (Out-of-Vocabulary) recovery through
    morphological analysis. When a word is not found in the dictionary,
    it attempts to decompose the word into a known root + suffixes,
    improving suggestion quality for inflected or derived words.

    **Interface Segregation**:
        This validator depends on WordRepository and SyllableRepository
        instead of the full DictionaryProvider. It uses:
        - WordRepository: is_valid_word, get_word_frequency, get_all_words
        - SyllableRepository: is_valid_syllable (for morphological analysis)

        This makes the validator more testable and reduces coupling.
    """

    def __init__(
        self,
        config: SpellCheckerConfig,
        segmenter: Segmenter,
        word_repository: WordRepository,
        syllable_repository: SyllableRepository,
        symspell: SymSpell | None,
        context_checker: NgramContextChecker | None = None,
        suggestion_strategy: SuggestionStrategy | None = None,
        reduplication_engine: "ReduplicationEngine" | None = None,
        compound_resolver: "CompoundResolver" | None = None,
    ):
        """
        Initialize WordValidator with required components.

        Args:
            config: SpellCheckerConfig with validation settings including
                confidence thresholds, max_suggestions, and feature flags.
            segmenter: Segmenter instance for breaking text into words and
                syllables. Used for tokenization and morphological analysis.
            word_repository: WordRepository for word-level dictionary lookups.
                Provides ``is_valid_word()`` and ``get_word_frequency()`` methods.
            syllable_repository: SyllableRepository for syllable lookups.
                Used for morphological analysis (root extraction) when a word
                is not found in the word dictionary.
            symspell: SymSpell instance for generating correction suggestions.
                Can be None when ``config.symspell.skip_init=True``.
            context_checker: Optional NgramContextChecker for context-aware
                suggestion ranking. When provided, suggestions are ranked
                based on bigram probability with surrounding words.
            suggestion_strategy: Optional SuggestionStrategy for unified
                suggestion generation. Aggregates suggestions from multiple
                sources (SymSpell, morphology, compound analysis) with
                deduplication and ranking.
            reduplication_engine: Optional ReduplicationEngine for productive
                reduplication validation (e.g., ကောင်းကောင်း from ကောင်း).
            compound_resolver: Optional CompoundResolver for compound word
                synthesis validation (e.g., ကျောင်းသား = ကျောင်း + သား).
        """
        super().__init__(config)
        self.segmenter = segmenter
        self.word_repository = word_repository
        self.syllable_repository = syllable_repository
        self.symspell = symspell  # Can be None when symspell.skip_init is True
        self.context_checker = context_checker
        self.suggestion_strategy = suggestion_strategy
        self._reduplication_engine = reduplication_engine
        self._compound_resolver = compound_resolver

        # Segmenter-merge skip guard tuning:
        # when a multi-syllable OOV token is split into mostly-valid parts,
        # still validate it if SymSpell exposes a strong whole-word candidate.
        self._segment_skip_candidate_min_freq = config.validation.segment_skip_min_freq
        self._segment_skip_candidate_min_ratio = config.validation.segment_skip_min_ratio
        self._segment_skip_candidate_max_edit_distance = (
            config.validation.segment_skip_max_edit_distance
        )
        self._segment_skip_candidate_max_length = config.validation.segment_skip_max_length

    def _get_refinement_bigram_probability(self, first: str, second: str) -> float:
        """Return bigram probability for token refinement scoring."""
        if self.context_checker is None:
            return 0.0
        getter = getattr(self.context_checker, "get_bigram_probability", None)
        if getter is None:
            return 0.0
        try:
            value = getter(first, second)
        except (RuntimeError, ValueError, TypeError, AttributeError):
            return 0.0
        return float(value) if isinstance(value, (int, float)) else 0.0

    def _is_in_dictionary(self, word: str) -> bool:
        """
        Check if a word is in the dictionary (as word or syllable).

        This is used as the dictionary_check callback for morphological analysis.

        Args:
            word: The word to check.

        Returns:
            True if the word is valid in the dictionary.
        """
        return self.word_repository.is_valid_word(
            word
        ) or self.syllable_repository.is_valid_syllable(word)

    def _recover_oov_root(self, word: str) -> WordAnalysis | None:
        """
        Attempt to recover a known root from an OOV word using morphology.

        This analyzes the morphological structure of an unknown word to find
        a valid root by stripping suffixes. This is particularly useful for:
        - Inflected verbs: စားခဲ့သည် → root: စား (with suffixes ခဲ့, သည်)
        - Derived nouns: ဆရာဝန်များ → root: ဆရာဝန် (with suffix များ)

        Args:
            word: The OOV word to analyze.

        Returns:
            WordAnalysis if a valid root was recovered (root != original),
            None if no recovery was possible.
        """
        if not word:
            return None

        # Analyze word with dictionary validation
        analysis = analyze_word(word, dictionary_check=self._is_in_dictionary)

        if (
            analysis.root
            and analysis.root != analysis.original
            and analysis.suffixes
            and self._is_in_dictionary(analysis.root)
        ):
            return analysis

        return None

    # Helper methods to reduce cyclomatic complexity

    def _merge_probe_adjacent_pairs(self, words: list[str]) -> list[str]:
        """Rescue over-segmented multi-token fragments by probing merges.

        For each adjacent pair (a, b), check if the concatenation `a+b` is:
          1. A known loan-word variant (loan_words.yaml + loan_words_mined.yaml)
          2. A valid dict word (only accepts if at least one side is NOT
             already a valid word — guards legit adjacent-word pairs)
          3. A valid dict word with asat (U+103A) appended (missing-asat typo)
          4. A bigram-associated pair above
             ``validation.segmenter_merge_bigram_threshold``

        Greedy left-to-right: a merged token can be merged further with its
        new right neighbour (i.e. three fragments can collapse to one).

        Does not mutate the input list.

        Workstream: segmenter-post-merge-rescue / task: seg-probe-01.
        """
        if len(words) < 2:
            return words

        # Lazy import: avoid import cycle if loan_word_variants imports us.
        from myspellchecker.core.loan_word_variants import get_loan_word_standard

        ngram_provider = None
        if self.context_checker is not None:
            ngram_provider = getattr(self.context_checker, "provider", None)
        bigram_threshold = self.config.validation.segmenter_merge_bigram_threshold

        # Work on a copy so we don't mutate the caller's list.
        work = list(words)
        out: list[str] = []
        i = 0
        while i < len(work):
            if i + 1 >= len(work):
                out.append(work[i])
                i += 1
                continue

            a = work[i]
            b = work[i + 1]
            if not a or not b or not a.strip() or not b.strip():
                out.append(a)
                i += 1
                continue

            # Only merge Myanmar-char pairs — skip punctuation, latin tokens.
            if not (self._is_myanmar_with_config(a) and self._is_myanmar_with_config(b)):
                out.append(a)
                i += 1
                continue

            merged = a + b
            hit = False

            # Probe 1: variant map (highest confidence).
            if get_loan_word_standard(merged):
                hit = True

            # Probe 2: dict word with at-least-one-fragment-OOV guard.
            if not hit and self.word_repository.is_valid_word(merged):
                a_valid = self.word_repository.is_valid_word(a)
                b_valid = self.word_repository.is_valid_word(b)
                if not (a_valid and b_valid):
                    hit = True

            # Probe 3: dict word with asat (missing-asat typo).
            if not hit and self.word_repository.is_valid_word(merged + "\u103a"):
                hit = True

            # Probe 4: bigram association (weakest — gated by threshold).
            # Disabled by default (threshold < 0). Even with fragment-rarity
            # guards a 2026-04-18 sweep showed FPR regressions from 8% → 22%
            # when enabled. Kept in code for future calibration.
            if (
                not hit
                and bigram_threshold >= 0
                and ngram_provider is not None
                and hasattr(ngram_provider, "get_bigram_probability")
            ):
                freq_floor = 100  # fragment rarity threshold
                fa = self.word_repository.get_word_frequency(a) or 0
                fb = self.word_repository.get_word_frequency(b) or 0
                one_rare = (
                    not isinstance(fa, (int, float))
                    or not isinstance(fb, (int, float))
                    or min(fa, fb) < freq_floor
                )
                if one_rare:
                    try:
                        pr = ngram_provider.get_bigram_probability(a, b)
                        if pr > bigram_threshold:
                            hit = True
                    except Exception:
                        pass

            if hit:
                # Update the working copy so the merged token can cascade —
                # next iteration sees `work[i+1] = merged` as the new `a` and
                # can attempt another merge with work[i+2].
                work[i + 1] = merged
                i += 1
                continue

            out.append(a)
            i += 1

        return out

    def _is_valid_compound(self, word: str) -> bool:
        """Check if word is a valid compound (splits into valid parts with no edits).

        Tightened validation: a compound split is only accepted when at least
        one part is a real word (not just a syllable) AND the parts have
        bigram evidence in the n-gram data. This prevents accepting arbitrary
        syllable combinations as valid compounds.
        """
        if self.symspell is None:
            return False

        # Fast path: if ALL syllables are individually valid words with
        # sufficient frequency AND have bigram association, skip the
        # expensive O(n*M) DP segmentation entirely.
        syllables = self.segmenter.segment_syllables(word)
        if len(syllables) >= 2:
            min_syllable_word_freq = self.config.validation.min_syllable_word_freq
            all_valid = all(
                self.word_repository.is_valid_word(s)
                and isinstance(f := self.word_repository.get_word_frequency(s), (int, float))
                and f >= min_syllable_word_freq
                for s in syllables
            )
            if all_valid and self.context_checker is not None:
                ngram_provider = getattr(self.context_checker, "provider", None)
                if ngram_provider is not None and hasattr(ngram_provider, "get_bigram_probability"):
                    has_bigram = any(
                        ngram_provider.get_bigram_probability(syllables[i], syllables[i + 1]) > 0
                        for i in range(len(syllables) - 1)
                    )
                    if has_bigram:
                        return True

        compound_check = self.symspell.lookup_compound(word, max_suggestions=1, max_edit_distance=0)
        if not compound_check:
            return False

        top_split = compound_check[0]
        if not isinstance(top_split, (list, tuple)) or len(top_split) < 3:
            return False

        parts, dist, score = top_split[0], top_split[1], top_split[2]

        # Must be zero-edit and perfect reconstruction
        if dist != 0 or "".join(parts) != word:
            return False

        # Single-part "compounds" with score=0 are just the original word
        # returned unsplit — not a real compound validation
        if len(parts) == 1 and score == 0:
            return False

        # For multi-part splits, require at least one part to be a real word
        # (not merely a valid syllable). This prevents accepting random
        # syllable-only combinations like ကေါက်+ဆွဲ as compounds.
        has_real_word = any(self.word_repository.is_valid_word(p) for p in parts)
        if not has_real_word:
            return False

        # Require bigram association evidence: at least one adjacent pair must
        # have positive PMI (Pointwise Mutual Information), meaning the words
        # co-occur more than chance.  PMI > 0  ⟺  P(w2|w1) > P(w2).
        # This prevents accepting spurious compounds from coincidental
        # co-occurrences (e.g. sentence-boundary artifacts in the corpus).
        if len(parts) >= 2 and self.context_checker is not None:
            ngram_provider = getattr(self.context_checker, "provider", None)
            if ngram_provider is not None and hasattr(ngram_provider, "get_bigram_probability"):
                has_association = False
                denom = self.context_checker.unigram_denominator
                for i in range(len(parts) - 1):
                    cond_prob = ngram_provider.get_bigram_probability(parts[i], parts[i + 1])
                    if cond_prob <= 0:
                        continue
                    # P(w2) = freq(w2) / total_tokens
                    w2_freq = ngram_provider.get_word_frequency(parts[i + 1])
                    if w2_freq <= 0:
                        # Rare word with observed bigram — give benefit of doubt
                        has_association = True
                        break
                    unigram_prob = w2_freq / denom
                    # PMI > 0  ⟺  P(w2|w1) > P(w2)
                    if cond_prob > unigram_prob:
                        has_association = True
                        break
                if not has_association:
                    return False

        return True

    def _is_valid_reduplication(self, word: str) -> bool:
        """Check if word is a valid productive reduplication of a known base word."""
        if self._reduplication_engine is None:
            return False
        result = self._reduplication_engine.analyze(
            word,
            dictionary_check=self._is_in_dictionary,
            frequency_check=self._get_word_frequency,
            pos_check=self._get_word_pos,
        )
        return result is not None and result.is_valid

    def _is_valid_compound_synthesis(self, word: str) -> bool:
        """Check if word is a valid compound by splitting into known morphemes."""
        if self._compound_resolver is None:
            return False
        result = self._compound_resolver.resolve(
            word,
            dictionary_check=self._is_in_dictionary,
            frequency_check=self._get_word_frequency,
            pos_check=self._get_word_pos,
        )
        return result is not None and result.is_valid

    def _get_word_frequency(self, word: str) -> int:
        """Get corpus frequency for a word."""
        return self.word_repository.get_word_frequency(word)

    def _get_word_pos(self, word: str) -> str | None:
        """Get POS tag for a word from the repository."""
        # Try to get POS from repository; return None if not available
        if hasattr(self.word_repository, "get_word_pos"):
            return cast("str | None", self.word_repository.get_word_pos(word))
        return None

    def _has_strong_whole_word_candidate(self, word: str) -> bool:
        """Return True when SymSpell has a strong correction candidate for `word`.

        Used to avoid over-skipping OOV words that segment into mostly-valid
        syllables but still have clear full-word corrections (e.g. missing
        diacritics inside compounds).
        """
        if self.symspell is None or not word:
            return False
        if len(word) > self._segment_skip_candidate_max_length:
            return False

        current_freq = 0.0
        if hasattr(self.word_repository, "get_word_frequency"):
            current = self.word_repository.get_word_frequency(word)
            if isinstance(current, (int, float)):
                current_freq = float(current)

        try:
            candidates = self.symspell.lookup(word, level="word", max_suggestions=5)
        except (RuntimeError, ValueError, TypeError, AttributeError):
            return False

        if not candidates:
            return False

        for cand in candidates:
            term = getattr(cand, "term", None)
            edit_distance = getattr(cand, "edit_distance", None)
            frequency = getattr(cand, "frequency", None)

            if not isinstance(term, str) or not term or term == word:
                continue
            if not isinstance(edit_distance, int):
                continue
            if edit_distance > self._segment_skip_candidate_max_edit_distance:
                continue
            if abs(len(term) - len(word)) > 2:
                continue
            if not self.word_repository.is_valid_word(term):
                continue
            if not isinstance(frequency, (int, float)):
                continue
            if float(frequency) < self._segment_skip_candidate_min_freq:
                continue

            ratio_base = max(current_freq, 1.0)
            if float(frequency) / ratio_base < self._segment_skip_candidate_min_ratio:
                continue

            return True

        return False

    def _generate_suggestions_via_strategy(
        self,
        word: str,
        prev_word: str | None,
        next_word: str | None = None,
        *,
        prev_words: list[str] | None = None,
        next_words: list[str] | None = None,
    ) -> list[str]:
        """Generate suggestions using the configured suggestion strategy.

        This method uses the composite suggestion strategy pipeline which:
        1. Aggregates suggestions from multiple sources (SymSpell, morphology, compound)
        2. Applies unified ranking with source attribution
        3. Deduplicates and returns the top suggestions

        Args:
            word: The word to generate suggestions for.
            prev_word: Previous word for left context (optional).
            next_word: Next word for right context (optional).
            prev_words: Extended left context words (up to 3), oldest-first.
                Falls back to [prev_word] if not provided.
            next_words: Extended right context words (up to 3), closest-first.
                Falls back to [next_word] if not provided.

        Returns:
            List of suggestion terms, ranked by the strategy.
        """
        if not self.suggestion_strategy:
            return []

        # Use extended context when available, fall back to single-word context
        ctx_prev = prev_words if prev_words else ([prev_word] if prev_word else [])
        ctx_next = next_words if next_words else ([next_word] if next_word else [])

        context = SuggestionContext(
            prev_words=ctx_prev,
            next_words=ctx_next,
            max_suggestions=self.config.max_suggestions,
            max_edit_distance=self.config.max_edit_distance,
        )

        result = self.suggestion_strategy.suggest(word, context)
        return result.terms

    def _check_colloquial_variant(self, word: str, position: int) -> WordError | None:
        """
        Check if word is a colloquial spelling variant.

        Handles colloquial variants based on the `colloquial_strictness` config:
        - 'strict': Flag as error with standard forms as suggestions
        - 'lenient': Return info note with low confidence
        - 'off': No handling, return None

        Args:
            word: The word to check.
            position: Position in the original text.

        Returns:
            WordError if colloquial variant detected (based on strictness),
            None otherwise.
        """
        strictness = self.config.validation.colloquial_strictness

        if strictness == "off":
            return None

        if not is_colloquial_variant(word):
            return None

        standard_forms = sorted(get_standard_forms(word))
        if not standard_forms:
            return None

        syllables = self.segmenter.segment_syllables(word)

        if strictness == "strict":
            # Flag as error with standard forms as suggestions
            return WordError(
                text=word,
                position=position,
                suggestions=standard_forms,
                confidence=self.config.validation.word_error_confidence,
                error_type=ET_COLLOQUIAL_VARIANT,
                syllable_count=len(syllables),
            )
        elif strictness == "lenient":
            # Suppress informational notes for very high-frequency words.
            # Words like မင်း (185K) are ubiquitous informal forms; flagging
            # them as colloquial_info is noise, not useful.
            # Exception: informal pronouns like ငါ are register-critical.
            if hasattr(self.word_repository, "get_word_frequency"):
                word_freq = self.word_repository.get_word_frequency(word)
                threshold = self.config.frequency_guards.colloquial_high_freq_suppression
                if (
                    isinstance(word_freq, (int, float))
                    and word_freq >= threshold
                    and word not in _REGISTER_CRITICAL_PRONOUNS
                ):
                    return None
            return WordError(
                text=word,
                position=position,
                suggestions=standard_forms,
                confidence=self.config.validation.colloquial_info_confidence,
                error_type=ET_COLLOQUIAL_INFO,
                syllable_count=len(syllables),
            )

        return None

    def validate(self, text: str) -> list[Error]:
        """
        Validate text for word-level errors with bidirectional context.

        This method segments the input text into words and validates each
        one against the dictionary, with support for:
            1. Direct dictionary lookup
            2. Compound word validation (splitting into valid parts)
            3. OOV recovery via morphological analysis (root + suffixes)
            4. Bidirectional context-aware suggestion ranking
            5. Colloquial variant detection

        Args:
            text: Normalized Myanmar text to validate. Should be preprocessed
                with normalization (e.g., via ``normalize_for_lookup()``).

        Returns:
            List of WordError objects, each containing:
                - text: The invalid word
                - position: Character position in input text
                - suggestions: Possible corrections (context-ranked)
                - confidence: Error confidence score (0.0-1.0)
                - syllable_count: Number of syllables in the word

        Note:
            Valid words are not included in the output. To check if a
            specific word is valid, use the ``is_valid_word()`` method
            on the word_repository directly.

        Example:
            >>> validator = WordValidator.create(
            ...     word_repository=provider,
            ...     syllable_repository=provider,
            ...     segmenter=segmenter,
            ...     symspell=symspell,
            ... )
            >>> errors = validator.validate("မြန်မာနိုင်ငံ")
            >>> for error in errors:
            ...     print(f"{error.text}: {error.suggestions[:3]}")
        """
        words = self.segmenter.segment_words(text)
        token_paths = build_validation_token_paths(
            words,
            is_valid_word=self.word_repository.is_valid_word,
            get_word_frequency=self.word_repository.get_word_frequency,
            get_bigram_probability=self._get_refinement_bigram_probability,
            segment_syllables=getattr(self.segmenter, "segment_syllables", None),
        )
        if not token_paths:
            return []

        merged_errors: list[Error] = []
        by_key: dict[tuple[int, str, str], WordError] = {}
        for path_words in token_paths:
            path_errors = self._validate_token_path(text, path_words)
            for error in path_errors:
                key = (error.position, error.text, error.error_type)
                existing = by_key.get(key)
                if existing is None:
                    by_key[key] = cast(WordError, error)
                    merged_errors.append(error)
                    continue
                if len(error.suggestions) > len(existing.suggestions):
                    existing.suggestions = list(error.suggestions)
                else:
                    for suggestion in error.suggestions:
                        if suggestion not in existing.suggestions:
                            existing.suggestions.append(suggestion)

        return merged_errors

    def _validate_token_path(self, text: str, words: list[str]) -> list[Error]:
        """Validate one tokenization path and return word-level errors."""
        errors: list[Error] = []
        current_idx = 0

        # Segmenter post-merge rescue (seg-probe-01). Rewrites `words` in place
        # by merging adjacent fragments whose concatenation hits a variant
        # map / dict / dict+asat / bigram probe. Off by default until FPR
        # calibration (seg-fpr-gate-01).
        if self.config.validation.use_segmenter_post_merge_rescue:
            words = self._merge_probe_adjacent_pairs(words)

        myanmar_words = [
            w
            for w in words
            if w and w.strip() and not self.is_punctuation(w) and self._is_myanmar_with_config(w)
        ]

        # Track position in myanmar_words list for correct bidirectional context
        myanmar_word_idx = 0

        for word in words:
            if not word:
                continue

            position, current_idx = self._find_token_position(text, word, current_idx)
            if position is None:
                continue

            # Skip non-Myanmar or punctuation
            is_valid_word = (
                word.strip()
                and not self.is_punctuation(word)
                and self._is_myanmar_with_config(word)
            )
            if not is_valid_word:
                continue

            # Strip non-Myanmar punctuation from token boundaries.
            # Corpus text often has quotes/brackets attached to words
            # (e.g., "word", word(, word)) which cause invalid_word FPs.
            stripped = _BOUNDARY_PUNCT_RE.sub("", word)
            if stripped and stripped != word and self._is_myanmar_with_config(stripped):
                word = stripped

            # Skip mixed-script tokens where non-Myanmar characters dominate.
            # Tokens like "။BYD" or "လက်ရှိ2" pass _is_myanmar_with_config because
            # they contain some Myanmar chars, but are not real Myanmar words.
            # Only applies when the token has BOTH Myanmar and non-Myanmar chars.
            myanmar_chars = sum(1 for c in word if "\u1000" <= c <= "\u109f")
            if myanmar_chars > 0 and myanmar_chars < len(word) * 0.5:
                myanmar_word_idx += 1
                continue

            syllables = self.segmenter.segment_syllables(word)

            # Skip segmenter artifacts: words starting with combining
            # characters (virama, vowel signs, medials) are clearly broken.
            first_ch = ord(word[0]) if word else 0
            if (
                0x1039 <= first_ch <= 0x103A
                or 0x102B <= first_ch <= 0x1038
                or 0x103B <= first_ch <= 0x103E
            ):
                myanmar_word_idx += 1
                continue
            # Skip virama stacking fragments (e.g., ဏ္ဍ from ကဏ္ဍ).
            # C + virama at position 1 means this is a bare stacking cluster
            # produced by syllable splitting — never a standalone word.
            if len(word) >= 2 and ord(word[1]) == 0x1039:
                myanmar_word_idx += 1
                continue
            # Skip segmenter merges: multi-syllable "words" where most
            # parts are independently valid words.  Word-level errors on
            # these produce poor suggestions that subsume better syllable
            # errors.  Legitimate compound typos (e.g. ကေါက်ဆွဲ where
            # one part is invalid) are recovered by text-level detectors.
            # Exception: don't skip if adding asat produces a high-freq
            # valid word — likely a missing-asat typo (e.g., တွင→တွင်).
            asat_freq_guard = self.config.validation.asat_freq_guard
            if len(syllables) >= 2:
                valid_parts = sum(1 for s in syllables if self.word_repository.is_valid_word(s))
                # For tokens with 4+ syllables where ALL are valid dictionary
                # words, skip unconditionally — these are segmenter merges of
                # valid words (verb chains, compound+particle), not real errors.
                if valid_parts == len(syllables) and len(syllables) >= 4:
                    myanmar_word_idx += 1
                    continue
                if valid_parts >= max(len(syllables) // 2, 1):
                    # Check whole word + asat
                    asat_form = word + "\u103a"
                    asat_freq = self.word_repository.get_word_frequency(asat_form)
                    if isinstance(asat_freq, (int, float)) and asat_freq >= asat_freq_guard:
                        pass  # don't skip — likely missing asat
                    else:
                        # Check prefix asat: join first N syllables + asat.
                        # Catches ကွနပျူတာ→ကွန်(ပျူတာ), ဖတရှု→ဖတ်(ရှု).
                        prefix_asat_found = False
                        for k in range(2, len(syllables) + 1):
                            prefix = "".join(syllables[:k])
                            pf = self.word_repository.get_word_frequency(prefix + "\u103a")
                            if isinstance(pf, (int, float)) and pf >= asat_freq_guard:
                                prefix_asat_found = True
                                break
                        if not prefix_asat_found and not self._has_strong_whole_word_candidate(
                            word
                        ):
                            myanmar_word_idx += 1
                            continue

            # Use tracked index instead of .index() to handle duplicate words correctly
            prev_word = myanmar_words[myanmar_word_idx - 1] if myanmar_word_idx > 0 else None
            if prev_word:
                prev_word = _BOUNDARY_PUNCT_RE.sub("", prev_word) or prev_word
            next_word = (
                myanmar_words[myanmar_word_idx + 1]
                if myanmar_word_idx < len(myanmar_words) - 1
                else None
            )
            if next_word:
                next_word = _BOUNDARY_PUNCT_RE.sub("", next_word) or next_word

            # Build extended context word lists for higher-order n-gram reranking
            prev_words: list[str] = []
            for k in range(min(myanmar_word_idx, 3), 0, -1):
                prev_words.append(myanmar_words[myanmar_word_idx - k])
            next_words: list[str] = []
            remaining = len(myanmar_words) - myanmar_word_idx - 1
            for k in range(1, min(remaining, 3) + 1):
                next_words.append(myanmar_words[myanmar_word_idx + k])

            # Valid word - check for colloquial variants before skipping
            if self.word_repository.is_valid_word(word):
                colloquial_error = self._check_colloquial_variant(word, position)
                if colloquial_error:
                    errors.append(colloquial_error)
                myanmar_word_idx += 1
                continue

            # Prong-3: known loan-word variant short-circuit. If this OOV word
            # is listed in loan_words.yaml or loan_words_mined.yaml as a
            # variant of a standard form, emit the correction directly rather
            # than falling through to SymSpell — which drops candidates with
            # edit distance > max_edit_distance (default 2). Workstream:
            # loan-word-db-mining / task: loanword-prong3-01.
            loan_standards = get_loan_word_standard(word)
            if loan_standards:
                standards_list = sorted(loan_standards)
                errors.append(
                    WordError(
                        text=word,
                        position=position,
                        suggestions=[standards_list[0]] + standards_list[1:],
                        confidence=self.config.validation.loan_word_detection_confidence,
                        syllable_count=len(syllables),
                    )
                )
                myanmar_word_idx += 1
                continue

            has_strong_whole_word_candidate = self._has_strong_whole_word_candidate(word)

            # Check valid compound (no edits needed)
            # But first: if a syllable-prefix + asat produces a high-freq
            # word, the compound is likely a segmentation artifact from
            # missing asat (e.g., ဖတရှု accepted as ဖ+တ+ရှု, but
            # really ဖတ်+ရှု with missing asat). Don't accept compound.
            if self._is_valid_compound(word):
                _prefix_asat_blocks_compound = False
                for k in range(2, len(syllables) + 1):
                    _pref = "".join(syllables[:k])
                    _pf = self.word_repository.get_word_frequency(_pref + "\u103a")
                    if isinstance(_pf, (int, float)) and _pf >= asat_freq_guard:
                        _prefix_asat_blocks_compound = True
                        break
                if not _prefix_asat_blocks_compound and not has_strong_whole_word_candidate:
                    # Also check for colloquial variants in compounds
                    colloquial_error = self._check_colloquial_variant(word, position)
                    if colloquial_error:
                        errors.append(colloquial_error)
                    myanmar_word_idx += 1
                    continue

            # Check productive reduplication (e.g., ကောင်းကောင်း from ကောင်း)
            if self._is_valid_reduplication(word):
                myanmar_word_idx += 1
                continue

            # Check compound synthesis (e.g., ကျောင်းသား = ကျောင်း + သား)
            if self._is_valid_compound_synthesis(word):
                # Same prefix-asat guard: don't accept if prefix+asat is
                # a high-freq word (missing asat creates fake compound).
                _synth_asat_blocks = False
                for k in range(2, len(syllables) + 1):
                    _sp = "".join(syllables[:k])
                    _sf = self.word_repository.get_word_frequency(_sp + "\u103a")
                    if isinstance(_sf, (int, float)) and _sf >= asat_freq_guard:
                        _synth_asat_blocks = True
                        break
                if not _synth_asat_blocks and not has_strong_whole_word_candidate:
                    myanmar_word_idx += 1
                    continue

            # Check morphological decomposition (e.g., ကိုယ့်ဟာ = ကိုယ့် + ဟာ)
            # Accept if OOV root recovery finds a valid known root with suffixes
            oov_analysis = self._recover_oov_root(word)
            if oov_analysis is not None and oov_analysis.root:
                myanmar_word_idx += 1
                continue

            # Generate suggestions via unified strategy pipeline
            if self.suggestion_strategy:
                suggestions = self._generate_suggestions_via_strategy(
                    word,
                    prev_word,
                    next_word,
                    prev_words=prev_words,
                    next_words=next_words,
                )
            else:
                # No strategy available - return empty suggestions
                # This happens when SymSpell is disabled (skip_init=True)
                suggestions = []

            suggestions = self._filter_suggestions(suggestions)

            errors.append(
                WordError(
                    text=word,
                    position=position,
                    suggestions=suggestions,
                    confidence=self.config.validation.word_error_confidence,
                    syllable_count=len(syllables),
                )
            )

            # Increment Myanmar word index for bidirectional context tracking
            myanmar_word_idx += 1

        return errors

    @classmethod
    def create(
        cls,
        word_repository: WordRepository,
        syllable_repository: SyllableRepository,
        segmenter: Segmenter,
        symspell: SymSpell,
        config: SpellCheckerConfig | None = None,
        context_checker: NgramContextChecker | None = None,
    ) -> "WordValidator":
        """
        Factory method for creating WordValidator instances.

        Provides a convenient way to create validators with sensible defaults.

        Args:
            word_repository: WordRepository for word lookup.
            syllable_repository: SyllableRepository for syllable lookup.
            segmenter: Segmenter for text tokenization.
            symspell: SymSpell instance for suggestions.
            config: Configuration (uses defaults if None).
            context_checker: Optional NgramContextChecker for context-aware suggestions.

        Returns:
            Configured WordValidator instance.

        Example:
            >>> from myspellchecker.providers import MemoryProvider
            >>> from myspellchecker.segmenters import DefaultSegmenter
            >>> from myspellchecker.algorithms import SymSpell
            >>>
            >>> provider = MemoryProvider(
            ...     syllables={"မြန်": 100, "မာ": 50},
            ...     words={"မြန်မာ": 80},
            ... )
            >>> segmenter = DefaultSegmenter()
            >>> symspell = SymSpell()
            >>> validator = WordValidator.create(
            ...     word_repository=provider,
            ...     syllable_repository=provider,
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
            word_repository=word_repository,
            syllable_repository=syllable_repository,
            symspell=symspell,
            context_checker=context_checker,
        )
