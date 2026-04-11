"""
Default implementation of the Segmenter interface for Myanmar text.

This module uses a hybrid approach:
- Syllable segmentation: Pure-Python, rule-based RegexSegmenter.
- Word segmentation: myWord-based Viterbi algorithm, CRF model, or Transformer model.

Word Segmentation Attribution:
    The word segmentation algorithms are based on research by Ye Kyaw Thu:
    - myWord: https://github.com/ye-kyaw-thu/myWord (Viterbi-based)
    - CRF models trained on myPOS corpus: https://github.com/ye-kyaw-thu/myPOS

    Transformer model by Chuu Htet Naing:
    - https://huggingface.co/chuuhtetnaing/myanmar-text-segmentation-model
"""

from __future__ import annotations

import re
from functools import cached_property, lru_cache
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from myspellchecker.providers.interfaces import WordRepository
    from myspellchecker.tokenizers.transformer_word_segmenter import TransformerWordSegmenter

from myspellchecker.core.constants import (
    SEGMENTER_ENGINE_CRF,
    SEGMENTER_ENGINE_MYWORD,
    SEGMENTER_ENGINE_TRANSFORMER,
    SENTENCE_SEPARATOR,
)
from myspellchecker.core.exceptions import MissingDependencyError, TokenizationError
from myspellchecker.grammar.config import GrammarRuleConfig, get_grammar_config
from myspellchecker.tokenizers import WordTokenizer
from myspellchecker.utils.cache import LRUCache
from myspellchecker.utils.logging_utils import get_logger

from .base import Segmenter
from .regex import RegexSegmenter

__all__ = [
    "DefaultSegmenter",
]


@lru_cache(maxsize=1)
def _get_grammar_config() -> GrammarRuleConfig:
    return get_grammar_config()


class DefaultSegmenter(Segmenter):
    """
    Myanmar text segmenter using a hybrid approach.

    Syllables: RegexSegmenter (pure Python, rule-based)
    Words: WordTokenizer (ML-based, myword or CRF engine) or TransformerWordSegmenter

    Example:
        >>> segmenter = DefaultSegmenter()
        >>> segmenter.segment_syllables("မြန်မာနိုင်ငံ")
        # Note: Regex segmenter might produce different results for complex words
        ['မြန်', 'မာ', 'နိုင်', 'ငံ']
    """

    def __init__(
        self,
        word_engine: str = SEGMENTER_ENGINE_MYWORD,
        allow_extended_myanmar: bool = False,
        seg_model: str | None = None,
        seg_device: int = -1,
    ) -> None:
        """
        Initialize the DefaultSegmenter.

        Args:
            word_engine: Word segmentation engine (default: "myword").
                         Options: "myword" (recommended), "crf", "transformer".
            allow_extended_myanmar: If True, accept Extended Myanmar characters
                in syllable segmentation, including:
                - Extended Core Block (U+1050-U+109F)
                - Extended-A (U+AA60-U+AA7F)
                - Extended-B (U+A9E0-U+A9FF)
                - Non-standard core chars (U+1022, U+1028, U+1033-U+1035)
            seg_model: Custom model name for transformer engine (optional).
                       Only used when word_engine="transformer".
            seg_device: Device for transformer inference (-1=CPU, 0+=GPU).
                       Only used when word_engine="transformer".
        """
        self.word_engine = word_engine
        self._allow_extended_myanmar = allow_extended_myanmar
        self._seg_model = seg_model
        self._seg_device = seg_device
        self.regex_segmenter = RegexSegmenter(
            allow_extended_myanmar=allow_extended_myanmar
        )  # Always available for syllables
        self.word_tokenizer: WordTokenizer | None = None
        self._transformer_segmenter: TransformerWordSegmenter | None = None
        self._word_tokenizer_initialized = False
        # Optional word repository for syllable-reassembly fallback.
        # Injected after construction via set_word_repository() because the
        # provider is resolved after the segmenter in the dependency graph.
        self._word_repository: WordRepository | None = None
        # Minimum syllable count to trigger reassembly fallback
        self._REASSEMBLY_MIN_SYLLABLES = 3
        # Maximum syllables to look ahead when merging
        self._REASSEMBLY_MAX_LOOKAHEAD = 4
        # Thread-safe bounded cache for syllable segmentation results to avoid
        # redundant calls across WordValidator, CompoundResolver,
        # ReduplicationEngine, etc.
        self._syllable_cache: LRUCache[list[str]] = LRUCache(maxsize=8192)

        # Word segmentation cache: text → tuple of words.
        # Context validator and multiple detectors call segment_words
        # on the same text; this avoids re-running the Viterbi/CRF segmenter.
        # Thread-safe bounded LRU cache for word segmentation.
        self._word_seg_cache: LRUCache[tuple[str, ...]] = LRUCache(maxsize=512)

        # Validate engine name early
        engine_map = {
            SEGMENTER_ENGINE_CRF: "CRF",
            SEGMENTER_ENGINE_MYWORD: "myword",
            SEGMENTER_ENGINE_TRANSFORMER: "transformer",
        }

        if word_engine.lower() not in engine_map:
            raise TokenizationError(
                f"Unsupported word_engine: {word_engine}. Supported: {list(engine_map.keys())}"
            )

        # Lazy initialization - defer creating WordTokenizer/TransformerWordSegmenter
        # until first segment_words() call to avoid network downloads at construction time

    @cached_property
    def _sfp_pattern(self) -> re.Pattern[str]:
        """Compile and cache the sentence-final particle regex pattern."""
        grammar_config = _get_grammar_config()
        sfps = grammar_config.sentence_final_particles
        sorted_sfps = sorted(sfps, key=len, reverse=True)
        escaped_sfps = [re.escape(p) for p in sorted_sfps]
        sfp_regex = "|".join(escaped_sfps)
        return re.compile(f"({SENTENCE_SEPARATOR})|({sfp_regex})(?=\\s|$)")

    def _validate_input(self, text: str) -> None:
        """
        Validate input text.

        Args:
            text: Input text.

        Raises:
            TypeError: If text is not a string.
            ValueError: If text is empty or whitespace-only.
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")

        if not text or not text.strip():
            raise TokenizationError("Text cannot be empty or whitespace-only")

    def segment_syllables(self, text: str) -> list[str]:
        """
        Segment Myanmar text into syllables using the internal RegexSegmenter.

        Results are cached (thread-safe LRU, bounded to 8192 entries) to avoid
        redundant calls across validation components (WordValidator,
        CompoundResolver, ReduplicationEngine, MorphemeSuggestionStrategy).
        """
        cached = self._syllable_cache.get(text)
        if cached is not None:
            return list(cached)
        self._validate_input(text)
        result = self.regex_segmenter.segment_syllables(text)
        self._syllable_cache.set(text, list(result))
        return result

    def _ensure_word_segmenter_initialized(self) -> None:
        """
        Lazily initialize word segmenter on first use.

        This avoids network downloads during SpellChecker construction,
        deferring resource loading until word segmentation is actually needed.
        """
        if self._word_tokenizer_initialized:
            return

        self._word_tokenizer_initialized = True

        if self.word_engine.lower() == SEGMENTER_ENGINE_TRANSFORMER:
            # Transformer engine uses TransformerWordSegmenter
            try:
                from myspellchecker.tokenizers.transformer_word_segmenter import (
                    TransformerWordSegmenter,
                )

                self._transformer_segmenter = TransformerWordSegmenter(
                    model_name=self._seg_model,
                    device=self._seg_device,
                )
                get_logger(__name__).debug(
                    f"Initialized TransformerWordSegmenter "
                    f"(model={self._transformer_segmenter.model_name})."
                )
            except (ValueError, ImportError, RuntimeError, OSError, TokenizationError) as e:
                self._transformer_segmenter = None
                self._word_tokenizer_initialized = False
                get_logger(__name__).error(
                    f"Failed to initialize TransformerWordSegmenter: {e}. "
                    f"Word segmentation will not be available."
                )
                raise
        else:
            # CRF/myword engines use WordTokenizer
            engine_map = {
                SEGMENTER_ENGINE_CRF: "CRF",
                SEGMENTER_ENGINE_MYWORD: "myword",
            }

            try:
                target_engine = engine_map[self.word_engine.lower()]
                self.word_tokenizer = WordTokenizer(engine=target_engine)
                get_logger(__name__).debug(
                    f"Initialized WordTokenizer with {target_engine} engine."
                )
            except (ValueError, ImportError, RuntimeError, OSError, TokenizationError) as e:
                self.word_tokenizer = None
                self._word_tokenizer_initialized = False
                get_logger(__name__).error(
                    f"Failed to initialize WordTokenizer engine '{self.word_engine}': {e}. "
                    f"Word segmentation will not be available."
                )
                raise

    def set_word_repository(self, word_repository: "WordRepository") -> None:
        """Inject a word repository for syllable-reassembly fallback.

        Called after construction (provider is resolved after segmenter in the
        DI graph). When set, ``segment_words()`` will use dictionary-guided
        syllable reassembly as a fallback when the statistical segmenter returns
        a single oversized token.

        Args:
            word_repository: WordRepository providing ``is_valid_word()`` and
                ``get_word_frequency()`` for dictionary lookups.
        """
        self._word_repository = word_repository

    def _syllable_reassembly_fallback(self, token: str) -> list[str]:
        """Re-segment an oversized token using syllable boundaries + dictionary.

        When the Viterbi/CRF segmenter returns a single token covering 3+
        syllables (typically because the text contains errors and no good
        statistical path exists), this fallback splits the token into
        syllables and greedily reassembles them into the longest valid
        dictionary words, left-to-right.

        Algorithm:
            1. Split token into syllables via RegexSegmenter.
            2. For each position, try joining the next 1..4 syllables.
            3. Pick the longest join that is a valid word in the dictionary.
               On ties, prefer higher corpus frequency.
            4. If no join produces a valid word, emit the single syllable
               (it will be caught downstream by the spell checker).
            5. Advance past the consumed syllables and repeat.

        This is O(n) in the number of syllables with a constant (4) lookahead
        factor, so effectively O(4n) = O(n).

        Args:
            token: A single oversized token from the statistical segmenter.

        Returns:
            List of words/syllables from greedy reassembly.
        """
        assert self._word_repository is not None  # Caller checks before calling

        syllables = self.regex_segmenter.segment_syllables(token)
        if len(syllables) < self._REASSEMBLY_MIN_SYLLABLES:
            return [token]

        result: list[str] = []
        i = 0
        n = len(syllables)
        max_look = self._REASSEMBLY_MAX_LOOKAHEAD

        while i < n:
            best_word: str | None = None
            best_len = 0
            best_freq = -1

            # Try joining k syllables starting at position i (k = max_look..1)
            # Longest match wins; ties broken by frequency.
            upper = min(max_look, n - i)
            for k in range(upper, 0, -1):
                candidate = "".join(syllables[i : i + k])
                if self._word_repository.is_valid_word(candidate):
                    freq = self._word_repository.get_word_frequency(candidate)
                    freq_val = int(freq) if isinstance(freq, (int, float)) else 0
                    if k > best_len or (k == best_len and freq_val > best_freq):
                        best_word = candidate
                        best_len = k
                        best_freq = freq_val

            if best_word is not None:
                result.append(best_word)
                i += best_len
            else:
                # No valid word found - emit single syllable for downstream
                # spell checking to handle.
                result.append(syllables[i])
                i += 1

        # Post-pass: merge adjacent single-syllable tokens whose
        # concatenation is OOV (not in dictionary).  When the greedy loop
        # splits e.g. "ကယာ" → ["က", "ယာ"] (both valid words), the
        # combined form is OOV and has an edit-distance-1 correction
        # ("ကရာ").  Keeping them merged lets the word validator flag the
        # OOV token and generate SymSpell suggestions.
        # Guard: at least one constituent must have word frequency below
        # _FUZZY_MERGE_FREQ_THRESHOLD to avoid merging common sequences.
        result = self._fuzzy_merge_single_syllables(result)

        return result

    def _is_single_syllable(self, token: str) -> bool:
        """Check if *token* is exactly one Myanmar syllable."""
        syls = self.regex_segmenter.segment_syllables(token)
        return len(syls) == 1

    # Frequency below which a single-syllable valid word is considered
    # "low-frequency" — i.e. more likely to be a fragment of a misspelled
    # multi-syllable word than an intentional standalone word.
    _FUZZY_MERGE_FREQ_THRESHOLD = 50_000

    # Forward-binding prefixes that must NOT be merged with the
    # *preceding* token.  These particles always attach to the
    # *following* word, so absorbing them backward produces nonsense.
    #   မ — negation prefix ("မစား" = don't eat)
    #   အ — nominalizer prefix ("အစား" = food/instead)
    _FORWARD_BINDING_PREFIXES: frozenset[str] = frozenset({"မ", "အ"})

    # Backward-binding suffixes / postpositions that must NOT be merged
    # with the *following* token.  These always attach to the preceding
    # noun/verb phrase.
    #   ကို — object marker
    #   ကား — topic marker
    #   နဲ့ — comitative ("with")
    #   လို့ — causal ("because")
    #   မှာ — locative / future
    #   က — subject marker (but note: က CAN be first syllable of words
    #       like ကရာ, so it's only protected as the first token of a pair)
    _BACKWARD_BINDING_SUFFIXES: frozenset[str] = frozenset({"ကို", "ကား", "နဲ့", "လို့", "မှာ"})

    def _fuzzy_merge_single_syllables(self, tokens: list[str]) -> list[str]:
        """Merge adjacent single-syllable tokens whose combination is OOV.

        After greedy reassembly, some misspelled words get split into
        individual valid syllables (e.g. ``ကယာ`` → ``['က', 'ယာ']``).
        This post-pass scans for pairs of adjacent single-syllable
        tokens and merges them when:

        1. **Both** constituents are valid dictionary words (so each was
           emitted by the greedy pass as a 1-syllable match).
        2. The concatenated form is **not** a valid word (OOV).
        3. At least one constituent has word frequency below
           ``_FUZZY_MERGE_FREQ_THRESHOLD`` — preventing merges of common
           function-word sequences like ``ရေး`` + ``က`` that happen to
           form an OOV bigram.
        4. Neither token is a forward-binding prefix (e.g. ``မ``
           negation — always standalone), and the first token is NOT
           a backward-binding suffix (e.g. ``ကို`` object marker).
           These particles always function independently — merging
           them produces nonsense tokens.

        Merged OOV tokens will be flagged by the downstream word
        validator which generates SymSpell suggestions
        (e.g. ``ကယာ`` → ``ကရာ`` at edit distance 1).

        Scanning is left-to-right; consumed tokens are skipped.
        """
        assert self._word_repository is not None
        if len(tokens) < 2:
            return tokens

        threshold = self._FUZZY_MERGE_FREQ_THRESHOLD
        fwd_prefixes = self._FORWARD_BINDING_PREFIXES
        bwd_suffixes = self._BACKWARD_BINDING_SUFFIXES
        merged: list[str] = []
        i = 0
        n = len(tokens)

        while i < n:
            found_merge = False

            if i + 1 < n:
                t1, t2 = tokens[i], tokens[i + 1]
                # Both must be single syllables AND valid words.
                if (
                    self._is_single_syllable(t1)
                    and self._is_single_syllable(t2)
                    and self._word_repository.is_valid_word(t1)
                    and self._word_repository.is_valid_word(t2)
                ):
                    # Guard: never merge if either token is a
                    # forward-binding prefix (e.g. မ negation — always
                    # standalone) or t1 is a backward-binding suffix
                    # (e.g. ကို object marker).
                    if t1 in fwd_prefixes or t2 in fwd_prefixes or t1 in bwd_suffixes:
                        merged.append(tokens[i])
                        i += 1
                        continue

                    candidate = t1 + t2
                    # Combined form must be OOV.
                    if not self._word_repository.is_valid_word(candidate):
                        f1 = self._word_repository.get_word_frequency(t1)
                        f2 = self._word_repository.get_word_frequency(t2)
                        fv1 = int(f1) if isinstance(f1, (int, float)) else 0
                        fv2 = int(f2) if isinstance(f2, (int, float)) else 0
                        if fv1 < threshold or fv2 < threshold:
                            merged.append(candidate)
                            i += 2
                            found_merge = True

            if not found_merge:
                merged.append(tokens[i])
                i += 1

        return merged

    # Explicit allowlist of colloquial-locative tokens where the merged form
    # appears in the dictionary at very low frequency but the canonical
    # reading is stem + colloquial locative particle. The merged form is
    # technically a valid word, so the segmenter retains it as one token,
    # but downstream homophone/word strategies then fire at the wrong span.
    # The allowlist forces a correct split at segmentation time.
    #
    # Each entry is added only after runtime verification that the merged
    # form has frequency below 1% of either constituent, preventing
    # accidental over-splitting of legitimate compounds.
    #
    # Wrapped in MappingProxyType so the class attribute is read-only at
    # runtime — prevents accidental mutation from other modules or tests.
    _COLLOQUIAL_LOCATIVE_MERGES: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
        {
            # "ရန်ကုန်မာ" → ["ရန်", "ကုန်မာ"] (Viterbi) → ["ရန်", "ကုန်", "မာ"]
            # ("ကုန်မာ" appears with frequency far below "ကုန်" + "မာ").
            "ကုန်မာ": ("ကုန်", "မာ"),
        }
    )

    def _maybe_reassemble(self, tokens: list[str]) -> list[str]:
        """Apply syllable-reassembly fallback to oversized tokens.

        Scans the token list for any token that:
            1. Is NOT a valid word in the dictionary.
            2. Has >= ``_REASSEMBLY_MIN_SYLLABLES`` syllables.
        and replaces it with the greedy reassembly result.

        This handles both single-token failures (Viterbi returns entire input
        unsplit) and partial failures (Viterbi splits into a few tokens but
        one chunk is still oversized, e.g. ``['ထွေ့ကယာရှောက်မဆား', 'နဲ့', 'လို့']``).

        Also processes tokens in ``_COLLOQUIAL_LOCATIVE_MERGES`` to split
        colloquial-locative merges that were retained by Viterbi due to
        their low-but-nonzero dictionary frequency.

        Args:
            tokens: Token list from the statistical segmenter.

        Returns:
            Token list with oversized tokens replaced by reassembled words.
        """
        if self._word_repository is None:
            return tokens

        result: list[str] = []
        changed = False
        for token in tokens:
            # Explicit colloquial-locative allowlist split
            allowlist_split = self._COLLOQUIAL_LOCATIVE_MERGES.get(token)
            if allowlist_split is not None:
                result.extend(allowlist_split)
                changed = True
                continue

            # Skip tokens that are already valid dictionary words
            if self._word_repository.is_valid_word(token):
                result.append(token)
                continue

            syllables = self.regex_segmenter.segment_syllables(token)

            # 2-syllable special case: always split invalid 2-syllable tokens.
            # The word validator handles each syllable independently — flagging
            # invalid ones and passing valid ones to context strategies.  This
            # catches merged pairs like "စြေးမှား" → ["စြေး", "မှား"] where
            # the segmenter failed to split at the word boundary.
            if len(syllables) == 2:
                result.extend(syllables)
                changed = True
                continue

            if len(syllables) < self._REASSEMBLY_MIN_SYLLABLES:
                result.append(token)
                continue

            # Only reassemble if at least one constituent syllable is also
            # OOV.  If ALL syllables are valid words, the token is likely a
            # segmenter merge of valid parts (e.g., compound with typo) and
            # the word validator handles it better as a single unit.
            has_oov_syllable = any(not self._word_repository.is_valid_word(s) for s in syllables)
            if not has_oov_syllable and len(syllables) < 5:
                # Try suffix-aware split: if the token ends with a known
                # grammatical suffix and the stem is a valid word, split it.
                # E.g., "အားလပ်ရင်" → ["အားလပ်", "ရင်"] (word + conditional).
                suffix_split = self._try_suffix_split(token, syllables)
                if suffix_split is not None:
                    result.extend(suffix_split)
                    changed = True
                    continue
                result.append(token)
                continue

            reassembled = self._syllable_reassembly_fallback(token)
            if len(reassembled) > 1:
                result.extend(reassembled)
                changed = True
            else:
                result.append(token)

        return result if changed else tokens

    # Known grammatical suffixes for suffix-aware re-segmentation.
    # Only used for OOV tokens where Viterbi merged word+suffix.
    _GRAMMATICAL_SUFFIXES: tuple[str, ...] = (
        # Sorted longest-first to prefer longer matches.
        "ကြောင့်",  # causal
        "ခြင်း",  # nominalization (formal)
        "လျှင်",  # conditional (formal)
        "များ",  # plural
        "ဆုံး",  # superlative
        "ရင်",  # conditional
        "ရန်",  # purpose infinitive
        "တာ",  # nominalizer (colloquial)
    )
    _NEGATION_PREFIX = "မ"

    def _try_suffix_split(self, token: str, syllables: list[str]) -> list[str] | None:
        """Try splitting an OOV token at a known grammatical suffix boundary.

        For tokens like "အားလပ်ရင်" (OOV) that end with a known suffix
        ("ရင်" = conditional), check if the stem ("အားလပ်") is a valid word.
        If so, return the split.  Also tries negation prefix "မ" + stem.

        Only called for OOV tokens with all-valid syllables — safe because
        valid dictionary words are already preserved by the caller.

        Returns:
            Split token list, or None if no valid split found.
        """
        if self._word_repository is None:
            return None

        # Try suffix detach (longest suffix first)
        for suffix in self._GRAMMATICAL_SUFFIXES:
            if not token.endswith(suffix) or len(token) <= len(suffix):
                continue
            stem = token[: -len(suffix)]
            if self._word_repository.is_valid_word(stem):
                return [stem, suffix]

        # Try negation prefix: မ + verb
        if token.startswith(self._NEGATION_PREFIX) and len(token) > len(self._NEGATION_PREFIX):
            rest = token[len(self._NEGATION_PREFIX) :]
            if self._word_repository.is_valid_word(rest):
                return [self._NEGATION_PREFIX, rest]

        return None

    def load_custom_dictionary(self, words: list[str]) -> None:
        """
        Load custom dictionary words into the segmenter.
        Only supported if using 'myword' engine.
        """
        self._ensure_word_segmenter_initialized()
        if self.word_tokenizer and hasattr(self.word_tokenizer, "add_custom_words"):
            self.word_tokenizer.add_custom_words(words)
        else:
            get_logger(__name__).warning(
                "Custom dictionary loading not supported for current engine configuration."
            )

    def segment_words(self, text: str) -> list[str]:
        """
        Segment Myanmar text into words using the configured engine.
        """
        # Fast path: return cached result for repeated segmentation of same text
        # (context_validator segments the same sentence for each lattice path,
        #  and multiple detectors also call segment_words on the same text).
        cached = self._word_seg_cache.get(text)
        if cached is not None:
            return list(cached)  # return copy to prevent mutation

        self._validate_input(text)

        # Lazy initialization on first call
        self._ensure_word_segmenter_initialized()

        if self.word_engine.lower() == SEGMENTER_ENGINE_TRANSFORMER:
            if self._transformer_segmenter:
                result = self._transformer_segmenter.segment(text)
                self._cache_word_seg(text, result)
                return result
            raise MissingDependencyError(
                "Transformer word segmentation is not available. "
                "The TransformerWordSegmenter failed to initialize. Check logs for details."
            )
        elif self.word_tokenizer:
            # Split at spaces first to preserve user-intended word boundaries,
            # then segment each chunk independently. Without this, the Viterbi
            # segmenter may merge adjacent chunks across space boundaries
            # (e.g., "ထမင်းက စားပါတယ်" → "ကစား" instead of keeping "က" separate).
            chunks = text.split()
            if len(chunks) <= 1:
                tokens = cast(list[str], self.word_tokenizer.tokenize(text))
                # Syllable-reassembly fallback: when Viterbi returns a single
                # oversized token (3+ syllables, not a valid word), re-segment
                # using syllable boundaries + dictionary lookup.
                tokens = self._maybe_reassemble(tokens)
                result = self._repair_orphan_vowel_fragments(tokens)
                self._cache_word_seg(text, result)
                return result
            all_tokens: list[str] = []
            for chunk in chunks:
                if chunk:
                    chunk_tokens = list(self.word_tokenizer.tokenize(chunk))
                    chunk_tokens = self._maybe_reassemble(chunk_tokens)
                    all_tokens.extend(chunk_tokens)
            result = self._repair_orphan_vowel_fragments(all_tokens)
            self._cache_word_seg(text, result)
            return result
        else:
            raise MissingDependencyError(
                "Word segmentation is not available. "
                "The WordTokenizer failed to initialize. Check logs for details."
            )

    def _cache_word_seg(self, text: str, result: list[str]) -> None:
        """Store a word segmentation result in the thread-safe LRU cache."""
        self._word_seg_cache.set(text, tuple(result))

    @staticmethod
    def _repair_orphan_vowel_fragments(tokens: list[str]) -> list[str]:
        """Merge orphan vowel/diacritic fragments with the preceding word.

        Catches segmentation artifacts from normalization mismatches where
        the Viterbi segmenter splits a word and produces a fragment consisting
        only of vowel signs and diacritics (no consonant onset). Valid Myanmar
        syllables always start with a consonant (U+1000-U+1021), independent
        vowel (U+1021-U+102A), digit (U+1040-U+1049), or punctuation
        (U+104A-U+104F). A fragment without any of these is an artifact.

        Example: 'ပြင်ပ' + 'ော°' → 'ပြင်ပေါ°' (merged back)
        """
        if len(tokens) <= 1:
            return tokens

        result: list[str] = []
        for token in tokens:
            if token and not any(
                "\u1000" <= c <= "\u102a"  # Consonants + independent vowels
                or "\u1040" <= c <= "\u104f"  # Digits + punctuation
                for c in token
            ):
                # Pure vowel/diacritic fragment — merge with previous word
                if result:
                    result[-1] = result[-1] + token
                    continue
            result.append(token)
        return result

    def segment_sentences(self, text: str) -> list[str]:
        """
        Segment Myanmar text into sentences using heuristics.

        Splits on:
        1. Standard separator '။' (preserved).
        2. Sentence Final Particles (SFPs) followed by space or newline.

        The separators are preserved at the end of the preceding sentence.
        """
        self._validate_input(text)

        # 1. Standard Separator Split (Fastest)
        # If text contains "။", it's the strongest signal.
        # But we also want to catch implicit endings.

        sfps = _get_grammar_config().sentence_final_particles

        # Use cached compiled pattern for SFP + separator splitting.
        # Result: [text, sep1, sep2, text, sep1, sep2, ...]
        # Group 1: The separator (။)
        # Group 2: The SFP that triggered the split (captured so we can re-attach)
        # If group 1 matches, sep2 is None, and vice-versa
        parts = self._sfp_pattern.split(text)

        sentences = []
        current_sentence = ""

        for part in parts:
            if part is None:
                continue

            if part == SENTENCE_SEPARATOR:
                # Append separator to current sentence and flush
                current_sentence += part
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
            elif part in sfps:
                # It's an SFP
                current_sentence += part
                # We only flush if the *next* part was a split boundary (whitespace)
                # But re.split consumes the separator?
                # Wait, (?=\s|$) is a lookahead, so the whitespace is NOT consumed
                # and is part of the NEXT 'text' chunk?
                # Actually, re.split consumes the 'match'.
                # My pattern matches the SFP itself. So the SFP is consumed from
                # the text and returned as a separator.
                # The whitespace remains in the next text chunk (start of next).

                # So we should flush here.
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
            else:
                # Regular text
                current_sentence += part

        # Flush remainder
        if current_sentence.strip():
            sentences.append(current_sentence.strip())

        return sentences
