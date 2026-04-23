"""
Word tokenizer for Myanmar (Burmese) text.

Supports two engines:
- CRF: Conditional Random Fields model for syllable-to-word tokenization
- myword: Viterbi-based word segmentation using unigram/bigram probabilities

Attribution:
    The word segmentation algorithms and models are based on research by Ye Kyaw Thu:
    - myWord project: https://github.com/ye-kyaw-thu/myWord
    - myPOS corpus: https://github.com/ye-kyaw-thu/myPOS
    - sylbreak: https://github.com/ye-kyaw-thu/sylbreak

    We gratefully acknowledge Ye Kyaw Thu for making these resources publicly
    available, enabling accurate Myanmar word segmentation.
"""

from __future__ import annotations

import pycrfsuite

from myspellchecker.core.constants import (
    ASAT,
    CONSONANTS,
    MYANMAR_NUMERALS,
    VIRAMA,
    VOWEL_CARRIER,
    VOWEL_SIGNS,
)
from myspellchecker.core.exceptions import TokenizationError
from myspellchecker.utils.logging_utils import get_logger

from .resource_loader import get_crf_model_path, get_segmentation_mmap_path
from .syllable import SyllableTokenizer

logger = get_logger(__name__)

# Myanmar consonants for fragment detection
# Use centralized constants from core/constants/myanmar_constants.py
# CONSONANTS includes all consonants U+1000-U+1020, plus VOWEL_CARRIER (အ) which
# functions as a syllable starter
_MYANMAR_CONSONANTS: set[str] = CONSONANTS | {VOWEL_CARRIER}
_ASAT: str = ASAT  # Myanmar asat (killer mark)
_VIRAMA: str = VIRAMA  # Myanmar virama (for stacking consonants)
_MYANMAR_VOWELS: set[str] = VOWEL_SIGNS  # Dependent vowel signs (U+102B-U+1032)
_MYANMAR_NUMERALS: set[str] = MYANMAR_NUMERALS  # Myanmar digits ၀-၉
_ZERO_NUMERAL = "၀"  # Myanmar numeral zero (U+1040)
_WA_LETTER = "ဝ"  # Myanmar letter wa (U+101D)


def _is_invalid_fragment(word: str) -> bool:
    """
    Check if a word is an invalid fragment.

    Invalid patterns include:
    1. Consonant + asat only (e.g., က်, င်, ည်)
    2. Consonant + tone + asat (e.g., င့်, ည့်)
    3. Words starting with consonant + asat (e.g., င်ငံ, န်မာ)
    4. Incomplete stacking: virama followed by vowel (e.g., ္ + ာ)

    These fragments should not exist as standalone words in Myanmar.
    They are typically segmentation artifacts from noisy training data.

    Args:
        word: Word to check.

    Returns:
        True if the word is an invalid fragment.
    """
    if not word:
        return False

    # Pattern 1: single consonant + asat (e.g., က်, င်, ည်)
    if len(word) == 2 and word[0] in _MYANMAR_CONSONANTS and word[1] == _ASAT:
        return True

    # Pattern 2: consonant + tone mark + asat (e.g., င့်, ည့်)
    if len(word) == 3 and word[0] in _MYANMAR_CONSONANTS and word[2] == _ASAT:
        if word[1] in "့း":  # Dot below or visarga before asat
            return True

    # Pattern 3: Words starting with consonant + asat (fragments like င်ငံ, န်မာ)
    # These are parts of words that got incorrectly segmented.
    # Exception: Kinzi sequences (C + Asat + Virama) are valid word starts.
    if len(word) > 2 and word[0] in _MYANMAR_CONSONANTS and word[1] == _ASAT:
        if len(word) > 2 and word[2] == _VIRAMA:
            return False  # Valid Kinzi prefix
        return True

    # Pattern 4: Incomplete stacking - virama (္) followed directly by vowel
    # Valid: ္ + consonant (e.g., ္က, ္မ)
    # Invalid: ္ + vowel (e.g., ္ာ, ္ု) - missing the stacked consonant
    if _VIRAMA in word:
        for i in range(len(word) - 1):
            if word[i] == _VIRAMA and word[i + 1] in _MYANMAR_VOWELS:
                return True

    # Pattern 5: Virama stacking fragment (e.g., ဏ္ဍ from ကဏ္ဍ, or ္ဘော)
    # C + U+1039 + C is always an atomic stacking cluster in Myanmar.
    # A standalone stacking cluster is a segmentation artifact — merge it.
    # Detect by checking: starts with virama, OR first consonant immediately stacks
    # (virama at position 1). Valid words like ကဏ္ဍ have virama deeper in the word.
    if _VIRAMA in word:
        if word[0] == _VIRAMA:
            return True
        if len(word) >= 2 and word[1] == _VIRAMA:
            return True

    return False


def _normalize_zero_to_wa(text: str) -> str:
    """
    Normalize Myanmar numeral zero (၀) to letter wa (ဝ) when not in numeric context.

    This fixes a common encoding issue where numeral zero (U+1040) is incorrectly
    used instead of letter wa (U+101D). They look similar in many fonts but are
    different characters.

    The function only replaces ၀ when it's NOT adjacent to other numerals,
    preserving actual numbers like ၂၀၂၃.

    Args:
        text: Input text that may contain zero/wa confusion.

    Returns:
        Text with zero normalized to wa in non-numeric contexts.
    """
    if not text or _ZERO_NUMERAL not in text:
        return text

    result = []
    for i, char in enumerate(text):
        if char == _ZERO_NUMERAL:
            # Check if adjacent to other numerals (part of a number)
            prev_is_numeral = i > 0 and text[i - 1] in _MYANMAR_NUMERALS
            next_is_numeral = i < len(text) - 1 and text[i + 1] in _MYANMAR_NUMERALS
            if prev_is_numeral or next_is_numeral:
                # Keep as numeral zero (part of a number)
                result.append(char)
            else:
                # Replace with letter wa
                result.append(_WA_LETTER)
        else:
            result.append(char)
    return "".join(result)


def _merge_invalid_fragments(tokens: list[str]) -> list[str]:
    """
    Merge invalid fragments with adjacent words.

    The Viterbi algorithm may produce consonant+asat fragments when the
    training dictionary contains these patterns with non-trivial probabilities.
    This post-processing step merges such fragments to produce valid words.

    Strategy:
    - If fragment has a previous word, merge with previous (preferred)
    - Otherwise, merge with next word
    - If isolated, keep as-is (rare edge case)

    Args:
        tokens: List of tokens from Viterbi segmentation.

    Returns:
        List of tokens with invalid fragments merged.
    """
    if not tokens:
        return tokens

    result: list[str] = []

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if _is_invalid_fragment(token):
            # Try to merge with previous word
            if result:
                result[-1] = result[-1] + token
            # Or merge with next word
            elif i + 1 < len(tokens):
                tokens[i + 1] = token + tokens[i + 1]
            else:
                # No adjacent word to merge with (rare edge case)
                result.append(token)
        else:
            result.append(token)

        i += 1

    return result


def _split_word_numeral_tokens(tokens: list[str]) -> list[str]:
    """
    Split tokens that contain both Myanmar letters and numerals.

    The mmap dictionary may contain word+numeral concatenations like:
    - လ၁ (month 1) → ['လ', '၁']
    - ကို၁ (to 1) → ['ကို', '၁']
    - မှ၁၁ (from 11) → ['မှ', '၁၁']

    This function splits such tokens at the letter-numeral boundary.

    Args:
        tokens: List of tokens from segmentation.

    Returns:
        List of tokens with word+numeral concatenations split.
    """
    if not tokens:
        return tokens

    result: list[str] = []

    for token in tokens:
        if not token:
            continue

        # Check if token contains both letters and numerals
        has_letters = any(c not in _MYANMAR_NUMERALS for c in token)
        has_numerals = any(c in _MYANMAR_NUMERALS for c in token)

        if has_letters and has_numerals:
            # Split at letter-numeral boundaries
            parts = _split_at_numeral_boundary(token)
            result.extend(parts)
        else:
            result.append(token)

    return result


def _split_at_numeral_boundary(token: str) -> list[str]:
    """
    Split a token at boundaries between letters and numerals.

    Examples:
        'လ၁' → ['လ', '၁']
        'ကို၁၂' → ['ကို', '၁၂']
        '၁၂လ' → ['၁၂', 'လ']
        'လ၁မှ' → ['လ', '၁', 'မှ']

    Args:
        token: A token containing both letters and numerals.

    Returns:
        List of split parts.
    """
    if not token:
        return []

    parts: list[str] = []
    current_part: list[str] = []
    prev_is_numeral: bool | None = None

    for char in token:
        is_numeral = char in _MYANMAR_NUMERALS

        if prev_is_numeral is not None and is_numeral != prev_is_numeral:
            # Boundary detected - save current part and start new one
            if current_part:
                parts.append("".join(current_part))
            current_part = [char]
        else:
            current_part.append(char)

        prev_is_numeral = is_numeral

    # Don't forget the last part
    if current_part:
        parts.append("".join(current_part))

    return parts


class WordTokenizer(SyllableTokenizer):
    """
    Word Tokenizer for Myanmar text.

    Supports two segmentation engines:
    - CRF: Uses a trained CRF model for syllable-based word segmentation
    - myword: Uses Viterbi algorithm with unigram/bigram probabilities

    Example:
        >>> tokenizer = WordTokenizer(engine="myword")
        >>> tokenizer.tokenize("မြန်မာနိုင်ငံ")
        ['မြန်မာ', 'နိုင်ငံ']
    """

    def __init__(self, engine: str = "myword") -> None:
        """
        Initialize the WordTokenizer.

        Args:
            engine: Segmentation engine to use. Options: "CRF", "myword".
                   Default is "myword" (recommended, most accurate).

        Raises:
            ValueError: If an unknown engine is specified.
        """
        super().__init__()
        self.engine = engine

        if engine == "CRF":
            self._init_crf()
        elif engine == "myword":
            self._init_myword()
        else:
            raise TokenizationError(f"Unknown engine: {engine}. Must be one of: CRF, myword")

    def _init_crf(self) -> None:
        """Initialize the CRF model for word tokenization."""
        model_path = get_crf_model_path()
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(str(model_path))

    def _init_myword(self) -> None:
        """Initialize the myword Viterbi segmenter."""
        mmap_path = get_segmentation_mmap_path()

        self._using_mmap = False
        self._using_cython = False

        try:
            from .cython.word_segment import initialize_mmap, viterbi, viterbi_topk

            if initialize_mmap(str(mmap_path)):
                # Debug level - avoid duplicate logging during model preload
                logger.debug("Using mmap-based Viterbi (fork-safe via COW).")
                self._viterbi_func = viterbi
                self._viterbi_topk_func = viterbi_topk
                self._using_mmap = True
                self._using_cython = True
                return
        except ImportError:
            logger.debug("Cython mmap module not available, trying pure Python...")
        except (RuntimeError, OSError, MemoryError) as e:
            logger.warning(f"Mmap initialization failed: {e}, falling back...")

        # No fallback available - mmap is required
        raise TokenizationError(
            "segmentation.mmap is required for myword engine. "
            "Ensure the file exists or can be downloaded from HuggingFace."
        )

    @staticmethod
    def _word2features(sent: str, i: int) -> dict[str, object]:
        """
        Extract features for CRF model at position i in the sentence.

        Args:
            sent: Input sentence (string of syllables).
            i: Position index.

        Returns:
            Dictionary of features for the CRF model.
        """
        word = sent[i]
        feats: dict[str, object] = {"number": word.isdigit()}

        if i > 0:
            prev = sent[i - 1]
            feats.update(
                {
                    "prev_word.lower()": prev.lower(),
                    "prev_number": prev.isdigit(),
                    "bigram": prev.lower() + "_" + word.lower(),
                }
            )
        else:
            feats["BOS"] = True

        if i < len(sent) - 1:
            nxt = sent[i + 1]
            feats.update({"next_word.lower()": nxt.lower(), "next_number": nxt.isdigit()})
        else:
            feats["EOS"] = True

        if i > 1:
            feats["trigram_1"] = (
                sent[i - 2].lower() + "_" + sent[i - 1].lower() + "_" + word.lower()
            )

        if i < len(sent) - 2:
            feats["trigram_2"] = (
                word.lower() + "_" + sent[i + 1].lower() + "_" + sent[i + 2].lower()
            )

        return feats

    def _crf_features(self, sent: str) -> list[dict]:
        """Generate CRF features for all positions in the sentence."""
        return [self._word2features(sent, i) for i in range(len(sent))]

    def add_custom_words(self, words: list[str]) -> None:
        """
        Check if custom words exist in the mmap dictionary (myword engine only).

        Note: With mmap-only mode, new words cannot be added dynamically.
        This method checks if the requested words already exist in the dictionary.

        Args:
            words: List of words to check.
        """
        if self.engine != "myword":
            logger.warning(
                f"add_custom_words is only supported for 'myword' engine, not {self.engine}"
            )
            return

        if not words:
            return

        logger.info(f"Checking {len(words)} custom words in segmentation dictionary...")

        if not self._using_mmap:
            logger.warning("Custom words require mmap mode which is not active.")
            return

        try:
            from .cython.mmap_reader import get_mmap_reader

            reader = get_mmap_reader()
            found_count = 0
            missing_words = []

            for word in words:
                word = word.strip()
                if not word:
                    continue
                log_prob = reader.get_unigram_log_prob(word)
                if log_prob < -15:  # Likely unknown word
                    missing_words.append(word)
                else:
                    found_count += 1

            if missing_words:
                preview = missing_words[:5]
                if len(missing_words) > 5:
                    logger.warning(f"  {len(missing_words)} words not in dictionary: {preview}...")
                else:
                    logger.warning(
                        f"  {len(missing_words)} words not in dictionary: {missing_words}"
                    )

            logger.info(f"  {found_count}/{len(words)} words found in dictionary.")

        except (RuntimeError, OSError, ImportError) as e:
            logger.warning(f"  Could not check mmap: {e}")

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize Myanmar text into words.

        Args:
            text: Input Myanmar text string.

        Returns:
            List of words.
        """
        # Pre-process: normalize zero to wa (fixes encoding issues)
        text = _normalize_zero_to_wa(text)
        chunks = [chunk for chunk in text.split() if chunk]
        if not chunks:
            return []

        all_tokens: list[str] = []
        if self.engine == "CRF":
            for chunk in chunks:
                preds = self.tagger.tag(self._crf_features(chunk))
                merged = "".join(
                    char + ("_" if tag == "|" else "")
                    for char, tag in zip(chunk, preds, strict=False)
                )
                all_tokens.extend(part for part in merged.split("_") if part)
            return all_tokens

        if self.engine == "myword":
            for chunk in chunks:
                _, tokens = self._viterbi_func(chunk)
                # Post-process step 1: merge invalid fragments (consonant+asat patterns)
                # that may result from noisy training data in the mmap dictionary
                tokens = _merge_invalid_fragments(tokens)
                # Post-process step 2: split word+numeral concatenations
                # that exist in the mmap dictionary (e.g., လ၁ → လ, ၁)
                tokens = _split_word_numeral_tokens(tokens)
                # Post-process step 3: merge fragments AGAIN
                # Splitting can recreate fragments (e.g., "မှု၃၂၅န့်" → ["မှု", "၃၂၅", "န့်"])
                tokens = _merge_invalid_fragments(tokens)
                all_tokens.extend(tokens)
            return all_tokens

        raise TokenizationError(f"Unknown engine: {self.engine}")


__all__ = ["WordTokenizer"]
