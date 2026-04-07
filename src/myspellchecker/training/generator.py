"""
Synthetic error generator for Myanmar model training.

Generates corrupted Myanmar text with token-level labels by applying
linguistically motivated error patterns (homophone swaps, medial confusions,
character deletions/insertions, and typo patterns).

Used for:
- Denoising data augmentation in MLM (semantic model) training
- Generating synthetic training data for any error detection approach
"""

from __future__ import annotations

import importlib.resources
import random
from typing import TYPE_CHECKING, Callable

import yaml

from myspellchecker.core.constants.myanmar_constants import (
    ANUSVARA,
    ASAT,
    DOT_BELOW,
    MEDIAL_HA,
    MEDIAL_RA,
    MEDIAL_WA,
    MEDIAL_YA,
    MEDIALS,
    VISARGA,
    VOWEL_SIGNS,
)
from myspellchecker.text.phonetic_data import VISUAL_SIMILAR
from myspellchecker.training.constants import (
    DEFAULT_CORRUPTION_RATIO,
    DEFAULT_CORRUPTION_WEIGHTS,
    DEFAULT_GENERATOR_MAX_SYLLABLES,
    LABEL_CORRECT,
    LABEL_ERROR,
)
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.segmenters.base import Segmenter

logger = get_logger(__name__)

# Pali stacked consonant marker — sentences containing this are skipped
# entirely because corrupting Pali syllables produces single-consonant ERROR
# labels (e.g. ဗ from ဗုဒ္ဓ) that are identical to valid Myanmar syllable
# starts, creating first-token ambiguity in BPE training.
_PALI_STACKED = "\u1039"  # U+1039 MYANMAR SIGN VIRAMA (stacking marker)

# Max errors injected per sentence regardless of sentence length.
# Real misspelled text has 1-3 errors; the ratio-based approach was producing
# 8-12 simultaneous errors in long sentences, causing distribution shift at
# inference time where users write sentences with at most 1-3 errors.
# Raised from 2→3 to expose the model to multi-error patterns matching
# real-world multi-error (2-3 per sentence) and dense-error patterns.
_MAX_ERRORS_PER_SENTENCE = 3

# Medial confusion pairs: ya-yit <-> ya-pin, wa-hswe <-> ha-htoe
_MEDIAL_PAIRS = {
    MEDIAL_YA: MEDIAL_RA,
    MEDIAL_RA: MEDIAL_YA,
    MEDIAL_WA: MEDIAL_HA,
    MEDIAL_HA: MEDIAL_WA,
}

# Deletable diacritics (asat, tone marks, select vowels, medials).
# Including medials enables ha-htoe/wa-hswe/ya-yit/ya-pin deletion errors
# (e.g. နှာ→နာ, ကျွန်→ကြန်) which are common real-world typos.
_DELETABLE_CHARS = {ASAT, ANUSVARA, DOT_BELOW, VISARGA} | VOWEL_SIGNS | MEDIALS

# Common particle confusions: correct ↔ incorrect pairs.
# Each key maps to a list of plausible wrong particles a user might type.
# Bidirectional: the generator randomly picks a direction.
_PARTICLE_CONFUSIONS: dict[str, list[str]] = {
    "ကို": ["ကိ", "ကု", "က"],  # object marker → truncated / subject marker
    "က": ["ကို"],  # subject marker → object marker
    "မှာ": ["မာ"],  # locative → "hard" (ha-htoe drop)
    "မာ": ["မှာ"],  # hard → locative (reverse)
    "ပါတယ်": ["ပါတယ"],  # past particle → missing asat
    "ပါမယ်": ["ပါမယ"],  # future particle → missing asat
    "တယ်": ["တယ"],  # colloquial past → missing asat
    "မယ်": ["မယ"],  # colloquial future → missing asat
    "နဲ့": ["နဲ", "နှင့်"],  # with-particle → truncated / formal
    "နှင့်": ["နဲ့", "နှင်"],  # formal with → colloquial / missing tone
    "၏": ["ရဲ့"],  # formal possessive → colloquial
    "ရဲ့": ["၏"],  # colloquial possessive → formal
    "ပါသည်": ["ပါတယ်"],  # formal ending → colloquial
    "ခဲ့": ["ခဲ"],  # past marker → missing tone
    "လျှင်": ["လျှင"],  # if-particle → missing asat
}

# Aspirated ↔ unaspirated consonant pairs for syllable-initial swaps.
_ASPIRATED_PAIRS: dict[str, str] = {
    "\u1000": "\u1001",  # က ↔ ခ
    "\u1001": "\u1000",
    "\u1002": "\u1003",  # ဂ ↔ ဃ
    "\u1003": "\u1002",
    "\u1005": "\u1006",  # စ ↔ ဆ
    "\u1006": "\u1005",
    "\u1010": "\u1011",  # တ ↔ ထ
    "\u1011": "\u1010",
    "\u1015": "\u1016",  # ပ ↔ ဖ
    "\u1016": "\u1015",
    "\u1017": "\u1018",  # ဗ ↔ ဘ
    "\u1018": "\u1017",
    "\u1012": "\u1013",  # ဒ ↔ ဓ
    "\u1013": "\u1012",
}

# Characters that can be doubled for insertion errors
_INSERTABLE_CHARS = {ASAT, ANUSVARA, DOT_BELOW, VISARGA}


def _load_homophones() -> dict[str, list[str]]:
    """Load homophone map from rules/homophones.yaml."""
    rules_path = importlib.resources.files("myspellchecker") / "rules"
    homophones_file = rules_path / "homophones.yaml"
    try:
        with importlib.resources.as_file(homophones_file) as f:
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
        result: dict[str, list[str]] = data.get("homophones", {})
        return result
    except (OSError, yaml.YAMLError) as e:
        logger.warning(f"Failed to load homophones.yaml: {e}")
        return {}


def _load_word_swap_pool() -> dict[str, list[str]]:
    """Load word swap pool from bundled JSON (generated from myPOS corpus).

    Returns a dict keyed by POS class ('n', 'v', 'adj', 'adv'), each containing
    a list of common Myanmar words sorted by corpus frequency descending.
    Words are real, in-context tagged, Pali-free, and frequency ≥ 2 in the corpus.
    """
    import json

    data_path = importlib.resources.files("myspellchecker") / "data" / "word_swap_pool.json"
    try:
        with importlib.resources.as_file(data_path) as f:
            result: dict[str, list[str]] = json.loads(f.read_text(encoding="utf-8"))
            return result
    except (OSError, json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to load word_swap_pool.json: {e}")
        return {}


def _load_inverse_typos() -> dict[str, str]:
    """Load typo corrections and invert them (correct -> incorrect) for corruption."""
    rules_path = importlib.resources.files("myspellchecker") / "rules"
    typo_file = rules_path / "typo_corrections.yaml"
    try:
        with importlib.resources.as_file(typo_file) as f:
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        logger.warning(f"Failed to load typo_corrections.yaml: {e}")
        return {}

    inverse: dict[str, str] = {}
    corrections = data.get("corrections", {})
    for _category, entries in corrections.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            correct = entry.get("correct")
            incorrect = entry.get("incorrect")
            if correct and incorrect and correct != incorrect:
                # Invert: correct -> incorrect (so we can corrupt correct text)
                inverse[correct] = incorrect
    return inverse


class SyntheticErrorGenerator:
    """
    Generates synthetic Myanmar spelling errors for training data augmentation.

    Takes clean sentences and returns corrupted versions with token-level labels
    (CORRECT=0, ERROR=1). Used for denoising in MLM training and synthetic
    error data generation.

    Args:
        corruption_ratio: Fraction of words to corrupt per sentence (0.0-1.0).
        corruption_weights: Dict mapping corruption type names to relative weights.
        seed: Random seed for reproducibility.
        clean_ratio: Fraction of sentences to emit without corruption (all-CORRECT).
            Teaches the model that some sentences have no errors, reducing FPs.
            Default: 0.0 (all sentences corrupted, original behavior).
    """

    def __init__(
        self,
        corruption_ratio: float = DEFAULT_CORRUPTION_RATIO,
        corruption_weights: dict[str, float] | None = None,
        seed: int | None = None,
        clean_ratio: float = 0.0,
    ):
        if not 0.0 < corruption_ratio < 1.0:
            raise ValueError("corruption_ratio must be between 0.0 and 1.0 (exclusive)")
        if not 0.0 <= clean_ratio < 1.0:
            raise ValueError("clean_ratio must be between 0.0 (inclusive) and 1.0 (exclusive)")

        self.corruption_ratio = corruption_ratio
        self.clean_ratio = clean_ratio
        self.corruption_weights = corruption_weights or dict(DEFAULT_CORRUPTION_WEIGHTS)
        self.rng = random.Random(seed)
        self.logger = get_logger(__name__)

        # Load linguistic resources lazily
        self._homophones: dict[str, list[str]] | None = None
        self._inverse_typos: dict[str, str] | None = None
        self._word_swap_pool: dict[str, list[str]] | None = None
        # Reverse index: word -> POS, built from the pool on first use
        self._word_to_pos: dict[str, str] | None = None
        self._method_map: dict[str, Callable[[str], str]] | None = None

        # Build weighted corruption method list
        self._corruption_methods = self._build_corruption_methods()

    @property
    def homophones(self) -> dict[str, list[str]]:
        if self._homophones is None:
            self._homophones = _load_homophones()
        return self._homophones

    @property
    def inverse_typos(self) -> dict[str, str]:
        if self._inverse_typos is None:
            self._inverse_typos = _load_inverse_typos()
        return self._inverse_typos

    @property
    def word_swap_pool(self) -> dict[str, list[str]]:
        if self._word_swap_pool is None:
            self._word_swap_pool = _load_word_swap_pool()
        return self._word_swap_pool

    @property
    def word_to_pos(self) -> dict[str, str]:
        """Reverse index from word → POS, built from the swap pool on first access."""
        if self._word_to_pos is None:
            self._word_to_pos = {
                word: pos for pos, words in self.word_swap_pool.items() for word in words
            }
        return self._word_to_pos

    def _build_corruption_methods(self) -> list[tuple[str, float]]:
        """Build sorted list of (method_name, cumulative_weight) for weighted selection."""
        methods = []
        total = sum(self.corruption_weights.values())
        cumulative = 0.0
        for name, weight in self.corruption_weights.items():
            cumulative += weight / total
            methods.append((name, cumulative))
        return methods

    def _select_corruption_method(self) -> str:
        """Select a corruption method based on weights."""
        r = self.rng.random()
        for name, threshold in self._corruption_methods:
            if r <= threshold:
                return name
        return self._corruption_methods[-1][0]

    def generate_one(
        self,
        sentence: str,
        segment_fn: Callable[[str], list[str]] | None = None,
    ) -> tuple[str, list[str], list[int]] | None:
        """
        Generate synthetic error for a single sentence.

        Args:
            sentence: A single clean Myanmar sentence.
            segment_fn: Callable that segments text into words/syllables.
                If not provided, uses RegexSegmenter.segment_syllables.

        Returns:
            (corrupted_sentence, words, labels) tuple, or None if sentence is empty.
        """
        if segment_fn is None:
            from myspellchecker.segmenters.regex import RegexSegmenter

            segment_fn = RegexSegmenter().segment_syllables

        sentence = sentence.strip()
        if not sentence:
            return None

        # Skip sentences with Pali stacked consonants.
        if _PALI_STACKED in sentence:
            return None

        words = segment_fn(sentence)
        if not words:
            return None

        # Skip sentences that exceed the model's context window.
        if len(words) > DEFAULT_GENERATOR_MAX_SYLLABLES:
            return None

        # Clean negative example: emit sentence as-is with all-CORRECT labels.
        if self.clean_ratio > 0.0 and self.rng.random() < self.clean_ratio:
            all_correct_labels = [LABEL_CORRECT] * len(words)
            return (sentence, words, all_correct_labels)

        corrupted_words: list[str] = []
        labels: list[int] = []

        # Decide which word indices to corrupt.
        n_corrupt = min(
            max(1, int(len(words) * self.corruption_ratio)),
            _MAX_ERRORS_PER_SENTENCE,
        )
        corrupt_indices = set(self.rng.sample(range(len(words)), min(n_corrupt, len(words))))

        for i, word in enumerate(words):
            if i in corrupt_indices:
                method_name = self._select_corruption_method()
                corrupted = self._apply_corruption(method_name, word)
                if corrupted != word:
                    corrupted_words.append(corrupted)
                    labels.append(LABEL_ERROR)
                else:
                    corrupted_words.append(word)
                    labels.append(LABEL_CORRECT)
            else:
                corrupted_words.append(word)
                labels.append(LABEL_CORRECT)

        corrupted_sentence = "".join(corrupted_words)
        return (corrupted_sentence, corrupted_words, labels)

    def generate(
        self,
        clean_sentences: list[str],
        segmenter: Segmenter | None = None,
    ) -> list[tuple[str, list[str], list[int]]]:
        """
        Generate corrupted training data from clean sentences.

        Args:
            clean_sentences: List of clean Myanmar sentences.
            segmenter: Optional segmenter with segment_words() method.
                Falls back to RegexSegmenter if not provided.

        Returns:
            List of (corrupted_sentence, words, labels) tuples where:
            - corrupted_sentence: The sentence with errors injected
            - words: List of words (some corrupted)
            - labels: List of int labels (0=CORRECT, 1=ERROR) per word
        """
        if segmenter is None:
            from myspellchecker.segmenters.regex import RegexSegmenter

            segmenter = RegexSegmenter()

        # Determine segmentation method: prefer segment_words, fall back to syllables
        if hasattr(segmenter, "segment_words"):
            try:
                segmenter.segment_words("test")
                segment_fn = segmenter.segment_words
            except NotImplementedError:
                segment_fn = segmenter.segment_syllables
        else:
            segment_fn = segmenter.segment_syllables

        results: list[tuple[str, list[str], list[int]]] = []

        for sentence in clean_sentences:
            result = self.generate_one(sentence, segment_fn=segment_fn)
            if result is not None:
                results.append(result)

        return results

    def _get_method_map(self) -> dict[str, Callable[[str], str]]:
        """Return (lazily-initialized) mapping from corruption name to method."""
        if self._method_map is None:
            self._method_map = {
                "homophone_swap": self._corrupt_homophone_swap,
                "medial_confusion": self._corrupt_medial_confusion,
                "similar_char_swap": self._corrupt_similar_char,
                "char_deletion": self._corrupt_char_deletion,
                "char_insertion": self._corrupt_char_insertion,
                "typo_pattern": self._corrupt_typo_pattern,
                "word_swap": self._corrupt_word_swap,
                "particle_confusion": self._corrupt_particle_confusion,
                "aspirated_confusion": self._corrupt_aspirated_confusion,
            }
        return self._method_map

    def _apply_corruption(self, method_name: str, word: str) -> str:
        """Apply the named corruption method to a word."""
        method = self._get_method_map().get(method_name)
        if method is None:
            return word
        return method(word)

    def _corrupt_homophone_swap(self, word: str) -> str:
        """Swap a word with one of its homophones."""
        alternatives = self.homophones.get(word)
        if alternatives:
            return self.rng.choice(alternatives)
        return word

    def _corrupt_medial_confusion(self, word: str) -> str:
        """Swap medial characters: ya-yit <-> ya-pin, wa-hswe <-> ha-htoe."""
        chars = list(word)
        medial_positions = [
            (i, c) for i, c in enumerate(chars) if c in MEDIALS and c in _MEDIAL_PAIRS
        ]
        if not medial_positions:
            return word
        idx, char = self.rng.choice(medial_positions)
        chars[idx] = _MEDIAL_PAIRS[char]
        return "".join(chars)

    def _corrupt_similar_char(self, word: str) -> str:
        """Swap a character with a visually similar one."""
        chars = list(word)
        swappable = [(i, c) for i, c in enumerate(chars) if c in VISUAL_SIMILAR]
        if not swappable:
            return word
        idx, char = self.rng.choice(swappable)
        alternatives = list(VISUAL_SIMILAR[char])
        if alternatives:
            chars[idx] = self.rng.choice(alternatives)
        return "".join(chars)

    def _corrupt_char_deletion(self, word: str) -> str:
        """Remove a deletable diacritic (asat, tone mark, vowel sign)."""
        chars = list(word)
        deletable = [i for i, c in enumerate(chars) if c in _DELETABLE_CHARS]
        if not deletable:
            return word
        idx = self.rng.choice(deletable)
        chars.pop(idx)
        return "".join(chars)

    def _corrupt_char_insertion(self, word: str) -> str:
        """Insert a duplicate diacritic (double asat, extra tone mark)."""
        chars = list(word)
        insertable = [i for i, c in enumerate(chars) if c in _INSERTABLE_CHARS]
        if not insertable:
            return word
        idx = self.rng.choice(insertable)
        chars.insert(idx + 1, chars[idx])
        return "".join(chars)

    def _corrupt_typo_pattern(self, word: str) -> str:
        """Apply an inverse typo pattern (correct -> incorrect)."""
        # Try whole-word match first
        if word in self.inverse_typos:
            return self.inverse_typos[word]
        # Try substring match for longer words
        for correct, incorrect in self.inverse_typos.items():
            if correct in word and len(correct) > 1:
                return word.replace(correct, incorrect, 1)
        return word

    def _corrupt_word_swap(self, word: str) -> str:
        """Replace word with a semantically plausible but contextually wrong word."""
        pool = self.word_swap_pool
        if not pool:
            return word

        word_pos = self.word_to_pos.get(word)
        candidates = pool.get(word_pos or "n", [])

        if not candidates:
            return word

        top_candidates = candidates[:500]
        choices = [w for w in top_candidates if w != word]
        if not choices:
            return word

        return self.rng.choice(choices)

    def _corrupt_particle_confusion(self, word: str) -> str:
        """Swap a particle with a commonly confused alternative."""
        alternatives = _PARTICLE_CONFUSIONS.get(word)
        if alternatives:
            return self.rng.choice(alternatives)
        return word

    def _corrupt_aspirated_confusion(self, word: str) -> str:
        """Swap the first consonant between aspirated and unaspirated."""
        if not word:
            return word
        first_char = word[0]
        replacement = _ASPIRATED_PAIRS.get(first_char)
        if replacement:
            return replacement + word[1:]
        return word
