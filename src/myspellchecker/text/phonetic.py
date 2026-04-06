"""
Phonetic hashing for Myanmar (Burmese) script.

This module implements phonetic similarity matching for Myanmar text, enabling
detection and correction of phonetically similar errors (e.g., confusing
characters that sound similar or look similar).

Myanmar script presents unique challenges for phonetic hashing:
1. **Consonant clusters**: Complex combinations like က္က, င်္ဂ
2. **Tone markers**: Medials and diacritics affect pronunciation
3. **Visual similarity**: Some characters look very similar (e.g., ေ vs ဲ)

The PhoneticHasher implements a Myanmar-specific phonetic encoding that:
- Groups phonetically similar characters
- Normalizes tone markers and medials
- Handles visual confusability

Performance optimization: Uses LRU caching for encoding to avoid recomputing
phonetic codes for frequently accessed terms.
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

from myspellchecker.algorithms.distance.edit_distance import levenshtein_distance
from myspellchecker.core.constants import TONE_MARKS
from myspellchecker.text.phonetic_data import (
    COLLOQUIAL_SUBSTITUTIONS,
    MYANMAR_SUBSTITUTION_COSTS,
    PHONETIC_GROUPS,
    TONAL_GROUPS,
    VISUAL_SIMILAR,
    get_phonetic_equivalents,
)

if TYPE_CHECKING:
    from myspellchecker.core.config.algorithm_configs import PhoneticConfig

# Local alias for readability in context
MYANMAR_TONES = TONE_MARKS

__all__ = ["PhoneticHasher"]

# Constants
NORM_FORM: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFC"


# Reverse mapping: character -> phonetic code
CHAR_TO_PHONETIC: dict[str, str] = {}
for phonetic_code, chars in PHONETIC_GROUPS.items():
    for char in chars:
        CHAR_TO_PHONETIC[char] = phonetic_code


class PhoneticHasher:
    """
    Phonetic hashing for Myanmar text.

    This class generates phonetic codes for Myanmar syllables and words,
    enabling fuzzy matching based on pronunciation rather than exact spelling.

    Use cases:
    1. Finding homophones (words that sound the same)
    2. Correcting phonetic errors (wrong character, same sound)
    3. Handling spelling variations (tone marks omitted/added)

    Example:
        >>> hasher = PhoneticHasher()
        >>> code1 = hasher.encode("မြန်")
        >>> code2 = hasher.encode("မျန်")  # Typo: wrong medial
        >>> hasher.similar(code1, code2)
        True  # Phonetically similar despite different medials
    """

    def __init__(
        self,
        ignore_tones: bool = True,
        normalize_length: bool = True,
        normalize_nasals: bool = False,
        max_code_length: int | None = None,
        adaptive_length: bool = True,
        chars_per_code_unit: int | None = None,
        cache_size: int | None = None,
        config: PhoneticConfig | None = None,
    ):
        """
        Initialize the phonetic hasher.

        Args:
            ignore_tones: If True, ignore tone marks in encoding (default: True)
                         This makes encoding more forgiving of tone mark errors
            normalize_length: If True, treat short/long vowels as same (default: True)
                             e.g., ိ (short i) and ီ (long ii) -> same code
            normalize_nasals: If True, normalize nasal endings (န်, မ်, င်) to Anusvara (ံ)
                             (default: False). Setting True increases recall but may cause
                             false positives between words with different nasal sounds
                             (e.g., /n/ vs /m/ vs /ŋ/)
            max_code_length: Maximum length of phonetic codes (default: 10)
                            Base limit for simple words; may be extended for compounds
            adaptive_length: If True, extend max_code_length for compound words (default: True)
                            This prevents information loss on longer Myanmar compounds
            chars_per_code_unit: Characters per phonetic code unit for adaptive calculation
                                (default: 6, meaning ~6 chars input per code unit output)
            cache_size: Maximum size of the phonetic encoding cache (default: 4096)
                       Set to 0 to disable caching
            config: Optional PhoneticConfig for similarity scoring weights.
                   When provided, encoding params (max_code_length,
                   chars_per_code_unit, cache_size) are taken from config
                   unless explicitly overridden via constructor kwargs.
        """
        # If config provided, use its values for encoding params
        # (constructor kwargs override config when explicitly passed)
        if config is not None:
            self._config = config
            self.max_code_length = (
                max_code_length if max_code_length is not None else config.max_code_length
            )
            self.chars_per_code_unit = (
                chars_per_code_unit
                if chars_per_code_unit is not None
                else config.chars_per_code_unit
            )
            self.cache_size = cache_size if cache_size is not None else config.cache_size
        else:
            from myspellchecker.core.config.algorithm_configs import (
                PhoneticConfig as _PhoneticConfig,
            )

            self._config = _PhoneticConfig()
            self.max_code_length = max_code_length if max_code_length is not None else 10
            self.chars_per_code_unit = chars_per_code_unit if chars_per_code_unit is not None else 6
            self.cache_size = cache_size if cache_size is not None else 4096

        self.ignore_tones = ignore_tones
        self.normalize_length = normalize_length
        self.normalize_nasals = normalize_nasals
        self.adaptive_length = adaptive_length

        # Closure-based LRU cache for encode()
        if self.cache_size > 0:
            self._encode_cached = lru_cache(maxsize=self.cache_size)(self._encode_impl)
        else:
            self._encode_cached = self._encode_impl  # type: ignore[assignment]

    def encode(self, text: str) -> str:
        """
        Generate phonetic code for Myanmar text.

        The encoding process:
        1. Normalize Unicode (NFC form)
        2. Map each character to phonetic group code
        3. Optionally ignore tone marks
        4. Optionally normalize vowel length
        5. Return concatenated phonetic code

        Uses LRU caching for performance optimization when encoding
        the same text multiple times.

        Args:
            text: Myanmar text (syllable or word)

        Returns:
            Phonetic code as string (e.g., "k-medial_r-vowel_a-n")

        Example:
            >>> hasher = PhoneticHasher()
            >>> hasher.encode("မြန်")
            'p-medial_r-vowel_a-n'
            >>> hasher.encode("မျန်")  # Wrong medial (ya instead of ra)
            'p-medial_y-vowel_a-n'
            >>> hasher.encode("မြန်မာ")
            'p-medial_r-vowel_a-n-p-vowel_a'
        """
        return self._encode_cached(text)

    def _encode_impl(self, text: str) -> str:
        """
        Internal implementation of phonetic encoding (cached).

        This method does the actual encoding work and is wrapped
        with LRU cache for performance.
        """
        if not text:
            return ""

        # Normalize to NFC form
        normalized = unicodedata.normalize(NORM_FORM, text)

        # Optional nasal normalization: Normalize common nasal endings to Anusvara (ံ)
        # This unifies phonetic representation of -an/-am sounds but loses distinction
        # between /n/, /m/, and /ŋ/ nasal sounds. Disabled by default.
        if self.normalize_nasals:
            # န် (U+1014 U+103A) -> ံ (U+1036)
            normalized = normalized.replace("\u1014\u103a", "\u1036")
            # မ် (U+1019 U+103A) -> ံ (U+1036)
            normalized = normalized.replace("\u1019\u103a", "\u1036")
            # င် (U+1004 U+103A) -> ံ (U+1036), BUT NOT when followed by Visarga (း)
            # Nga+Asat+Visarga (င်း) represents /ing/ sound (different from Anusvara)
            # Nga+Asat alone (င်) at syllable end represents nasalization (same as Anusvara)
            # First, protect င်း pattern by temporarily replacing it
            normalized = normalized.replace("\u1004\u103a\u1038", "\ue000")  # Temp marker
            # Now normalize remaining င် to ံ
            normalized = normalized.replace("\u1004\u103a", "\u1036")
            # Restore protected pattern
            normalized = normalized.replace("\ue000", "\u1004\u103a\u1038")

        phonetic_codes: list[str] = []

        for char in normalized:
            # Skip Virama (stacking marker) to allow matching stacked/unstacked forms
            if char == "\u1039":
                continue

            # Skip if should ignore tones
            if self.ignore_tones and char in MYANMAR_TONES:
                continue

            # Get phonetic code for character
            if char in CHAR_TO_PHONETIC:
                code = CHAR_TO_PHONETIC[char]

                phonetic_codes.append(code)
            else:
                # Character not in phonetic mapping
                # Keep it as-is (might be punctuation, space, etc.)
                if char.strip():  # Not whitespace
                    # Use Unicode code point as fallback
                    phonetic_codes.append(f"U{ord(char):04X}")

        # Join codes with separator
        phonetic_str = "-".join(phonetic_codes)

        # Calculate adaptive code length for compound words
        effective_max_length = self.max_code_length
        if self.adaptive_length:
            # Scale max length based on input text length
            # Compound words (longer input) get proportionally longer codes
            input_length = len(normalized)
            # Adaptive formula: base + (input_length / chars_per_code_unit) * scale_factor
            # This ensures compound words preserve more phonetic information
            min_input = self._config.adaptive_min_input_length
            scale = self._config.adaptive_scale_factor
            adaptive_bonus = max(0, (input_length - min_input) // self.chars_per_code_unit) * scale
            effective_max_length = self.max_code_length + adaptive_bonus
            # Cap at reasonable maximum to prevent excessive memory usage
            cap = self._config.adaptive_code_length_cap
            effective_max_length = min(effective_max_length, cap)

        # Truncate at code boundary when possible to avoid splitting mid-token.
        # Only use boundary-aware cut if it preserves at least half the limit;
        # Myanmar code names like "medial_r" are long, so early separators
        # would discard too much information.
        if len(phonetic_str) > effective_max_length:
            cut_pos = phonetic_str.rfind("-", 0, effective_max_length)
            if cut_pos > effective_max_length // 2:
                phonetic_str = phonetic_str[:cut_pos]
            else:
                phonetic_str = phonetic_str[:effective_max_length]

        return phonetic_str

    def clear_cache(self) -> None:
        """
        Clear the phonetic encoding cache.

        Call this method if the hasher configuration changes or
        to free memory.
        """
        if self.cache_size > 0 and hasattr(self._encode_cached, "cache_clear"):
            self._encode_cached.cache_clear()

    def cache_info(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache hits, misses, size, and maxsize
            or empty dict if caching is disabled.
        """
        if self.cache_size > 0 and hasattr(self._encode_cached, "cache_info"):
            info = self._encode_cached.cache_info()
            return {
                "hits": info.hits,
                "misses": info.misses,
                "size": info.currsize,
                "maxsize": info.maxsize,
            }
        return {}

    def similar(self, code1: str, code2: str, max_distance: int = 1) -> bool:
        """
        Check if two phonetic codes are similar.

        Uses simple string comparison with edit distance.

        Args:
            code1: First phonetic code
            code2: Second phonetic code
            max_distance: Maximum edit distance to consider similar (default: 1)

        Returns:
            True if codes are similar, False otherwise

        Example:
            >>> hasher = PhoneticHasher()
            >>> code1 = hasher.encode("မြန်")
            >>> code2 = hasher.encode("မျန်")
            >>> hasher.similar(code1, code2, max_distance=2)
            True
        """
        if code1 == code2:
            return True

        # Use Levenshtein distance for phonetic code comparison
        distance = levenshtein_distance(code1, code2)
        return distance <= max_distance

    def get_phonetic_variants(self, text: str) -> set[str]:
        """
        Generate phonetically similar variants of a text.

        Creates variations by substituting characters with phonetically
        or visually similar ones.

        Args:
            text: Myanmar text (syllable or word)

        Returns:
            Set of phonetically similar variants (including original)

        Example:
            >>> hasher = PhoneticHasher()
            >>> variants = hasher.get_phonetic_variants("မြန်")
            >>> print(variants)
            {'မြန်', 'မျန်', 'ဗြန်', 'ပြန်', ...}

        Note:
            This generates many variants and should be used carefully.
            For large-scale usage, consider limiting to most common confusions.
        """
        if not text:
            return set()

        variants: set[str] = {text}  # Include original

        # Normalize to NFC
        normalized = unicodedata.normalize(NORM_FORM, text)

        # Add nasal-normalized variant (န်/မ် -> ံ)
        # This helps match words like နိုင်ငံ (official) vs နိုင်ငန် (typo)
        nasal_normalized = normalized.replace("\u1014\u103a", "\u1036").replace(
            "\u1019\u103a", "\u1036"
        )
        if nasal_normalized != normalized:
            variants.add(nasal_normalized)

        # Check for whole-word colloquial substitutions
        if text in COLLOQUIAL_SUBSTITUTIONS:
            variants.update(COLLOQUIAL_SUBSTITUTIONS[text])

        # Generate variants by substituting each character
        for i, char in enumerate(normalized):
            if char in CHAR_TO_PHONETIC:
                phonetic_code = CHAR_TO_PHONETIC[char]
                similar_chars = PHONETIC_GROUPS.get(phonetic_code, [])

                # Substitute with each similar character
                for similar_char in similar_chars:
                    if similar_char != char:
                        variant = normalized[:i] + similar_char + normalized[i + 1 :]
                        variants.add(variant)

            if char in VISUAL_SIMILAR:
                for confusable in VISUAL_SIMILAR[char]:
                    variant = normalized[:i] + confusable + normalized[i + 1 :]
                    variants.add(variant)

            # G2P-based phonetic equivalents (covers aspiration, voicing,
            # retroflex mergers, and other systematic confusions from
            # g2p_mappings.yaml homophone groups)
            g2p_equivalents = get_phonetic_equivalents(char)
            for equiv_char in g2p_equivalents:
                if equiv_char != char:
                    variant = normalized[:i] + equiv_char + normalized[i + 1 :]
                    variants.add(variant)

        return variants

    def get_tonal_variants(self, text: str) -> set[str]:
        """
        Generate tonal variants of a text.

        Creates variations by modifying tone marks and vowel lengths.
        This is critical for catching "Contextual Homophones" (Real-word errors).

        Args:
            text: Myanmar text (syllable or word)

        Returns:
            Set of tonal variants (including original)

        Example:
            >>> hasher = PhoneticHasher()
            >>> variants = hasher.get_tonal_variants("ကား")
            >>> print(variants)
            {'ကား', 'ကာ', 'က'}
        """
        if not text:
            return set()

        variants: set[str] = {text}

        # Normalize to NFC
        normalized = unicodedata.normalize(NORM_FORM, text)

        # Simple permutation strategy:
        # Iterate through chars, if char is in TONAL_GROUPS, generate variants
        # Limit to 1 change per call to avoid explosion? No, let's try all
        # combinations for short strings.
        # For now, let's just substitute single tonal positions to keep it fast.

        for i, char in enumerate(normalized):
            if char in TONAL_GROUPS:
                for tone_var in TONAL_GROUPS[char]:
                    if tone_var != char:
                        variant = normalized[:i] + tone_var + normalized[i + 1 :]
                        variants.add(variant)

        # Handle multi-character tonal group keys (e.g., "ော") that the
        # single-char iteration above cannot reach.
        for multi_key, tone_vars in TONAL_GROUPS.items():
            if len(multi_key) < 2:
                continue
            idx = normalized.find(multi_key)
            while idx != -1:
                for tone_var in tone_vars:
                    if tone_var != multi_key:
                        variant = normalized[:idx] + tone_var + normalized[idx + len(multi_key) :]
                        variants.add(variant)
                idx = normalized.find(multi_key, idx + 1)

        # Also handle appending tone marks to the end if it ends in a vowel that can take a tone
        # (This is a simplified heuristic)
        if normalized:
            last_char = normalized[-1]
            # If ends in vowel 'aa' (102C/102B), try adding visarga
            if last_char in ["\u102c", "\u102b"]:
                variants.add(normalized + "\u1038")  # add visarga
                variants.add(normalized + "\u1037")  # add dot below

            # If ends in visarga, try removing it
            if last_char == "\u1038":
                variants.add(normalized[:-1])
                variants.add(normalized[:-1] + "\u1037")

            # If ends in asat (U+103A), try adding visarga, dot-below insertion,
            # or swapping to dot-below.
            # E.g., တန် → တန်း (missing visarga), ဖြည် → ဖြည့် (insert dot-below)
            if last_char == "\u103a":
                variants.add(normalized + "\u1038")  # append visarga after asat
                variants.add(normalized[:-1] + "\u1037")  # swap asat → dot-below
                # Insert dot-below before asat (creaky tone + final stop)
                variants.add(normalized[:-1] + "\u1037\u103a")

            # If dot-below + asat at end (pre-normalization order), remove dot-below
            if len(normalized) >= 2 and normalized[-2:] == "\u1037\u103a":
                variants.add(normalized[:-2] + "\u103a")

            # If asat + dot-below at end (post-normalization order), remove dot-below
            if len(normalized) >= 2 and normalized[-2:] == "\u103a\u1037":
                variants.add(normalized[:-1])  # remove dot-below, keep asat

            # If ends in dot-below (U+1037) without asat adjacent, try swapping to asat
            if last_char == "\u1037" and (len(normalized) < 2 or normalized[-2] not in ("\u103a",)):
                variants.add(normalized[:-1] + "\u103a")  # swap dot-below → asat

        return variants

    def compute_phonetic_similarity(self, s1: str, s2: str) -> float:
        """
        Compute graded phonetic similarity score between two Myanmar strings.

        This method returns a continuous similarity score in [0.0, 1.0] that
        accounts for Myanmar-specific phonetic relationships:
        - Characters in the same phonetic group (e.g., aspirated pairs) get higher similarity
        - Characters with known substitution costs get proportionally weighted scores
        - Visual confusables are treated as highly similar
        - Different character classes (consonant vs vowel) reduce similarity

        Note: Uses positional character alignment which can underestimate
        similarity for medial insertion/deletion pairs. The phonetic
        code comparison (secondary signal) partially compensates.

        Args:
            s1: First Myanmar string
            s2: Second Myanmar string

        Returns:
            Float in [0.0, 1.0] where:
            - 1.0 = phonetically identical
            - 0.8-0.99 = highly similar (same phonetic group, minor variations)
            - 0.5-0.79 = moderately similar (related sounds)
            - 0.0-0.49 = dissimilar (different sounds/classes)

        Example:
            >>> hasher = PhoneticHasher()
            >>> # Same phonetic group (aspirated pairs)
            >>> hasher.compute_phonetic_similarity("က", "ခ")
            0.85
            >>> # Medial confusion (very common error)
            >>> hasher.compute_phonetic_similarity("မျ", "မြ")
            0.9
            >>> # Different consonant classes
            >>> hasher.compute_phonetic_similarity("က", "သ")
            0.3
        """
        if s1 == s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        # Normalize to NFC
        s1 = unicodedata.normalize(NORM_FORM, s1)
        s2 = unicodedata.normalize(NORM_FORM, s2)

        if s1 == s2:
            return 1.0

        # Get phonetic codes for both strings
        code1 = self.encode(s1)
        code2 = self.encode(s2)

        # If phonetic codes match exactly, very high similarity
        if code1 == code2:
            return self._config.matching_code_similarity

        # Calculate character-level similarity with Myanmar-specific weighting
        max_len = max(len(s1), len(s2))
        min_len = min(len(s1), len(s2))

        # Length difference penalty (proportional to length difference)
        length_penalty = (max_len - min_len) / max_len * self._config.length_penalty_weight

        # Calculate weighted character similarity
        total_score = 0.0
        comparisons = 0

        for i in range(min_len):
            c1 = s1[i]
            c2 = s2[i]

            if c1 == c2:
                total_score += 1.0
            else:
                # Check Myanmar substitution costs (most specific)
                char_sim = self._get_character_similarity(c1, c2)
                total_score += char_sim

            comparisons += 1

        # Handle extra characters as partial mismatches
        extra_chars = max_len - min_len
        for _ in range(extra_chars):
            total_score += 0.0  # Extra chars contribute 0
            comparisons += 1

        # Calculate base similarity score
        if comparisons == 0:
            base_score = 0.0
        else:
            base_score = total_score / comparisons

        # Apply length penalty and ensure bounds
        final_score = max(0.0, min(1.0, base_score - length_penalty))

        # Boost if phonetic codes are similar (within edit distance 1-2)
        code_distance = levenshtein_distance(code1, code2)
        max_code_len = max(len(code1), len(code2))
        if max_code_len > 0:
            code_similarity = 1.0 - (code_distance / max_code_len)
            # Blend character-level and code-level similarity
            # Weight code similarity higher for longer strings
            code_weight = min(
                self._config.code_weight_cap,
                len(s1) / self._config.code_weight_divisor,
            )
            final_score = (1.0 - code_weight) * final_score + code_weight * code_similarity

        return max(0.0, min(1.0, final_score))

    def _get_character_similarity(self, c1: str, c2: str) -> float:
        """
        Get phonetic similarity between two Myanmar characters.

        Returns a score in [0.0, 1.0] based on:
        1. Myanmar substitution costs (if defined)
        2. Phonetic group membership (same group = similar)
        3. Visual confusability
        4. Character class (consonant, vowel, medial, tone)

        Args:
            c1: First character
            c2: Second character

        Returns:
            Similarity score in [0.0, 1.0]
        """
        if c1 == c2:
            return 1.0

        # 1. Check Myanmar substitution costs (highest priority)
        # MYANMAR_SUBSTITUTION_COSTS maps cost values where lower = more similar
        multiplier = self._config.cost_to_similarity_multiplier
        if c1 in MYANMAR_SUBSTITUTION_COSTS:
            costs = MYANMAR_SUBSTITUTION_COSTS[c1]
            if c2 in costs:
                # Convert cost to similarity: cost 0.1 -> sim 0.925
                cost = costs[c2]
                return max(0.0, 1.0 - cost * multiplier)

        if c2 in MYANMAR_SUBSTITUTION_COSTS:
            costs = MYANMAR_SUBSTITUTION_COSTS[c2]
            if c1 in costs:
                cost = costs[c1]
                return max(0.0, 1.0 - cost * multiplier)

        # 2. Check visual confusability
        confusable_sim = self._config.confusable_char_similarity
        if c1 in VISUAL_SIMILAR and c2 in VISUAL_SIMILAR[c1]:
            return confusable_sim
        if c2 in VISUAL_SIMILAR and c1 in VISUAL_SIMILAR[c2]:
            return confusable_sim

        # 3. Check phonetic group membership
        code1 = CHAR_TO_PHONETIC.get(c1)
        code2 = CHAR_TO_PHONETIC.get(c2)

        if code1 and code2:
            if code1 == code2:
                # Same phonetic group (e.g., both labial consonants)
                return self._config.same_phonetic_group_similarity

            # Different groups but related categories
            # Consonant-consonant: moderate similarity
            # Vowel-vowel: moderate similarity
            # Cross-category: low similarity
            consonant_prefixes = (
                "p",
                "t",
                "k",
                "c",
                "l",
                "r",
                "y",
                "w",
                "s",
                "h",
                "m",
                "n",
            )

            is_c1_consonant = any(code1.startswith(p) for p in consonant_prefixes)
            is_c2_consonant = any(code2.startswith(p) for p in consonant_prefixes)
            is_c1_vowel = code1.startswith("vowel_")
            is_c2_vowel = code2.startswith("vowel_")
            is_c1_medial = code1.startswith("medial_")
            is_c2_medial = code2.startswith("medial_")

            # Same category (consonant-consonant, vowel-vowel)
            if (is_c1_consonant and is_c2_consonant) or (is_c1_vowel and is_c2_vowel):
                return self._config.same_category_diff_group_similarity
            if is_c1_medial and is_c2_medial:
                return self._config.medial_confusion_similarity

            # Cross-category: low similarity
            return self._config.cross_category_similarity

        # 4. Fallback: no phonetic info, use low similarity
        return self._config.fallback_similarity
