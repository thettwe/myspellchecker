"""
Myanmar Tone Disambiguation Module.

This module handles tone-ambiguous words in Myanmar text by using context
to infer the correct tonal spelling or meaning.

Myanmar Tone System:
1. Low tone (unmarked)
2. Creaky tone (with ့, dot below / auk-myit)
3. High tone (with း, visarga)
4. Checked tone (syllable closed by ်, asat)

Many syllables are written identically but have different meanings based on tone.
This module provides context-based disambiguation to improve spell checking accuracy.
"""

from __future__ import annotations

from typing import Any

from myspellchecker.core.config.text_configs import ToneConfig

__all__ = [
    "TONE_AMBIGUOUS",
    "TONE_MARK_ERRORS",
    "ToneDisambiguator",
    "create_disambiguator",
]

# Default Tone configuration (module-level singleton)
_default_tone_config = ToneConfig()

# Tone-ambiguous words with their context patterns
# Format: {word: {context_type: (patterns, correct_form, meaning)}}
TONE_AMBIGUOUS: dict[str, dict[str, tuple[tuple[str, ...], str, str]]] = {
    # ကျ - "to fall" (verb)
    # Note: ကျောင်း/ကြောင်း confusion is handled in ambiguous_words.yaml, not here.
    "ကျ": {
        "verb": (
            ("တယ်", "သည်", "မယ်", "ခဲ့", "နေ", "ပြီ", "လိုက်", "သွား"),
            "ကျ",
            "to fall (verb)",
        ),
    },
    # မ - female prefix vs negative prefix
    "မ": {
        "female": (
            ("သမီး", "မိန်းမ", "မယား", "မိဘ", "အမ"),
            "မ",
            "female prefix",
        ),
        "negative": (
            ("ဘူး", "ဟုတ်ဘူး", "ရဘူး", "တော့ဘူး"),
            "မ",
            "negative prefix",
        ),
    },
    # သား - son/offspring (correct as-is; tiger is "ကျား", not a tonal variant of "သား")
    "သား": {
        "family": (
            ("သမီး", "မိသား", "သားသမီး", "လင်", "မယား", "မိဘ", "အဖေ", "အမေ"),
            "သား",
            "son",
        ),
    },
    # က - subject marker vs letter name
    "က": {
        "particle": (
            ("ကို", "ကြောင့်", "နဲ့", "မှာ", "တွင်", "၌"),
            "က",
            "subject marker",
        ),
        "letter": (
            ("အက္ခရာ", "စာလုံး", "ခ", "ဂ", "ဃ", "င"),
            "က",
            "letter Ka",
        ),
    },
    # ငါ - I/me vs fish (ငါး)
    "ငါ": {
        "pronoun": (
            ("ငါ့", "က", "ကို", "အတွက်", "ရဲ့"),
            "ငါ",
            "I/me",
        ),
        "fish_missing_tone": (
            ("ကောင်", "ငုတ်", "ပင်လယ်", "ရေ", "ချက်", "ကြော်"),
            "ငါး",  # Fish needs high tone mark
            "fish",
        ),
    },
    # သံ - sound/iron vs three (သုံး)
    "သံ": {
        "sound_metal": (
            ("သံစဉ်", "သံရုံး", "သံချပ်", "ကြား", "ထွက်"),
            "သံ",
            "sound/iron",
        ),
        "number_error": (
            ("ခု", "ယောက်", "ကောင်", "လုံး", "စု"),
            "သုံး",  # Three (number)
            "three",
        ),
    },
    # ခု - classifier vs now
    "ခု": {
        "classifier": (
            ("တစ်", "နှစ်", "သုံး", "လေး", "ငါး", "ယောက်"),
            "ခု",
            "classifier",
        ),
        "now": (
            ("အခု", "ယခု", "ဒီ"),
            "ခု",
            "now/this moment",
        ),
    },
    # ပဲ - only/bean
    "ပဲ": {
        "emphasis": (
            ("တယ်", "ပါ", "သည်", "ဘဲ", "လိုပဲ"),
            "ပဲ",
            "only/just (emphasis)",
        ),
        "bean": (
            ("စိုက်", "ပင်", "စေ့", "ထောင်း", "ဆီ"),
            "ပဲ",
            "bean/pea",
        ),
    },
    # တော် - royal/suitable vs forest (တော)
    "တော်": {
        "royal": (
            ("မင်း", "နန်း", "ဘုရင်", "မိဖုရား"),
            "တော်",
            "royal/honorific",
        ),
        "suitable": (
            ("ပြီ", "သင့်", "ကောင်း", "လောက်"),
            "တော်",
            "suitable/enough",
        ),
    },
    "တော": {
        "forest": (
            ("ထဲ", "အုပ်", "တောင်", "သစ်", "ပင်", "တိရစ္ဆာန်"),
            "တော",
            "forest",
        ),
    },
}

# Common tone mark errors (missing or wrong tone marks)
TONE_MARK_ERRORS: dict[str, str] = {
    # Missing visarga (း)
    # "ငါ" removed: valid first-person pronoun; handled via TONE_AMBIGUOUS with context
    # Removed: "သား": "သား့" — "သား" (son/offspring) is correct; tiger is "ကျား"
    "သုံ": "သုံး",  # three (not a valid standalone syllable, always a typo)
    "လေ": "လေး",  # four/wind
    # "ခြေး" removed: medial confusion (ya-yit vs wa-hswe), not a tone error
    # Common particle errors
    # "လာ" removed: valid high-frequency verb ("to come"); handled via TONE_AMBIGUOUS with context
    "သလာ": "သလား",  # question particle
}


class ToneDisambiguator:
    """
    Context-based tone disambiguation for Myanmar text.

    Uses surrounding words to infer the correct tonal spelling
    when multiple interpretations are possible.
    """

    def __init__(self, config: ToneConfig | None = None):
        """
        Initialize the tone disambiguator.

        Args:
            config: ToneConfig instance for settings. If config contains
                tone_ambiguous_map or tone_errors_map, they will be used
                instead of the hard-coded defaults.
        """
        self.config = config or _default_tone_config
        self.context_window = self.config.context_window
        self.min_confidence = self.config.min_confidence

        # Use config-provided maps if available, otherwise use hard-coded defaults
        # This allows tone_rules.yaml to be wired in via GrammarRuleConfig
        if self.config.tone_ambiguous_map:
            self.ambiguous_words = self._convert_yaml_to_runtime_format(
                self.config.tone_ambiguous_map
            )
        else:
            self.ambiguous_words = TONE_AMBIGUOUS

        if self.config.tone_errors_map:
            self.tone_errors = self.config.tone_errors_map
        else:
            self.tone_errors = TONE_MARK_ERRORS

    def _convert_yaml_to_runtime_format(
        self, yaml_map: dict[str, Any]
    ) -> dict[str, dict[str, tuple[tuple[str, ...], str, str]]]:
        """
        Convert YAML-loaded tone ambiguous map to runtime format.

        The YAML format uses nested dicts with 'patterns', 'correct_form', 'meaning'.
        The runtime format uses tuples: (patterns_tuple, correct_form, meaning).

        Args:
            yaml_map: Dict loaded from tone_rules.yaml ambiguous_words section.

        Returns:
            Dict in the format expected by disambiguate() and other methods.
        """
        result: dict[str, dict[str, tuple[tuple[str, ...], str, str]]] = {}

        for word, contexts in yaml_map.items():
            if not isinstance(contexts, dict):
                continue

            word_contexts: dict[str, tuple[tuple[str, ...], str, str]] = {}
            for context_type, data in contexts.items():
                if not isinstance(data, dict):
                    continue

                patterns = data.get("patterns", [])
                correct_form = data.get("correct_form", word)
                meaning = data.get("meaning", "")

                # Convert patterns list to tuple for consistency
                if isinstance(patterns, list):
                    patterns_tuple = tuple(patterns)
                else:
                    patterns_tuple = (patterns,) if patterns else ()

                word_contexts[context_type] = (patterns_tuple, correct_form, meaning)

            if word_contexts:
                result[word] = word_contexts

        return result

    def _get_context_words(self, words: list[str], index: int) -> tuple[list[str], list[str]]:
        """Get context words around the target index."""
        start = max(0, index - self.context_window)
        end = min(len(words), index + self.context_window + 1)

        left_context = words[start:index]
        right_context = words[index + 1 : end]

        return left_context, right_context

    def _match_context(self, patterns: tuple[str, ...], context: list[str]) -> float:
        """
        Calculate how well context matches the patterns.

        Returns:
            Score between 0.0 and 1.0
        """
        if not patterns or not context:
            return 0.0

        matches = 0
        for word in context:
            for pattern in patterns:
                if pattern in word or word in pattern:
                    matches += 1
                    break

        return matches / max(len(patterns), len(context))

    def disambiguate(self, words: list[str], index: int) -> tuple[str, str, float] | None:
        """
        Disambiguate a word at the given index using context.

        Args:
            words: List of words in the sentence
            index: Index of the word to disambiguate

        Returns:
            Tuple of (correct_form, meaning, confidence) or None if not ambiguous
        """
        if index < 0 or index >= len(words):
            return None

        word = words[index]
        if word not in self.ambiguous_words:
            return None

        left_ctx, right_ctx = self._get_context_words(words, index)
        full_context = left_ctx + right_ctx

        best_match: tuple[str, str, float] | None = None
        best_score = 0.0

        for _context_type, (patterns, form, meaning) in self.ambiguous_words[word].items():
            score = self._match_context(patterns, full_context)

            if score > best_score:
                best_score = score
                best_match = (form, meaning, score)

        # Only return if we have reasonable confidence
        if best_match and best_score >= self.min_confidence:
            return best_match

        return None

    def suggest_tone_correction(self, word: str, context: list[str]) -> tuple[str, float] | None:
        """
        Suggest tone mark correction based on context.

        Args:
            word: The word to check
            context: Surrounding words for context

        Returns:
            Tuple of (corrected_word, confidence) or None if no correction needed
        """
        # Check for common tone mark errors
        if word in self.tone_errors:
            correction = self.tone_errors[word]

            # Validate with context if available
            if context:
                # Check if context suggests the correction is appropriate
                context_str = " ".join(context)

                # Number context for tone-marked numbers
                if correction in ("ငါး", "သုံး", "လေး"):
                    number_patterns = ("ခု", "ယောက်", "ကောင်", "လုံး", "ခါ")
                    if any(p in context_str for p in number_patterns):
                        return (correction, 0.85)

                # Question context for လား / သလား
                if correction in ("လား", "သလား"):
                    question_patterns = ("ဘာ", "ဘယ်", "ဘယ်သူ", "ဘယ်လို")
                    if any(p in context_str for p in question_patterns):
                        return (correction, 0.90)

            # Fallback for unconditional corrections: words that are never valid
            # standalone syllables (e.g., "သုံ" → "သုံး", "သလာ" → "သလား")
            if correction in ("သုံး", "သလား"):
                return (correction, 0.60)

        return None

    def check_sentence(self, words: list[str]) -> list[tuple[int, str, str, float]]:
        """
        Check a sentence for potential tone-related corrections.

        Args:
            words: List of words in the sentence

        Returns:
            List of (index, original, suggestion, confidence) tuples
        """
        corrections = []

        for i, word in enumerate(words):
            # Check tone-ambiguous words
            result = self.disambiguate(words, i)
            if result:
                correct_form, meaning, confidence = result
                if correct_form != word:
                    corrections.append((i, word, correct_form, confidence))
                continue

            # Check for missing tone marks
            context = self._get_context_words(words, i)
            full_context = context[0] + context[1]
            tone_result = self.suggest_tone_correction(word, full_context)
            if tone_result:
                correction, confidence = tone_result
                corrections.append((i, word, correction, confidence))

        return corrections


def create_disambiguator(config: ToneConfig | None = None) -> ToneDisambiguator:
    """
    Factory function to create a ToneDisambiguator instance.

    Args:
        config: ToneConfig instance for settings.

    Returns:
        ToneDisambiguator instance
    """
    return ToneDisambiguator(config=config)
