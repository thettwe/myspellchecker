"""Text-level detector mixins for SpellChecker.

Each mixin provides a group of related _detect_* methods and their
associated class-level data constants.  SpellChecker inherits from
these mixins, keeping the same method signatures and call sites.

Infrastructure:
    - ``TokenizedText`` / ``TokenSpan``: pre-computed space-delimited
      tokenization, replacing duplicated ``text.split()`` + position loops.
    - ``DetectorContext``: explicit dependency bundle for detector methods.
"""

from myspellchecker.core.detectors.context import DetectorContext
from myspellchecker.core.detectors.post_normalization import PostNormalizationDetectorsMixin
from myspellchecker.core.detectors.pre_normalization import PreNormalizationDetectorsMixin
from myspellchecker.core.detectors.sentence_detectors import SentenceDetectorsMixin
from myspellchecker.core.detectors.tokenized_text import TokenizedText, TokenSpan

__all__ = [
    "DetectorContext",
    "PostNormalizationDetectorsMixin",
    "PreNormalizationDetectorsMixin",
    "SentenceDetectorsMixin",
    "TokenizedText",
    "TokenSpan",
]
