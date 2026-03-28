"""Sentence-level detector sub-mixins for SentenceDetectorsMixin decomposition.

Each mixin provides a group of related methods extracted from
``sentence_detectors.py`` to reduce file size while preserving the exact
same method signatures and behaviour.
"""

from myspellchecker.core.detectors.sentence_mixins.register_mixing_mixin import (
    RegisterMixingMixin,
)
from myspellchecker.core.detectors.sentence_mixins.structure_detection_mixin import (
    StructureDetectionMixin,
)
from myspellchecker.core.detectors.sentence_mixins.tense_detection_mixin import (
    TenseDetectionMixin,
)

__all__ = [
    "RegisterMixingMixin",
    "StructureDetectionMixin",
    "TenseDetectionMixin",
]
