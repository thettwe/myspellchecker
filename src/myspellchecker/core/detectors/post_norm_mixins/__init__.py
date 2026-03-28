"""Post-normalization detector sub-mixins for PostNormalizationDetectorsMixin.

Each mixin provides a group of related methods extracted from
``post_normalization.py`` to reduce file size while preserving the exact
same method signatures and behaviour.
"""

from myspellchecker.core.detectors.post_norm_mixins.collocation_detection_mixin import (
    CollocationDetectionMixin,
)
from myspellchecker.core.detectors.post_norm_mixins.compound_detection_mixin import (
    CompoundDetectionMixin,
)
from myspellchecker.core.detectors.post_norm_mixins.medial_confusion_mixin import (
    MedialConfusionMixin,
)
from myspellchecker.core.detectors.post_norm_mixins.particle_detection_mixin import (
    ParticleDetectionMixin,
)

__all__ = [
    "CollocationDetectionMixin",
    "CompoundDetectionMixin",
    "MedialConfusionMixin",
    "ParticleDetectionMixin",
]
