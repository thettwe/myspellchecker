"""Lightweight NER configuration (no heavy dependencies).

Separated from ``ner_model`` so that ``SpellCheckerConfig`` can import
``NERConfig`` without pulling in segmenter / transformer modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NERConfig:
    """
    Configuration for NER models.

    Attributes:
        enabled: Whether NER is enabled.
        model_type: Model type ("heuristic" or "transformer").
        model_name: HuggingFace model name for transformer NER.
        device: Device to run on (-1 for CPU, 0+ for GPU).
        confidence_threshold: Minimum confidence to accept entity.
        heuristic_confidence: Confidence score for heuristic NER results (default: 0.7).
        batch_size: Batch size for inference.
        cache_size: LRU cache size for predictions.
        fallback_to_heuristic: Use heuristic NER if transformer fails.
        ner_entity_types: Entity types to suppress FPs for (default: ["PER"]).
            Add "LOC" to also suppress place-name false positives.
        loc_confidence_threshold: Higher confidence threshold for LOC entities
            due to common-noun/place-name ambiguity in Myanmar (default: 0.85).
    """

    enabled: bool = True
    model_type: str = "heuristic"  # "heuristic" or "transformer"
    model_name: str = "chuuhtetnaing/myanmar-ner-model"  # HuggingFace model name
    device: int = -1  # -1 for CPU, 0+ for GPU
    confidence_threshold: float = 0.5
    heuristic_confidence: float = 0.7  # Confidence for heuristic NER results
    batch_size: int = 32
    cache_size: int = 1000
    fallback_to_heuristic: bool = True
    ner_entity_types: list[str] = field(default_factory=lambda: ["PER"])
    loc_confidence_threshold: float = 0.85

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.model_type not in ("heuristic", "transformer"):
            raise ValueError(f"Invalid model_type: {self.model_type}")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0 <= self.heuristic_confidence <= 1:
            raise ValueError("heuristic_confidence must be between 0 and 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not 0 <= self.loc_confidence_threshold <= 1:
            raise ValueError("loc_confidence_threshold must be between 0 and 1")
        valid_entity_types = {"PER", "LOC", "ORG", "DATE", "NUM", "TIME"}
        for t in self.ner_entity_types:
            if t not in valid_entity_types:
                raise ValueError(f"Invalid entity type: {t}. Must be one of {valid_entity_types}")
