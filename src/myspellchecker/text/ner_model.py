"""
Learning-Based Named Entity Recognition (NER) for Myanmar Text.

This module provides transformer-based NER models for identifying named entities
in Myanmar text. It supports multiple entity types and provides confidence scores.

Entity Types:
- PER (Person): Personal names
- LOC (Location): Place names, geographic locations
- ORG (Organization): Companies, institutions, groups
- MISC (Miscellaneous): Other named entities

Example:
    >>> from myspellchecker.text.ner_model import TransformerNER, NERConfig
    >>>
    >>> # Using pre-trained model
    >>> config = NERConfig(model_name="thettwe/myanmar-ner-base")
    >>> ner = TransformerNER(config)
    >>>
    >>> entities = ner.extract_entities("ကိုအောင်သည် ရန်ကုန်မြို့တွင် နေသည်။")
    >>> for entity in entities:
    ...     print(f"{entity.text}: {entity.label} ({entity.confidence:.2f})")

    >>> # Using with spell checker
    >>> from myspellchecker import SpellChecker
    >>> from myspellchecker.core.config import SpellCheckerConfig, NERConfig
    >>>
    >>> config = SpellCheckerConfig(
    ...     ner=NERConfig(enabled=True, model_name="thettwe/myanmar-ner-base")
    ... )
    >>> checker = SpellChecker(config=config)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from myspellchecker.segmenters import DefaultSegmenter, Segmenter
from myspellchecker.text.ner_config import NERConfig
from myspellchecker.utils.cache import LRUCache
from myspellchecker.utils.logging_utils import get_logger

__all__ = [
    "EntityType",
    "Entity",
    "NERConfig",
    "NERModel",
    "HeuristicNER",
    "HybridNER",
    "TransformerNER",
    "NERFactory",
]


class EntityType(str, Enum):
    """Named entity types for Myanmar NER."""

    PERSON = "PER"
    LOCATION = "LOC"
    ORGANIZATION = "ORG"
    DATE = "DATE"
    NUMBER = "NUM"
    TIME = "TIME"
    MISCELLANEOUS = "MISC"
    OTHER = "O"  # Not an entity

    @classmethod
    def from_bio_tag(cls, tag: str) -> "EntityType":
        """
        Convert BIO tag to EntityType.

        Args:
            tag: BIO tag (e.g., "B-PER", "I-LOC", "O")

        Returns:
            Corresponding EntityType.
        """
        if not tag or tag == "O":
            return cls.OTHER
        # Extract entity type from B-XXX or I-XXX format
        # Use maxsplit=1 and validate to prevent IndexError
        if "-" in tag:
            parts = tag.split("-", 1)
            entity_type = parts[1] if len(parts) > 1 and parts[1] else tag
        else:
            entity_type = tag

        mapping = {
            "PER": cls.PERSON,
            "LOC": cls.LOCATION,
            "ORG": cls.ORGANIZATION,
            "DATE": cls.DATE,
            "NUM": cls.NUMBER,
            "TIME": cls.TIME,
            "MISC": cls.MISCELLANEOUS,
        }
        return mapping.get(entity_type, cls.OTHER)


@dataclass
class Entity:
    """
    Represents a detected named entity.

    Attributes:
        text: The entity text.
        label: Entity type.
        start: Start character position in original text.
        end: End character position in original text.
        confidence: Confidence score (0.0 to 1.0).
        metadata: Additional metadata (e.g., alternative labels).
    """

    text: str
    label: EntityType
    start: int
    end: int
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert entity to dictionary for serialization."""
        return {
            "text": self.text,
            "label": self.label.value,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class NERModel(ABC):
    """
    Abstract base class for NER models.

    Subclasses must implement extract_entities and extract_entities_batch.
    """

    @abstractmethod
    def extract_entities(self, text: str) -> list[Entity]:
        """
        Extract named entities from text.

        Args:
            text: Input text.

        Returns:
            List of detected entities.
        """
        raise NotImplementedError

    @abstractmethod
    def extract_entities_batch(self, texts: list[str]) -> list[list[Entity]]:
        """
        Extract entities from multiple texts.

        Args:
            texts: List of input texts.

        Returns:
            List of entity lists, one per input text.
        """
        raise NotImplementedError


class HeuristicNER(NERModel):
    """
    Heuristic-based NER using patterns and rules.

    This is the fallback NER that uses honorifics, common name patterns,
    and whitelists to detect named entities. It's fast but less accurate
    than transformer-based models.
    """

    def __init__(
        self,
        config: NERConfig | None = None,
        segmenter: Segmenter | None = None,
        allow_extended_myanmar: bool = False,
    ):
        """
        Initialize heuristic NER.

        Args:
            config: Optional NER configuration.
            segmenter: Optional segmenter for word tokenization. If not provided,
                      DefaultSegmenter will be used (supports dependency injection).
            allow_extended_myanmar: Whether to allow Extended Myanmar characters
                when creating the default segmenter.
        """
        self.config = config or NERConfig()
        self.logger = get_logger(__name__)
        self._segmenter = segmenter
        self._allow_extended_myanmar = allow_extended_myanmar
        # Cache a default segmenter instance to avoid recreating on every call
        self._default_segmenter: DefaultSegmenter | None = None

        # Import the existing NameHeuristic
        from myspellchecker.text.ner import NameHeuristic

        self._heuristic = NameHeuristic()

        # Known entity patterns (from named_entities.yaml gazetteer)
        from myspellchecker.text.ner import get_gazetteer_data

        gaz = get_gazetteer_data()
        self._location_suffixes = set(gaz.location_suffixes)
        self._org_patterns = set(gaz.org_patterns)

        # Place-name dictionary (townships + states + short state names + ethnic groups)
        self._known_places = gaz.all_places
        self._state_short = gaz.states_regions | gaz.ethnic_groups

    def extract_entities(self, text: str) -> list[Entity]:
        """Extract entities using heuristics."""
        if not text or not text.strip():
            return []

        # Use injected segmenter or lazily-created default instance
        if self._segmenter:
            segmenter = self._segmenter
        else:
            if self._default_segmenter is None:
                self._default_segmenter = DefaultSegmenter(
                    allow_extended_myanmar=self._allow_extended_myanmar
                )
            segmenter = self._default_segmenter
        words = segmenter.segment_words(text)

        entities = []
        current_pos = 0

        for i, word in enumerate(words):
            # Find word position in text
            word_start = text.find(word, current_pos)
            if word_start == -1:
                current_pos += len(word)
                continue
            word_end = word_start + len(word)
            current_pos = word_end

            prev_word = words[i - 1] if i > 0 else None
            prev_prev_word = words[i - 2] if i > 1 else None

            # Check for person names
            if self._heuristic.is_potential_name(word, prev_word, prev_prev_word):
                entities.append(
                    Entity(
                        text=word,
                        label=EntityType.PERSON,
                        start=word_start,
                        end=word_end,
                        confidence=self.config.heuristic_confidence,
                        metadata={"source": "heuristic"},
                    )
                )
                continue

            # Check for locations — dictionary match first (high confidence),
            # then suffix-based fallback (lower confidence).
            loc_matched = False

            # Dictionary: exact match on word or word+next (e.g. "မိုးကုတ်"+"မြို့")
            if word in self._known_places or word in self._state_short:
                # Check if next word is a location suffix → extend span
                next_word = words[i + 1] if i + 1 < len(words) else None
                span_text = word
                span_end = word_end
                if next_word and next_word in self._location_suffixes:
                    nw_start = text.find(next_word, word_end)
                    if nw_start != -1 and nw_start - word_end <= 1:
                        span_text = text[word_start : nw_start + len(next_word)]
                        span_end = nw_start + len(next_word)
                entities.append(
                    Entity(
                        text=span_text,
                        label=EntityType.LOCATION,
                        start=word_start,
                        end=span_end,
                        confidence=0.95,
                        metadata={"source": "heuristic", "pattern": "dictionary"},
                    )
                )
                loc_matched = True

            if not loc_matched:
                for suffix in self._location_suffixes:
                    if word.endswith(suffix):
                        entities.append(
                            Entity(
                                text=word,
                                label=EntityType.LOCATION,
                                start=word_start,
                                end=word_end,
                                confidence=self.config.heuristic_confidence,
                                metadata={"source": "heuristic", "pattern": "suffix"},
                            )
                        )
                        break

            # Check for organizations
            for pattern in self._org_patterns:
                if pattern in word:
                    entities.append(
                        Entity(
                            text=word,
                            label=EntityType.ORGANIZATION,
                            start=word_start,
                            end=word_end,
                            confidence=self.config.heuristic_confidence,
                            metadata={"source": "heuristic", "pattern": "contains"},
                        )
                    )
                    break

        return entities

    def extract_entities_batch(self, texts: list[str]) -> list[list[Entity]]:
        """Extract entities from multiple texts."""
        return [self.extract_entities(text) for text in texts]


class TransformerNER(NERModel):
    """
    Transformer-based NER using HuggingFace models.

    Uses a token classification model for sequence labeling with BIO tags.
    Supports any HuggingFace token classification model.

    Example:
        >>> ner = TransformerNER.from_pretrained("thettwe/myanmar-ner-base")
        >>> entities = ner.extract_entities("ကိုအောင်သည် ရန်ကုန်တွင် နေသည်။")

        >>> # Or with custom config
        >>> config = NERConfig(model_name="thettwe/myanmar-ner-base", device=0)
        >>> ner = TransformerNER(config)
    """

    def __init__(self, config: NERConfig):
        """
        Initialize transformer NER.

        Args:
            config: NER configuration with model_name.
        """
        self.config = config
        self.logger = get_logger(__name__)
        self._model = None
        self._tokenizer: Any = None
        self._pipeline = None
        self._cache: LRUCache[list[Entity]] = LRUCache(maxsize=config.cache_size)

        if not config.model_name:
            raise ValueError("model_name is required for TransformerNER")

        self._load_model()

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: int = -1,
        confidence_threshold: float = 0.5,
        **kwargs: Any,
    ) -> "TransformerNER":
        """
        Create TransformerNER from a pretrained HuggingFace model.

        This is a convenience factory method for loading models without
        explicitly creating an NERConfig.

        Args:
            model_name: HuggingFace model name (e.g., "thettwe/myanmar-ner-base").
            device: Device to run on (-1 for CPU, 0+ for GPU).
            confidence_threshold: Minimum confidence to accept entity.
            **kwargs: Additional arguments passed to NERConfig.

        Returns:
            Configured TransformerNER instance.

        Example:
            >>> ner = TransformerNER.from_pretrained("thettwe/myanmar-ner-base")
            >>> ner = TransformerNER.from_pretrained(
            ...     "thettwe/myanmar-ner-base",
            ...     device=0,
            ...     confidence_threshold=0.7
            ... )
        """
        config = NERConfig(
            model_type="transformer",
            model_name=model_name,
            device=device,
            confidence_threshold=confidence_threshold,
            **kwargs,
        )
        return cls(config)

    def _load_model(self) -> None:
        """Load the transformer model and tokenizer."""
        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

            self.logger.info(f"Loading NER model: {self.config.model_name}")

            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self._model = AutoModelForTokenClassification.from_pretrained(self.config.model_name)

            # Move to device
            device = self.config.device
            if device >= 0:
                import torch

                if torch.cuda.is_available():
                    self._model = self._model.to(f"cuda:{device}")  # type: ignore[union-attr,attr-defined]
                else:
                    self.logger.warning("CUDA not available, using CPU")
                    device = -1

            # Create pipeline for easy inference
            self._pipeline = pipeline(  # type: ignore[call-overload]
                "ner",
                model=self._model,
                tokenizer=self._tokenizer,
                device=device,
                aggregation_strategy="simple",
            )

            self.logger.info("NER model loaded successfully")

        except ImportError as e:
            raise ImportError(
                "transformers package is required for TransformerNER. "
                "Install with: pip install transformers torch"
            ) from e
        except (OSError, RuntimeError, ValueError) as e:
            self.logger.error(f"Failed to load NER model: {e}")
            raise

    def extract_entities(self, text: str) -> list[Entity]:
        """Extract entities using transformer model."""
        if not text or not text.strip():
            return []

        # Check cache (thread-safe LRU)
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        if self._pipeline is None:
            self.logger.debug("NER pipeline not initialized - returning empty results")
            return []

        try:
            results = self._pipeline(text)

            entities = []
            for result in results:
                entity_type = EntityType.from_bio_tag(result["entity_group"])
                if entity_type == EntityType.OTHER:
                    continue

                confidence = result["score"]
                if confidence < self.config.confidence_threshold:
                    continue

                entities.append(
                    Entity(
                        text=result["word"],
                        label=entity_type,
                        start=result["start"],
                        end=result["end"],
                        confidence=confidence,
                        metadata={"source": "transformer"},
                    )
                )

            # Cache result (thread-safe LRU with auto-eviction)
            self._cache.set(text, entities)

            return entities

        except (RuntimeError, KeyError, IndexError, TypeError) as e:
            self.logger.warning(f"NER extraction failed: {e}")
            return []

    def extract_entities_batch(self, texts: list[str]) -> list[list[Entity]]:
        """Extract entities from multiple texts efficiently."""
        if not texts:
            return []

        if self._pipeline is None:
            self.logger.debug("NER pipeline not initialized - returning empty batch results")
            return [[] for _ in texts]

        try:
            # Use pipeline batch processing
            all_results = self._pipeline(texts, batch_size=self.config.batch_size)

            all_entities = []
            for _text, results in zip(texts, all_results, strict=False):
                entities = []
                for result in results:
                    entity_type = EntityType.from_bio_tag(result["entity_group"])
                    if entity_type == EntityType.OTHER:
                        continue

                    confidence = result["score"]
                    if confidence < self.config.confidence_threshold:
                        continue

                    entities.append(
                        Entity(
                            text=result["word"],
                            label=entity_type,
                            start=result["start"],
                            end=result["end"],
                            confidence=confidence,
                            metadata={"source": "transformer"},
                        )
                    )
                all_entities.append(entities)

            return all_entities

        except (RuntimeError, KeyError, IndexError, TypeError) as e:
            self.logger.warning(f"Batch NER extraction failed: {e}")
            return [[] for _ in texts]


class HybridNER(NERModel):
    """
    Hybrid NER combining transformer and heuristic models.

    Uses transformer model as primary, falls back to heuristics
    for texts where transformer fails or has low confidence.
    """

    def __init__(self, config: NERConfig, allow_extended_myanmar: bool = False):
        """Initialize hybrid NER."""
        self.config = config
        self.logger = get_logger(__name__)
        self._transformer: TransformerNER | None = None
        self._heuristic = HeuristicNER(config, allow_extended_myanmar=allow_extended_myanmar)

        # Try to load transformer
        if config.model_name:
            try:
                self._transformer = TransformerNER(config)
            except (ImportError, OSError, RuntimeError, ValueError) as e:
                self.logger.warning(f"Transformer NER unavailable, using heuristics only: {e}")

    def extract_entities(self, text: str) -> list[Entity]:
        """Extract entities using hybrid approach."""
        # Try transformer first
        if self._transformer is not None:
            try:
                entities = self._transformer.extract_entities(text)
                if entities:
                    return entities
            except (RuntimeError, KeyError, IndexError, TypeError) as e:
                self.logger.debug(f"Transformer NER failed, falling back to heuristics: {e}")

        # Fallback to heuristics
        if self.config.fallback_to_heuristic:
            return self._heuristic.extract_entities(text)

        return []

    def extract_entities_batch(self, texts: list[str]) -> list[list[Entity]]:
        """Extract entities from multiple texts."""
        # Try batch processing with transformer
        if self._transformer is not None:
            try:
                results = self._transformer.extract_entities_batch(texts)
                return results
            except (RuntimeError, KeyError, IndexError, TypeError) as e:
                # Log the exception instead of silently suppressing
                self.logger.debug(f"Transformer batch extraction failed, falling back: {e}")

        # Fallback to individual heuristic processing
        if self.config.fallback_to_heuristic:
            return self._heuristic.extract_entities_batch(texts)

        return [[] for _ in texts]


class NERFactory:
    """Factory for creating NER models."""

    @staticmethod
    def create(config: NERConfig, allow_extended_myanmar: bool = False) -> NERModel:
        """
        Create NER model based on configuration.

        Args:
            config: NER configuration.
            allow_extended_myanmar: Whether to allow Extended Myanmar characters
                when creating segmenters for heuristic NER.

        Returns:
            NER model instance.
        """
        if not config.enabled:
            # Return a no-op NER that returns empty results
            return HeuristicNER(
                NERConfig(enabled=False),
                allow_extended_myanmar=allow_extended_myanmar,
            )

        if config.model_type == "heuristic":
            return HeuristicNER(config, allow_extended_myanmar=allow_extended_myanmar)

        if config.model_type == "transformer":
            if config.fallback_to_heuristic:
                return HybridNER(config, allow_extended_myanmar=allow_extended_myanmar)
            return TransformerNER(config)

        raise ValueError(f"Unknown model_type: {config.model_type}")
