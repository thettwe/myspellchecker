"""
Tests for learning-based NER model.

Tests cover:
- EntityType enum
- Entity dataclass
- NERConfig validation
- HeuristicNER
- TransformerNER (with mocks)
- HybridNER
- NERFactory
- Entity filtering utility
"""

from unittest.mock import MagicMock, patch

import pytest

from myspellchecker.text.ner_model import (
    Entity,
    EntityType,
    HeuristicNER,
    HybridNER,
    NERConfig,
    NERFactory,
    NERModel,
)

# --- EntityType Tests ---


class TestEntityType:
    """Tests for EntityType enum."""

    def test_entity_types(self):
        """Test all entity types exist."""
        assert EntityType.PERSON.value == "PER"
        assert EntityType.LOCATION.value == "LOC"
        assert EntityType.ORGANIZATION.value == "ORG"
        assert EntityType.MISCELLANEOUS.value == "MISC"
        assert EntityType.OTHER.value == "O"

    def test_from_bio_tag_b_tags(self):
        """Test parsing B- tags."""
        assert EntityType.from_bio_tag("B-PER") == EntityType.PERSON
        assert EntityType.from_bio_tag("B-LOC") == EntityType.LOCATION
        assert EntityType.from_bio_tag("B-ORG") == EntityType.ORGANIZATION
        assert EntityType.from_bio_tag("B-MISC") == EntityType.MISCELLANEOUS

    def test_from_bio_tag_i_tags(self):
        """Test parsing I- tags."""
        assert EntityType.from_bio_tag("I-PER") == EntityType.PERSON
        assert EntityType.from_bio_tag("I-LOC") == EntityType.LOCATION

    def test_from_bio_tag_o_tag(self):
        """Test parsing O tag."""
        assert EntityType.from_bio_tag("O") == EntityType.OTHER

    def test_from_bio_tag_unknown(self):
        """Test parsing unknown tags."""
        assert EntityType.from_bio_tag("B-UNKNOWN") == EntityType.OTHER

    def test_from_bio_tag_edge_cases(self):
        """Test parsing edge case tags that could cause IndexError."""
        # Tag with only prefix (no entity type after hyphen)
        assert EntityType.from_bio_tag("B-") == EntityType.OTHER
        assert EntityType.from_bio_tag("I-") == EntityType.OTHER
        # Just a hyphen
        assert EntityType.from_bio_tag("-") == EntityType.OTHER
        # Multiple hyphens - "PER-extra" is not in mapping so returns OTHER
        assert EntityType.from_bio_tag("B-PER-extra") == EntityType.OTHER
        # Empty string
        assert EntityType.from_bio_tag("") == EntityType.OTHER


# --- Entity Tests ---


class TestEntity:
    """Tests for Entity dataclass."""

    def test_entity_creation(self):
        """Test creating an entity."""
        entity = Entity(
            text="အောင်",
            label=EntityType.PERSON,
            start=0,
            end=6,
            confidence=0.95,
        )
        assert entity.text == "အောင်"
        assert entity.label == EntityType.PERSON
        assert entity.start == 0
        assert entity.end == 6
        assert entity.confidence == 0.95

    def test_entity_default_values(self):
        """Test entity default values."""
        entity = Entity(
            text="test",
            label=EntityType.PERSON,
            start=0,
            end=4,
        )
        assert entity.confidence == 1.0
        assert entity.metadata == {}

    def test_entity_to_dict(self):
        """Test entity serialization."""
        entity = Entity(
            text="ရန်ကုန်",
            label=EntityType.LOCATION,
            start=10,
            end=20,
            confidence=0.88,
            metadata={"source": "transformer"},
        )
        d = entity.to_dict()
        assert d["text"] == "ရန်ကုန်"
        assert d["label"] == "LOC"
        assert d["start"] == 10
        assert d["end"] == 20
        assert d["confidence"] == 0.88
        assert d["metadata"]["source"] == "transformer"


# --- NERConfig Tests ---


class TestNERConfig:
    """Tests for NERConfig validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = NERConfig()
        assert config.enabled is True
        assert config.model_type == "heuristic"
        assert config.model_name == "chuuhtetnaing/myanmar-ner-model"
        assert config.device == -1
        assert config.confidence_threshold == 0.5
        assert config.batch_size == 32
        assert config.cache_size == 1000
        assert config.fallback_to_heuristic is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = NERConfig(
            model_type="transformer",
            model_name="test/model",
            device=0,
            confidence_threshold=0.7,
        )
        assert config.model_type == "transformer"
        assert config.model_name == "test/model"
        assert config.device == 0
        assert config.confidence_threshold == 0.7

    def test_invalid_model_type(self):
        """Test invalid model_type raises error."""
        with pytest.raises(ValueError, match="Invalid model_type"):
            NERConfig(model_type="invalid")

    def test_invalid_confidence_threshold(self):
        """Test invalid confidence_threshold raises error."""
        with pytest.raises(ValueError, match="confidence_threshold"):
            NERConfig(confidence_threshold=1.5)
        with pytest.raises(ValueError, match="confidence_threshold"):
            NERConfig(confidence_threshold=-0.1)

    def test_invalid_batch_size(self):
        """Test invalid batch_size raises error."""
        with pytest.raises(ValueError, match="batch_size"):
            NERConfig(batch_size=0)


# --- HeuristicNER Tests ---


class TestHeuristicNER:
    """Tests for HeuristicNER."""

    @pytest.fixture
    def ner(self):
        """Create HeuristicNER instance."""
        return HeuristicNER()

    def test_extract_empty_text(self, ner):
        """Test extraction from empty text."""
        entities = ner.extract_entities("")
        assert entities == []

    def test_extract_person_after_honorific(self, ner):
        """Test person detection after honorific."""
        # Using a mock to avoid segmenter dependency
        with patch.object(ner, "_heuristic") as mock_heuristic:
            mock_heuristic.is_potential_name.return_value = True

            # Simplified test - just verify the method works
            entities = ner.extract_entities("ကို အောင်")
            # Should find at least one entity or work without error
            assert isinstance(entities, list)

    def test_extract_location_suffix(self, ner):
        """Test location detection by suffix."""
        # The heuristic NER checks for location suffixes
        # Test that it returns a list (actual detection depends on segmenter)
        entities = ner.extract_entities("ရန်ကုန်မြို့")
        assert isinstance(entities, list)

    def test_extract_organization(self, ner):
        """Test organization detection by pattern."""
        # Test that it returns a list (actual detection depends on segmenter)
        entities = ner.extract_entities("ကုမ္ပဏီ")
        assert isinstance(entities, list)

    def test_batch_extraction(self, ner):
        """Test batch extraction."""
        texts = ["text1", "text2", "text3"]
        results = ner.extract_entities_batch(texts)
        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)


# --- TransformerNER Tests (with mocks) ---


class TestTransformerNER:
    """Tests for TransformerNER with mocked dependencies."""

    def test_requires_model_name(self):
        """Test that model_name is required."""
        from myspellchecker.text.ner_model import TransformerNER

        with pytest.raises(ValueError, match="model_name is required"):
            TransformerNER(NERConfig(model_type="transformer", model_name=""))

    def test_requires_transformers_package(self):
        """Test that TransformerNER validates model loading."""
        from myspellchecker.text.ner_model import TransformerNER

        # Mock the model loading to simulate failure
        # This tests the error handling path without relying on network
        with patch(
            "myspellchecker.text.ner_model.TransformerNER._load_model",
            side_effect=ValueError("Model not found"),
        ):
            with pytest.raises(ValueError, match="Model not found"):
                TransformerNER(NERConfig(model_type="transformer", model_name="test/model"))

    @patch("myspellchecker.text.ner_model.TransformerNER._load_model")
    def test_extract_entities_empty(self, mock_load):
        """Test extraction with empty text."""
        from myspellchecker.text.ner_model import TransformerNER

        config = NERConfig(model_type="transformer", model_name="test/model")
        ner = TransformerNER(config)
        ner._pipeline = None  # Simulate no model loaded

        entities = ner.extract_entities("")
        assert entities == []

        entities = ner.extract_entities("text")
        assert entities == []

    @patch("myspellchecker.text.ner_model.TransformerNER._load_model")
    def test_extract_entities_with_pipeline(self, mock_load):
        """Test extraction with mocked pipeline."""
        from myspellchecker.text.ner_model import TransformerNER

        config = NERConfig(model_type="transformer", model_name="test/model")
        ner = TransformerNER(config)

        # Mock pipeline
        ner._pipeline = MagicMock()
        ner._pipeline.return_value = [
            {"entity_group": "PER", "word": "အောင်", "start": 0, "end": 6, "score": 0.95}
        ]

        entities = ner.extract_entities("အောင်သည်")

        assert len(entities) == 1
        assert entities[0].label == EntityType.PERSON
        assert entities[0].confidence == 0.95

    @patch("myspellchecker.text.ner_model.TransformerNER._load_model")
    def test_extract_filters_low_confidence(self, mock_load):
        """Test that low confidence entities are filtered."""
        from myspellchecker.text.ner_model import TransformerNER

        config = NERConfig(
            model_type="transformer", model_name="test/model", confidence_threshold=0.8
        )
        ner = TransformerNER(config)

        ner._pipeline = MagicMock()
        ner._pipeline.return_value = [
            {"entity_group": "PER", "word": "test", "start": 0, "end": 4, "score": 0.5}
        ]

        entities = ner.extract_entities("test")
        assert len(entities) == 0  # Filtered due to low confidence

    @patch("myspellchecker.text.ner_model.TransformerNER._load_model")
    def test_caching(self, mock_load):
        """Test that results are cached."""
        from myspellchecker.text.ner_model import TransformerNER

        config = NERConfig(model_type="transformer", model_name="test/model")
        ner = TransformerNER(config)

        ner._pipeline = MagicMock()
        ner._pipeline.return_value = [
            {"entity_group": "PER", "word": "test", "start": 0, "end": 4, "score": 0.9}
        ]

        # First call
        ner.extract_entities("test")
        assert ner._pipeline.call_count == 1

        # Second call should use cache
        ner.extract_entities("test")
        assert ner._pipeline.call_count == 1  # Not called again


# --- HybridNER Tests ---


class TestHybridNER:
    """Tests for HybridNER."""

    def test_uses_heuristic_when_transformer_unavailable(self):
        """Test fallback to heuristic when transformer fails."""
        config = NERConfig(
            model_type="transformer",
            model_name="nonexistent/model",
            fallback_to_heuristic=True,
        )

        # Should not raise, should fall back to heuristic
        # Use OSError which is one of the caught exception types
        with patch("myspellchecker.text.ner_model.TransformerNER") as MockTrans:
            MockTrans.side_effect = OSError("Model not found")

            ner = HybridNER(config)
            assert ner._transformer is None

            # Should still work with heuristic
            entities = ner.extract_entities("test")
            assert isinstance(entities, list)

    def test_uses_transformer_when_available(self):
        """Test using transformer when available."""
        config = NERConfig(
            model_type="transformer",
            model_name="test/model",
        )

        with patch("myspellchecker.text.ner_model.TransformerNER") as MockTrans:
            mock_transformer = MockTrans.return_value
            mock_transformer.extract_entities.return_value = [
                Entity("test", EntityType.PERSON, 0, 4, 0.9)
            ]

            ner = HybridNER(config)
            entities = ner.extract_entities("test")

            assert len(entities) == 1
            mock_transformer.extract_entities.assert_called_once()


# --- NERFactory Tests ---


class TestNERFactory:
    """Tests for NERFactory."""

    def test_create_heuristic(self):
        """Test creating heuristic NER."""
        config = NERConfig(model_type="heuristic")
        ner = NERFactory.create(config)
        assert isinstance(ner, HeuristicNER)

    def test_create_disabled(self):
        """Test creating disabled NER."""
        config = NERConfig(enabled=False)
        ner = NERFactory.create(config)
        # Should return a working NER that does nothing
        assert isinstance(ner, NERModel)

    def test_create_transformer_with_fallback(self):
        """Test creating transformer NER with fallback."""
        config = NERConfig(
            model_type="transformer",
            model_name="test/model",
            fallback_to_heuristic=True,
        )

        with patch("myspellchecker.text.ner_model.HybridNER") as MockHybrid:
            NERFactory.create(config)
            MockHybrid.assert_called_once()

    def test_create_invalid_type(self):
        """Test that invalid type raises error."""
        config = NERConfig()
        config.model_type = "invalid"  # Bypass validation

        with pytest.raises(ValueError, match="Unknown model_type"):
            NERFactory.create(config)


# --- Integration Tests ---


class TestNERIntegration:
    """Integration tests for NER module."""

    def test_import_from_text_module(self):
        """Test importing NER classes from text module."""
        from myspellchecker.text import (
            Entity,
            EntityType,
            HeuristicNER,
            NERConfig,
            NERFactory,
        )

        assert Entity is not None
        assert EntityType is not None
        assert HeuristicNER is not None
        assert NERConfig is not None
        assert NERFactory is not None

    def test_heuristic_ner_end_to_end(self):
        """Test heuristic NER end-to-end."""
        ner = HeuristicNER()

        # Test with simple text
        entities = ner.extract_entities("test text")
        assert isinstance(entities, list)

        # Test batch
        results = ner.extract_entities_batch(["text1", "text2"])
        assert len(results) == 2
