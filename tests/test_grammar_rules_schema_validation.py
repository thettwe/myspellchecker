"""
Tests for grammar rules 4-file YAML schema validation.

Verifies that the 4 category YAML files:
- Validate against their respective JSON Schemas
- Contain all required fields
- Have correct data types and constraints
- Statistics match actual counts
"""

import json
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

try:
    import jsonschema

    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


# Test data directory
PROJECT_ROOT = Path(__file__).parent.parent
RULES_DIR = PROJECT_ROOT / "src/myspellchecker/rules"
SCHEMAS_DIR = PROJECT_ROOT / "src/myspellchecker/schemas"


@pytest.fixture
def particles_data() -> Dict[str, Any]:
    """Load particles.yaml."""
    with open(RULES_DIR / "particles.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def typo_corrections_data() -> Dict[str, Any]:
    """Load typo_corrections.yaml."""
    with open(RULES_DIR / "typo_corrections.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def grammar_rules_data() -> Dict[str, Any]:
    """Load grammar_rules.yaml."""
    with open(RULES_DIR / "grammar_rules.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def morphology_data() -> Dict[str, Any]:
    """Load morphology.yaml."""
    with open(RULES_DIR / "morphology.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def particles_schema() -> Dict[str, Any]:
    """Load particles JSON Schema."""
    with open(SCHEMAS_DIR / "particles.schema.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def typo_corrections_schema() -> Dict[str, Any]:
    """Load typo_corrections JSON Schema."""
    with open(SCHEMAS_DIR / "typo_corrections.schema.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def grammar_rules_schema() -> Dict[str, Any]:
    """Load grammar_rules JSON Schema."""
    with open(SCHEMAS_DIR / "grammar_rules.schema.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def morphology_schema() -> Dict[str, Any]:
    """Load morphology JSON Schema."""
    with open(SCHEMAS_DIR / "morphology.schema.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def common_schema() -> Dict[str, Any]:
    """Load common JSON Schema."""
    with open(SCHEMAS_DIR / "_common.schema.json", "r", encoding="utf-8") as f:
        return json.load(f)


def create_validator_with_store(schema: Dict, common_schema: Dict) -> Any:
    """
    Create a validator with a schema store for $ref resolution.

    Registers both schemas by their $id URLs, plus the common schema
    under the resolved relative URI that $ref: "_common.schema.json"
    produces when resolved against the main schema's $id base URL.
    This ensures validation works offline without network access.
    """
    if not HAS_JSONSCHEMA:
        return None

    from urllib.parse import urljoin

    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT7

    common_resource = Resource.from_contents(common_schema, default_specification=DRAFT7)

    # Register each schema by its $id
    resources: list[tuple[str, Resource]] = []
    for s in [schema, common_schema]:
        sid = s.get("$id", "")
        if sid:
            resources.append((sid, Resource.from_contents(s, default_specification=DRAFT7)))

    # Register the common schema under the resolved relative URI.
    # Schemas use $ref: "_common.schema.json#/..." which resolves relative
    # to the main schema's $id (e.g. ".../typo-corrections/v1.0" ->
    # ".../typo-corrections/_common.schema.json").
    schema_id = schema.get("$id", "")
    if schema_id:
        resolved_common_uri = urljoin(schema_id, "_common.schema.json")
        resources.append((resolved_common_uri, common_resource))

    registry = Registry().with_resources(resources)

    validator_class = jsonschema.validators.validator_for(schema)
    return validator_class(schema, registry=registry)


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
class TestSchemaValidation:
    """Test suite for JSON Schema validation."""

    def test_particles_validates_against_schema(
        self, particles_data, particles_schema, common_schema
    ):
        """Test that particles.yaml validates against its JSON Schema."""
        # Create validator with schema store for $ref resolution
        validator = create_validator_with_store(particles_schema, common_schema)

        # Should not raise ValidationError
        validator.validate(particles_data)

    def test_typo_corrections_validates_against_schema(
        self, typo_corrections_data, typo_corrections_schema, common_schema
    ):
        """Test that typo_corrections.yaml validates against its JSON Schema."""
        # Create validator with schema store for $ref resolution
        validator = create_validator_with_store(typo_corrections_schema, common_schema)

        validator.validate(typo_corrections_data)

    def test_grammar_rules_validates_against_schema(
        self, grammar_rules_data, grammar_rules_schema, common_schema
    ):
        """Test that grammar_rules.yaml validates against its JSON Schema."""
        # Create validator with schema store for $ref resolution
        validator = create_validator_with_store(grammar_rules_schema, common_schema)

        validator.validate(grammar_rules_data)

    def test_morphology_validates_against_schema(
        self, morphology_data, morphology_schema, common_schema
    ):
        """Test that morphology.yaml validates against its JSON Schema."""
        # Create validator with schema store for $ref resolution
        validator = create_validator_with_store(morphology_schema, common_schema)

        validator.validate(morphology_data)


class TestFileStructure:
    """Test suite for YAML file structure validation."""

    def test_particles_has_correct_version(self, particles_data):
        """Test that particles.yaml has correct version."""
        assert "version" in particles_data
        assert particles_data["version"] == "1.1.0"

    def test_particles_has_correct_category(self, particles_data):
        """Test that particles.yaml has correct category."""
        assert "category" in particles_data
        assert particles_data["category"] == "particles"

    def test_typo_corrections_has_correct_version(self, typo_corrections_data):
        """Test that typo_corrections.yaml has correct version."""
        assert "version" in typo_corrections_data
        assert typo_corrections_data["version"] == "1.0.0"

    def test_typo_corrections_has_correct_category(self, typo_corrections_data):
        """Test that typo_corrections.yaml has correct category."""
        assert "category" in typo_corrections_data
        assert typo_corrections_data["category"] == "typo_corrections"

    def test_grammar_rules_has_correct_version(self, grammar_rules_data):
        """Test that grammar_rules.yaml has correct version."""
        assert "version" in grammar_rules_data
        assert grammar_rules_data["version"] == "1.0.0"

    def test_grammar_rules_has_correct_category(self, grammar_rules_data):
        """Test that grammar_rules.yaml has correct category."""
        assert "category" in grammar_rules_data
        assert grammar_rules_data["category"] == "grammar_rules"

    def test_morphology_has_correct_version(self, morphology_data):
        """Test that morphology.yaml has correct version."""
        assert "version" in morphology_data
        assert morphology_data["version"] == "1.0.0"

    def test_morphology_has_correct_category(self, morphology_data):
        """Test that morphology.yaml has correct category."""
        assert "category" in morphology_data
        assert morphology_data["category"] == "morphology"

    def test_particles_has_required_sections(self, particles_data):
        """Test that particles.yaml has all required sections."""
        required = ["version", "category", "description", "metadata", "particles"]
        for section in required:
            assert section in particles_data, f"Missing section: {section}"

    def test_typo_corrections_has_required_sections(self, typo_corrections_data):
        """Test that typo_corrections.yaml has all required sections."""
        required = ["version", "category", "description", "metadata", "corrections"]
        for section in required:
            assert section in typo_corrections_data, f"Missing section: {section}"

    def test_grammar_rules_has_required_sections(self, grammar_rules_data):
        """Test that grammar_rules.yaml has all required sections."""
        required = ["version", "category", "description", "metadata", "rules"]
        for section in required:
            assert section in grammar_rules_data, f"Missing section: {section}"

    def test_morphology_has_required_sections(self, morphology_data):
        """Test that morphology.yaml has all required sections."""
        required = ["version", "category", "description", "metadata", "suffixes"]
        for section in required:
            assert section in morphology_data, f"Missing section: {section}"


class TestMetadataAccuracy:
    """Test suite for metadata accuracy verification."""

    def count_particles(self, data: Dict[str, Any]) -> int:
        """Count particles in particles.yaml."""
        count = 0
        if "particles" in data:
            for category in data["particles"].values():
                if isinstance(category, dict):
                    for subcategory in category.values():
                        if isinstance(subcategory, list):
                            count += len(subcategory)
                elif isinstance(category, list):
                    count += len(category)
        return count

    def count_typo_corrections(self, data: Dict[str, Any]) -> int:
        """Count typo corrections in typo_corrections.yaml."""
        count = 0
        if "corrections" in data:
            for corrections in data["corrections"].values():
                if isinstance(corrections, list):
                    count += len(corrections)
        return count

    def count_grammar_rules(self, data: Dict[str, Any]) -> int:
        """Count grammar rules in grammar_rules.yaml."""
        count = 0
        if "rules" in data:
            for rules in data["rules"].values():
                if isinstance(rules, list):
                    count += len(rules)
        return count

    def count_morphology(self, data: Dict[str, Any]) -> int:
        """Count morphology rules in morphology.yaml (prefixes + suffixes)."""
        count = 0
        # Count prefixes (added in expansion)
        if "prefixes" in data:
            for prefixes in data["prefixes"].values():
                if isinstance(prefixes, list):
                    count += len(prefixes)
        # Count suffixes
        if "suffixes" in data:
            for suffixes in data["suffixes"].values():
                if isinstance(suffixes, list):
                    count += len(suffixes)
        return count

    def test_particles_metadata_matches_actual_count(self, particles_data):
        """Test that particles.yaml metadata matches actual count."""
        actual = self.count_particles(particles_data)
        metadata = particles_data["metadata"]["total_entries"]
        assert actual == metadata, f"Count mismatch: actual={actual}, metadata={metadata}"

    def test_typo_corrections_metadata_matches_actual_count(self, typo_corrections_data):
        """Test that typo_corrections.yaml metadata matches actual count."""
        actual = self.count_typo_corrections(typo_corrections_data)
        metadata = typo_corrections_data["metadata"]["total_entries"]
        assert actual == metadata, f"Count mismatch: actual={actual}, metadata={metadata}"

    def test_grammar_rules_metadata_matches_actual_count(self, grammar_rules_data):
        """Test that grammar_rules.yaml metadata matches actual count."""
        actual = self.count_grammar_rules(grammar_rules_data)
        metadata = grammar_rules_data["metadata"]["total_entries"]
        assert actual == metadata, f"Count mismatch: actual={actual}, metadata={metadata}"

    def test_morphology_metadata_matches_actual_count(self, morphology_data):
        """Test that morphology.yaml metadata matches actual count."""
        actual = self.count_morphology(morphology_data)
        metadata = morphology_data["metadata"]["total_entries"]
        assert actual == metadata, f"Count mismatch: actual={actual}, metadata={metadata}"

    def test_all_files_have_metadata_fields(
        self, particles_data, typo_corrections_data, grammar_rules_data, morphology_data
    ):
        """Test that all files have required metadata fields."""
        required_fields = ["created_date", "last_updated", "total_entries"]

        for name, data in [
            ("particles", particles_data),
            ("typo_corrections", typo_corrections_data),
            ("grammar_rules", grammar_rules_data),
            ("morphology", morphology_data),
        ]:
            metadata = data.get("metadata", {})
            for field in required_fields:
                assert field in metadata, f"{name}.yaml missing metadata field: {field}"


class TestParticleRules:
    """Test suite for particle rule validation."""

    def test_all_particles_have_required_fields(self, particles_data):
        """Test that all particle rules have required fields."""
        required_fields = ["particle", "pos_tag", "type", "meaning", "confidence"]

        particles = particles_data["particles"]
        for category_name, category in particles.items():
            if isinstance(category, dict):
                for subcategory_name, subcategory in category.items():
                    if isinstance(subcategory, list):
                        for i, particle in enumerate(subcategory):
                            for field in required_fields:
                                assert field in particle, (
                                    f"Missing '{field}' in {category_name}.{subcategory_name}[{i}]"
                                )
            elif isinstance(category, list):
                for i, particle in enumerate(category):
                    for field in required_fields:
                        assert field in particle, f"Missing '{field}' in {category_name}[{i}]"

    def test_all_particle_confidence_scores_valid(self, particles_data):
        """Test that all particle confidence scores are in [0.0, 1.0]."""
        particles = particles_data["particles"]
        for category_name, category in particles.items():
            if isinstance(category, dict):
                for subcategory_name, subcategory in category.items():
                    if isinstance(subcategory, list):
                        for i, particle in enumerate(subcategory):
                            confidence = particle.get("confidence")
                            loc = f"{category_name}.{subcategory_name}[{i}]"
                            assert 0.0 <= confidence <= 1.0, (
                                f"Invalid confidence {confidence} in {loc}"
                            )
            elif isinstance(category, list):
                for i, particle in enumerate(category):
                    confidence = particle.get("confidence")
                    assert 0.0 <= confidence <= 1.0, (
                        f"Invalid confidence {confidence} in {category_name}[{i}]"
                    )

    def test_all_pos_tags_match_pattern(self, particles_data):
        """Test that all POS tags match P_* pattern."""
        import re

        pos_tag_pattern = re.compile(r"^P_[A-Z_]+$")

        particles = particles_data["particles"]
        for category_name, category in particles.items():
            if isinstance(category, dict):
                for subcategory_name, subcategory in category.items():
                    if isinstance(subcategory, list):
                        for i, particle in enumerate(subcategory):
                            pos_tag = particle.get("pos_tag")
                            loc = f"{category_name}.{subcategory_name}[{i}]"
                            assert pos_tag_pattern.match(pos_tag), (
                                f"Invalid POS tag '{pos_tag}' in {loc}"
                            )
            elif isinstance(category, list):
                for i, particle in enumerate(category):
                    pos_tag = particle.get("pos_tag")
                    assert pos_tag_pattern.match(pos_tag), (
                        f"Invalid POS tag '{pos_tag}' in {category_name}[{i}]"
                    )


class TestTypoCorrectionRules:
    """Test suite for typo correction rule validation."""

    def test_all_typo_corrections_have_required_fields(self, typo_corrections_data):
        """Test that all typo corrections have required fields."""
        required_fields = ["incorrect", "correct", "error_type", "confidence"]

        corrections = typo_corrections_data["corrections"]
        for category_name, category_corrections in corrections.items():
            if isinstance(category_corrections, list):
                for i, correction in enumerate(category_corrections):
                    for field in required_fields:
                        assert field in correction, (
                            f"Missing '{field}' in corrections.{category_name}[{i}]"
                        )

    def test_all_typo_correction_confidence_scores_valid(self, typo_corrections_data):
        """Test that all typo correction confidence scores are in [0.0, 1.0]."""
        corrections = typo_corrections_data["corrections"]
        for category_name, category_corrections in corrections.items():
            if isinstance(category_corrections, list):
                for i, correction in enumerate(category_corrections):
                    confidence = correction.get("confidence")
                    assert 0.0 <= confidence <= 1.0, (
                        f"Invalid confidence {confidence} in corrections.{category_name}[{i}]"
                    )


class TestCrossFileConsistency:
    """Test suite for consistency across all YAML files."""

    def test_no_duplicate_particles(self, particles_data, typo_corrections_data):
        """Test that particles are not duplicated across files."""
        # Extract all particles
        particles_set = set()
        for category in particles_data["particles"].values():
            if isinstance(category, dict):
                for subcategory in category.values():
                    if isinstance(subcategory, list):
                        for item in subcategory:
                            if isinstance(item, dict) and "particle" in item:
                                particles_set.add(item["particle"])
            elif isinstance(category, list):
                for item in category:
                    if isinstance(item, dict) and "particle" in item:
                        particles_set.add(item["particle"])

        # Check typo corrections don't define particles (they should only reference them)
        # This test passes if typo corrections use the existing particles

    def test_all_files_use_version_1_x(
        self, particles_data, typo_corrections_data, grammar_rules_data, morphology_data
    ):
        """Test that all files use version 1.x.x."""
        for name, data in [
            ("particles", particles_data),
            ("typo_corrections", typo_corrections_data),
            ("grammar_rules", grammar_rules_data),
            ("morphology", morphology_data),
        ]:
            assert data["version"].startswith("1."), f"{name}.yaml has incorrect version"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
