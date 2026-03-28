"""Tests for detection_registry.py, detection_rules.py, and detector_data.py."""

from pathlib import Path

import pytest

from myspellchecker.core.detection_registry import (
    POST_NORM_DETECTOR_SEQUENCE,
    DetectorEntry,
)
from myspellchecker.core.detection_rules import (
    DetectionRules,
    _load_yaml,
    load_collocation_rules,
    load_compound_confusion,
    load_orthographic_corrections,
    load_particle_confusion,
)
from myspellchecker.core.detector_data import (
    STRUCTURAL_ERROR_TYPES,
    TEXT_DETECTOR_CONFIDENCES,
    norm_dict,
    norm_dict_context,
    norm_dict_tuple,
    norm_set,
)

# ---------------------------------------------------------------------------
# DetectorEntry and POST_NORM_DETECTOR_SEQUENCE
# ---------------------------------------------------------------------------


class TestDetectorEntry:
    def test_entry_has_method_name_and_description(self):
        entry = DetectorEntry(method_name="_detect_foo", description="Test detector")
        assert entry.method_name == "_detect_foo"
        assert entry.description == "Test detector"

    def test_entry_is_frozen(self):
        entry = DetectorEntry(method_name="_detect_foo")
        with pytest.raises(AttributeError):
            entry.method_name = "_detect_bar"

    def test_default_description_is_empty(self):
        entry = DetectorEntry(method_name="_detect_foo")
        assert entry.description == ""


class TestPostNormDetectorSequence:
    def test_sequence_is_not_empty(self):
        assert len(POST_NORM_DETECTOR_SEQUENCE) > 0

    def test_all_entries_are_detector_entry_instances(self):
        for entry in POST_NORM_DETECTOR_SEQUENCE:
            assert isinstance(entry, DetectorEntry)

    def test_all_method_names_start_with_detect(self):
        for entry in POST_NORM_DETECTOR_SEQUENCE:
            assert entry.method_name.startswith("_detect_")

    def test_all_method_names_are_unique(self):
        names = [e.method_name for e in POST_NORM_DETECTOR_SEQUENCE]
        assert len(names) == len(set(names))

    def test_broken_stacking_precedes_colloquial_contractions(self):
        names = [e.method_name for e in POST_NORM_DETECTOR_SEQUENCE]
        stacking_idx = names.index("_detect_broken_stacking")
        colloquial_idx = names.index("_detect_colloquial_contractions")
        assert stacking_idx < colloquial_idx

    def test_particle_confusion_follows_medial_confusion(self):
        names = [e.method_name for e in POST_NORM_DETECTOR_SEQUENCE]
        medial_idx = names.index("_detect_medial_confusion")
        particle_idx = names.index("_detect_particle_confusion")
        assert medial_idx < particle_idx

    def test_register_mixing_runs_after_particle_confusion(self):
        names = [e.method_name for e in POST_NORM_DETECTOR_SEQUENCE]
        particle_idx = names.index("_detect_particle_confusion")
        register_idx = names.index("_detect_register_mixing")
        assert register_idx > particle_idx

    def test_punctuation_errors_is_last(self):
        last = POST_NORM_DETECTOR_SEQUENCE[-1]
        assert last.method_name == "_detect_punctuation_errors"


# ---------------------------------------------------------------------------
# detector_data.py — normalization helpers
# ---------------------------------------------------------------------------


class TestNormHelpers:
    def test_norm_set_normalizes_myanmar_text(self):
        result = norm_set(["ကို", "မြန်"])
        assert isinstance(result, frozenset)
        assert len(result) == 2

    def test_norm_set_with_empty_iterable(self):
        result = norm_set([])
        assert result == frozenset()

    def test_norm_dict_normalizes_keys_and_string_values(self):
        result = norm_dict({"ကို": "ခို"})
        assert len(result) == 1

    def test_norm_dict_normalizes_list_values(self):
        result = norm_dict({"ကို": ["ခို", "ကိုး"]})
        values = list(result.values())
        assert isinstance(values[0], list)
        assert len(values[0]) == 2

    def test_norm_dict_tuple_normalizes_tuple_values(self):
        result = norm_dict_tuple({"pattern": ("wrong", "correct")})
        assert len(result) == 1
        val = list(result.values())[0]
        assert isinstance(val, tuple) and len(val) == 2

    def test_norm_dict_context_normalizes_nested_tuples(self):
        result = norm_dict_context({"ကို": ("ခို", ("ပါ", "တယ်"))})
        assert len(result) == 1
        val = list(result.values())[0]
        assert isinstance(val, tuple) and len(val) == 2
        assert isinstance(val[1], tuple)


# ---------------------------------------------------------------------------
# detector_data.py — constants
# ---------------------------------------------------------------------------


class TestDetectorDataConstants:
    def test_structural_error_types_contains_known_types(self):
        assert "dangling_particle" in STRUCTURAL_ERROR_TYPES
        assert "tense_mismatch" in STRUCTURAL_ERROR_TYPES
        assert "register_mixing" in STRUCTURAL_ERROR_TYPES

    def test_text_detector_confidences_has_entries(self):
        assert len(TEXT_DETECTOR_CONFIDENCES) > 0

    def test_all_confidence_values_are_between_0_and_1(self):
        for key, val in TEXT_DETECTOR_CONFIDENCES.items():
            assert 0.0 <= val <= 1.0, f"{key} has confidence {val} outside [0, 1]"


# ---------------------------------------------------------------------------
# detection_rules.py — YAML loading
# ---------------------------------------------------------------------------


class TestLoadYaml:
    def test_returns_none_for_missing_file(self):
        result = _load_yaml(Path("/nonexistent/path.yaml"), "test")
        assert result is None

    def test_returns_none_for_invalid_yaml(self, tmp_path):
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text(": :\n  - [\n", encoding="utf-8")
        result = _load_yaml(bad_file, "test")
        assert result is None


class TestLoadOrthographicCorrections:
    def test_returns_expected_keys_with_empty_data(self, tmp_path):
        yaml_file = tmp_path / "ortho.yaml"
        yaml_file.write_text("{}", encoding="utf-8")
        result = load_orthographic_corrections(yaml_file)
        assert "medial_confusion_unconditional" in result
        assert "colloquial_contractions" in result
        assert "stacking_completions" in result

    def test_returns_defaults_for_missing_file(self):
        result = load_orthographic_corrections(Path("/nonexistent.yaml"))
        assert result["medial_confusion_unconditional"] == {}


class TestLoadCompoundConfusion:
    def test_returns_expected_keys(self, tmp_path):
        yaml_file = tmp_path / "compounds.yaml"
        yaml_file.write_text("{}", encoding="utf-8")
        result = load_compound_confusion(yaml_file)
        assert "ha_htoe_compounds" in result
        assert "aspirated_compounds" in result
        assert "consonant_confusion_compounds" in result


class TestLoadParticleConfusion:
    def test_returns_expected_keys(self, tmp_path):
        yaml_file = tmp_path / "particles.yaml"
        yaml_file.write_text("{}", encoding="utf-8")
        result = load_particle_confusion(yaml_file)
        assert "particle_confusion" in result
        assert "dangling_particles" in result
        assert "missing_asat_particles" in result
        assert "particle_misuse_rules" in result


class TestLoadCollocationRules:
    def test_returns_empty_for_missing_file(self):
        result = load_collocation_rules(Path("/nonexistent.yaml"))
        assert result == []

    def test_parses_valid_collocation_entry(self, tmp_path):
        import yaml

        yaml_file = tmp_path / "collocations.yaml"
        data = {
            "collocations": [
                {
                    "wrong_word": "ကို",
                    "correct_word": "ခို",
                    "context_words": ["ပါ"],
                    "direction": "left",
                    "window": 3,
                }
            ]
        }
        yaml_file.write_text(yaml.dump(data, allow_unicode=True), encoding="utf-8")
        result = load_collocation_rules(yaml_file)
        assert len(result) == 1
        assert result[0]["direction"] == "left"
        assert result[0]["window"] == 3


# ---------------------------------------------------------------------------
# DetectionRules container
# ---------------------------------------------------------------------------


class TestDetectionRulesContainer:
    def test_loads_from_real_yaml_files(self):
        rules = DetectionRules()
        # Should have loaded without error; check a few attributes exist
        assert hasattr(rules, "medial_confusion_unconditional")
        assert hasattr(rules, "particle_confusion")
        assert hasattr(rules, "collocation_rules")

    def test_particle_confusion_is_dict(self):
        rules = DetectionRules()
        assert isinstance(rules.particle_confusion, dict)

    def test_dangling_particles_is_frozenset(self):
        rules = DetectionRules()
        assert isinstance(rules.dangling_particles, frozenset)
