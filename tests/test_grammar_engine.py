from unittest.mock import MagicMock

from myspellchecker.core.config.grammar_configs import GrammarEngineConfig
from myspellchecker.grammar.engine import SyntacticRuleChecker


def test_syntactic_rule_checker_helpers():
    """Test helper methods."""
    checker = SyntacticRuleChecker(MagicMock())

    # _has_tag
    assert checker._has_tag("N|V", "N") is True
    assert checker._has_tag("N|V", "ADJ") is False
    assert checker._has_tag(None, "N") is False

    # _get_all_tags
    assert checker._get_all_tags("N|V") == {"N", "V"}
    assert checker._get_all_tags(None) == set()

    # _get_primary_tag
    assert checker._get_primary_tag("N|V") == "N"
    assert checker._get_primary_tag(None) == ""

    # Mock particle_tags config
    checker.config.particle_tags = {"word": "P_SENT"}
    assert checker._get_primary_tag(None, "word") == "P_SENT"


def test_check_particle_typos():
    """Test particle typo checking."""
    checker = SyntacticRuleChecker(MagicMock())

    # Mock config.get_particle_typo
    checker.config.get_particle_typo = MagicMock()

    # Case 1: After verb
    checker.config.get_particle_typo.return_value = {
        "correction": "correct",
        "context": "after_verb",
        "meaning": "m",
    }
    res = checker._check_particle_typos("typo", "V")
    assert res == ("correct", "After verb, should be correct (m)", 0.9)

    # Case 2: After noun
    checker.config.get_particle_typo.return_value = {
        "correction": "correct",
        "context": "after_noun",
        "meaning": "m",
    }
    res = checker._check_particle_typos("typo", "N")
    assert res == ("correct", "Likely correct (m)", 0.9)

    # Case 3: Default
    checker.config.get_particle_typo.return_value = {"correction": "correct", "meaning": "m"}
    res = checker._check_particle_typos("typo", None)
    assert res == ("correct", "m", 0.9)


def test_check_verb_particle_agreement():
    """Test verb-particle agreement."""
    checker = SyntacticRuleChecker(MagicMock())

    # Case 1: Typo after verb
    checker.config.get_particle_typo = MagicMock(
        return_value={"correction": "c", "context": "after_verb", "meaning": "m"}
    )
    res = checker._check_verb_particle_agreement("word", "V", None)
    assert res[0] == "c"

    # Case 2: Verb particle after non-verb
    checker.config.get_particle_typo.return_value = None
    checker.config.is_verb_particle = MagicMock(return_value=True)
    res = checker._check_verb_particle_agreement("မယ်", "N", None)
    assert res[0] == "မယ်"

    # Case 3: Word correction
    checker.config.is_verb_particle.return_value = False
    checker.config.get_word_correction = MagicMock(return_value={"correction": "c", "meaning": "m"})
    res = checker._check_verb_particle_agreement("word", None, None)
    assert res[0] == "c"


def test_check_medial_confusions():
    """Test medial confusion logic."""
    checker = SyntacticRuleChecker(MagicMock())
    checker.config.get_medial_confusion = MagicMock()

    # Case 1: After verb
    checker.config.get_medial_confusion.return_value = {
        "correction": "c",
        "context": "after_verb",
        "meaning": "m",
    }
    res = checker._check_medial_confusions("w", "V")
    assert res == ("c", "After verb: c (m)", 0.90)

    # Case 2: After noun
    checker.config.get_medial_confusion.return_value = {
        "correction": "c",
        "context": "after_noun",
        "meaning": "m",
    }
    res = checker._check_medial_confusions("w", "N")
    assert res == ("c", "After noun: c (m)", 0.85)

    # Case 3: Context dependent
    checker.config.get_medial_confusion.return_value = {
        "correction": "c",
        "context": "context_dependent",
        "meaning": "m",
    }
    res = checker._check_medial_confusions("w", None)
    assert res == ("c", "Possible: c (m)", 0.65)

    # Legacy hardcoded check removed - now config-driven
    # When config returns None, no correction is applied
    checker.config.get_medial_confusion.return_value = None
    res = checker._check_medial_confusions("ကျောင်း", "V")
    assert res is None  # Config-driven: no fallback to hardcoded patterns


def test_check_sentence_structure():
    """Test sentence structure rules."""
    checker = SyntacticRuleChecker(MagicMock())
    checker.config.is_sentence_final = MagicMock(return_value=False)

    # Rule 1: Missing final particle after verb (needs 2+ words)
    words = ["noun", "verb"]
    tags = ["N", "V"]
    errors = checker._check_sentence_structure(words, tags)
    assert any(e[2] == "verbတယ်" for e in errors)

    # Rule 3: Missing subject marker (requires 3+ words — 2-word "N V" is valid SOV)
    words = ["noun", "verb", "particle"]
    tags = ["N", "V", "PPM"]
    checker.config.get_invalid_sequence_error = MagicMock(return_value=None)
    errors = checker._check_sentence_structure(words, tags)
    # Should find missing subject marker error
    assert any(e[2] == "nounက" for e in errors)

    # Rule 4: Question without particle
    words = ["ဘယ်", "သွား"]
    tags = ["Q", "V"]
    # "သွား" is not final particle
    errors = checker._check_sentence_structure(words, tags)
    assert any(e[2] == "သွားလဲ" for e in errors)


def test_check_sentence_structure_question_context_prefers_question_form():
    """Merged question words should get question completions, not declarative fallback."""
    checker = SyntacticRuleChecker(MagicMock())
    checker.config.is_sentence_final = MagicMock(return_value=False)
    checker.config.get_invalid_sequence_error = MagicMock(return_value=None)

    # Merged question forms.
    errors_476 = checker._check_sentence_structure(["ခင်ဗျား", "ဘယ်သွား"], ["N", "V"])
    suggestions_476 = [e[2] for e in errors_476]
    assert "ဘယ်သွားလဲ" in suggestions_476
    assert "ဘယ်သွားတယ်" not in suggestions_476

    errors_477 = checker._check_sentence_structure(["ဒါ", "ဘာလုပ်"], ["N", "V"])
    suggestions_477 = [e[2] for e in errors_477]
    assert "ဘာလုပ်လဲ" in suggestions_477
    assert "ဘာလုပ်တယ်" not in suggestions_477


def test_check_sentence_structure_second_person_modal_question_fallback():
    """Second-person modal/future implicit questions should suggest yes/no endings."""
    checker = SyntacticRuleChecker(MagicMock())
    checker.config.is_sentence_final = MagicMock(return_value=False)
    checker.config.get_invalid_sequence_error = MagicMock(return_value=None)

    errors = checker._check_sentence_structure(
        ["နင်", "မနက်ဖြန်", "လာနိုင်", "တယ်"],
        ["N", "N", "V", "V"],
    )
    suggestions = [e[2] for e in errors]
    assert "မလား" in suggestions
    assert "တယ်တယ်" not in suggestions


def test_check_sentence_structure_skips_fallback_when_question_particle_already_present():
    """Split question particle earlier in sentence should block trailing fallback."""
    checker = SyntacticRuleChecker(MagicMock())
    checker.config.is_sentence_final = MagicMock(return_value=False)
    checker.config.get_invalid_sequence_error = MagicMock(return_value=None)

    words = ["ဒါ", "က", "ဘာ", "လဲ", "ပြော", "ပါ", "ဦး"]
    tags = ["N", "P_SUBJ", "Q", "P_SENT", "V", "P_POL", "P_POL"]
    errors = checker._check_sentence_structure(words, tags)
    suggestions = [e[2] for e in errors]

    assert "ဦးလဲ" not in suggestions
    assert "ဦးလား" not in suggestions


def test_check_sentence_boundaries_returns_concrete_completions():
    """Sentence end constraints should emit concrete completions, not generic labels."""
    checker = SyntacticRuleChecker(MagicMock())
    checker.config.is_sentence_final = MagicMock(return_value=False)

    errors_470 = checker._check_sentence_boundaries(["ဒီကိစ္စက", "အရမ်းကို"], ["N", "P_BEN"])
    assert any(e[2] == "အရေးကြီးတယ်" for e in errors_470)
    errors_470_split = checker._check_sentence_boundaries(
        ["ဒီကိစ္စက", "အရမ်း", "ကို"],
        ["N", "ADV", "P_OBJ"],
    )
    assert any(e[2] == "အရေးကြီးတယ်" for e in errors_470_split)

    errors_471 = checker._check_sentence_boundaries(["သူ", "တကယ်ကို"], ["N", "P_BEN"])
    assert any(e[2] == "တကယ်ကို မှန်တယ်" for e in errors_471)
    errors_471_split = checker._check_sentence_boundaries(
        ["သူ", "တကယ်", "ကို"],
        ["N", "ADV", "P_OBJ"],
    )
    assert any(e[2] == "တကယ်ကို မှန်တယ်" for e in errors_471_split)

    errors_480 = checker._check_sentence_boundaries(["သူ", "က"], ["N", "P_SUBJ"])
    assert any(e[2] == "ပါ" for e in errors_480)


def test_check_sentence_boundaries_respects_template_toggle():
    """Disabling targeted templates should fall back to generic completion suggestions."""
    grammar_config = GrammarEngineConfig(enable_targeted_grammar_completion_templates=False)
    checker = SyntacticRuleChecker(MagicMock(), grammar_config=grammar_config)
    checker.config.is_sentence_final = MagicMock(return_value=False)

    errors = checker._check_sentence_boundaries(["ဒီကိစ္စက", "အရမ်းကို"], ["N", "P_BEN"])
    assert any(e[2] == "ပါ" for e in errors)


def test_check_sentence_boundaries():
    """Test boundary constraints."""
    checker = SyntacticRuleChecker(MagicMock())
    checker.config.sentence_start_constraints = [
        {"forbidden_words": ["bad_start"], "confidence": 0.9}
    ]
    checker.config.sentence_end_constraints = [{"forbidden_tags": ["BAD_END"], "confidence": 0.9}]

    # Start error
    errors = checker._check_sentence_boundaries(["bad_start"], ["ANY"])
    assert len(errors) == 1
    assert errors[0][1] == "bad_start"

    # End error
    errors = checker._check_sentence_boundaries(["ok"], ["BAD_END"])
    assert len(errors) == 1
    assert errors[0][1] == "ok"


def test_check_sequence_full():
    """Test check_sequence main loop with hardcoded rules."""
    provider = MagicMock()
    # Mock get_word_pos
    provider.get_word_pos.side_effect = lambda w: "V" if w == "word" else None

    checker = SyntacticRuleChecker(provider)

    # "ပါတယ" -> "ပါတယ်"
    res = checker.check_sequence(["ပါတယ"])
    assert res[0][:3] == (0, "ပါတယ", "ပါတယ်")

    # "သည" -> "သည်"
    res = checker.check_sequence(["သည"])
    assert res[0][:3] == (0, "သည", "သည်")

    # "မယ" -> "မယ်"
    res = checker.check_sequence(["မယ"])
    assert res[0][:3] == (0, "မယ", "မယ်")

    # "ကြေးဇူး" -> "ကျေးဇူး"
    res = checker.check_sequence(["ကြေးဇူး"])
    assert res[0][:3] == (0, "ကြေးဇူး", "ကျေးဇူး")
