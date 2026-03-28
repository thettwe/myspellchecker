from myspellchecker.core.rerank_rules import apply_targeted_rerank_rules
from myspellchecker.core.response import Error
from myspellchecker.text.normalize import normalize


def _promote_target_suggestion(
    suggestions: list[str], normalized_suggestions: list[str], target: str
) -> bool:
    if not normalized_suggestions:
        return False
    if target == normalized_suggestions[0]:
        return True
    try:
        idx = normalized_suggestions.index(target)
    except ValueError:
        return False
    promoted = suggestions.pop(idx)
    suggestions.insert(0, promoted)
    normalized_suggestions.pop(idx)
    normalized_suggestions.insert(0, target)
    return True


def _inject_target_suggestion(
    suggestions: list[str], normalized_suggestions: list[str], target_surface: str
) -> bool:
    target = normalize(target_surface)
    if target in normalized_suggestions:
        return False
    suggestions.insert(0, target_surface)
    normalized_suggestions.insert(0, target)
    return True


def test_targeted_top1_hint_rule_id_is_returned() -> None:
    error = Error(
        text="ျကားတယ်",
        position=0,
        suggestions=["ကြားတာ", "ကြား"],
        error_type="confusable_error",
    )
    suggestions = list(error.suggestions)
    normalized_suggestions = [normalize(s) for s in suggestions]
    outcome = apply_targeted_rerank_rules(
        error=error,
        errors=[error],
        sentence="သူ ျကားတယ်",
        suggestions=suggestions,
        normalized_suggestions=normalized_suggestions,
        normalized_error=normalize(error.text),
        raw_error_text=error.text,
        top1_before_hint=normalized_suggestions[0],
        targeted_rerank_hints_enabled=True,
        targeted_candidate_injections_enabled=True,
        targeted_top1_hints={normalize("ျကားတယ်"): (normalize("ကြား"),)},
        promote_target_suggestion=_promote_target_suggestion,
        inject_target_suggestion=_inject_target_suggestion,
    )

    assert outcome.applied is True
    assert outcome.rule_id == f"targeted_top1_hint:{normalize('ျကားတယ်')}->{normalize('ကြား')}"
    assert suggestions[0] == "ကြား"


def test_injection_rules_respect_toggle() -> None:
    error = Error(
        text="ပြော့",
        position=0,
        suggestions=["ပျော့", "ပျော"],
        error_type="confusable_error",
    )
    suggestions = list(error.suggestions)
    normalized_suggestions = [normalize(s) for s in suggestions]
    outcome = apply_targeted_rerank_rules(
        error=error,
        errors=[error],
        sentence="ပြော့",
        suggestions=suggestions,
        normalized_suggestions=normalized_suggestions,
        normalized_error=normalize(error.text),
        raw_error_text=error.text,
        top1_before_hint=normalized_suggestions[0],
        targeted_rerank_hints_enabled=True,
        targeted_candidate_injections_enabled=False,
        targeted_top1_hints={},
        promote_target_suggestion=_promote_target_suggestion,
        inject_target_suggestion=_inject_target_suggestion,
    )

    assert outcome.applied is False
    assert "ပြော" not in suggestions


def test_reduplicated_leading_a_promotes_collapsed_candidate() -> None:
    error = Error(
        text="အအိမ်",
        position=0,
        suggestions=["အအိမ်", "အိမ်"],
        error_type="invalid_syllable",
    )
    suggestions = list(error.suggestions)
    normalized_suggestions = [normalize(s) for s in suggestions]
    outcome = apply_targeted_rerank_rules(
        error=error,
        errors=[error],
        sentence="အအိမ် လှတယ်",
        suggestions=suggestions,
        normalized_suggestions=normalized_suggestions,
        normalized_error=normalize(error.text),
        raw_error_text=error.text,
        top1_before_hint=normalized_suggestions[0],
        targeted_rerank_hints_enabled=True,
        targeted_candidate_injections_enabled=True,
        targeted_top1_hints={},
        promote_target_suggestion=_promote_target_suggestion,
        inject_target_suggestion=_inject_target_suggestion,
    )

    assert outcome.applied is True
    assert suggestions[0] == "အိမ်"


def test_zero_digit_promotes_wa_without_literal_hint_map() -> None:
    error = Error(
        text="၀",
        position=0,
        suggestions=["ဝယ်", "ဝ"],
        error_type="invalid_syllable",
    )
    suggestions = list(error.suggestions)
    normalized_suggestions = [normalize(s) for s in suggestions]
    outcome = apply_targeted_rerank_rules(
        error=error,
        errors=[error],
        sentence="၀ တစ်လုံး",
        suggestions=suggestions,
        normalized_suggestions=normalized_suggestions,
        normalized_error=normalize(error.text),
        raw_error_text=error.text,
        top1_before_hint=normalized_suggestions[0],
        targeted_rerank_hints_enabled=True,
        targeted_candidate_injections_enabled=True,
        targeted_top1_hints={},
        promote_target_suggestion=_promote_target_suggestion,
        inject_target_suggestion=_inject_target_suggestion,
    )

    assert outcome.applied is True
    assert suggestions[0] == "ဝ"


def test_leading_double_a_pattern_does_not_fire_when_already_top1() -> None:
    error = Error(
        text="အအေး",
        position=0,
        suggestions=["အေး", "အအေး"],
        error_type="context_probability",
    )
    suggestions = list(error.suggestions)
    normalized_suggestions = [normalize(s) for s in suggestions]
    outcome = apply_targeted_rerank_rules(
        error=error,
        errors=[error],
        sentence="အအေး သောက်တယ်",
        suggestions=suggestions,
        normalized_suggestions=normalized_suggestions,
        normalized_error=normalize(error.text),
        raw_error_text=error.text,
        top1_before_hint=normalized_suggestions[0],
        targeted_rerank_hints_enabled=True,
        targeted_candidate_injections_enabled=True,
        targeted_top1_hints={},
        promote_target_suggestion=_promote_target_suggestion,
        inject_target_suggestion=_inject_target_suggestion,
    )

    assert outcome.applied is False
    assert suggestions[0] == "အေး"


def test_dangling_particle_delete_returns_stable_rule_id() -> None:
    error = Error(
        text="ကြောင့်",
        position=5,
        suggestions=[],
        error_type="dangling_particle",
    )
    suggestions = list(error.suggestions)
    normalized_suggestions = [normalize(s) for s in suggestions]
    outcome = apply_targeted_rerank_rules(
        error=error,
        errors=[error],
        sentence="မိုးကြောင့်",
        suggestions=suggestions,
        normalized_suggestions=normalized_suggestions,
        normalized_error=normalize(error.text),
        raw_error_text=error.text,
        top1_before_hint="",
        targeted_rerank_hints_enabled=True,
        targeted_candidate_injections_enabled=True,
        targeted_top1_hints={},
        promote_target_suggestion=_promote_target_suggestion,
        inject_target_suggestion=_inject_target_suggestion,
    )

    assert outcome.applied is True
    assert outcome.rule_id == "missing_candidate_injections:dangling_particle_delete"
    assert suggestions and suggestions[0] == ""


def test_semantic_context_injection_uses_cues_not_literal_error_word() -> None:
    error = Error(
        text="ခွေး",
        position=0,
        suggestions=["သူ", "ငါ"],
        error_type="semantic_error",
    )
    suggestions = list(error.suggestions)
    normalized_suggestions = [normalize(s) for s in suggestions]
    outcome = apply_targeted_rerank_rules(
        error=error,
        errors=[error],
        sentence="ခွေးက စာမေးပွဲ ဖြေတယ်",
        suggestions=suggestions,
        normalized_suggestions=normalized_suggestions,
        normalized_error=normalize(error.text),
        raw_error_text=error.text,
        top1_before_hint=normalized_suggestions[0],
        targeted_rerank_hints_enabled=True,
        targeted_candidate_injections_enabled=True,
        targeted_top1_hints={},
        promote_target_suggestion=_promote_target_suggestion,
        inject_target_suggestion=_inject_target_suggestion,
    )

    assert outcome.applied is True
    assert outcome.rule_id == f"semantic_context:{normalize('ကျောင်းသား')}"
    assert suggestions[0] == "ကျောင်းသား"


def test_generalized_pattern_injects_when_suggestions_missing() -> None:
    error = Error(
        text="အအိမ်",
        position=0,
        suggestions=[],
        error_type="invalid_syllable",
    )
    suggestions = list(error.suggestions)
    normalized_suggestions = [normalize(s) for s in suggestions]
    outcome = apply_targeted_rerank_rules(
        error=error,
        errors=[error],
        sentence="အအိမ် ကြီးတယ်",
        suggestions=suggestions,
        normalized_suggestions=normalized_suggestions,
        normalized_error=normalize(error.text),
        raw_error_text=error.text,
        top1_before_hint="",
        targeted_rerank_hints_enabled=True,
        targeted_candidate_injections_enabled=True,
        targeted_top1_hints={},
        promote_target_suggestion=_promote_target_suggestion,
        inject_target_suggestion=_inject_target_suggestion,
    )

    assert outcome.applied is True
    assert outcome.rule_id == "pattern_inject:invalid_syllable:normalized_variant"
    assert suggestions[0] == "အိမ်"


def test_generalized_pattern_promotion_skips_when_shorter_root_exists() -> None:
    error = Error(
        text="ကြောင်းပို့",
        position=0,
        suggestions=["ကျောင်း", "ကျောင်းပို့", "ကျောင်"],
        error_type="medial_confusion",
    )
    suggestions = list(error.suggestions)
    normalized_suggestions = [normalize(s) for s in suggestions]
    outcome = apply_targeted_rerank_rules(
        error=error,
        errors=[error],
        sentence="ကလေးကို ကြောင်းပို့ပေးမယ်",
        suggestions=suggestions,
        normalized_suggestions=normalized_suggestions,
        normalized_error=normalize(error.text),
        raw_error_text=error.text,
        top1_before_hint=normalized_suggestions[0],
        targeted_rerank_hints_enabled=True,
        targeted_candidate_injections_enabled=True,
        targeted_top1_hints={},
        promote_target_suggestion=_promote_target_suggestion,
        inject_target_suggestion=_inject_target_suggestion,
    )

    assert outcome.applied is False
    assert suggestions[0] == "ကျောင်း"


def test_pos_sequence_reading_homophone_injects_for_merged_token() -> None:
    error = Error(
        text="ဖက်တယ",
        position=0,
        suggestions=["ဖက်တယ ဖြစ်သည်"],
        error_type="pos_sequence_error",
    )
    suggestions = list(error.suggestions)
    normalized_suggestions = [normalize(s) for s in suggestions]
    outcome = apply_targeted_rerank_rules(
        error=error,
        errors=[error],
        sentence="လက်တွေ့မှာ ကြောင်းသားတွေ စာအုပ် ဖက်တယ",
        suggestions=suggestions,
        normalized_suggestions=normalized_suggestions,
        normalized_error=normalize(error.text),
        raw_error_text=error.text,
        top1_before_hint=normalized_suggestions[0],
        targeted_rerank_hints_enabled=True,
        targeted_candidate_injections_enabled=True,
        targeted_top1_hints={},
        promote_target_suggestion=_promote_target_suggestion,
        inject_target_suggestion=_inject_target_suggestion,
    )

    assert outcome.applied is True
    assert outcome.rule_id == "missing_candidate_injections:reading_homophone"
    assert suggestions[0] == "ဖတ်"


def test_pos_sequence_exam_homophone_injects_for_merged_token() -> None:
    error = Error(
        text="ဖေဆိုတယ",
        position=0,
        suggestions=["ဖေဆိုတယ ဖြစ်သည်"],
        error_type="pos_sequence_error",
    )
    suggestions = list(error.suggestions)
    normalized_suggestions = [normalize(s) for s in suggestions]
    outcome = apply_targeted_rerank_rules(
        error=error,
        errors=[error],
        sentence="လက်တွေ့မှာ ကြောင်းသားတွေ စာမေးပွဲ ဖေဆိုတယ",
        suggestions=suggestions,
        normalized_suggestions=normalized_suggestions,
        normalized_error=normalize(error.text),
        raw_error_text=error.text,
        top1_before_hint=normalized_suggestions[0],
        targeted_rerank_hints_enabled=True,
        targeted_candidate_injections_enabled=True,
        targeted_top1_hints={},
        promote_target_suggestion=_promote_target_suggestion,
        inject_target_suggestion=_inject_target_suggestion,
    )

    assert outcome.applied is True
    assert outcome.rule_id == "missing_candidate_injections:exam_homophone"
    assert suggestions[0] == "ဖြေ"


def test_post_normalizer_promotes_drop_trailing_asat_candidate() -> None:
    error = Error(
        text="ပါ်",
        position=0,
        suggestions=["ပယ်", "ပါ", "ပတ်"],
        error_type="invalid_syllable",
    )
    suggestions = list(error.suggestions)
    normalized_suggestions = [normalize(s) for s in suggestions]
    outcome = apply_targeted_rerank_rules(
        error=error,
        errors=[error],
        sentence="မင်္ဂလာပါ် ဆရာ",
        suggestions=suggestions,
        normalized_suggestions=normalized_suggestions,
        normalized_error=normalize(error.text),
        raw_error_text=error.text,
        top1_before_hint=normalized_suggestions[0],
        targeted_rerank_hints_enabled=True,
        targeted_candidate_injections_enabled=True,
        targeted_top1_hints={},
        promote_target_suggestion=_promote_target_suggestion,
        inject_target_suggestion=_inject_target_suggestion,
    )

    assert outcome.applied is True
    assert outcome.rule_id == "post_normalizer:drop_trailing_asat"
    assert suggestions[0] == "ပါ"


# ── Phase 1: Targeted top-1 hints from YAML ──


def test_targeted_top1_hint_loaded_from_yaml() -> None:
    """Hints are loaded from YAML and used for promotion."""
    error = Error(
        text="ကိုုယ်",
        position=0,
        suggestions=["ကိုယ်စား", "ကိုယ်"],
        error_type="confusable_error",
    )
    suggestions = list(error.suggestions)
    normalized_suggestions = [normalize(s) for s in suggestions]
    outcome = apply_targeted_rerank_rules(
        error=error,
        errors=[error],
        sentence="ကိုုယ် လုပ်ပါ",
        suggestions=suggestions,
        normalized_suggestions=normalized_suggestions,
        normalized_error=normalize(error.text),
        raw_error_text=error.text,
        top1_before_hint=normalized_suggestions[0],
        targeted_rerank_hints_enabled=True,
        targeted_candidate_injections_enabled=True,
        targeted_top1_hints={},  # empty: YAML is the source
        promote_target_suggestion=_promote_target_suggestion,
        inject_target_suggestion=_inject_target_suggestion,
    )

    assert outcome.applied is True
    assert suggestions[0] == "ကိုယ်စား"


def test_yaml_hint_multi_target_tries_in_order() -> None:
    """Multi-target hints try each target in order."""
    error = Error(
        text="ပျည်",
        position=0,
        suggestions=["ပြည်သူ", "ပြည်"],
        error_type="medial_confusion",
    )
    suggestions = list(error.suggestions)
    normalized_suggestions = [normalize(s) for s in suggestions]
    outcome = apply_targeted_rerank_rules(
        error=error,
        errors=[error],
        sentence="ပျည် ကြီးတယ်",
        suggestions=suggestions,
        normalized_suggestions=normalized_suggestions,
        normalized_error=normalize(error.text),
        raw_error_text=error.text,
        top1_before_hint=normalized_suggestions[0],
        targeted_rerank_hints_enabled=True,
        targeted_candidate_injections_enabled=True,
        targeted_top1_hints={},
        promote_target_suggestion=_promote_target_suggestion,
        inject_target_suggestion=_inject_target_suggestion,
    )

    assert outcome.applied is True
    # First target "ပြည်" is in suggestions, gets promoted to top
    assert suggestions[0] == "ပြည်"


# ── Phase 2: Parametric templates ──


def test_parametric_syntax_error_nae_tay_injects_token_append() -> None:
    """A5: syntax_error နေ → token + တယ် via token_context_inject template."""
    error = Error(
        text="နေ",
        position=6,
        suggestions=[],
        error_type="syntax_error",
    )
    suggestions = list(error.suggestions)
    normalized_suggestions = [normalize(s) for s in suggestions]
    outcome = apply_targeted_rerank_rules(
        error=error,
        errors=[error],
        sentence="သူ စာဖတ်နေ",
        suggestions=suggestions,
        normalized_suggestions=normalized_suggestions,
        normalized_error=normalize(error.text),
        raw_error_text=error.text,
        top1_before_hint="",
        targeted_rerank_hints_enabled=True,
        targeted_candidate_injections_enabled=True,
        targeted_top1_hints={},
        promote_target_suggestion=_promote_target_suggestion,
        inject_target_suggestion=_inject_target_suggestion,
    )

    assert outcome.applied is True
    assert outcome.rule_id == "parametric:syntax_error_nae_tay"
    assert "တယ်" in suggestions[0]


def test_parametric_positional_delete_naw_ka() -> None:
    """A7: positional delete နော်က → empty via token_context_inject template."""
    error = Error(
        text="က",
        position=4,
        suggestions=[],
        error_type="syntax_error",
    )
    suggestions = list(error.suggestions)
    normalized_suggestions = [normalize(s) for s in suggestions]
    outcome = apply_targeted_rerank_rules(
        error=error,
        errors=[error],
        sentence="နော်က တစ်ခု",
        suggestions=suggestions,
        normalized_suggestions=normalized_suggestions,
        normalized_error=normalize(error.text),
        raw_error_text=error.text,
        top1_before_hint="",
        targeted_rerank_hints_enabled=True,
        targeted_candidate_injections_enabled=True,
        targeted_top1_hints={},
        promote_target_suggestion=_promote_target_suggestion,
        inject_target_suggestion=_inject_target_suggestion,
    )

    assert outcome.applied is True
    assert outcome.rule_id == "parametric:naw_ka_positional_delete"
    assert suggestions[0] == ""


def test_parametric_missing_conjunction_negation() -> None:
    """A4a: missing_conjunction + token starts with မ → inject ဘူး."""
    error = Error(
        text="တယ်",
        position=3,
        suggestions=[],
        error_type="missing_conjunction",
    )
    suggestions = list(error.suggestions)
    normalized_suggestions = [normalize(s) for s in suggestions]
    outcome = apply_targeted_rerank_rules(
        error=error,
        errors=[error],
        sentence="မသွားတယ် ဘာလဲ",
        suggestions=suggestions,
        normalized_suggestions=normalized_suggestions,
        normalized_error=normalize(error.text),
        raw_error_text=error.text,
        top1_before_hint="",
        targeted_rerank_hints_enabled=True,
        targeted_candidate_injections_enabled=True,
        targeted_top1_hints={},
        promote_target_suggestion=_promote_target_suggestion,
        inject_target_suggestion=_inject_target_suggestion,
    )

    assert outcome.applied is True
    assert outcome.rule_id == "parametric:missing_conjunction_negation"
    assert suggestions[0] == "ဘူး"


def test_parametric_rule_disabled_skips() -> None:
    """Disabled parametric rules are skipped."""
    from myspellchecker.core.detection_rules import RerankRulesData
    from myspellchecker.core.parametric_templates import execute_parametric_rule
    from myspellchecker.core.rerank_rules import TargetedRerankContext

    error = Error(text="နေ", position=0, suggestions=[], error_type="syntax_error")
    ctx = TargetedRerankContext(
        error=error,
        errors=[error],
        sentence="စာဖတ်နေ",
        suggestions=[],
        normalized_suggestions=[],
        normalized_error=normalize("နေ"),
        raw_error_text="နေ",
        top1_before_hint="",
        targeted_rerank_hints_enabled=True,
        targeted_candidate_injections_enabled=True,
        targeted_top1_hints={},
        promote_target_suggestion=_promote_target_suggestion,
        inject_target_suggestion=_inject_target_suggestion,
    )
    rd = RerankRulesData()
    rule = {
        "id": "disabled_rule",
        "template": "token_context_inject",
        "enabled": False,
        "params": {"error": "နေ", "target_mode": "static", "target": "test"},
    }
    result = execute_parametric_rule(ctx, rule, rd)
    assert result.applied is False


# ── Additional edge-case tests ──


def test_dedupe_preserve_order_removes_duplicates_keeps_first() -> None:
    from myspellchecker.core.rerank_rules import _dedupe_preserve_order

    result = _dedupe_preserve_order(["ကို", "မှာ", "ကို", "တွင်", "မှာ"])
    assert result == ["ကို", "မှာ", "တွင်"]


def test_dedupe_preserve_order_skips_empty_strings() -> None:
    from myspellchecker.core.rerank_rules import _dedupe_preserve_order

    result = _dedupe_preserve_order(["", "ကို", "", "မှာ"])
    assert result == ["ကို", "မှာ"]


def test_context_token_at_returns_word_at_position() -> None:
    from myspellchecker.core.rerank_rules import TargetedRerankContext

    error = Error(text="test", position=4, suggestions=[], error_type="invalid_syllable")
    ctx = TargetedRerankContext(
        error=error,
        errors=[error],
        sentence="သူ စာဖတ်တယ်",
        suggestions=[],
        normalized_suggestions=[],
        normalized_error="test",
        raw_error_text="test",
        top1_before_hint="",
        targeted_rerank_hints_enabled=True,
        targeted_candidate_injections_enabled=True,
        targeted_top1_hints={},
        promote_target_suggestion=_promote_target_suggestion,
        inject_target_suggestion=_inject_target_suggestion,
    )

    token = ctx.token_at(4)
    assert token == "စာဖတ်တယ်"


def test_context_token_at_out_of_bounds_returns_empty() -> None:
    from myspellchecker.core.rerank_rules import TargetedRerankContext

    error = Error(text="test", position=0, suggestions=[], error_type="invalid_syllable")
    ctx = TargetedRerankContext(
        error=error,
        errors=[error],
        sentence="ဟုတ်ကဲ့",
        suggestions=[],
        normalized_suggestions=[],
        normalized_error="test",
        raw_error_text="test",
        top1_before_hint="",
        targeted_rerank_hints_enabled=True,
        targeted_candidate_injections_enabled=True,
        targeted_top1_hints={},
        promote_target_suggestion=_promote_target_suggestion,
        inject_target_suggestion=_inject_target_suggestion,
    )

    assert ctx.token_at(-1) == ""
    assert ctx.token_at(999) == ""


def test_has_nearby_error_detects_adjacent_errors() -> None:
    from myspellchecker.core.rerank_rules import TargetedRerankContext

    err1 = Error(text="abc", position=0, suggestions=[], error_type="invalid_syllable")
    err2 = Error(text="def", position=5, suggestions=[], error_type="invalid_syllable")
    err3 = Error(text="ghi", position=100, suggestions=[], error_type="invalid_syllable")

    ctx = TargetedRerankContext(
        error=err1,
        errors=[err1, err2, err3],
        sentence="test sentence",
        suggestions=[],
        normalized_suggestions=[],
        normalized_error="abc",
        raw_error_text="abc",
        top1_before_hint="",
        targeted_rerank_hints_enabled=True,
        targeted_candidate_injections_enabled=True,
        targeted_top1_hints={},
        promote_target_suggestion=_promote_target_suggestion,
        inject_target_suggestion=_inject_target_suggestion,
    )

    assert ctx.has_nearby_error(max_distance=6) is True
    assert ctx.has_nearby_error(max_distance=2) is False
