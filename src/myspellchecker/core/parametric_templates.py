"""Parametric template executors for YAML-driven rerank rules.

Each template function implements a reusable algorithmic pattern that can be
configured via YAML parameters. New *instances* of existing patterns only need
YAML — new *pattern types* need a new template function registered here.

Template types:
  - trailing_context_promote: Scan trailing text, check conditions, promote
  - stem_exclusion_promote: Strip suffix, check exclusion list, promote
  - tail_cue_promote: Match token/sentence against cue list, promote
  - token_context_inject: Match token shape, construct target, inject
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from myspellchecker.text.normalize import normalize

if TYPE_CHECKING:
    from myspellchecker.core.detection_rules import RerankRulesData
    from myspellchecker.core.rerank_rules import RuleResult, TargetedRerankContext


def execute_trailing_context_promote(
    ctx: "TargetedRerankContext",
    params: dict[str, Any],
    rd: "RerankRulesData",
) -> "RuleResult":
    """Scan trailing text window, check conditions, promote target.

    Params:
      error: str — normalized error to match
      required_suggestions: list[str] — all must be present in suggestions
      trailing_window: int — chars to scan from error position (includes error text)
      trailing_patterns: list[str] — any starting pattern triggers promotion
      nearby_error_fallback: bool — also promote if nearby error exists
      no_adjacent_digits: bool — only promote if no adjacent digits
      target: str — suggestion to promote
    """
    from myspellchecker.core.rerank_rules import RuleResult

    norm_error = normalize(params["error"])
    if ctx.normalized_error != norm_error:
        return RuleResult(applied=False)

    required = [normalize(s) for s in params.get("required_suggestions", [])]
    if not all(r in ctx.normalized_suggestions for r in required):
        return RuleResult(applied=False)

    target = normalize(params["target"])

    # Check for adjacent digit exclusion
    if params.get("no_adjacent_digits"):
        prev_ch = ctx.sentence[ctx.error.position - 1] if ctx.error.position > 0 else ""
        next_pos = ctx.error.position + len(ctx.raw_error_text)
        next_ch = ctx.sentence[next_pos] if next_pos < len(ctx.sentence) else ""
        if any(ch and ch.isdigit() for ch in (prev_ch, next_ch)):
            return RuleResult(applied=False)

    # Check trailing text patterns (scans from error position, including
    # the error text itself — YAML patterns are written accordingly).
    trailing_window = params.get("trailing_window", 6)
    trailing_patterns = [normalize(p) for p in params.get("trailing_patterns", [])]
    if trailing_patterns:
        trailing_text = ctx.sentence[ctx.error.position : ctx.error.position + trailing_window]
        pattern_match = any(trailing_text.startswith(p) for p in trailing_patterns)
        nearby_fallback = params.get("nearby_error_fallback", False) and ctx.has_nearby_error()
        if not pattern_match and not nearby_fallback:
            return RuleResult(applied=False)

    rule_id = params.get("rule_id", f"parametric:trailing_context_promote:{norm_error}")
    if ctx.maybe_promote(target):
        return RuleResult(applied=True, rule_id=rule_id)
    return RuleResult(applied=False)


def execute_stem_exclusion_promote(
    ctx: "TargetedRerankContext",
    params: dict[str, Any],
    rd: "RerankRulesData",
) -> "RuleResult":
    """Strip token suffix, check against exclusion list, promote.

    Params:
      error: str — normalized error to match
      error_type: str — error type to match
      required_suggestion: str — must be in suggestions
      token_ends_with: str — token at error pos must end with this
      exclusion_list_attr: str — RerankRulesData attribute name for exclusion list
      target: str — suggestion to promote
      promote_if_needed: bool — use promote_if_needed vs promote
    """
    from myspellchecker.core.rerank_rules import RuleResult

    norm_error = normalize(params["error"])
    if ctx.normalized_error != norm_error:
        return RuleResult(applied=False)

    if params.get("error_type") and ctx.error.error_type != params["error_type"]:
        return RuleResult(applied=False)

    required = normalize(params["required_suggestion"])
    if required not in ctx.normalized_suggestions:
        return RuleResult(applied=False)

    suffix = normalize(params["token_ends_with"])
    token = ctx.token_at(ctx.error.position)
    if not token.endswith(suffix):
        return RuleResult(applied=False)

    stem = token[: -len(suffix)]
    exclusion_attr = params.get("exclusion_list_attr", "")
    if exclusion_attr and stem:
        exclusions = getattr(rd, exclusion_attr, ())
        if any(stem.endswith(exc) for exc in exclusions):
            return RuleResult(applied=False)
    elif not stem:
        return RuleResult(applied=False)

    target = normalize(params["target"])
    rule_id = params.get("rule_id", f"parametric:stem_exclusion_promote:{norm_error}")
    promote_fn = (
        ctx.maybe_promote_if_needed if params.get("promote_if_needed") else ctx.maybe_promote
    )
    if promote_fn(target):
        return RuleResult(applied=True, rule_id=rule_id)
    return RuleResult(applied=False)


def execute_tail_cue_promote(
    ctx: "TargetedRerankContext",
    params: dict[str, Any],
    rd: "RerankRulesData",
) -> "RuleResult":
    """Match token/sentence against cue list, promote target.

    Params:
      error_any: list[str] — any of these error texts matches
      error_type: str — error type to match
      required_suggestion: str — must be in suggestions
      tail_cue_list_attr: str — RerankRulesData attribute name for cue list
      target: str — suggestion to promote
      promote_if_needed: bool — use promote_if_needed vs promote
    """
    from myspellchecker.core.rerank_rules import RuleResult

    error_any = [normalize(e) for e in params.get("error_any", [])]
    if error_any and ctx.normalized_error not in error_any:
        return RuleResult(applied=False)

    if params.get("error_type") and ctx.error.error_type != params["error_type"]:
        return RuleResult(applied=False)

    required = normalize(params["required_suggestion"])
    if required not in ctx.normalized_suggestions:
        return RuleResult(applied=False)

    cue_attr = params.get("tail_cue_list_attr", "")
    cues = getattr(rd, cue_attr, ()) if cue_attr else ()
    if not cues:
        return RuleResult(applied=False)

    token = ctx.token_at(ctx.error.position)
    if not any(token.endswith(cue) for cue in cues) and not any(
        cue in ctx.sentence for cue in cues
    ):
        return RuleResult(applied=False)

    target = normalize(params["target"])
    rule_id = params.get("rule_id", f"parametric:tail_cue_promote:{ctx.normalized_error}")
    promote_fn = (
        ctx.maybe_promote_if_needed if params.get("promote_if_needed") else ctx.maybe_promote
    )
    if promote_fn(target):
        return RuleResult(applied=True, rule_id=rule_id)
    return RuleResult(applied=False)


def execute_token_context_inject(
    ctx: "TargetedRerankContext",
    params: dict[str, Any],
    rd: "RerankRulesData",
) -> "RuleResult":
    """Match token shape, construct target, inject.

    Params:
      error: str — normalized error to match
      error_type: str — error type to match (optional)
      token_starts_with: str — token must start with this (optional)
      token_ends_with: str — token must end with this (optional)
      sentence_starts_with: str — sentence[error.position:] must start with this (optional)
      lookback_window: int — chars to look back from error position (optional)
      lookback_pattern: str — lookback text must end with this (optional)
      no_suggestions: bool — only fire when no suggestions (optional)
      target_mode: str — "static" | "token_append" | "token_strip_append"
      target: str — static target (for "static" mode)
      target_suffix: str — suffix to append (for "token_append" mode)
      strip_suffix: str — suffix to strip before appending (for "token_strip_append")
      append_suffix: str — suffix to add after stripping (for "token_strip_append")
    """
    from myspellchecker.core.rerank_rules import RuleResult

    norm_error = normalize(params["error"])
    if ctx.normalized_error != norm_error:
        return RuleResult(applied=False)

    if params.get("error_type") and ctx.error.error_type != params["error_type"]:
        return RuleResult(applied=False)

    if params.get("no_suggestions") and ctx.suggestions:
        return RuleResult(applied=False)

    token = ctx.token_at(ctx.error.position)

    if params.get("token_starts_with"):
        prefix = normalize(params["token_starts_with"])
        if not token.startswith(prefix):
            return RuleResult(applied=False)

    if params.get("token_ends_with"):
        suffix = normalize(params["token_ends_with"])
        if not token.endswith(suffix):
            return RuleResult(applied=False)

    if params.get("sentence_starts_with"):
        pattern = normalize(params["sentence_starts_with"])
        if not ctx.sentence[ctx.error.position :].startswith(pattern):
            return RuleResult(applied=False)

    if params.get("lookback_window"):
        window = params["lookback_window"]
        lookback_pattern = normalize(params["lookback_pattern"])
        lookback_text = ctx.sentence[max(0, ctx.error.position - window) : ctx.error.position + 1]
        if not lookback_text.endswith(lookback_pattern):
            return RuleResult(applied=False)

    # Construct target based on mode
    target_mode = params.get("target_mode", "static")
    if target_mode == "token_append":
        target_surface = token + params["target_suffix"]
    elif target_mode == "token_strip_append":
        strip = normalize(params["strip_suffix"])
        if not token.endswith(strip):
            return RuleResult(applied=False)
        target_surface = token[: -len(strip)] + params["append_suffix"]
    else:  # static
        target_surface = params.get("target", "")

    rule_id = params.get("rule_id", f"parametric:token_context_inject:{norm_error}")
    if ctx.maybe_inject(target_surface):
        return RuleResult(applied=True, rule_id=rule_id)
    return RuleResult(applied=False)


# Registry mapping template names to executor functions.
_TEMPLATE_REGISTRY: dict[str, Any] = {
    "trailing_context_promote": execute_trailing_context_promote,
    "stem_exclusion_promote": execute_stem_exclusion_promote,
    "tail_cue_promote": execute_tail_cue_promote,
    "token_context_inject": execute_token_context_inject,
}

# Templates that perform promotions (run during contextual promotion phase).
PROMOTION_TEMPLATES = frozenset(
    {"trailing_context_promote", "stem_exclusion_promote", "tail_cue_promote"}
)

# Templates that perform injections (run during injection phase).
INJECTION_TEMPLATES = frozenset({"token_context_inject"})


def execute_parametric_rule(
    ctx: "TargetedRerankContext",
    rule: dict[str, Any],
    rd: "RerankRulesData",
) -> "RuleResult":
    """Dispatch a parametric rule to its template executor."""
    from myspellchecker.core.rerank_rules import RuleResult

    if not rule.get("enabled", True):
        return RuleResult(applied=False)

    template_name = rule.get("template", "")
    executor = _TEMPLATE_REGISTRY.get(template_name)
    if executor is None:
        return RuleResult(applied=False)

    return executor(ctx, rule.get("params", {}), rd)
