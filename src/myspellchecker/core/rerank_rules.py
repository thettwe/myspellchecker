"""Targeted rerank rule registry for suggestion top-1 recovery."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from myspellchecker.core.constants import (
    ET_CONFUSABLE_ERROR,
    ET_MEDIAL_CONFUSION,
    ET_SEMANTIC_ERROR,
    ET_SYLLABLE,
)
from myspellchecker.core.myanmar_confusables import (
    ASPIRATION_SWAP_MAP as _ASPIRATED_SWAP_MAP,
)
from myspellchecker.core.myanmar_confusables import (
    MEDIAL_SWAP_PAIRS as _MEDIAL_SWAP_PAIRS,
)
from myspellchecker.core.myanmar_confusables import (
    TONE_MARK_PAIRS as _TONE_MARK_PAIRS,
)
from myspellchecker.core.myanmar_confusables import (
    VOWEL_LENGTH_PAIRS as _VOWEL_LENGTH_PAIRS,
)
from myspellchecker.core.response import Error
from myspellchecker.text.normalize import normalize

if TYPE_CHECKING:
    from myspellchecker.core.detection_rules import RerankRulesData

_DEFAULT_RERANK_DATA: RerankRulesData | None = None
_rerank_lock = threading.Lock()


def _default_rerank_data() -> RerankRulesData:
    """Lazy-load module-level default RerankRulesData singleton."""
    global _DEFAULT_RERANK_DATA
    if _DEFAULT_RERANK_DATA is None:
        with _rerank_lock:
            if _DEFAULT_RERANK_DATA is None:
                from myspellchecker.core.detection_rules import RerankRulesData

                _DEFAULT_RERANK_DATA = RerankRulesData()
    return _DEFAULT_RERANK_DATA


def _get_rd(ctx: TargetedRerankContext) -> RerankRulesData:
    """Get RerankRulesData from context or module default."""
    if ctx.rerank_data is not None:
        return ctx.rerank_data
    return _default_rerank_data()


PromoteFn = Callable[[list[str], list[str], str], bool]
InjectFn = Callable[[list[str], list[str], str], bool]
GuardFn = Callable[["TargetedRerankContext"], bool]
RuleActionFn = Callable[["TargetedRerankContext"], "RuleResult"]


@dataclass
class RuleResult:
    """Outcome of one targeted rerank rule application."""

    applied: bool
    rule_id: str | None = None


@dataclass(frozen=True)
class TargetedRerankRule:
    """Declarative targeted rerank rule entry."""

    id: str
    guard: GuardFn
    action: RuleActionFn


@dataclass
class TargetedRerankContext:
    """Mutable context for targeted rerank rule evaluation."""

    error: Error
    errors: list[Error]
    sentence: str
    suggestions: list[str]
    normalized_suggestions: list[str]
    normalized_error: str
    raw_error_text: str
    top1_before_hint: str
    targeted_rerank_hints_enabled: bool
    targeted_candidate_injections_enabled: bool
    targeted_top1_hints: dict[str, tuple[str, ...]]
    promote_target_suggestion: PromoteFn
    inject_target_suggestion: InjectFn
    rerank_data: RerankRulesData | None = field(default=None, repr=False)

    def maybe_promote(self, target: str) -> bool:
        if not self.targeted_rerank_hints_enabled:
            return False
        return self.promote_target_suggestion(self.suggestions, self.normalized_suggestions, target)

    def maybe_promote_if_needed(self, target: str) -> bool:
        """Promote only when target is present and not already top-1."""
        if not self.normalized_suggestions:
            return False
        if self.normalized_suggestions[0] == target:
            return False
        return self.maybe_promote(target)

    def maybe_inject(self, target_surface: str) -> bool:
        if not self.targeted_candidate_injections_enabled:
            return False
        return self.inject_target_suggestion(
            self.suggestions,
            self.normalized_suggestions,
            target_surface,
        )

    def token_at(self, pos: int) -> str:
        if not self.sentence or pos < 0 or pos >= len(self.sentence):
            return ""
        start = self.sentence.rfind(" ", 0, pos) + 1
        end = self.sentence.find(" ", pos)
        if end < 0:
            end = len(self.sentence)
        return self.sentence[start:end]

    def has_nearby_error(self, max_distance: int = 6) -> bool:
        return any(
            other is not self.error and abs(other.position - self.error.position) <= max_distance
            for other in self.errors
        )


def _always(_: TargetedRerankContext) -> bool:
    return True


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _generate_generalized_candidates(
    ctx: TargetedRerankContext,
) -> tuple[list[str], list[str]]:
    """Generate generalized candidate surfaces for typo-heavy buckets.

    Returns two lists:
      - **core**: high-confidence transforms used for both promotion and injection
        (medial swaps, digit-to-letter, double-prefix, trailing asat).
      - **extended**: broader transforms used only for injection when no
        suggestions exist (aspirated swaps, vowel length, tone marks).
    """
    etype = ctx.error.error_type
    if etype not in {
        ET_SYLLABLE,
        ET_CONFUSABLE_ERROR,
        ET_MEDIAL_CONFUSION,
        ET_SEMANTIC_ERROR,
    }:
        return [], []
    token = ctx.normalized_error
    if not token:
        return [], []

    core: list[str] = []
    if "၀" in token:
        core.append(token.replace(normalize("၀"), normalize("ဝ")))
    if token.startswith(normalize("အအ")) and len(token) > 1:
        core.append(token[1:])
    if token.endswith(normalize("််")):
        core.append(token[:-1])
    # --- Medial swaps (ျ↔ြ, ွ↔ှ) from shared confusable pairs ---
    for medial_a, medial_b in _MEDIAL_SWAP_PAIRS:
        if medial_a in token:
            core.append(token.replace(medial_a, medial_b))
        if medial_b in token:
            core.append(token.replace(medial_b, medial_a))

    extended: list[str] = []

    # --- Aspirated consonant swaps (per-position to avoid replacing all) ---
    for i, ch in enumerate(token):
        swap = _ASPIRATED_SWAP_MAP.get(ch)
        if swap is not None:
            extended.append(token[:i] + swap + token[i + 1 :])

    # --- Vowel length swaps ---
    for short_vowel, long_vowel in _VOWEL_LENGTH_PAIRS:
        if short_vowel in token:
            extended.append(token.replace(short_vowel, long_vowel))
        if long_vowel in token:
            extended.append(token.replace(long_vowel, short_vowel))

    # --- Tone mark swaps ---
    for tone_a, tone_b in _TONE_MARK_PAIRS:
        if tone_a in token:
            extended.append(token.replace(tone_a, tone_b))
        if tone_b in token:
            extended.append(token.replace(tone_b, tone_a))

    core_deduped = [v for v in _dedupe_preserve_order(core) if v != token]
    # Extended candidates exclude anything already in core.
    core_set = set(core_deduped)
    extended_deduped = [
        v for v in _dedupe_preserve_order(extended) if v != token and v not in core_set
    ]
    return core_deduped, extended_deduped


def _apply_generalized_candidate_generation(ctx: TargetedRerankContext) -> RuleResult:
    core, extended = _generate_generalized_candidates(ctx)
    if not core and not extended:
        return RuleResult(applied=False)

    def _has_shorter_prefix_candidate(candidate: str) -> bool:
        return any(
            other != candidate and len(other) < len(candidate) and candidate.startswith(other)
            for other in ctx.normalized_suggestions
        )

    # Core variants participate in promotion (reorder existing suggestions).
    for candidate in core:
        # Avoid over-promoting compound continuations when a shorter root
        # candidate is already present (e.g., "ကျောင်း" over "ကျောင်းပို့").
        if _has_shorter_prefix_candidate(candidate):
            continue
        if candidate in ctx.normalized_suggestions and ctx.maybe_promote_if_needed(candidate):
            return RuleResult(
                applied=True,
                rule_id=f"pattern_promote:{ctx.error.error_type}:normalized_variant",
            )

    # Digit-zero injection: when ၀→ဝ transform produces a valid word
    # missing from suggestions, inject it. This is the highest-confidence
    # core transform (visually identical characters). Other core transforms
    # (medial swaps) are too broad for unconditional injection.
    if "၀" in ctx.normalized_error and core:
        digit_zero_candidate = core[0]  # digit-zero is always first in core list
        if digit_zero_candidate not in ctx.normalized_suggestions:
            if ctx.maybe_inject(digit_zero_candidate):
                return RuleResult(
                    applied=True,
                    rule_id=f"pattern_inject:{ctx.error.error_type}:digit_zero",
                )

    # When no suggestions exist, try core candidates first (higher confidence),
    # then extended variants as fallback.
    if not ctx.suggestions:
        for candidate in core:
            if ctx.maybe_inject(candidate):
                return RuleResult(
                    applied=True,
                    rule_id=f"pattern_inject:{ctx.error.error_type}:normalized_variant",
                )
        for candidate in extended:
            if ctx.maybe_inject(candidate):
                return RuleResult(
                    applied=True,
                    rule_id=f"pattern_inject:{ctx.error.error_type}:extended_variant",
                )

    return RuleResult(applied=False)


def _apply_semantic_context_injections(ctx: TargetedRerankContext) -> RuleResult:
    if ctx.error.error_type != ET_SEMANTIC_ERROR:
        return RuleResult(applied=False)

    rd = _get_rd(ctx)
    for rule in rd.semantic_context_injections:
        if not all(cue in ctx.sentence for cue in rule["sentence_requires_all"]):
            continue
        target = rule["target_normalized"]
        target_surface = rule["target_surface"]
        if target in ctx.normalized_suggestions and ctx.maybe_promote_if_needed(target):
            return RuleResult(applied=True, rule_id=f"semantic_context:{target}")
        if ctx.maybe_inject(target_surface):
            return RuleResult(applied=True, rule_id=f"semantic_context:{target}")

    return RuleResult(applied=False)


def _match_contextual_promotions(ctx: TargetedRerankContext) -> RuleResult:
    """Evaluate data-driven contextual promotions from YAML."""
    rd = _get_rd(ctx)
    for promo in rd.contextual_promotions:
        if ctx.normalized_error != promo["error"]:
            continue
        promote_fn = (
            ctx.maybe_promote_if_needed if promo.get("promote_if_needed") else ctx.maybe_promote
        )
        matched_cue = False
        for cue in promo.get("cues", []):
            cue_or = cue.get("sentence_contains", ())
            cue_and = cue.get("sentence_contains_all", ())
            if (cue_or and any(c in ctx.sentence for c in cue_or)) or (
                cue_and and all(c in ctx.sentence for c in cue_and)
            ):
                if promote_fn(cue["target"]):
                    return RuleResult(applied=True, rule_id=cue.get("rule_id"))
                matched_cue = True
                break
        if not matched_cue and promo.get("default_target"):
            if promote_fn(promo["default_target"]):
                return RuleResult(applied=True, rule_id=promo.get("default_rule_id"))
    return RuleResult(applied=False)


def _apply_contextual_promotions(ctx: TargetedRerankContext) -> RuleResult:
    # --- Data-driven contextual promotions from YAML ---
    result = _match_contextual_promotions(ctx)
    if result.applied:
        return result

    # --- Parametric promotion templates ---
    rd = _get_rd(ctx)
    from myspellchecker.core.parametric_templates import (
        PROMOTION_TEMPLATES,
        execute_parametric_rule,
    )

    for rule in rd.parametric_rules:
        if rule.get("template") in PROMOTION_TEMPLATES:
            outcome = execute_parametric_rule(ctx, rule, rd)
            if outcome.applied:
                return outcome

    return RuleResult(applied=False)


def _apply_targeted_top1_hint_map(ctx: TargetedRerankContext) -> RuleResult:
    rd = _get_rd(ctx)
    hint_targets = rd.targeted_top1_hints.get(ctx.normalized_error, ())
    if not hint_targets:
        hint_targets = ctx.targeted_top1_hints.get(ctx.normalized_error, ())
    for target in hint_targets:
        # Phase 1: Promote if target already in suggestions
        if ctx.maybe_promote(target):
            return RuleResult(
                applied=True,
                rule_id=f"targeted_top1_hint:{ctx.normalized_error}->{target}",
            )

        # Phase 2: Inject if target has morpheme-of-compound relationship
        # Direction 1: target is substring of a suggestion (extract morpheme)
        is_morpheme_of_compound = any(
            target in sugg for sugg in ctx.normalized_suggestions if sugg != target
        )
        # Direction 2: suggestion is prefix of target (extend morpheme to compound)
        # Guarded: the suffix added must also appear in the error text to prevent
        # cascade (e.g. ကျောင်းတိုက် where တိုက် is NOT in error ကြောင်း).
        if not is_morpheme_of_compound:
            for sugg in ctx.normalized_suggestions:
                if sugg == target or not target.startswith(sugg):
                    continue
                suffix = target[len(sugg) :]
                if suffix and suffix in ctx.normalized_error:
                    is_morpheme_of_compound = True
                    break
        if is_morpheme_of_compound and ctx.maybe_inject(target):
            return RuleResult(
                applied=True,
                rule_id=f"targeted_top1_morpheme_inject:{ctx.normalized_error}->{target}",
            )

    return RuleResult(applied=False)


def _apply_disambiguation_and_rewrite_promotions(ctx: TargetedRerankContext) -> RuleResult:
    rd = _get_rd(ctx)

    # --- Surface rewrites (modify suggestions in-place) ---
    for rewrite in rd.surface_rewrites:
        if ctx.normalized_error != rewrite["trigger_error"]:
            continue
        malformed = rewrite["malformed"]
        if malformed in ctx.normalized_suggestions:
            malformed_idx = next(
                (
                    idx
                    for idx, candidate in enumerate(ctx.suggestions)
                    if normalize(candidate) == malformed
                ),
                None,
            )
            if malformed_idx is not None:
                ctx.suggestions[malformed_idx] = rewrite["corrected_surface"]
                ctx.normalized_suggestions[malformed_idx] = rewrite["corrected_normalized"]
                if ctx.maybe_promote(rewrite["corrected_normalized"]):
                    return RuleResult(applied=True)

    # --- Disambiguation promotions from YAML ---
    for rule in rd.disambiguation_promotions:
        if ctx.normalized_error != rule["error"]:
            continue

        if "suggestion_contains" in rule:
            if rule["suggestion_contains"] not in ctx.normalized_suggestions:
                continue
        if "suggestion_contains_any" in rule:
            if not any(s in ctx.normalized_suggestions for s in rule["suggestion_contains_any"]):
                continue
        if "suggestion_prefix" in rule:
            if not any(c.startswith(rule["suggestion_prefix"]) for c in ctx.normalized_suggestions):
                continue
        if "suggestion_contains_all" in rule:
            if not all(s in ctx.normalized_suggestions for s in rule["suggestion_contains_all"]):
                continue
        if "top1_was" in rule:
            if ctx.top1_before_hint != rule["top1_was"]:
                continue
        if "sentence_contains" in rule:
            if rule["sentence_contains"] not in ctx.sentence:
                continue

        target = rule["target"]
        action = rule.get("action", "promote")
        if action == "inject":
            if ctx.maybe_inject(target):
                return RuleResult(applied=True)
        elif ctx.maybe_promote(target):
            return RuleResult(applied=True)

    # --- Algorithmic: leading double အ pattern ---
    if ctx.normalized_error.startswith(normalize("အအ")) and len(ctx.normalized_error) > 1:
        collapsed = ctx.normalized_error[1:]
        if collapsed in ctx.normalized_suggestions and ctx.maybe_promote_if_needed(collapsed):
            return RuleResult(applied=True, rule_id="pattern_promote:leading_double_a")

    return RuleResult(applied=False)


def _apply_injection_rules(ctx: TargetedRerankContext) -> RuleResult:
    generalized_outcome = _apply_generalized_candidate_generation(ctx)
    if generalized_outcome.applied:
        return generalized_outcome

    semantic_outcome = _apply_semantic_context_injections(ctx)
    if semantic_outcome.applied:
        return semantic_outcome

    rd = _get_rd(ctx)

    # --- YAML-driven: exact simple injections ---
    target = rd.simple_injections.get(ctx.normalized_error)
    if target is not None:
        if ctx.maybe_inject(target):
            return RuleResult(applied=True)

    # --- YAML-driven: no-suggestions injections ---
    if not ctx.suggestions:
        ns_target = rd.simple_injections_no_suggestions.get(ctx.normalized_error)
        if ns_target is not None:
            if ctx.maybe_inject(ns_target):
                return RuleResult(applied=True)

    # --- Algorithmic: ZWS syllable error ---
    if (
        ctx.error.error_type == ET_SYLLABLE
        and ctx.normalized_error == ""
        and any(ch in ctx.raw_error_text for ch in ("\u200b", "\u200c", "\u200d", "\ufeff"))
    ):
        if ctx.maybe_inject(""):
            return RuleResult(applied=True)

    # --- YAML-driven: delete-on-error-type ---
    if ctx.error.error_type in rd.delete_on_error_types and not ctx.suggestions:
        rule_id_map = rd.delete_on_error_type_rule_ids
        if ctx.maybe_inject(""):
            return RuleResult(
                applied=True,
                rule_id=rule_id_map.get(ctx.error.error_type),
            )

    # --- YAML-driven: error-type conditioned injections ---
    for rule in rd.error_type_injections:
        if rule.get("error_type") and ctx.error.error_type != rule["error_type"]:
            continue

        rule_error = rule["error"]
        match_mode = rule.get("match_mode", "exact")
        if match_mode == "exact" and ctx.normalized_error != rule_error:
            continue
        elif match_mode == "startswith" and not ctx.normalized_error.startswith(rule_error):
            continue
        elif match_mode == "endswith" and not ctx.normalized_error.endswith(rule_error):
            continue

        if rule.get("no_suggestions") and ctx.suggestions:
            continue

        if "suggestion_exact" in rule:
            if ctx.normalized_suggestions != list(rule["suggestion_exact"]):
                continue

        if "sentence_contains" in rule:
            if not any(c in ctx.sentence for c in rule["sentence_contains"]):
                continue

        if "sentence_contains_all" in rule:
            if not all(c in ctx.sentence for c in rule["sentence_contains_all"]):
                continue

        if "token_contains" in rule:
            token = ctx.token_at(ctx.error.position)
            if rule["token_contains"] not in normalize(token):
                continue

        if rule.get("second_person_context"):
            if not any(pron in ctx.sentence for pron in rd.second_person_pronouns):
                continue
        if ctx.maybe_inject(rule["target"]):
            return RuleResult(
                applied=True,
                rule_id=rule.get("rule_id"),
            )

    # --- YAML-driven: conditional simple injections (startswith/error_type) ---
    for rule in rd.simple_injections_conditional:
        rule_error = rule["error"]
        match_mode = rule.get("match_mode", "exact")
        if match_mode == "startswith" and not ctx.normalized_error.startswith(rule_error):
            continue
        elif match_mode == "exact" and ctx.normalized_error != rule_error:
            continue

        if "error_type" in rule and ctx.error.error_type != rule["error_type"]:
            continue

        if ctx.maybe_inject(rule["target"]):
            return RuleResult(applied=True)

    # --- Parametric injection templates ---
    from myspellchecker.core.parametric_templates import (
        INJECTION_TEMPLATES,
        execute_parametric_rule,
    )

    for rule in rd.parametric_rules:
        if rule.get("template") in INJECTION_TEMPLATES:
            outcome = execute_parametric_rule(ctx, rule, rd)
            if outcome.applied:
                return outcome

    # --- Algorithmic: token context (ပါတယ်် + လာခဲ့) ---
    if ctx.normalized_error == normalize("တယ််") and ctx.error.error_type == ET_SYLLABLE:
        token = ctx.token_at(ctx.error.position)
        if token.endswith(normalize("ပါတယ််")) and token.startswith(normalize("လာခဲ့")):
            if ctx.maybe_inject("ပါတယ်"):
                return RuleResult(applied=True)
        elif ctx.maybe_inject("တယ်"):
            return RuleResult(applied=True)

    return RuleResult(applied=False)


def _apply_post_injection_normalizers(ctx: TargetedRerankContext) -> RuleResult:
    rd = _get_rd(ctx)

    # --- YAML-driven: post-injection normalizers ---
    for rule in rd.post_normalizers:
        if ctx.normalized_error != rule["error"]:
            continue

        if rule.get("top1_was") is not None and ctx.top1_before_hint != rule["top1_was"]:
            continue

        action = rule.get("action", "inject")
        if action == "promote":
            if ctx.maybe_promote(rule["target_normalized"]):
                return RuleResult(applied=True)
        elif action == "inject":
            if ctx.maybe_inject(rule["target_surface"]):
                return RuleResult(applied=True)

    # --- Algorithmic: trailing asat normalizer ---
    if (
        ctx.error.error_type == ET_SYLLABLE
        and ctx.normalized_error.endswith("\u103a")
        and len(ctx.normalized_error) > 1
    ):
        without_trailing_asat = ctx.normalized_error[:-1]
        if ctx.maybe_promote(without_trailing_asat):
            return RuleResult(applied=True, rule_id="post_normalizer:drop_trailing_asat")

    return RuleResult(applied=False)


TARGETED_RERANK_RULES: tuple[TargetedRerankRule, ...] = (
    TargetedRerankRule(
        id="contextual_promotions",
        guard=_always,
        action=_apply_contextual_promotions,
    ),
    TargetedRerankRule(
        id="targeted_top1_hint_map",
        guard=_always,
        action=_apply_targeted_top1_hint_map,
    ),
    TargetedRerankRule(
        id="disambiguation_and_rewrite_promotions",
        guard=_always,
        action=_apply_disambiguation_and_rewrite_promotions,
    ),
    TargetedRerankRule(
        id="missing_candidate_injections",
        guard=_always,
        action=_apply_injection_rules,
    ),
    TargetedRerankRule(
        id="post_injection_normalizers",
        guard=_always,
        action=_apply_post_injection_normalizers,
    ),
)


def apply_targeted_rerank_rules(
    *,
    error: Error,
    errors: list[Error],
    sentence: str,
    suggestions: list[str],
    normalized_suggestions: list[str],
    normalized_error: str,
    raw_error_text: str,
    top1_before_hint: str,
    targeted_rerank_hints_enabled: bool,
    targeted_candidate_injections_enabled: bool,
    targeted_top1_hints: dict[str, tuple[str, ...]],
    promote_target_suggestion: PromoteFn,
    inject_target_suggestion: InjectFn,
    rerank_data: RerankRulesData | None = None,
) -> RuleResult:
    """Apply targeted rerank rules in registry order."""
    ctx = TargetedRerankContext(
        error=error,
        errors=errors,
        sentence=sentence,
        suggestions=suggestions,
        normalized_suggestions=normalized_suggestions,
        normalized_error=normalized_error,
        raw_error_text=raw_error_text,
        top1_before_hint=top1_before_hint,
        targeted_rerank_hints_enabled=targeted_rerank_hints_enabled,
        targeted_candidate_injections_enabled=targeted_candidate_injections_enabled,
        targeted_top1_hints=targeted_top1_hints,
        promote_target_suggestion=promote_target_suggestion,
        inject_target_suggestion=inject_target_suggestion,
        rerank_data=rerank_data,
    )

    for rule in TARGETED_RERANK_RULES:
        if not rule.guard(ctx):
            continue
        outcome = rule.action(ctx)
        if outcome.applied:
            return outcome

    return RuleResult(applied=False)
