from myspellchecker.algorithms.ranker import (
    DefaultRanker,
    EditDistanceOnlyRanker,
    FrequencyFirstRanker,
    PhoneticFirstRanker,
    SuggestionData,
    UnifiedRanker,
)
from myspellchecker.core.config import RankerConfig


def test_default_ranker_bonuses():
    ranker = DefaultRanker()

    # Base case
    base = SuggestionData("term", edit_distance=1, frequency=100)
    base_score = ranker.score(base)

    # Test nasal variant bonus
    nasal = SuggestionData("term", edit_distance=1, frequency=100, is_nasal_variant=True)
    nasal_score = ranker.score(nasal)
    assert nasal_score < base_score

    # Test same nasal ending bonus
    same_nasal = SuggestionData("term", edit_distance=1, frequency=100, has_same_nasal_ending=True)
    same_nasal_score = ranker.score(same_nasal)
    assert same_nasal_score < base_score

    # Test multiplicative plausibility via weighted_distance
    medial_swap = SuggestionData("term", edit_distance=1, frequency=100, weighted_distance=0.3)
    random_sub = SuggestionData("term", edit_distance=1, frequency=100, weighted_distance=1.0)
    assert ranker.score(medial_swap) < ranker.score(random_sub)
    # Gap should be substantial (multiplicative: 0.3 vs 1.0 base)
    gap = ranker.score(random_sub) - ranker.score(medial_swap)
    assert gap > 0.5, f"Plausibility gap {gap} too small"

    # Test syllable_distance as plausibility fallback
    syllable = SuggestionData("term", edit_distance=2, frequency=100, syllable_distance=1.0)
    no_syllable = SuggestionData("term", edit_distance=2, frequency=100)
    assert ranker.score(syllable) < ranker.score(no_syllable)

    # Test POS fit bonus
    with_pos = SuggestionData("term", edit_distance=1, frequency=100, pos_fit_score=0.8)
    without_pos = SuggestionData("term", edit_distance=1, frequency=100)
    assert ranker.score(with_pos) < ranker.score(without_pos)


def test_frequency_first_ranker():
    config = RankerConfig(frequency_first_edit_weight=0.5, frequency_first_scale=0.1)
    ranker = FrequencyFirstRanker(ranker_config=config)
    assert ranker.name == "frequency_first"

    # Higher frequency should have lower score (better)
    low_freq = SuggestionData("low", edit_distance=1, frequency=10)
    high_freq = SuggestionData("high", edit_distance=1, frequency=1000)

    assert ranker.score(high_freq) < ranker.score(low_freq)

    # Edit distance impact
    dist1 = SuggestionData("d1", edit_distance=1, frequency=100)
    dist2 = SuggestionData("d2", edit_distance=2, frequency=100)
    assert ranker.score(dist1) < ranker.score(dist2)


def test_edit_distance_only_ranker():
    ranker = EditDistanceOnlyRanker()
    assert ranker.name == "edit_distance_only"

    s1 = SuggestionData("s1", edit_distance=1, frequency=1000)
    s2 = SuggestionData("s2", edit_distance=2, frequency=1)

    assert ranker.score(s1) == 1.0
    assert ranker.score(s2) == 2.0
    # Frequency shouldn't matter
    s3 = SuggestionData("s3", edit_distance=1, frequency=1)
    assert ranker.score(s3) == ranker.score(s1)


def test_phonetic_first_ranker():
    config = RankerConfig(phonetic_first_weight=1.0, phonetic_first_edit_weight=0.3)
    ranker = PhoneticFirstRanker(ranker_config=config)
    assert ranker.name == "phonetic_first"

    # Higher phonetic score -> lower total score (better)
    p_low = SuggestionData("p_low", edit_distance=1, frequency=100, phonetic_score=0.5)
    p_high = SuggestionData("p_high", edit_distance=1, frequency=100, phonetic_score=0.9)

    assert ranker.score(p_high) < ranker.score(p_low)


def test_unified_ranker_deduplication():
    ranker = UnifiedRanker()

    # Same term, different sources
    s1 = SuggestionData("term", edit_distance=1, frequency=100, source="symspell", confidence=1.0)
    s2 = SuggestionData(
        "term", edit_distance=1, frequency=100, source="particle_typo", confidence=1.0
    )

    # particle_typo has higher weight (1.2) than symspell (1.0)
    deduped = ranker._deduplicate([s1, s2])
    assert len(deduped) == 1
    assert deduped[0].source == "particle_typo"

    # Reverse order input
    deduped = ranker._deduplicate([s2, s1])
    assert len(deduped) == 1
    assert deduped[0].source == "particle_typo"


def test_unified_ranker_sorting():
    ranker = UnifiedRanker()

    s1 = SuggestionData("bad", edit_distance=2, frequency=1, source="symspell")
    s2 = SuggestionData("good", edit_distance=1, frequency=100, source="symspell")

    ranked = ranker.rank_suggestions([s1, s2])
    assert ranked[0].term == "good"
    assert ranked[1].term == "bad"


def test_unified_ranker_deterministic_tie_break_ordering():
    """Near-equal scores should be resolved deterministically."""
    ranker = UnifiedRanker()

    # Force near-equal scores to exercise deterministic tie-break keys.
    ranker.score = lambda _: 0.1234567  # type: ignore[method-assign]

    s_high_freq = SuggestionData(
        "high_freq",
        edit_distance=1,
        weighted_distance=0.2,
        frequency=300,
    )
    s_low_freq = SuggestionData(
        "low_freq",
        edit_distance=1,
        weighted_distance=0.2,
        frequency=100,
    )
    s_higher_weighted_distance = SuggestionData(
        "higher_weighted_distance",
        edit_distance=1,
        weighted_distance=0.4,
        frequency=1000,
    )
    s_worse_edit = SuggestionData(
        "worse_edit",
        edit_distance=2,
        weighted_distance=0.1,
        frequency=1000,
    )

    ranked = ranker.rank_suggestions(
        [s_worse_edit, s_low_freq, s_higher_weighted_distance, s_high_freq],
        deduplicate=False,
        enforce_diversity=False,
    )

    assert [s.term for s in ranked] == [
        "high_freq",
        "low_freq",
        "higher_weighted_distance",
        "worse_edit",
    ]
