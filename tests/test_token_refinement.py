"""Tests for validation-time token refinement."""

from myspellchecker.core.token_refinement import (
    build_validation_token_paths,
    refine_validation_tokens,
)


def test_refine_detaches_suffix_variant_from_oov_token() -> None:
    words = ["လပ်ဖက်ရည်ဆိုင်မာ"]
    valid_words = {"လပ်ဖက်ရည်ဆိုင်"}
    freqs = {"လပ်ဖက်ရည်ဆိုင်": 9_500}

    refined = refine_validation_tokens(
        words,
        is_valid_word=lambda w: w in valid_words,
        get_word_frequency=lambda w: freqs.get(w, 0),
    )

    assert refined == ["လပ်ဖက်ရည်ဆိုင်", "မာ"]


def test_refine_detaches_negation_prefix_for_merged_token() -> None:
    words = ["မဖျေ"]
    valid_words = {"ဖြေ"}
    freqs = {"ဖြေ": 50_000}

    refined = refine_validation_tokens(
        words,
        is_valid_word=lambda w: w in valid_words,
        get_word_frequency=lambda w: freqs.get(w, 0),
    )

    assert refined == ["မ", "ဖျေ"]


def test_refine_keeps_high_frequency_known_word() -> None:
    words = ["စာအုပ်မှာ"]
    valid_words = {"စာအုပ်မှာ", "စာအုပ်"}
    freqs = {"စာအုပ်မှာ": 12_000, "စာအုပ်": 15_000}

    refined = refine_validation_tokens(
        words,
        is_valid_word=lambda w: w in valid_words,
        get_word_frequency=lambda w: freqs.get(w, 0),
    )

    assert refined == ["စာအုပ်မှာ"]


def test_build_validation_token_paths_emits_alternate_syllable_split_path() -> None:
    words = ["ကျောင်က"]
    valid_words = {"ကျောင်", "က"}
    freqs = {"ကျောင်": 12_000, "က": 110_000}

    paths = build_validation_token_paths(
        words,
        is_valid_word=lambda w: w in valid_words,
        get_word_frequency=lambda w: freqs.get(w, 0),
        segment_syllables=lambda _w: ["ကျောင်", "က"],
    )

    assert len(paths) >= 1
    assert paths[0] == ["ကျောင်က"]
    assert any(path == ["ကျောင်", "က"] for path in paths)


# ── Additional behavioral tests ──


def test_refine_empty_input_returns_empty() -> None:
    result = refine_validation_tokens(
        [],
        is_valid_word=lambda w: True,
        get_word_frequency=lambda w: 100,
    )
    assert result == []


def test_refine_no_lexical_signals_returns_copy() -> None:
    words = ["ကျောင်းသား", "တွေ"]
    result = refine_validation_tokens(words)
    assert result == words
    assert result is not words


def test_refine_skips_token_with_digits() -> None:
    words = ["အမှတ်123သည်"]
    valid_words = {"အမှတ်"}
    freqs = {"အမှတ်": 5_000}

    refined = refine_validation_tokens(
        words,
        is_valid_word=lambda w: w in valid_words,
        get_word_frequency=lambda w: freqs.get(w, 0),
    )

    assert refined == ["အမှတ်123သည်"]


def test_refine_detaches_particle_ကို_from_oov_stem() -> None:
    words = ["ကလေးကို"]
    valid_words = {"ကလေး", "ကို"}
    freqs = {"ကလေး": 30_000, "ကို": 200_000}

    refined = refine_validation_tokens(
        words,
        is_valid_word=lambda w: w in valid_words,
        get_word_frequency=lambda w: freqs.get(w, 0),
    )

    assert refined == ["ကလေး", "ကို"]


def test_refine_detaches_suffix_တယ_from_verb() -> None:
    words = ["သွားတယ်"]
    valid_words = {"သွား", "တယ်"}
    freqs = {"သွား": 80_000, "တယ်": 500_000}

    refined = refine_validation_tokens(
        words,
        is_valid_word=lambda w: w in valid_words,
        get_word_frequency=lambda w: freqs.get(w, 0),
    )

    assert refined == ["သွား", "တယ်"]


def test_refine_negation_plus_suffix_three_way_split() -> None:
    words = ["မသွားဘူး"]
    valid_words = {"သွား", "ဘူး"}
    freqs = {"သွား": 80_000, "ဘူး": 100_000}

    refined = refine_validation_tokens(
        words,
        is_valid_word=lambda w: w in valid_words,
        get_word_frequency=lambda w: freqs.get(w, 0),
    )

    assert "မ" in refined
    assert "ဘူး" in refined


def test_refine_preserves_multiple_tokens_independently() -> None:
    words = ["ကျောင်း", "သွားတယ်"]
    valid_words = {"ကျောင်း", "သွား", "တယ်"}
    freqs = {"ကျောင်း": 50_000, "သွား": 80_000, "တယ်": 500_000}

    refined = refine_validation_tokens(
        words,
        is_valid_word=lambda w: w in valid_words,
        get_word_frequency=lambda w: freqs.get(w, 0),
    )

    assert refined[0] == "ကျောင်း"
    assert "တယ်" in refined


def test_build_validation_token_paths_empty_returns_empty() -> None:
    paths = build_validation_token_paths(
        [],
        is_valid_word=lambda w: True,
        get_word_frequency=lambda w: 100,
    )
    assert paths == []


def test_build_validation_token_paths_single_path_when_lattice_max_is_one() -> None:
    from myspellchecker.core.config.algorithm_configs import TokenRefinementConfig

    words = ["ကျောင်းသွားတယ်"]
    valid_words = {"ကျောင်း", "သွား", "တယ်"}
    freqs = {"ကျောင်း": 50_000, "သွား": 80_000, "တယ်": 500_000}

    cfg = TokenRefinementConfig(lattice_max_paths=1)
    paths = build_validation_token_paths(
        words,
        is_valid_word=lambda w: w in valid_words,
        get_word_frequency=lambda w: freqs.get(w, 0),
        config=cfg,
    )

    assert len(paths) == 1
    assert paths[0] == words
