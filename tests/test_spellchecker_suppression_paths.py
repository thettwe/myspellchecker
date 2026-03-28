from unittest.mock import MagicMock

from myspellchecker import SpellChecker
from myspellchecker.core.response import Error


class TestSuppressionPaths:
    """Tests for SpellChecker error suppression and filtering methods."""

    def test_suppress_low_value_context_probability_drops_high_freq_no_suggestion(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.is_valid_word.return_value = True
        checker.provider.get_word_frequency.return_value = 3000
        errors = [
            Error(
                text="ကြွေးမြီ",
                position=11,
                suggestions=[],
                error_type="context_probability",
            ),
            Error(
                text="ကွာ",
                position=0,
                suggestions=["ပါ"],
                error_type="dangling_word",
            ),
        ]

        checker._suppress_low_value_context_probability(errors)

        assert len(errors) == 1
        assert errors[0].error_type == "dangling_word"

    def test_suppress_low_value_context_probability_keeps_rare_token(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.is_valid_word.return_value = True
        checker.provider.get_word_frequency.return_value = 20
        errors = [
            Error(
                text="ရှားပါးစကား",
                position=0,
                suggestions=[],
                error_type="context_probability",
            )
        ]

        checker._suppress_low_value_context_probability(errors)

        assert len(errors) == 1
        assert errors[0].error_type == "context_probability"

    def test_suppress_low_value_context_probability_drops_reduplicated_valid_word(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.is_valid_word.return_value = True
        checker.provider.get_word_frequency.return_value = 3000
        errors = [
            Error(
                text="ဖြေးဖြေး",
                position=0,
                suggestions=["တဖြေးဖြေး", "ဖွေးဖွေး"],
                error_type="context_probability",
            )
        ]

        checker._suppress_low_value_context_probability(errors, text="ဖြေးဖြေး စကားပြောပါ")

        assert len(errors) == 0

    def test_suppress_low_value_context_probability_drops_fragment_inside_compound(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.is_valid_word.return_value = True
        checker.provider.get_word_frequency.return_value = 5000
        errors = [
            Error(
                text="ညနေ",
                position=0,
                suggestions=["နေ", "ယနေ့"],
                error_type="context_probability",
            )
        ]

        checker._suppress_low_value_context_probability(errors, text="ညနေချိန် နေဝင်အရောင်အဝါသည်")

        assert len(errors) == 0

    def test_suppress_low_value_context_probability_drops_short_function_suggestions(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.is_valid_word.return_value = True
        checker.provider.get_word_frequency.return_value = 8000
        checker._KEEP_ATTACHED_SUFFIXES = ()
        errors = [
            Error(
                text="စနစ်",
                position=10,
                suggestions=["စစ်", "စိတ်", "ရန်"],
                error_type="context_probability",
            )
        ]

        checker._suppress_low_value_context_probability(errors, text="အပ်ဒိတ်ကို စနစ်မှတ်တမ်းတွင် ဖော်ပြထားသည်")

        assert len(errors) == 0

    def test_suppress_low_value_syntax_errors_drops_quotative_invalid_start(self):
        checker = SpellChecker.__new__(SpellChecker)
        text = "သူ မနက်က လာတယ် လို့ ပြောတယ်"
        errors = [
            Error(
                text="လို့",
                position=text.find("လို့"),
                suggestions=["Invalid start"],
                error_type="syntax_error",
            ),
            Error(
                text="ဆေးဆိုင်",
                position=8,
                suggestions=[""],
                error_type="context_probability",
            ),
        ]

        checker._suppress_low_value_syntax_errors(errors, text=text)

        assert len(errors) == 1
        assert errors[0].error_type == "context_probability"

    def test_suppress_low_value_syntax_errors_drops_thi_to_mi_modal_swap(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.is_valid_word.return_value = True
        checker.provider.get_word_frequency.return_value = 5000
        text = "ယနေ့ အစီရင်ခံစာတွင် ဖော်ပြထားသည်။"
        errors = [
            Error(
                text="သည်",
                position=text.find("သည်။"),
                suggestions=["မည်"],
                error_type="syntax_error",
                confidence=0.8,
            ),
            Error(
                text="ကျောင်း",
                position=0,
                suggestions=[""],
                error_type="context_probability",
            ),
        ]

        checker._suppress_low_value_syntax_errors(errors, text=text)

        assert len(errors) == 1
        assert errors[0].error_type == "context_probability"

    def test_suppress_low_value_pos_sequence_errors_drops_known_artifact_tokens(self):
        checker = SpellChecker.__new__(SpellChecker)
        errors = [
            Error(
                text="ဆအတည်ပြု",
                position=0,
                suggestions=["က", "ခု"],
                error_type="pos_sequence_error",
            ),
            Error(
                text="တိတိကျကျမှတ်တမ်းတင်",
                position=10,
                suggestions=[],
                error_type="pos_sequence_error",
            ),
            Error(
                text="မျို့တော်အကြောင်း",
                position=20,
                suggestions=["မြို့တော်"],
                error_type="pos_sequence_error",
            ),
        ]

        checker._suppress_low_value_pos_sequence_errors(errors)

        assert len(errors) == 1
        assert errors[0].text == "မျို့တော်အကြောင်း"

    def test_suppress_low_value_syllable_errors_drops_non_boundary_loan_fragment(self):
        checker = SpellChecker.__new__(SpellChecker)
        text = "အင်ဂျင်နီယာအဖွဲ့က ဆာဗာလော့ဂ်များကို ယနေ့ည စစ်ဆေးမည်။"
        errors = [
            Error(
                text="လော့ဂ်",
                position=text.find("လော့ဂ်"),
                suggestions=["လောဂ်"],
                error_type="invalid_syllable",
            ),
            Error(
                text="မည်",
                position=text.find("မည်"),
                suggestions=["မီ"],
                error_type="invalid_syllable",
            ),
        ]

        checker._suppress_low_value_syllable_errors(errors, text=text)

        assert len(errors) == 1
        assert errors[0].text == "မည်"

    def test_suppress_low_value_confusable_errors_drops_fragment_inside_word(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.get_word_frequency.return_value = 100
        errors = [
            Error(
                text="မှ",
                position=0,
                suggestions=["မ"],
                error_type="confusable_error",
            )
        ]

        checker._suppress_low_value_confusable_errors(errors, text="မှတ်တမ်း")

        assert errors == []

    def test_suppress_low_value_confusable_errors_drops_high_freq_single_char_swap(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.get_word_frequency.side_effect = lambda token: (
            50000 if token in {"က", "မ"} else 100
        )
        errors = [
            Error(
                text="က",
                position=2,
                suggestions=["မ"],
                error_type="confusable_error",
            )
        ]

        checker._suppress_low_value_confusable_errors(errors, text="သူ က လာတယ်")

        assert errors == []

    def test_suppress_low_value_confusable_errors_keeps_non_trivial_confusable(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.get_word_frequency.return_value = 300
        errors = [
            Error(
                text="နင့်",
                position=3,
                suggestions=["နှင့်"],
                error_type="confusable_error",
            )
        ]

        checker._suppress_low_value_confusable_errors(errors, text="သူ နင့် အတူ သွားတယ်")

        assert len(errors) == 1
        assert errors[0].text == "နင့်"

    def test_suppress_low_value_confusable_errors_keeps_non_boundary_with_attached_suffix(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.get_word_frequency.return_value = 5000
        checker.provider.is_valid_word.return_value = True
        checker._KEEP_ATTACHED_SUFFIXES = ("ကို",)
        text = "ဒီမနက် ကျွန်တော်ကို အခွင့်အရေကို သူငယ်ချင်းကို ပြောလိုက်တယ်။"
        token_pos = text.find("အရေကို")
        errors = [
            Error(
                text="အရေ",
                position=token_pos,
                suggestions=["အရေး", "အရေ"],
                error_type="confusable_error",
                confidence=0.75,
            )
        ]

        checker._suppress_low_value_confusable_errors(errors, text=text)

        assert len(errors) == 1
        assert errors[0].text == "အရေ"

    def test_suppress_low_value_confusable_errors_drops_self_suggestion_non_boundary(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.get_word_frequency.return_value = 100
        errors = [
            Error(
                text="သည်",
                position=2,
                suggestions=["သည်း", "သည်"],
                error_type="confusable_error",
            )
        ]

        checker._suppress_low_value_confusable_errors(errors, text="မိုးသည်းထန်စွာ")

        assert errors == []

    def test_suppress_low_value_confusable_errors_drops_boundary_self_suggestion_short_particle(
        self,
    ):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.is_valid_word.return_value = True
        checker.provider.get_word_frequency.return_value = 50000
        text = "ယနေ့ ရန်ကုန်မြို့တွင် မိုးသည်းထန်စွာ ရွာသွန်းခဲ့သည်။"
        token_pos = text.find("သည်။")
        errors = [
            Error(
                text="သည်",
                position=token_pos,
                suggestions=["သည်း", "သည်"],
                error_type="confusable_error",
                confidence=0.8,
            )
        ]

        checker._suppress_low_value_confusable_errors(errors, text=text)

        assert errors == []

    def test_suppress_low_value_semantic_errors_drops_short_candidate_drift_on_long_noun(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.is_valid_word.return_value = True
        checker.provider.get_word_frequency.return_value = 5000
        text = "ကျောင်းသားများသည် စာကြည့်တိုက်တွင် တိတ်တဆိတ် လေ့လာနေကြသည် ဟု ပါမောက္ခက ထပ်မံရှင်းပြခဲ့သည်။"
        token_pos = text.find("ပါမောက္ခ")
        errors = [
            Error(
                text="ပါမောက္ခ",
                position=token_pos,
                suggestions=["သူ", "သူမ", "ဆရာမ"],
                error_type="semantic_error",
                confidence=0.8,
            )
        ]

        checker._suppress_low_value_semantic_errors(errors, text=text)

        assert errors == []

    def test_suppress_generic_pos_sequence_errors_prefers_specific_root_cause(self):
        errors = [
            Error(
                text="ကြေးဇူးတင်ပါတယ််",
                position=0,
                suggestions=["ကြေးဇူးတင်ပါတယ်် ဖြစ်သည်", "ကြေးဇူးတင်ပါတယ်် ဖြစ်ပါသည်"],
                error_type="pos_sequence_error",
            ),
            Error(
                text="ကြေးဇူး",
                position=0,
                suggestions=["ကျေးဇူး"],
                error_type="medial_confusion",
            ),
            Error(
                text="တယ််",
                position=12,
                suggestions=["တယ်"],
                error_type="invalid_syllable",
            ),
        ]

        SpellChecker._suppress_generic_pos_sequence_errors(errors)

        assert len(errors) == 2
        assert all(e.error_type != "pos_sequence_error" for e in errors)
