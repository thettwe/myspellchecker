from unittest.mock import MagicMock, patch

from myspellchecker import SpellChecker
from myspellchecker.core.response import Error, SyllableError


class TestDetectionPaths:
    """Tests for SpellChecker text-level detection methods."""

    def test_detect_particle_confusion_flags_suffix_ka_after_pronoun_object_pattern(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.get_word_pos.return_value = "V"
        errors: list[Error] = []

        checker._detect_particle_confusion("ဒီနေ့ ကျွန်တော် ထမင်းက စားပါတယ်", errors)

        particle_errors = [
            e for e in errors if e.error_type == "particle_confusion" and e.text == "က"
        ]
        assert particle_errors
        assert particle_errors[0].suggestions[0] == "ကို"

    def test_detect_particle_confusion_no_fp_on_repeated_subject_ka(self):
        """Multiple က subject markers in multi-clause sentences are valid.

        Repeated-ka alone should NOT trigger particle_confusion.  Only the
        pronoun-object pattern should fire.  This prevents FPs on sentences
        like "ကော်မရှင်က ... ရလဒ်များက ..." where both nouns are subjects.
        """
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.get_word_pos.return_value = "V"
        errors: list[Error] = []
        sentence = "စမ်းသပ်ချက် - ရွေးကောက်ပွဲ ကော်မရှင်က ရလဒ်များက ကြေညာခဲ့သည်"

        checker._detect_particle_confusion(sentence, errors)

        particle_errors = [
            e for e in errors if e.error_type == "particle_confusion" and e.text == "က"
        ]
        assert not particle_errors, "Repeated subject-marker က should not be flagged"

    def test_detect_particle_confusion_flags_nai_connector_missing_dotbelow(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        errors: list[Error] = []

        checker._detect_particle_confusion("ကျွန်တော်နဲ ဆရာ တွေ့မယ်", errors)

        particle_errors = [
            e for e in errors if e.error_type == "particle_confusion" and e.text == "နဲ"
        ]
        assert particle_errors
        assert particle_errors[0].suggestions[0] == "နဲ့"

    def test_detect_particle_confusion_flags_pronoun_ko_in_subject_slot(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.get_word_pos.return_value = "V"
        errors: list[Error] = []
        sentence = "ဒီမနက် ကျွန်တော်ကို သူငယ်ချင်းကို ပြောလိုက်တယ်"

        checker._detect_particle_confusion(sentence, errors)

        particle_errors = [
            e for e in errors if e.error_type == "particle_confusion" and e.text == "ကျွန်တော်ကို"
        ]
        assert particle_errors
        expected_pos = sentence.find("ကျွန်တော်ကို")
        assert any(
            e.position == expected_pos and e.suggestions[0] == "ကျွန်တော်က" for e in particle_errors
        )

    def test_detect_particle_confusion_flags_pronoun_ko_with_sentence_punctuation(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.get_word_pos.side_effect = lambda token: (
            "V" if token == "ပြောလိုက်တယ်" else "N"
        )
        errors: list[Error] = []
        sentence = "ဒီမနက် ကျွန်တော်ကို သူငယ်ချင်းကို ပြောလိုက်တယ်။"

        checker._detect_particle_confusion(sentence, errors)

        particle_errors = [
            e for e in errors if e.error_type == "particle_confusion" and e.text == "ကျွန်တော်ကို"
        ]
        assert particle_errors
        expected_pos = sentence.find("ကျွန်တော်ကို")
        assert any(
            e.position == expected_pos and e.suggestions[0] == "ကျွန်တော်က" for e in particle_errors
        )

    def test_detect_particle_confusion_flags_pronoun_ko_with_formal_verb_suffix(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.get_word_pos.return_value = None
        errors: list[Error] = []
        sentence = "အားလပ်ရက်မှာ သူကို သူငယ်ချင်းကို စာအုပ်ပေး၏ ဆိုပြီး မှတ်ထားတယ်။"

        checker._detect_particle_confusion(sentence, errors)

        particle_errors = [
            e for e in errors if e.error_type == "particle_confusion" and e.text == "သူကို"
        ]
        assert particle_errors
        expected_pos = sentence.find("သူကို")
        assert any(e.position == expected_pos and e.suggestions[0] == "သူက" for e in particle_errors)

    def test_detect_particle_confusion_keeps_valid_pronoun_object_ko(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.get_word_pos.return_value = "V"
        errors: list[Error] = []
        sentence = "သူက ကျွန်တော်ကို စာအုပ် ပေးတယ်"

        checker._detect_particle_confusion(sentence, errors)

        assert not [
            e for e in errors if e.error_type == "particle_confusion" and e.text == "ကျွန်တော်ကို"
        ]

    def test_detect_compound_confusion_typos_detects_ngat_ko_missing_ha_htoe(self):
        checker = SpellChecker.__new__(SpellChecker)
        errors: list[Error] = []

        checker._detect_compound_confusion_typos("ချိုသာစွာ မြည်တွန်သော ငက်ကို အားလုံး ချစ်ခင်ကြသည်", errors)

        compound_errors = [
            e for e in errors if e.error_type == "ha_htoe_confusion" and "ငက်" in e.text
        ]
        assert compound_errors
        assert "ငှက်" in compound_errors[0].suggestions[0]

    def test_detect_broken_compound_space_flags_prefix_join_before_tail(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.get_compound_confusion.return_value = None
        checker.segmenter = MagicMock()
        errors: list[Error] = []

        freq_map = {
            "အိမ်": 145010,
            "စာ": 140399,
            "အကြောင်း": 40000,
            "စာအကြောင်း": 227,
            "အိမ်စာ": 1307,
        }
        valid_words = set(freq_map)
        checker.provider.is_valid_word.side_effect = lambda w: w in valid_words
        checker.provider.get_word_frequency.side_effect = lambda w: freq_map.get(w, 0)
        checker.segmenter.segment_syllables.side_effect = lambda token: (
            ["စာ", "အကြောင်း"] if token == "စာအကြောင်း" else [token]
        )

        sentence = "ဌာနမှူးက အိမ် စာအကြောင်း မှတ်စုတစ်ပုဒ် ရေးသားခဲ့သည်"
        checker._detect_broken_compound_space(sentence, errors)

        compound_errors = [e for e in errors if e.error_type == "broken_compound"]
        assert compound_errors
        assert any(e.text == "အိမ် စာ" and e.suggestions[0] == "အိမ်စာ" for e in compound_errors)

    def test_detect_broken_compound_space_avoids_overjoin_on_high_freq_right_token(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.get_compound_confusion.return_value = None
        checker.segmenter = MagicMock()
        errors: list[Error] = []

        freq_map = {
            "မြန်မာ": 185000,
            "စာ": 140399,
            "အုပ်": 20140,
            "စာအုပ်": 96463,
            "မြန်မာစာ": 13030,
        }
        valid_words = set(freq_map)
        checker.provider.is_valid_word.side_effect = lambda w: w in valid_words
        checker.provider.get_word_frequency.side_effect = lambda w: freq_map.get(w, 0)
        checker.segmenter.segment_syllables.side_effect = lambda token: (
            ["စာ", "အုပ်"] if token == "စာအုပ်" else [token]
        )

        checker._detect_broken_compound_space("သူ မြန်မာ စာအုပ် ဖတ်တယ်", errors)

        assert not [e for e in errors if e.error_type == "broken_compound"]

    def test_detect_broken_compound_space_skips_verbal_prefix_join(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.get_compound_confusion.return_value = None
        checker.segmenter = MagicMock()
        errors: list[Error] = []

        freq_map = {
            "အလိုအလျောက်": 61262,
            "ပြုလုပ်": 244299,
            "သည်": 300000,
            "ပြုလုပ်သည်": 0,
            "အလိုအလျောက်ပြုလုပ်": 629,
        }
        pos_map = {
            "အလိုအလျောက်ပြုလုပ်": "ADV",
        }
        valid_words = {"အလိုအလျောက်", "ပြုလုပ်", "သည်", "အလိုအလျောက်ပြုလုပ်"}
        checker.provider.is_valid_word.side_effect = lambda w: w in valid_words
        checker.provider.get_word_frequency.side_effect = lambda w: freq_map.get(w, 0)
        checker.provider.get_word_pos.side_effect = lambda w: pos_map.get(w)
        checker.segmenter.segment_syllables.side_effect = lambda token: (
            ["ပြုလုပ်", "သည်"] if token == "ပြုလုပ်သည်" else [token]
        )

        checker._detect_broken_compound_space("စနစ်က အလိုအလျောက် ပြုလုပ်သည်", errors)

        assert not [e for e in errors if e.error_type == "broken_compound"]

    def test_detect_broken_compound_space_handles_attached_particle_tail(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.get_compound_confusion.return_value = None
        checker.segmenter = MagicMock()
        errors: list[Error] = []

        freq_map = {
            "ဆေးရုံ": 62000,
            "အုပ်": 21000,
            "အုပ်၏": 0,
            "ဆေးရုံအုပ်": 9800,
        }
        valid_words = {"ဆေးရုံ", "အုပ်", "ဆေးရုံအုပ်"}
        checker.provider.is_valid_word.side_effect = lambda w: w in valid_words
        checker.provider.get_word_frequency.side_effect = lambda w: freq_map.get(w, 0)
        checker.provider.get_word_pos.return_value = "N"
        checker.segmenter.segment_syllables.side_effect = lambda token: [token]

        sentence = "လေ့လာရေးစာတမ်းတွင် ဆေးရုံ အုပ်၏ တာဝန်ကို ဖော်ပြထားသည်"
        checker._detect_broken_compound_space(sentence, errors)

        compound_errors = [e for e in errors if e.error_type == "broken_compound"]
        assert compound_errors
        assert any(e.text == "ဆေးရုံ အုပ်" and e.suggestions[0] == "ဆေးရုံအုပ်" for e in compound_errors)

    def test_detect_suffix_confusion_typos_flags_invalid_nominal_suffixes(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        errors: list[Error] = []

        valid_words = {"စွမ်းဆောင်ရည်", "ထိရောက်မှု", "ညွှန်ကြားချက်"}
        freq_map = {
            "စွမ်းဆောင်ရည": 0,
            "စွမ်းဆောင်ရည်": 48971,
            "ထိရောက်မူ": 0,
            "ထိရောက်မှု": 8693,
            "ညွှန်ကြားချတ်": 0,
            "ညွှန်ကြားချက်": 7235,
        }
        checker.provider.is_valid_word.side_effect = lambda w: w in valid_words
        checker.provider.get_word_frequency.side_effect = lambda w: freq_map.get(w, 0)

        sentence = "စာတမ်းတွင် စွမ်းဆောင်ရည၊ ထိရောက်မူ၊ ညွှန်ကြားချတ် ကို စစ်ဆေးတယ်"
        checker._detect_suffix_confusion_typos(sentence, errors)

        word_errors = [e for e in errors if e.error_type == "invalid_word"]
        assert any(e.text == "စွမ်းဆောင်ရည" and e.suggestions[0] == "စွမ်းဆောင်ရည်" for e in word_errors)
        assert any(e.text == "ထိရောက်မူ" and e.suggestions[0] == "ထိရောက်မှု" for e in word_errors)
        assert any(e.text == "ညွှန်ကြားချတ်" and e.suggestions[0] == "ညွှန်ကြားချက်" for e in word_errors)

    def test_detect_suffix_confusion_typos_skips_when_source_token_is_valid(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        errors: list[Error] = []

        valid_words = {"ထိရောက်မူ", "ထိရောက်မှု"}
        freq_map = {"ထိရောက်မူ": 4200, "ထိရောက်မှု": 8693}
        checker.provider.is_valid_word.side_effect = lambda w: w in valid_words
        checker.provider.get_word_frequency.side_effect = lambda w: freq_map.get(w, 0)

        checker._detect_suffix_confusion_typos("ထိရောက်မူကို လေ့လာတယ်", errors)

        assert not [e for e in errors if e.error_type == "invalid_word"]

    def test_detect_suffix_confusion_typos_handles_attached_particle_suffix(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        errors: list[Error] = []

        valid_words = {"ထိရောက်မှု"}
        freq_map = {"ထိရောက်မူ": 0, "ထိရောက်မှု": 8693}
        checker.provider.is_valid_word.side_effect = lambda w: w in valid_words
        checker.provider.get_word_frequency.side_effect = lambda w: freq_map.get(w, 0)

        checker._detect_suffix_confusion_typos("သိုလှောင်မှုစနစ်တွင် ထိရောက်မူအတွက် ပြင်ဆင်တယ်", errors)

        word_errors = [e for e in errors if e.error_type == "invalid_word"]
        assert word_errors
        assert word_errors[0].text == "ထိရောက်မူ"
        assert word_errors[0].suggestions[0] == "ထိရောက်မှု"

    def test_detect_invalid_token_with_strong_candidates_recovers_high_freq_base_forms(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.syllable_validator = MagicMock()
        checker.syllable_validator.symspell = MagicMock()
        errors: list[Error] = []

        valid_words = {
            "စီမံကိန်း",
            "ကုန်ကျစရိတ်",
            "ဗီတာမင်",
        }
        freq_map = {
            "စီမံကိန်း": 84474,
            "ကုန်ကျစရိတ်": 27677,
            "ဗီတာမင်": 52254,
        }

        checker.provider.is_valid_word.side_effect = lambda w: w in valid_words
        checker.provider.get_word_frequency.side_effect = lambda w: freq_map.get(w, 0)

        def _mk(term: str, edit_distance: int, frequency: int) -> MagicMock:
            m = MagicMock()
            m.term = term
            m.edit_distance = edit_distance
            m.frequency = frequency
            return m

        def lookup_side_effect(word: str, **_kwargs):
            table = {
                "စမံကိန်း": [_mk("စီမံကိန်း", 1, 84474)],
                "ခုန်ကျစရိတ်": [_mk("ကုန်ကျစရိတ်", 1, 27677)],
                "ဗီတမင်": [_mk("ဗီတာမင်", 1, 52254)],
            }
            return table.get(word, [])

        checker.syllable_validator.symspell.lookup.side_effect = lookup_side_effect

        sentence = "စမံကိန်းအတွက် ခုန်ကျစရိတ်အကြောင်း ဗီတမင်အတွက်"
        checker._detect_invalid_token_with_strong_candidates(sentence, errors)

        word_errors = [e for e in errors if e.error_type == "invalid_word"]
        assert any(e.text == "စမံကိန်း" and e.suggestions[0] == "စီမံကိန်း" for e in word_errors)
        assert any(e.text == "ခုန်ကျစရိတ်" and e.suggestions[0] == "ကုန်ကျစရိတ်" for e in word_errors)
        assert any(e.text == "ဗီတမင်" and e.suggestions[0] == "ဗီတာမင်" for e in word_errors)

    def test_detect_invalid_token_with_strong_candidates_prefers_high_freq_variant(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.syllable_validator = MagicMock()
        checker.syllable_validator.symspell = MagicMock()
        errors: list[Error] = []

        valid_words = {"သံဃါ", "သံဃာ"}
        freq_map = {"သံဃါ": 64, "သံဃာ": 11270}
        checker.provider.is_valid_word.side_effect = lambda w: w in valid_words
        checker.provider.get_word_frequency.side_effect = lambda w: freq_map.get(w, 0)

        low = MagicMock()
        low.term = "သံဃါ"
        low.edit_distance = 1
        low.frequency = 64
        high = MagicMock()
        high.term = "သံဃာ"
        high.edit_distance = 2
        high.frequency = 11270
        checker.syllable_validator.symspell.lookup.return_value = [low, high]

        checker._detect_invalid_token_with_strong_candidates("သံဂါအတွက်", errors)

        assert errors
        assert errors[0].text == "သံဂါ"
        assert errors[0].suggestions[0] == "သံဃာ"

    def test_detect_invalid_token_with_strong_candidates_skips_finite_to_nonfinite_suffix_swaps(
        self,
    ):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.syllable_validator = MagicMock()
        checker.syllable_validator.symspell = MagicMock()
        errors: list[Error] = []

        valid_words = {"ရှင်းပြခဲ့သည့်", "အကြံပြုသည့်", "မှတ်တမ်းတင်ရန်"}
        freq_map = {
            "ရှင်းပြခဲ့သည့်": 72550,
            "အကြံပြုသည့်": 63110,
            "မှတ်တမ်းတင်ရန်": 44220,
        }
        checker.provider.is_valid_word.side_effect = lambda w: w in valid_words
        checker.provider.get_word_frequency.side_effect = lambda w: freq_map.get(w, 0)

        def _mk(term: str, edit_distance: int, frequency: int) -> MagicMock:
            m = MagicMock()
            m.term = term
            m.edit_distance = edit_distance
            m.frequency = frequency
            return m

        def lookup_side_effect(word: str, **_kwargs):
            table = {
                "ရှင်းပြခဲ့သည်": [_mk("ရှင်းပြခဲ့သည့်", 1, 72550)],
                "အကြံပြုသည်": [_mk("အကြံပြုသည့်", 1, 63110)],
                "မှတ်တမ်းတင်ရသည်": [_mk("မှတ်တမ်းတင်ရန်", 1, 44220)],
            }
            return table.get(word, [])

        checker.syllable_validator.symspell.lookup.side_effect = lookup_side_effect

        sentence = "ဆရာမက အဆင့်လိုက် ရှင်းပြခဲ့သည် ဟု မှတ်တမ်းတင်ရသည် ဆိုပြီး အကြံပြုသည်။"
        checker._detect_invalid_token_with_strong_candidates(sentence, errors)

        assert not [e for e in errors if e.error_type == "invalid_word"]

    def test_detect_frequency_dominant_valid_variants_uses_semantic_confirmation(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker._semantic_checker = MagicMock()
        errors: list[Error] = []

        valid_words = {"ကွန်ပြူတာ", "ကွန်ပျူတာ", "ကွန်ပျူတာဆိုင်ရာ", "ကွန်ပြူတာဆိုင်ရာ"}
        freq_map = {
            "ကွန်ပြူတာ": 15610,
            "ကွန်ပျူတာ": 75867,
            "ကွန်ပြူတာဆိုင်ရာ": 0,
            "ကွန်ပျူတာဆိုင်ရာ": 159,
        }
        checker.provider.is_valid_word.side_effect = lambda w: w in valid_words
        checker.provider.get_word_frequency.side_effect = lambda w: freq_map.get(w, 0)

        with patch(
            "myspellchecker.core.detectors.post_norm_mixins.compound_detection_mixin."
            "generate_confusable_variants",
            return_value={"ကွန်ပျူတာ"},
        ):
            checker._semantic_checker.score_mask_candidates.return_value = {
                "ကွန်ပြူတာဆိုင်ရာ": 2.739,
                "ကွန်ပျူတာဆိုင်ရာ": 3.114,
            }
            checker._detect_frequency_dominant_valid_variants(
                "ကွန်ပြူတာဆိုင်ရာ ပြဿနာကို တင်ပြသည်",
                errors,
            )

        confusable_errors = [e for e in errors if e.error_type == "confusable_error"]
        assert confusable_errors
        assert confusable_errors[0].text == "ကွန်ပြူတာဆိုင်ရာ"
        assert confusable_errors[0].suggestions[0] == "ကွန်ပျူတာဆိုင်ရာ"

    def test_detect_missing_asat_recovers_stem_before_attached_particle(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker._KEEP_ATTACHED_SUFFIXES = ("ကို", "က", "မှာ", "မှ")
        errors: list[Error] = []

        valid_words = {"စနစ်"}
        freq_map = {"စနစ်": 187197, "စနစ": 0}
        checker.provider.is_valid_word.side_effect = lambda w: w in valid_words
        checker.provider.get_word_frequency.side_effect = lambda w: freq_map.get(w, 0)

        text = "ယနေ့နံနက် လုံခြုံရေးအဖွဲ့က စနစကို စမ်းသပ်ပြီး"
        checker._detect_missing_asat(text, errors)

        asat_errors = [e for e in errors if e.error_type == "missing_asat" and e.text == "စနစ"]
        assert asat_errors
        assert asat_errors[0].suggestions[0] == "စနစ်"

    def test_detect_missing_asat_keeps_valid_stem_before_attached_particle(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker._KEEP_ATTACHED_SUFFIXES = ("ကို", "က", "မှာ", "မှ")
        errors: list[Error] = []

        valid_words = {"စနစ်"}
        freq_map = {"စနစ်": 187197}
        checker.provider.is_valid_word.side_effect = lambda w: w in valid_words
        checker.provider.get_word_frequency.side_effect = lambda w: freq_map.get(w, 0)

        checker._detect_missing_asat("ယနေ့ စနစ်ကို စမ်းသပ်တယ်", errors)

        assert not [e for e in errors if e.error_type == "missing_asat"]

    def test_detect_merged_classifier_mismatch_flags_inanimate_noun(self):
        checker = SpellChecker.__new__(SpellChecker)
        errors: list[Error] = []

        sentence = "စမ်းသပ်မှုအတွင်း လုံခြုံရေးအဖွဲ့က ရူတာ ငါးယောက် ပြင်ဆင်ခဲ့သည်"
        checker._detect_merged_classifier_mismatch(sentence, errors)

        classifier_errors = [e for e in errors if e.error_type == "classifier_error"]
        assert classifier_errors
        assert any(e.text == "ငါးယောက်" and e.suggestions[0] == "ငါးလုံး" for e in classifier_errors)

    def test_detect_merged_classifier_mismatch_flags_animate_noun(self):
        checker = SpellChecker.__new__(SpellChecker)
        errors: list[Error] = []

        sentence = "မနေ့က စစ်ကိုင်းတိုင်းတွင် ကော်မတီဝင် သုံးကောင် တက်ရောက်ခဲ့သည်"
        checker._detect_merged_classifier_mismatch(sentence, errors)

        classifier_errors = [e for e in errors if e.error_type == "classifier_error"]
        assert classifier_errors
        assert any(e.text == "သုံးကောင်" and e.suggestions[0] == "သုံးဦး" for e in classifier_errors)

    def test_detect_merged_classifier_mismatch_keeps_human_classifier(self):
        checker = SpellChecker.__new__(SpellChecker)
        errors: list[Error] = []

        checker._detect_merged_classifier_mismatch("ကျောင်းသား နှစ်ယောက် တက်ရောက်ခဲ့သည်", errors)

        assert not [e for e in errors if e.error_type == "classifier_error"]

    def test_filter_syllable_errors_preserves_classifier_error_inside_valid_token(self):
        from myspellchecker.core.correction_utils import filter_syllable_errors_in_valid_words

        text = "လုံခြုံရေးအဖွဲ့က ဆာဗာ နှစ်ယောက် ပြင်ဆင်ခဲ့သည်"
        words = ["လုံခြုံရေးအဖွဲ့က", "ဆာဗာ", "နှစ်ယောက်", "ပြင်ဆင်ခဲ့သည်"]
        validity_map = {w: True for w in words}
        errors = [
            SyllableError(
                text="နှစ်ယောက်",
                position=text.find("နှစ်ယောက်"),
                suggestions=["နှစ်လုံး"],
                error_type="classifier_error",
            )
        ]

        filtered = filter_syllable_errors_in_valid_words(text, errors, words, validity_map)
        assert filtered
        assert filtered[0].error_type == "classifier_error"

    def test_detect_formal_yi_in_colloquial_context_flags_verb_token(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.get_word_pos.return_value = "V"
        errors: list[Error] = []

        sentence = "အားလပ်ရက်မှာ သူငယ်ချင်းကို စာအုပ်ပေး၏ ဆိုပြီး မှတ်ထားတယ်"
        checker._detect_formal_yi_in_colloquial_context(sentence, errors)

        register_errors = [e for e in errors if e.error_type == "register_mixing"]
        assert register_errors
        assert any(
            e.text == "စာအုပ်ပေး၏" and e.suggestions[0] == "စာအုပ်ပေးတယ်" for e in register_errors
        )

    def test_detect_formal_yi_in_colloquial_context_skips_possessive_usage(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.get_word_pos.return_value = "N"
        errors: list[Error] = []

        checker._detect_formal_yi_in_colloquial_context("သူ၏ အိမ်ကို သူ ကြည့်တယ်", errors)

        assert not [e for e in errors if e.error_type == "register_mixing"]

    def test_detect_formal_yi_in_colloquial_context_handles_punctuation_and_case_marker(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.get_word_pos.return_value = "V"
        errors: list[Error] = []

        checker._detect_formal_yi_in_colloquial_context(
            "အားလပ်ရက်မှာ သူငယ်ချင်းကို စာအုပ်ပေး၏။",
            errors,
        )

        register_errors = [e for e in errors if e.error_type == "register_mixing"]
        assert register_errors
        assert register_errors[0].text == "စာအုပ်ပေး၏"

    def test_detect_semantic_agent_implausibility_flags_cat_exam_subject(self):
        checker = SpellChecker.__new__(SpellChecker)
        errors: list[Error] = []

        checker._detect_semantic_agent_implausibility("ကြောင်က စာမေးပွဲ ဖြေတယ်", errors)

        semantic_errors = [
            e for e in errors if e.error_type == "semantic_error" and e.text == "ကြောင်"
        ]
        assert semantic_errors
        assert semantic_errors[0].suggestions[0] == "ကျောင်းသား"

    def test_detect_sentence_structure_issues_suggests_polite_particle_for_kwa(self):
        checker = SpellChecker.__new__(SpellChecker)
        errors: list[Error] = []

        checker._detect_sentence_structure_issues("ဆရာမ ပြောပြီးပြီ ကွာ", errors)

        assert errors
        assert errors[0].error_type == "dangling_word"
        assert errors[0].suggestions
        assert errors[0].suggestions[0] == "ပါ"

    def test_detect_sentence_structure_issues_flags_verb_fronted_object_phrase(self):
        checker = SpellChecker.__new__(SpellChecker)
        errors: list[Error] = []

        checker._detect_sentence_structure_issues(
            "အတန်းပြီးနောက် သင်ကြားရေးအဖွဲ့က ဖြေကြားခဲ့သည် မေးခွန်းများကို တစ်ခုပြီးတစ်ခု ဟု ဆရာမက သတိပေးခဲ့သည်။",
            errors,
        )

        order_errors = [
            e for e in errors if e.error_type == "pos_sequence_error" and "ဖြေကြားခဲ့သည်" in e.text
        ]
        assert order_errors
        assert order_errors[0].suggestions
        assert order_errors[0].suggestions[0] == "မေးခွန်းများကို တစ်ခုပြီးတစ်ခု ဖြေကြားခဲ့သည်"

    def test_detect_sentence_structure_issues_skips_canonical_object_before_verb(self):
        checker = SpellChecker.__new__(SpellChecker)
        errors: list[Error] = []

        checker._detect_sentence_structure_issues(
            "အတန်းပြီးနောက် သင်ကြားရေးအဖွဲ့က မေးခွန်းများကို တစ်ခုပြီးတစ်ခု ဖြေကြားခဲ့သည်။",
            errors,
        )

        assert not [e for e in errors if e.error_type == "pos_sequence_error"]

    def test_detect_sentence_structure_issues_skips_verb_when_tail_is_compound_split(self):
        """When the first two tail tokens form a valid compound word,
        the G04 word-order detector should suppress the error entirely —
        the 'case marker' is on the compound, not a displaced argument."""
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.is_valid_word.side_effect = lambda w: w == "လူနာတင်ယာဉ်"
        errors: list[Error] = []

        checker._detect_sentence_structure_issues(
            "အတန်းပြီးနောက် သင်ကြားရေးအဖွဲ့က ရှင်းပြခဲ့သည် လူနာတင် ယာဉ်ကို အသေးစိတ်။",
            errors,
        )

        order_errors = [e for e in errors if e.error_type == "pos_sequence_error"]
        assert not order_errors  # Suppressed: tail tokens form compound လူနာတင်ယာဉ်

    def test_detect_tense_mismatch_flags_ne_khae_with_present_adverb(self):
        checker = SpellChecker.__new__(SpellChecker)
        errors: list[Error] = []

        checker._detect_tense_mismatch("သူ ဘောလုံးကို ကန်နေခဲ့တယ် အခုတော့ ရပ်သွားပြီ", errors)

        assert errors
        assert errors[0].text == "နေခဲ့"
        assert errors[0].suggestions
        assert errors[0].suggestions[0] == "နေ"

    def test_detect_tense_mismatch_flags_present_adverb_with_past_marker(self):
        checker = SpellChecker.__new__(SpellChecker)
        errors: list[Error] = []

        checker._detect_tense_mismatch("လက်ရှိတွင် အဖွဲ့က စုံစမ်းစစ်ဆေးခဲ့သည် ဟု သိရသည်", errors)

        assert errors
        # Now reports the full suffix (ခဲ့သည်) and its present replacement
        assert errors[0].text == "ခဲ့သည်"
        assert errors[0].error_type == "aspect_adverb_conflict"
        assert errors[0].suggestions
        assert errors[0].suggestions[0] == "နေသည်"

    def test_detect_tense_mismatch_recognizes_past_adverb_variant_with_future_marker(self):
        checker = SpellChecker.__new__(SpellChecker)
        errors: list[Error] = []

        checker._detect_tense_mismatch("ပြီးခဲ့သောလတွင် ဖွံ့ဖြိုးရေးအဖွဲ့က မိတ်ဆက်မည် ဟု ဆိုသည်", errors)

        assert errors
        # Now reports the full verb+marker token and its full corrected form
        assert errors[0].text == "မိတ်ဆက်မည်"
        assert errors[0].error_type == "tense_mismatch"
        assert errors[0].suggestions
        assert errors[0].suggestions[0] == "မိတ်ဆက်ခဲ့သည်"

    def test_detect_ha_htoe_particle_typos_skips_upehma_discourse_marker(self):
        checker = SpellChecker.__new__(SpellChecker)
        errors: list[Error] = []

        checker._detect_ha_htoe_particle_typos("ဥပမာ - စနစ်ကို ပြန်စမယ်", errors)

        assert not errors

    def test_detect_informal_h_after_completive_suggests_naw(self):
        checker = SpellChecker.__new__(SpellChecker)
        errors: list[Error] = []

        checker._detect_informal_h_after_completive("ညစာစားပြီးပြီ ဟ", errors)

        assert errors
        assert errors[0].text == "ဟ"
        assert errors[0].suggestions
        assert errors[0].suggestions[0] == "နော်"

    def test_detect_register_mixing_contextual_formal_colloquial_suffix(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.get_word_frequency.return_value = 0
        errors: list[Error] = []

        checker._detect_register_mixing("ကျွန်တော် အစီရင်ခံစာ တင်ပြတယ်", errors)

        assert errors
        assert errors[0].text == "တင်ပြတယ်"
        assert errors[0].suggestions
        assert errors[0].suggestions[0] == "တင်ပြပါသည်"

    def test_detect_register_mixing_full_token_formal_rewrite(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.get_word_frequency.return_value = 0
        errors: list[Error] = []

        checker._detect_register_mixing("ကျွန်တော် အစည်းအဝေးတက်ပါသည် နောက်မှ ထွက်တယ်", errors)

        assert errors
        assert errors[0].text == "ထွက်တယ်"
        assert errors[0].suggestions
        assert errors[0].suggestions[0] == "ထွက်ပါသည်"

    def test_detect_register_mixing_keeps_suffix_rewrite_for_long_stem(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.get_word_frequency.return_value = 0
        errors: list[Error] = []

        checker._detect_register_mixing("ကျွန်တော် သွားပါသည် ပြီးတော့ ပြန်လာတယ်", errors)

        assert errors
        assert errors[0].text == "တယ်"
        assert errors[0].suggestions
        assert errors[0].suggestions[0] == "ပါသည်"

    def test_detect_register_mixing_prefers_thi_for_literary_non_first_person(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.get_word_frequency.return_value = 0
        errors: list[Error] = []

        checker._detect_register_mixing(
            "ကဗျာဆရာ၏ နှလုံးသားတွင် နက်နဲသော ဝမ်းနည်းမှုဖြင့် ပြည့်နေတယ်",
            errors,
        )

        assert errors
        assert errors[0].text == "တယ်"
        assert errors[0].suggestions
        assert errors[0].suggestions[0] == "သည်"

    def test_detect_register_mixing_keeps_suffix_rewrite_for_future_modal(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.get_word_frequency.return_value = 0
        errors: list[Error] = []

        checker._detect_register_mixing("ယနေ့ ကျွန်တော် အစီရင်ခံစာ တင်ပြမယ်", errors)

        assert errors
        assert errors[0].text == "တင်ပြမယ်"
        assert errors[0].suggestions
        assert errors[0].suggestions[0] == "တင်ပြမည်"

    def test_detect_register_mixing_flags_formal_event_context_modal(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.get_word_frequency.return_value = 0
        errors: list[Error] = []

        checker._detect_register_mixing("ယနေ့ အစည်းအဝေး ကျင်းပမယ်", errors)

        assert errors
        assert errors[0].text == "မယ်"
        assert errors[0].suggestions
        assert errors[0].suggestions[0] == "မည်"

    def test_detect_register_mixing_does_not_fire_for_single_literary_adverb(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.get_word_frequency.return_value = 0
        errors: list[Error] = []

        checker._detect_register_mixing("ယနေ့ မိုးရွာလို့ လမ်းတွေ ရွှံ့ထူနေတယ်", errors)

        assert not errors

    def test_detect_register_mixing_does_not_fire_for_literary_adverb_plus_neutral_pronoun(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.get_word_frequency.return_value = 0
        errors: list[Error] = []

        checker._detect_register_mixing("ယနေ့ ကျွန်တော် ကျန်းမာပါတယ်", errors)

        assert not errors

    def test_detect_register_mixing_does_not_fire_for_literary_adverb_plus_single_formal_context(
        self,
    ):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.get_word_frequency.return_value = 0
        errors: list[Error] = []

        checker._detect_register_mixing("ယနေ့ သူ့ရဲ့ လက်ထပ်မင်္ဂလာပွဲ ကျင်းပခဲ့တယ်", errors)

        assert not errors

    def test_detect_punctuation_errors_overrides_register_mixing_at_same_position(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = MagicMock()
        checker.provider.get_word_frequency.return_value = 0
        errors: list[Error] = []
        sentence = "ယနေ့ အစည်းအဝေး ပြီးသွားတယ် နောက်မှ ပြန်လာမယ်"

        checker._detect_register_mixing(sentence, errors)
        checker._detect_punctuation_errors(sentence, errors)

        punctuation_errors = [e for e in errors if e.error_type == "missing_punctuation"]
        assert punctuation_errors
        assert punctuation_errors[0].text == "တယ် နောက်မှ"
        assert punctuation_errors[0].suggestions
        assert punctuation_errors[0].suggestions[0] == "တယ်။ နောက်"

    def test_detect_informal_with_honorific_prefers_shin_for_kwa(self):
        checker = SpellChecker.__new__(SpellChecker)
        errors: list[Error] = []

        checker._detect_informal_with_honorific("ဒော်ခင်မာ လာပြီကွာ", errors)

        assert errors
        assert errors[0].text == "ကွာ"
        assert errors[0].suggestions
        assert errors[0].suggestions[0] == "ရှင်"

    def test_detect_informal_with_honorific_prefers_shint_for_completive_kwa(self):
        checker = SpellChecker.__new__(SpellChecker)
        errors: list[Error] = []

        checker._detect_informal_with_honorific("ဒော်ခင်မာ ထမင်းစားပြီးပြီကွာ", errors)

        assert errors
        assert errors[0].text == "ကွာ"
        assert errors[0].suggestions
        assert errors[0].suggestions[0] == "ရှင့်"

    def test_detect_informal_with_honorific_prefers_pa_for_ha(self):
        checker = SpellChecker.__new__(SpellChecker)
        errors: list[Error] = []

        checker._detect_informal_with_honorific("ဆရာကြီး ထိုင်ပါဦး ဟ", errors)

        assert errors
        assert errors[0].text == "ဟ"
        assert errors[0].suggestions
        assert errors[0].suggestions[0] == "ပါ"

    def test_detect_informal_with_honorific_prefers_khinbya_after_question(self):
        checker = SpellChecker.__new__(SpellChecker)
        errors: list[Error] = []

        checker._detect_informal_with_honorific("ဆရာကြီး ထမင်းစားပြီးပြီလား ဟ", errors)

        assert errors
        assert errors[0].text == "ဟ"
        assert errors[0].suggestions
        assert errors[0].suggestions[0] == "ခင်ဗျာ"

    def test_detect_punctuation_errors_detects_boundary_before_nauk_hma(self):
        checker = SpellChecker.__new__(SpellChecker)
        errors: list[Error] = []

        checker._detect_punctuation_errors("အစည်းအဝေး ပြီးသွားတယ် နောက်မှ ပြန်လာမယ်", errors)

        assert errors
        assert errors[0].error_type == "missing_punctuation"
        assert "တယ် နောက်မှ" in errors[0].text
        assert errors[0].suggestions
        assert errors[0].suggestions[0] == "တယ်။ နောက်"

    def test_detect_punctuation_errors_detects_boundary_after_pyi_before_pronoun(self):
        checker = SpellChecker.__new__(SpellChecker)
        errors: list[Error] = []

        checker._detect_punctuation_errors("ဆရာမ ပြောပြီး ကျွန်တော် မှတ်သားတယ်", errors)

        assert errors
        assert errors[0].error_type == "missing_punctuation"
        assert "ပြီး ကျွန်တော်" in errors[0].text
        assert errors[0].suggestions
        assert errors[0].suggestions[0] == "ပြီး။ ကျွန်"

    def test_detect_punctuation_errors_detects_boundary_before_nga_clause(self):
        checker = SpellChecker.__new__(SpellChecker)
        errors: list[Error] = []

        checker._detect_punctuation_errors("သူ သွားတယ် ငါ နေခဲ့တယ်", errors)

        assert errors
        assert errors[0].error_type == "missing_punctuation"
        assert "တယ် ငါ" in errors[0].text
        assert errors[0].suggestions
        assert errors[0].suggestions[0] == "တယ်။ ငါ"

    def test_detect_sentence_structure_issues_treats_sotar_as_valid_connector(self):
        checker = SpellChecker.__new__(SpellChecker)
        errors: list[Error] = []

        checker._detect_sentence_structure_issues(
            "ဒီမနက် မိုးရွာပေမဲ့ ကျွန်တော် စျေးကို သွားခဲ့တယ် ဆိုတာ အိမ်သားတွေ သိတယ်။",
            errors,
        )

        assert not [e for e in errors if e.error_type == "dangling_word"]
