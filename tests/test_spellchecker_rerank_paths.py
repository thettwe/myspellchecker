from myspellchecker import SpellChecker
from myspellchecker.core.response import Error


class TestRerankDetectorSuggestions:
    """Tests for _rerank_detector_suggestions_by_distance method."""

    def test_rerank_detector_suggestions_promotes_close_form_candidate(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = None
        error = Error(
            text="ပျုတာ",
            position=0,
            suggestions=["ကွန်ပျူတာ", "ပျူတာ", "ကွန်"],
            error_type="confusable_error",
        )

        checker._rerank_detector_suggestions_by_distance([error])

        assert error.suggestions[0] == "ပျူတာ"

    def test_rerank_detector_suggestions_does_not_promote_identity_text(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = None
        error = Error(
            text="ကျန်",
            position=0,
            suggestions=["ကျန်းမာ", "ကျန်", "ကျန်း"],
            error_type="confusable_error",
        )

        checker._rerank_detector_suggestions_by_distance([error])

        assert error.suggestions[0] == "ကျန်း"

    def test_rerank_detector_suggestions_applies_phrase_targeted_hint(self):
        checker = SpellChecker.__new__(SpellChecker)
        error = Error(
            text="ဘဏ္ာ",
            position=0,
            suggestions=["ဘဏ္ဍာ", "ဘဏ္ဍာရေး"],
            error_type="incomplete_stacking",
        )

        checker._rerank_detector_suggestions_by_distance([error])

        assert error.suggestions[0] == "ဘဏ္ဍာရေး"

    def test_rerank_detector_suggestions_promotes_khaung_for_headache_context(self):
        checker = SpellChecker.__new__(SpellChecker)
        error = Error(
            text="ကေါင်း",
            position=0,
            suggestions=["ကောင်း", "ခေါင်း", "ခေါင်းကိုက်"],
            error_type="invalid_syllable",
        )

        checker._rerank_detector_suggestions_by_distance([error])

        assert error.suggestions[0] == "ခေါင်း"

    def test_rerank_detector_suggestions_keeps_kaung_without_headache_cue(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = None
        error = Error(
            text="ကေါင်း",
            position=0,
            suggestions=["ကောင်း", "ခေါင်း", "ကောင်းတယ်"],
            error_type="invalid_syllable",
        )

        checker._rerank_detector_suggestions_by_distance([error])

        assert error.suggestions[0] == "ကောင်း"

    def test_rerank_detector_suggestions_promotes_khe_from_vowel_after_asat(self):
        checker = SpellChecker.__new__(SpellChecker)
        error = Error(
            text="လာခဲ့ေသည်",
            position=0,
            suggestions=["လာခဲ့သည်", "လာခဲ့", "ခဲ့", "သည်"],
            error_type="vowel_after_asat",
        )

        checker._rerank_detector_suggestions_by_distance([error])

        assert error.suggestions[0] == "ခဲ့"

    def test_rerank_detector_records_literal_hint_telemetry(self):
        checker = SpellChecker.__new__(SpellChecker)
        error = Error(
            text="ကွာ",
            position=0,
            suggestions=["ခင်ဗျာ", "ပါ"],
            error_type="dangling_word",
        )

        checker._rerank_detector_suggestions_by_distance([error], sentence_text="အဆင်ပြေပြီ ကွာ")

        assert error.suggestions[0] == "ပါ"
        telemetry = checker._last_rerank_rule_telemetry
        assert telemetry
        rule_id = next(iter(telemetry))
        assert rule_id.startswith("literal_hint:")
        assert telemetry[rule_id]["fires"] == 1
        assert telemetry[rule_id]["top1_changes"] == 1

    def test_rerank_detector_records_distance_rerank_telemetry(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = None
        error = Error(
            text="abc",
            position=0,
            suggestions=["xyz", "abcd"],
            error_type="invalid_syllable",
        )

        checker._rerank_detector_suggestions_by_distance([error], sentence_text="abc")

        assert error.suggestions[0] == "abcd"
        telemetry = checker._last_rerank_rule_telemetry
        assert telemetry["distance_rerank:invalid_syllable"]["fires"] == 1
        assert telemetry["distance_rerank:invalid_syllable"]["top1_changes"] == 1

    def test_rerank_detector_suggestions_promotes_correct_leader_form(self):
        checker = SpellChecker.__new__(SpellChecker)
        error = Error(
            text="ကောင်း",
            position=0,
            suggestions=["ခောင်း", "ခောင်းဆောင်", "ခောင်းဆေ"],
            error_type="ha_htoe_confusion",
        )

        checker._rerank_detector_suggestions_by_distance([error])

        assert error.suggestions[0] == "ခေါင်းဆောင်"

    def test_rerank_detector_suggestions_keeps_way_for_zero_wa_token(self):
        checker = SpellChecker.__new__(SpellChecker)
        error = Error(
            text="၀",
            position=0,
            suggestions=["ဝယ်", "ဝ"],
            error_type="invalid_syllable",
        )

        checker._rerank_detector_suggestions_by_distance([error])

        assert error.suggestions[0] == "ဝ"

    def test_rerank_detector_suggestions_keeps_kabar_base_form_for_kamar(self):
        checker = SpellChecker.__new__(SpellChecker)
        error = Error(
            text="ကမ္ာ့",
            position=0,
            suggestions=["ကမ္ဘာ", "ကမ္ဘာ့", "ကမာ့"],
            error_type="broken_virama",
        )

        checker._rerank_detector_suggestions_by_distance([error])

        assert error.suggestions[0] == "ကမ္ဘာ့"

    def test_rerank_detector_suggestions_prefers_lar_over_ye_lar_for_bare_lae(self):
        checker = SpellChecker.__new__(SpellChecker)
        error = Error(
            text="လဲ",
            position=0,
            suggestions=["ရဲ့လား", "လား", "လဲ"],
            error_type="question_structure",
        )

        checker._rerank_detector_suggestions_by_distance([error])

        assert error.suggestions[0] == "လား"

    def test_rerank_detector_suggestions_promotes_ye_lar_for_yes_no_question(self):
        checker = SpellChecker.__new__(SpellChecker)
        error = Error(
            text="ရဲ့လဲ",
            position=0,
            suggestions=["လား", "ရဲ့လား", "လဲ"],
            error_type="confusable_error",
        )

        checker._rerank_detector_suggestions_by_distance([error])

        assert error.suggestions[0] == "ရဲ့လား"

    def test_rerank_detector_suggestions_promotes_kaung_for_koing_classifier(self):
        checker = SpellChecker.__new__(SpellChecker)
        error = Error(
            text="ကိုင်",
            position=0,
            suggestions=["ကွိုင်", "ကွို", "ကွိ", "ကွ"],
            error_type="confusable_error",
        )

        checker._rerank_detector_suggestions_by_distance([error])

        assert error.suggestions[0] == "ကောင်"

    def test_rerank_detector_suggestions_promotes_formal_particle_from_parsaye(self):
        checker = SpellChecker.__new__(SpellChecker)
        error = Error(
            text="ပါစေ",
            position=0,
            suggestions=["ပါရစေ", "ပါ", "ပေ", "စေ"],
            error_type="context_probability",
        )

        checker._rerank_detector_suggestions_by_distance([error])

        assert error.suggestions[0] == "ပါသည်"

    def test_rerank_detector_suggestions_canonicalizes_classifier_chaung(self):
        checker = SpellChecker.__new__(SpellChecker)
        error = Error(
            text="ယောက်",
            position=0,
            suggestions=["ခြောင်း", "ယောက်"],
            error_type="syntax_error",
        )

        checker._rerank_detector_suggestions_by_distance([error])

        assert error.suggestions[0] == "ချောင်း"

    def test_rerank_detector_suggestions_injects_missing_candidate_for_byaut(self):
        checker = SpellChecker.__new__(SpellChecker)
        error = Error(
            text="ပြော့",
            position=0,
            suggestions=["ပျော့", "ပျော"],
            error_type="confusable_error",
        )

        checker._rerank_detector_suggestions_by_distance([error])

        assert error.suggestions[0] == "ပြော"

    def test_rerank_detector_suggestions_respects_hint_toggle(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker._enable_targeted_rerank_hints = False
        checker._enable_targeted_candidate_injections = True
        error = Error(
            text="ကွာ",
            position=0,
            suggestions=["ကွာ", "ပါ"],
            error_type="question_structure",
        )

        checker._rerank_detector_suggestions_by_distance([error])

        assert error.suggestions[0] == "ကွာ"

    def test_rerank_detector_suggestions_respects_injection_toggle(self):
        checker = SpellChecker.__new__(SpellChecker)
        checker.provider = None
        checker._enable_targeted_rerank_hints = True
        checker._enable_targeted_candidate_injections = False
        error = Error(
            text="ပြော့",
            position=0,
            suggestions=["ပျော့", "ပျော"],
            error_type="confusable_error",
        )

        checker._rerank_detector_suggestions_by_distance([error])

        assert "ပြော" not in error.suggestions

    def test_rerank_detector_suggestions_injects_missing_candidate_for_pos_sequence(self):
        checker = SpellChecker.__new__(SpellChecker)
        error = Error(
            text="သွားခဲ့မည်",
            position=0,
            suggestions=[],
            error_type="pos_sequence_error",
        )

        checker._rerank_detector_suggestions_by_distance([error])

        assert error.suggestions[0] == "သွားမည်"

    def test_rerank_detector_suggestions_injects_empty_for_zero_width_error(self):
        checker = SpellChecker.__new__(SpellChecker)
        error = Error(
            text="\u200b",
            position=0,
            suggestions=["မြန်မာ"],
            error_type="invalid_syllable",
        )

        checker._rerank_detector_suggestions_by_distance([error])

        assert error.suggestions[0] == ""

    def test_rerank_detector_suggestions_promotes_way_for_buy_context(self):
        checker = SpellChecker.__new__(SpellChecker)
        error = Error(
            text="၀",
            position=7,
            suggestions=["ဝ", "ဝယ်", "ဝယ်စရာ"],
            error_type="invalid_syllable",
        )

        checker._rerank_detector_suggestions_by_distance(
            [error], sentence_text="ဈေးမှာ ၀ယ်စရာ များတယ်"
        )

        assert error.suggestions[0] == "ဝယ်"

    def test_rerank_detector_suggestions_injects_minem_for_miinirm(self):
        checker = SpellChecker.__new__(SpellChecker)
        error = Error(
            text="နိ်း",
            position=6,
            suggestions=["နန်း", "နည်း"],
            error_type="invalid_syllable",
        )

        checker._rerank_detector_suggestions_by_distance(
            [error], sentence_text="ထို မိန်ိးမက ဆေးရုံတွင် အလုပ်လုပ်သည်"
        )

        assert error.suggestions[0] == "မိန်းမ"

    def test_rerank_detector_suggestions_injects_dammahta_from_wide_token(self):
        checker = SpellChecker.__new__(SpellChecker)
        error = Error(
            text="တ",
            position=13,
            suggestions=["တာ", "တူ"],
            error_type="invalid_syllable",
        )

        checker._rerank_detector_suggestions_by_distance(
            [error], sentence_text="နိုင်ငံတော်သမတကြီးသည် နိုင်ငံခြားခရီးစဉ်မှ ပြန်လည်ရောက်ရှိခဲ့သည်"
        )

        assert error.suggestions[0] == "သမ္မတ"

    def test_rerank_detector_suggestions_injects_empty_for_naw_ka_particle(self):
        checker = SpellChecker.__new__(SpellChecker)
        error = Error(
            text="က",
            position=21,
            suggestions=["ပါ"],
            error_type="syntax_error",
        )

        checker._rerank_detector_suggestions_by_distance([error], sentence_text="အဲဒီလို မလုပ်နဲ့ နော်က")

        assert error.suggestions[0] == ""

    def test_rerank_detector_suggestions_prefers_lar_for_kyanmar_ye_lae(self):
        checker = SpellChecker.__new__(SpellChecker)
        error = Error(
            text="ရဲ့လဲ",
            position=12,
            suggestions=["ရဲ့လား", "လား", "လဲ"],
            error_type="question_structure",
        )

        checker._rerank_detector_suggestions_by_distance([error], sentence_text="မင်း ကျန်းမာရဲ့လဲ")

        assert error.suggestions[0] == "လား"

    def test_rerank_detector_suggestions_injects_hin_for_chicken_car(self):
        checker = SpellChecker.__new__(SpellChecker)
        error = Error(
            text="ကား",
            position=8,
            suggestions=["က", "ဆား", "စား"],
            error_type="context_probability",
        )

        checker._rerank_detector_suggestions_by_distance([error], sentence_text="ကြက်သား ကား ချက်တယ်")

        assert error.suggestions[0] == "ဟင်း"

    def test_rerank_detector_suggestions_injects_nay_tal_for_sathin_pyi(self):
        checker = SpellChecker.__new__(SpellChecker)
        error = Error(
            text="ပြီ",
            position=25,
            suggestions=["ပြီး", "ပြီ", "ပြ"],
            error_type="confusable_error",
        )

        checker._rerank_detector_suggestions_by_distance(
            [error], sentence_text="အခု ဆရာမ အတန်းထဲမှာ စာသင်ပြီ"
        )

        assert error.suggestions[0] == "နေတယ်"

    def test_rerank_detector_suggestions_prefers_pyu_for_system_software_sentence(self):
        checker = SpellChecker.__new__(SpellChecker)
        error = Error(
            text="ကွန်ပျုတာ",
            position=0,
            suggestions=["ကွန်ပျူတာ", "ကွန်", "ပျူတာ", "ပျူ"],
            error_type="medial_confusion",
        )

        checker._rerank_detector_suggestions_by_distance(
            [error],
            sentence_text="ကွန်ပျုတာ စနစ်ကို ပြန်လည်တည်ဆောက်ရန် ဆော့ဖ်၀ဲအသစ် ထည်သွင်းရမည်",
        )

        assert error.suggestions[0] == "ပျူတာ"

    def test_rerank_detector_konpyuta_noop_when_full_form_already_top1(self):
        checker = SpellChecker.__new__(SpellChecker)
        error = Error(
            text="ကွန်ပျုတာ",
            position=0,
            suggestions=["ကွန်ပျူတာ", "ပျူတာ", "ကွန်"],
            error_type="medial_confusion",
        )

        checker._rerank_detector_suggestions_by_distance([error], sentence_text="ကွန်ပျုတာ သင်ခန်းစာ")

        assert error.suggestions[0] == "ကွန်ပျူတာ"
        telemetry = getattr(checker, "_last_rerank_rule_telemetry", {})
        assert "literal_hint:medial_confusion:ကွန်ပျုတာ" not in telemetry

    def test_rerank_detector_suggestions_injects_tal_for_generic_double_asat_variant(self):
        checker = SpellChecker.__new__(SpellChecker)
        error = Error(
            text="တယ််",
            position=10,
            suggestions=["တယ်", "တယ့်", "တယ်း"],
            error_type="invalid_syllable",
        )

        checker._rerank_detector_suggestions_by_distance([error], sentence_text="သူ သွားပါတယ််")

        assert error.suggestions[0] == "တယ်"

    def test_rerank_detector_suggestions_injects_patay_for_larkhae_patay_double_asat(self):
        checker = SpellChecker.__new__(SpellChecker)
        error = Error(
            text="တယ််",
            position=10,
            suggestions=["တယ်", "တယ့်", "တယ်း"],
            error_type="invalid_syllable",
        )

        checker._rerank_detector_suggestions_by_distance([error], sentence_text="သူ လာခဲ့ပါတယ််")

        assert error.suggestions[0] == "ပါတယ်"

    def test_rerank_detector_suggestions_injects_patay_for_patay_double_asat_error(self):
        checker = SpellChecker.__new__(SpellChecker)
        error = Error(
            text="ပါတယ််",
            position=6,
            suggestions=["ပါတယ့်", "ပါတ်", "ပါတယ်း"],
            error_type="invalid_syllable",
        )

        checker._rerank_detector_suggestions_by_distance([error], sentence_text="သူ လာခဲ့ပါတယ််")

        assert error.suggestions[0] == "ပါတယ်"

    def test_rerank_detector_suggestions_injects_thaw_for_pyaw_thaw_pattern(self):
        checker = SpellChecker.__new__(SpellChecker)
        error = Error(
            text="ပြေော",
            position=5,
            suggestions=["ပြော", "ပျော"],
            error_type="invalid_syllable",
        )

        checker._rerank_detector_suggestions_by_distance(
            [error], sentence_text="ဆရာမ ပြောေသာ စာကို နားထောင်ပါ"
        )

        assert error.suggestions[0] == "သော"

    def test_rerank_detector_suggestions_injects_student_for_cat_exam_context(self):
        checker = SpellChecker.__new__(SpellChecker)
        error = Error(
            text="ကြောင်",
            position=0,
            suggestions=["ကျွန်တော်", "သူ", "ငါ"],
            error_type="semantic_error",
        )

        checker._rerank_detector_suggestions_by_distance(
            [error], sentence_text="ကြောင်က စာမေးပွဲ ဖြေတယ်"
        )

        assert error.suggestions[0] == "ကျောင်းသား"

    def test_rerank_detector_suggestions_injects_lu_for_fish_coffee_context(self):
        checker = SpellChecker.__new__(SpellChecker)
        error = Error(
            text="ငါး",
            position=0,
            suggestions=["သူ", "အမေ", "မင်း"],
            error_type="semantic_error",
        )

        checker._rerank_detector_suggestions_by_distance(
            [error], sentence_text="ငါးက ကော်ဖီ ချက်ပေးတယ်"
        )

        assert error.suggestions[0] == "လူ"
