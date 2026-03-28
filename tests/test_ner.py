from myspellchecker.text.ner import NameHeuristic


class TestNameHeuristic:
    """Unit tests for NameHeuristic class."""

    def test_basic_honorifics(self):
        ner = NameHeuristic()
        # "Kyaw" after "U" should be a name
        assert ner.is_potential_name("Kyaw", prev_word="ဦး")
        # "Su" after "Daw"
        assert ner.is_potential_name("Su", prev_word="ဒေါ်")

    def test_whitelist(self):
        whitelist = {"Yangon", "Mandalay"}
        ner = NameHeuristic(whitelist=whitelist)
        assert ner.is_potential_name("Yangon")
        # Use Myanmar text that is NOT in whitelist and NOT an honorific/regex match
        assert not ner.is_potential_name("နေပြည်တော်")

    def test_regex_english(self):
        ner = NameHeuristic()
        assert ner.is_potential_name("Hello")
        assert ner.is_potential_name("Testing")
        assert not ner.is_potential_name("မြန်မာ")

    def test_regex_number(self):
        ner = NameHeuristic()
        assert ner.is_potential_name("123")
        assert ner.is_potential_name("၁၂၃")
        assert ner.is_potential_name("abc")  # English text is valid entity

    def test_regex_date(self):
        ner = NameHeuristic()
        assert ner.is_potential_name("12/12/2024")
        assert ner.is_potential_name("12-12-2024")
        assert ner.is_potential_name("၁၂-၁၂-၂၀၂၄")

    def test_analyze_sentence(self):
        ner = NameHeuristic()
        # Change "Ko" (ambiguous) to "Hnint" (With) which is not an honorific
        words = ["ကျွန်တော်", "ဦး", "ကျော်", "နှင့်", "တွေ့", "သည်"]
        # Should flag "ကျော်" as name (index 2) because it follows "ဦး"
        is_name = ner.analyze_sentence(words)

        assert not is_name[0]  # Kyun-taw
        assert not is_name[1]  # U
        assert is_name[2]  # Kyaw
        assert not is_name[4]  # Tway (Now follows "Hnint", so shouldn't be flagged)

    def test_ambiguous_honorific_ko_as_particle(self):
        """ကို after noun = object particle, not honorific."""
        ner = NameHeuristic()
        # "ဒီသတင်းကို ကျား" — ကို is object marker after သတင်း (news)
        words = ["ဒီ", "သတင်း", "ကို", "ကျား", "ဖူး", "လား"]
        is_name = ner.analyze_sentence(words)
        assert not is_name[3]  # ကျား should NOT be masked as name

    def test_ambiguous_honorific_ko_as_honorific(self):
        """ကို at sentence start = honorific "Ko"."""
        ner = NameHeuristic()
        # "ကို ကျော်" — ကို is honorific at sentence start
        words = ["ကို", "ကျော်", "ကို", "တွေ့", "တယ်"]
        is_name = ner.analyze_sentence(words)
        assert is_name[1]  # ကျော် after sentence-initial ကို is a name

    def test_ambiguous_honorific_ma_as_negation(self):
        """မ before verb = negation, not honorific."""
        ner = NameHeuristic()
        # "ကိစ္စတော့ မ ရိ ပါ ဘူး" — မ is negation prefix
        words = ["ကိစ္စတော့", "မ", "ရိ", "ပါ", "ဘူး"]
        is_name = ner.analyze_sentence(words)
        assert not is_name[2]  # ရိ should NOT be masked as name

    def test_ambiguous_honorific_ma_as_honorific(self):
        """မ at sentence start = honorific "Ma"."""
        ner = NameHeuristic()
        # "မ ခင်" — မ is honorific for Ms.
        words = ["မ", "ခင်", "ကို", "တွေ့", "တယ်"]
        is_name = ner.analyze_sentence(words)
        assert is_name[1]  # ခင် after sentence-initial မ is a name
