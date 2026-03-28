"""
Sample corpus for testing and demonstration.

This module provides a comprehensive sample corpus of ~270 Myanmar sentences
covering all supported grammar rules and patterns. It is used by the
`myspellchecker build --sample` command to generate a functional test database.

Usage:
    from myspellchecker.data_pipeline.sample_corpus import get_sample_corpus

    sentences = get_sample_corpus()

For production use, build with a real corpus of 10,000+ sentences.
"""

from __future__ import annotations


def get_sample_corpus() -> list[str]:
    """
    Get the sample corpus for testing and demonstration.

    This corpus includes ~270 diverse Myanmar sentences covering:

    **Pronouns & Register:**
    - First person: ကျွန်တော် (male polite), ကျွန်မ (female polite), ငါ (informal)
    - Second person: ခင်ဗျား (polite), မင်း (informal), ရှင် (female polite)
    - Third person: သူ, သူမ, သူတို့
    - Honorifics: ဦး, ဒေါ်, ကို, မ, ဆရာ, ဆရာမ

    **Particles & Tense:**
    - Past: ခဲ့, ပြီ
    - Future: မည်, မယ်, လိမ့်မည်
    - Progressive: နေ
    - Completion: ပြီး

    **Classifiers:**
    - People: ယောက်, ဦး, ပါး
    - Animals: ကောင်
    - Objects: ခု, လုံး, ချပ်
    - Vehicles: စီး, စင်း

    **Postpositions:**
    - Location: မှာ, တွင်
    - Direction: ကို, သို့
    - Source: မှ, က

    **Sentence Types:**
    - Statements, questions, commands
    - Negation patterns
    - Compound sentences
    - Relative clauses

    Returns:
        List of Myanmar sentences suitable for building a test database.

    Note:
        For production use, build with a real corpus of 10,000+ sentences.
    """
    return _SAMPLE_SENTENCES.copy()


# ============================================================
# SAMPLE CORPUS DATA
# Organized by grammatical category for maintainability
# ============================================================

_PRONOUNS_SENTENCES = [
    # First person - Male polite
    "ကျွန်တော် ကျောင်းသွားသည်",
    "ကျွန်တော် စာဖတ်နေသည်",
    "ကျွန်တော် အလုပ်လုပ်သည်",
    "ကျွန်တော် ထမင်းစားပြီးပြီ",
    "ကျွန်တော် မနက်ဖြန် လာမည်",
    # First person - Female polite
    "ကျွန်မ အိမ်ပြန်မည်",
    "ကျွန်မ စာရေးနေသည်",
    "ကျွန်မ ဈေးသွားခဲ့သည်",
    "ကျွန်မ ကျောင်းမှာ သင်သည်",
    "ကျွန်မ နေကောင်းပါသည်",
    # First person - Informal
    "ငါ သွားမယ်",
    "ငါ စားပြီးပြီ",
    "ငါ မလာဘူး",
    # First person - Plural
    "ကျွန်တော်တို့ သွားကြမည်",
    "ကျွန်မတို့ လာကြသည်",
    "သူတို့ စားကြပြီ",
    # Second person - Polite
    "ခင်ဗျား နေကောင်းပါသလား",
    "ခင်ဗျား ဘယ်သွားမလဲ",
    "ရှင် ဘာလုပ်နေလဲ",
    "ရှင် စားပြီးပြီလား",
    # Second person - Informal
    "မင်း လာခဲ့",
    "မင်း ဘာလုပ်နေတာလဲ",
    "နင် သွားတော့",
    # Third person
    "သူ စာဖတ်နေသည်",
    "သူမ ရေးနေသည်",
    "သူတို့ ကစားနေကြသည်",
    "သူ မလာသေးပါ",
    "သူမ ရောက်ပြီ",
    # Honorifics as subjects
    "ဦးမောင်မောင် ရောက်ပြီ",
    "ဒေါ်ခင်ခင် သွားပြီ",
    "ကိုအောင် လာမည်",
    "မမြ စာဖတ်သည်",
    "ဆရာ သင်ပြသည်",
    "ဆရာမ ရှင်းပြသည်",
]

_TENSE_SENTENCES = [
    # Past tense with ခဲ့
    "ကျွန်တော် သွားခဲ့သည်",
    "သူမ စားခဲ့သည်",
    "သူ ရေးခဲ့သည်",
    "ကလေးများ ကစားခဲ့ကြသည်",
    "ဆရာ သင်ခဲ့သည်",
    # Past tense with ပြီ (completion)
    "စားပြီးပြီ",
    "သွားပြီးပြီ",
    "ရေးပြီးပြီ",
    "ဖတ်ပြီးပြီ",
    "လုပ်ပြီးပြီ",
    # Future tense - Formal (မည်)
    "ကျွန်တော် သွားမည်",
    "သူမ လာမည်",
    "သူ ရေးမည်",
    "ကျွန်မတို့ စားကြမည်",
    # Future tense - Colloquial (မယ်)
    "ငါ သွားမယ်",
    "သူ လာမယ်",
    "စားမယ်",
    "ဖတ်မယ်",
    # Probable future (လိမ့်မည်)
    "သူ ရောက်လိမ့်မည်",
    "မိုးရွာလိမ့်မည်",
    "အောင်မြင်လိမ့်မည်",
    # Progressive (နေ)
    "ကျွန်တော် စားနေသည်",
    "သူမ ရေးနေသည်",
    "သူ ဖတ်နေသည်",
    "ကလေးများ အိပ်နေကြသည်",
    "မိုး ရွာနေသည်",
]

_ASPECT_SENTENCES = [
    # Completion with ပြီး (and then)
    "စာဖတ်ပြီး အိပ်မည်",
    "ထမင်းစားပြီး သွားမည်",
    "အလုပ်လုပ်ပြီး အိမ်ပြန်မည်",
    "ရေချိုးပြီး အဝတ်လဲမည်",
    # Habitual (တတ်)
    "သူ စာဖတ်တတ်သည်",
    "ကလေး ငိုတတ်သည်",
    "ကျွန်တော် စောစောထတတ်သည်",
    # Resultative (ထား)
    "တံခါးပိတ်ထားသည်",
    "စာရေးထားသည်",
    "ထမင်းချက်ထားပြီ",
    "ပိုက်ဆံသိမ်းထားသည်",
]

_CLASSIFIER_SENTENCES = [
    # People classifiers
    "လူ တစ်ယောက်",
    "ကလေး နှစ်ယောက်",
    "ကျောင်းသား သုံးယောက်",
    "ဆရာ တစ်ဦး",
    "ဧည့်သည် နှစ်ဦး",
    "ဘုန်းကြီး တစ်ပါး",
    "ရဟန်း သုံးပါး",
    # Animal classifiers
    "ခွေး တစ်ကောင်",
    "ကြောင် နှစ်ကောင်",
    "ငှက် သုံးကောင်",
    "ငါး လေးကောင်",
    "ဆင် ငါးကောင်",
    # Object classifiers - General
    "စားပွဲ တစ်ခု",
    "ကုလားထိုင် နှစ်ခု",
    "အိမ် သုံးခု",
    # Object classifiers - Round
    "ဘောလုံး တစ်လုံး",
    "ပန်းသီး နှစ်လုံး",
    "ဥ သုံးလုံး",
    # Object classifiers - Flat
    "စက္ကူ တစ်ရွက်",
    "ဓာတ်ပုံ နှစ်ချပ်",
    "အဝတ် သုံးထည်",
    # Object classifiers - Long
    "ဘောပင် တစ်ခြောင်း",
    "ခဲတံ နှစ်ခြောင်း",
    # Book classifiers
    "စာအုပ် တစ်အုပ်",
    "စာအုပ် နှစ်အုပ်",
    "ဂျာနယ် သုံးစောင်",
    # Vehicle classifiers
    "ကား တစ်စီး",
    "လှေ နှစ်စင်း",
    "လေယာဉ် တစ်စင်း",
    # Time classifiers
    "တစ်ကြိမ်",
    "နှစ်ကြိမ်",
    "သုံးခါ",
    "တစ်နေ့",
    "နှစ်ရက်",
]

_POSTPOSITION_SENTENCES = [
    # Location (မှာ/တွင်)
    "ကျွန်တော် ကျောင်းမှာ ရှိသည်",
    "သူမ ရုံးမှာ အလုပ်လုပ်သည်",
    "ကလေး အိမ်မှာ ရှိသည်",
    "စာအုပ် စားပွဲပေါ်မှာ ရှိသည်",
    "သူ ရန်ကုန်တွင် နေသည်",
    "အစည်းအဝေး ရုံးတွင် ကျင်းပသည်",
    # Direction (ကို/သို့)
    "ကျွန်မ ဈေးကို သွားမည်",
    "သူ ကျောင်းကို သွားသည်",
    "မိဘများ အိမ်သို့ ပြန်လာသည်",
    "ကျွန်တော် ရန်ကုန်သို့ သွားမည်",
    "စာကို ဆရာထံ ပေးသည်",
    # Source (မှ/က)
    "သူ ကျောင်းမှ ပြန်လာသည်",
    "ကျွန်တော် ရုံးမှ ထွက်သည်",
    "စာ ဆရာ့ထံမှ ရသည်",
    "အသံ အိမ်ထဲက ကြားသည်",
]

_NEGATION_SENTENCES = [
    # Standard negation (မ...ပါ)
    "ကျွန်တော် မသွားပါ",
    "သူမ မစားပါ",
    "သူ မရေးပါ",
    "ကလေး မအိပ်ပါ",
    # Negation with သေး (not yet)
    "ကလေး မအိပ်သေးပါ",
    "ဆရာ မရောက်သေးပါ",
    "ထမင်း မချက်သေးပါ",
    "စာ မရေးသေးပါ",
    # Colloquial negation (မ...ဘူး)
    "ငါ မသွားဘူး",
    "သူ မလာဘူး",
    "မစားဘူး",
    "မသိဘူး",
]

_QUESTION_SENTENCES = [
    # WH-questions
    "ဘယ်သူလဲ",
    "ဘာလဲ",
    "ဘယ်မှာလဲ",
    "ဘယ်တော့လဲ",
    "ဘယ်လိုလဲ",
    "ဘာကြောင့်လဲ",
    "ဘယ်နှစ်ယောက်လဲ",
    "ဘယ်လောက်လဲ",
    # Yes/No questions (လား)
    "သွားမလား",
    "စားပြီးပြီလား",
    "နေကောင်းပါသလား",
    "ရေးပြီးပြီလား",
    "နားလည်လား",
    # Softer questions (လဲ)
    "ဘာလုပ်နေလဲ",
    "ဘယ်သွားမလဲ",
    "ဘာစားမလဲ",
]

_COMMAND_SENTENCES = [
    # Polite commands (ပါ)
    "ကျေးဇူးပြု၍ ထိုင်ပါ",
    "စာဖတ်ပါ",
    "ရေးပါ",
    "သွားပါ",
    "လာပါ",
    "စားပါ",
    "နားထောင်ပါ",
    # Informal commands
    "လာခဲ့",
    "သွားတော့",
    "ထိုင်",
    "စား",
    # Requests with ပေး
    "ကူညီပေးပါ",
    "ပြောပြပေးပါ",
    "ရေးပေးပါ",
    "ပို့ပေးပါ",
]

_COMPLEX_SENTENCES = [
    # Sequential actions
    "ကျွန်တော် နံနက်စာစားပြီး ကျောင်းသွားသည်",
    "သူမ အလုပ်လုပ်ပြီး အိမ်ပြန်သည်",
    "ရေချိုးပြီး အဝတ်လဲပြီး ထွက်သည်",
    # Relative clauses
    "စာဖတ်နေသော ကလေး",
    "သွားမည့် လူ",
    "စားခဲ့သော အစားအစာ",
    "ရေးထားသော စာ",
    "ဝယ်လာသော ပစ္စည်း",
    # Conditional
    "မိုးရွာရင် မသွားဘူး",
    "အချိန်ရရင် လာမယ်",
]

_POLITENESS_SENTENCES = [
    # Polite expressions
    "ကျေးဇူးတင်ပါသည်",
    "ဝမ်းနည်းပါသည်",
    "တောင်းပန်ပါသည်",
    "ဂုဏ်ယူပါသည်",
    "ဝမ်းသာပါသည်",
    "ကြိုဆိုပါသည်",
    # Formal register
    "ဤအချက်ကို သတိပြုပါ",
    "ထိုသို့ ဆောင်ရွက်မည်",
    # Literary/gazette register
    "၎င်းသည် မှန်ကန်ပါသည်",
]

_VOCABULARY_SENTENCES = [
    # Places
    "ရန်ကုန်မြို့ ကြီးမားသည်",
    "မန္တလေးမြို့ လှပသည်",
    "နေပြည်တော်သည် မြို့တော် ဖြစ်သည်",
    "မြန်မာနိုင်ငံ လှပသည်",
    # Actions - High frequency
    "စာ ရေးသည်",
    "စာ ဖတ်သည်",
    "စာ သင်သည်",
    "အလုပ် လုပ်သည်",
    "ထမင်း စားသည်",
    "ရေ သောက်သည်",
    "အိပ်ယာဝင်သည်",
    "နိုးထသည်",
    "လမ်းလျှောက်သည်",
    "ပြေးသည်",
    # Adjectives
    "ကောင်းသည်",
    "လှသည်",
    "ကြီးသည်",
    "ငယ်သည်",
    "မြန်သည်",
    "နှေးသည်",
    "ခက်သည်",
    "လွယ်သည်",
    "သစ်လွင်သည်",
    "ဟောင်းသည်",
    # Time expressions
    "ယနေ့ သွားမည်",
    "မနက်ဖြန် လာမည်",
    "မနေ့က သွားခဲ့သည်",
    "အခု စားနေသည်",
    "နောက်မှ ပြောမည်",
    "မနက် စောစော ထမည်",
    "ညနေ ပြန်မည်",
]

_DEMONSTRATIVE_SENTENCES = [
    "ဤစာအုပ်သည် ကောင်းသည်",
    "ထိုလူသည် ဆရာဖြစ်သည်",
    "ဒီကလေးက စာတော်သည်",
    "ဟိုအိမ်က ကြီးသည်",
    "ဤနေရာတွင် ထိုင်ပါ",
]

_POSSESSIVE_SENTENCES = [
    "ကျွန်တော်၏ စာအုပ်",
    "သူ၏ အိမ်",
    "သူမ၏ ကား",
    "ကျွန်မရဲ့ မိသားစု",
    "သူတို့ရဲ့ ကျောင်း",
]

_FREQUENCY_SENTENCES = [
    # Repeated patterns for n-gram training
    "ကျွန်တော် သွားသည်",
    "ကျွန်တော် လာသည်",
    "ကျွန်တော် စားသည်",
    "ကျွန်တော် ဖတ်သည်",
    "ကျွန်တော် ရေးသည်",
    "သူ သွားသည်",
    "သူ လာသည်",
    "သူ စားသည်",
    "သူ ဖတ်သည်",
    "သူ ရေးသည်",
    "သူမ သွားသည်",
    "သူမ လာသည်",
    "သူမ စားသည်",
    "သူမ ဖတ်သည်",
    "သူမ ရေးသည်",
    # Common sentence endings
    "သွားပါသည်",
    "လာပါသည်",
    "စားပါသည်",
    "ရေးပါသည်",
    "ဖတ်ပါသည်",
]

_GRAMMAR_SENTENCES = [
    # Comparison
    "ဒီစာအုပ်က ပိုကောင်းသည်",
    "သူက ကျွန်တော်ထက် မြန်သည်",
    "ဒီဟာက အကောင်းဆုံး",
    # Ability (နိုင်)
    "ကျွန်တော် မြန်မာစာ ရေးနိုင်သည်",
    "သူ ရေကူးနိုင်သည်",
    "ကလေး စာဖတ်နိုင်ပြီ",
    # Permission/possibility (ရ)
    "သွားလို့ရသည်",
    "စားလို့ရသလား",
    "ဒီမှာ ထိုင်လို့ရသည်",
    # Want/desire (ချင်)
    "ကျွန်တော် သွားချင်သည်",
    "စားချင်သည်",
    "အိပ်ချင်သည်",
    # Must/have to (ရမည်)
    "သွားရမည်",
    "စားရမည်",
    "လုပ်ရမည်",
]

# Combine all sentences into the main corpus
_SAMPLE_SENTENCES: list[str] = (
    _PRONOUNS_SENTENCES
    + _TENSE_SENTENCES
    + _ASPECT_SENTENCES
    + _CLASSIFIER_SENTENCES
    + _POSTPOSITION_SENTENCES
    + _NEGATION_SENTENCES
    + _QUESTION_SENTENCES
    + _COMMAND_SENTENCES
    + _COMPLEX_SENTENCES
    + _POLITENESS_SENTENCES
    + _VOCABULARY_SENTENCES
    + _DEMONSTRATIVE_SENTENCES
    + _POSSESSIVE_SENTENCES
    + _FREQUENCY_SENTENCES
    + _GRAMMAR_SENTENCES
)
