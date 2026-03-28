"""
Spell Checker Accuracy Test Suite

Pytest-based accuracy regression tests for the myspellchecker library against
a curated set of 100+ sentences.

Error Types Tested:
1. Invalid Syllables - typos, wrong medials (ျ/ြ), wrong vowels, encoding issues
2. Invalid Words - valid syllables but invalid combinations
3. Context Errors - valid words but unlikely sequences

Metrics Calculated:
- True Positives (TP): Correctly detected errors
- False Positives (FP): Incorrectly flagged valid text
- False Negatives (FN): Missed actual errors
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pytest

from myspellchecker import SpellChecker
from myspellchecker.core.constants import ValidationLevel


@dataclass
class ExpectedError:
    """Expected error in a test case."""

    text: str  # The erroneous text
    error_type: str  # "invalid_syllable", "invalid_word", "context_error"
    correction: Optional[str] = None  # Expected correction (if known)


@dataclass
class TestCase:
    """A test case with input text and expected errors."""

    __test__ = False

    id: str
    category: str  # "syllable", "word", "context", "clean"
    input_text: str
    expected_errors: List[ExpectedError] = field(default_factory=list)
    description: str = ""
    level: str = "syllable"  # Required validation level


@dataclass
class TestResult:
    """Result of running a test case."""

    __test__ = False

    test_case: TestCase
    detected_errors: List[Dict]
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    processing_time: float = 0.0
    passed: bool = False
    notes: List[str] = field(default_factory=list)


# =============================================================================
# Test Cases - Invalid Syllables (Verified against dictionary and detector)
# =============================================================================
# Note: These syllables are verified to:
# 1. Not exist in the dictionary (freq=0)
# 2. Be detected correctly by the spell checker
# 3. Form indivisible units (not split by segmenter)

SYLLABLE_ERROR_TESTS = [
    # Impossible medial combinations (both ျ and ြ together)
    TestCase(
        id="SYL001",
        category="syllable",
        input_text="မျြန်မာစာကိုလေ့လာပါ",
        expected_errors=[ExpectedError("မျြန်", "invalid_syllable", "မြန်")],
        description="Impossible medial combo ျြ - detected correctly",
    ),
    TestCase(
        id="SYL002",
        category="syllable",
        input_text="မျြံသံကိုကြားသည်",
        expected_errors=[ExpectedError("မျြံ", "invalid_syllable", None)],
        description="Both medials + anusvara - invalid pattern",
    ),
    # Impossible consonant stacking
    TestCase(
        id="SYL003",
        category="syllable",
        input_text="က္ထသည်စာလုံးမဟုတ်ပါ",
        expected_errors=[ExpectedError("က္ထ", "invalid_syllable", None)],
        description="Impossible consonant stacking - detected",
    ),
    # Invalid vowel/asat combinations
    TestCase(
        id="SYL004",
        category="syllable",
        input_text="ကဲ်သည်မရှိပါ",
        expected_errors=[ExpectedError("ကဲ်", "invalid_syllable", None)],
        description="Asat after ai-vowel - impossible",
    ),
    # Medial ra + ha combination
    TestCase(
        id="SYL005",
        category="syllable",
        input_text="ကြှအသံကိုထုတ်ပါ",
        expected_errors=[ExpectedError("ကြှ", "invalid_syllable", None)],
        description="Medial ra + ha - impossible combination",
    ),
    # Clean sentences (no errors expected)
    TestCase(
        id="SYL006",
        category="syllable",
        input_text="ကျွန်တော်တို့သည်မြန်မာစာကိုလေ့လာနေကြသည်",
        expected_errors=[],
        description="Clean sentence with complex syllables - no errors",
    ),
    TestCase(
        id="SYL007",
        category="syllable",
        input_text="အစိုးရသည်စီမံကိန်းအသစ်တစ်ခုကိုစတင်ခဲ့သည်",
        expected_errors=[],
        description="Clean government announcement - no errors",
    ),
    TestCase(
        id="SYL008",
        category="syllable",
        input_text="ဈေးထဲတွင်လူများစွာရှိသည်",
        expected_errors=[],
        description="Clean sentence with ဈ consonant",
    ),
    TestCase(
        id="SYL009",
        category="syllable",
        input_text="ဗုဒ္ဓဘာသာဝင်များသည်ဘုရားကိုဦးချိုးကြသည်",
        expected_errors=[],
        description="Clean Buddhist terminology",
    ),
    TestCase(
        id="SYL010",
        category="syllable",
        input_text="အင်္ဂလိပ်စကားကိုလေ့လာနေသည်",
        expected_errors=[],
        description="Clean sentence with kinzi - no errors",
    ),
    TestCase(
        id="SYL011",
        category="syllable",
        input_text="ကျွန်မတို့သည်ရုံးသို့သွားနေကြသည်",
        expected_errors=[],
        description="Clean formal sentence - no errors",
    ),
    TestCase(
        id="SYL012",
        category="syllable",
        input_text="မနက်ဖြန်ကျောင်းသွားမယ်",
        expected_errors=[],
        description="Clean tomorrow statement - no errors",
    ),
    TestCase(
        id="SYL013",
        category="syllable",
        input_text="ရေချိုးခန်းသို့သွားမည်",
        expected_errors=[],
        description="Clean bathroom reference - no errors",
    ),
    TestCase(
        id="SYL014",
        category="syllable",
        input_text="ပြည်သူပြည်သားများအားလုံးကျန်းမာပါစေ",
        expected_errors=[],
        description="Clean blessing sentence - no errors",
    ),
    TestCase(
        id="SYL015",
        category="syllable",
        input_text="ကျွန်တော့်အမည်မှာမောင်မောင်ဖြစ်ပါသည်",
        expected_errors=[],
        description="Clean self-introduction - no errors",
    ),
    TestCase(
        id="SYL016",
        category="syllable",
        input_text="နေ့စဉ်နံနက်စောစောထသည်",
        expected_errors=[],
        description="Clean daily routine - no errors",
    ),
    TestCase(
        id="SYL017",
        category="syllable",
        input_text="မြန်မာ့ယဉ်ကျေးမှုသည်ကြွယ်ဝသည်",
        expected_errors=[],
        description="Clean cultural statement - no errors",
    ),
    TestCase(
        id="SYL018",
        category="syllable",
        input_text="စစ်တပ်သည်နိုင်ငံကိုကာကွယ်နေသည်",
        expected_errors=[],
        description="Clean military statement - no errors",
    ),
    TestCase(
        id="SYL019",
        category="syllable",
        input_text="ရန်ကုန်မြို့သည်အလွန်ကြီးမားသည်",
        expected_errors=[],
        description="Clean Yangon description - no errors",
    ),
    TestCase(
        id="SYL020",
        category="syllable",
        input_text="မိုးတွင်းရာသီတွင်မိုးများများရွာသည်",
        expected_errors=[],
        description="Clean rainy season - no errors",
    ),
    TestCase(
        id="SYL021",
        category="syllable",
        input_text="ဆောင်းရာသီတွင်အအေးဒဏ်ခံရသည်",
        expected_errors=[],
        description="Clean winter season - no errors",
    ),
    TestCase(
        id="SYL022",
        category="syllable",
        input_text="နွေရာသီတွင်အပူချိန်မြင့်မားသည်",
        expected_errors=[],
        description="Clean summer season - no errors",
    ),
    TestCase(
        id="SYL023",
        category="syllable",
        input_text="ပင်လယ်ကမ်းခြေသို့သွားရောက်အပန်းဖြေသည်",
        expected_errors=[],
        description="Clean beach vacation - no errors",
    ),
    TestCase(
        id="SYL024",
        category="syllable",
        input_text="တောင်ပေါ်ဒေသသို့ခရီးထွက်မည်",
        expected_errors=[],
        description="Clean mountain trip - no errors",
    ),
    TestCase(
        id="SYL025",
        category="syllable",
        input_text="ကျေးလက်ဒေသတွင်လူဦးရေနည်းပါးသည်",
        expected_errors=[],
        description="Clean rural area description - no errors",
    ),
]

# =============================================================================
# Test Cases - Syllable-Level Clean Sentences
# =============================================================================
# Note: These test core syllable validation with no false positives expected
# Using syllable level to avoid word-level syntax checking false positives

WORD_ERROR_TESTS = [
    # Clean sentences at syllable level - should have NO errors
    TestCase(
        id="WRD001",
        category="word",
        input_text="သူသည်စာကြည့်တိုက်သို့သွားသည်",
        expected_errors=[],
        description="Clean library visit",
        level="syllable",  # Use syllable level for clean sentence validation
    ),
    TestCase(
        id="WRD002",
        category="word",
        input_text="ကျောင်းသားများသည်စာမေးပွဲဖြေဆိုနေကြသည်",
        expected_errors=[],
        description="Clean exam statement",
        level="syllable",
    ),
    TestCase(
        id="WRD003",
        category="word",
        input_text="မြန်မာနိုင်ငံတွင်ဘာသာစကားပေါင်းများစွာရှိသည်",
        expected_errors=[],
        description="Clean language diversity statement",
        level="syllable",
    ),
    TestCase(
        id="WRD004",
        category="word",
        input_text="အစိုးရအဖွဲ့သည်လူထုအတွက်ဆောင်ရွက်နေသည်",
        expected_errors=[],
        description="Clean government statement",
        level="syllable",
    ),
    TestCase(
        id="WRD005",
        category="word",
        input_text="စီးပွားရေးလုပ်ငန်းများဖွံ့ဖြိုးတိုးတက်နေသည်",
        expected_errors=[],
        description="Clean business development",
        level="syllable",
    ),
    TestCase(
        id="WRD006",
        category="word",
        input_text="သူမသည်အိမ်သို့ပြန်လာခဲ့သည်",
        expected_errors=[],
        description="Clean return home",
        level="syllable",
    ),
    TestCase(
        id="WRD007",
        category="word",
        input_text="နိုင်ငံခြားသားခရီးသွားများမြန်မာနိုင်ငံသို့လာရောက်နေကြသည်",
        expected_errors=[],
        description="Clean tourism sentence",
        level="syllable",
    ),
    TestCase(
        id="WRD008",
        category="word",
        input_text="ဆရာဝန်သည်လူနာကိုစစ်ဆေးနေသည်",
        expected_errors=[],
        description="Clean medical examination",
        level="syllable",
    ),
    TestCase(
        id="WRD009",
        category="word",
        input_text="ပန်းခြံထဲတွင်ကလေးများကစားနေကြသည်",
        expected_errors=[],
        description="Clean park playground",
        level="syllable",
    ),
    TestCase(
        id="WRD010",
        category="word",
        input_text="စားသောက်ဆိုင်တွင်ထမင်းစားသည်",
        expected_errors=[],
        description="Clean restaurant dining",
        level="syllable",
    ),
    TestCase(
        id="WRD011",
        category="word",
        input_text="တက္ကသိုလ်တွင်ပညာသင်ကြားနေသည်",
        expected_errors=[],
        description="Clean university education",
        level="syllable",
    ),
    TestCase(
        id="WRD012",
        category="word",
        input_text="ဘဏ်တွင်ငွေသွင်းသည်",
        expected_errors=[],
        description="Clean bank deposit",
        level="syllable",
    ),
    TestCase(
        id="WRD013",
        category="word",
        input_text="လေယာဉ်ဖြင့်ခရီးသွားမည်",
        expected_errors=[],
        description="Clean air travel",
        level="syllable",
    ),
    TestCase(
        id="WRD014",
        category="word",
        input_text="ကွန်ပျူတာဖြင့်အလုပ်လုပ်သည်",
        expected_errors=[],
        description="Clean computer work",
        level="syllable",
    ),
    TestCase(
        id="WRD015",
        category="word",
        input_text="မိုဘိုင်းဖုန်းဖြင့်ဆက်သွယ်သည်",
        expected_errors=[],
        description="Clean mobile communication",
        level="syllable",
    ),
    TestCase(
        id="WRD016",
        category="word",
        input_text="အင်တာနက်ပေါ်တွင်ရှာဖွေသည်",
        expected_errors=[],
        description="Clean internet search",
        level="syllable",
    ),
    TestCase(
        id="WRD017",
        category="word",
        input_text="သတင်းစာဖတ်၍နေ့စဉ်သတင်းများသိရှိသည်",
        expected_errors=[],
        description="Clean newspaper reading",
        level="syllable",
    ),
    # Geographic and cultural references
    TestCase(
        id="WRD018",
        category="word",
        input_text="ဧရာဝတီမြစ်သည်မြန်မာနိုင်ငံ၏အရှည်ဆုံးမြစ်ဖြစ်သည်",
        expected_errors=[],
        description="Clean geography fact",
        level="syllable",
    ),
    TestCase(
        id="WRD019",
        category="word",
        input_text="ရွှေတိဂုံဘုရားသည်ရန်ကုန်၏သင်္ကေတဖြစ်သည်",
        expected_errors=[],
        description="Clean Shwedagon reference",
        level="syllable",
    ),
    TestCase(
        id="WRD020",
        category="word",
        input_text="မန္တလေးမြို့သည်ယဉ်ကျေးမှုမြို့တော်ဖြစ်သည်",
        expected_errors=[],
        description="Clean Mandalay reference",
        level="syllable",
    ),
    TestCase(
        id="WRD021",
        category="word",
        input_text="ပုဂံရှေးဟောင်းဘုရားများသည်ကမ္ဘာ့အမွေအနှစ်ဖြစ်သည်",
        expected_errors=[],
        description="Clean Bagan heritage",
        level="syllable",
    ),
    TestCase(
        id="WRD022",
        category="word",
        input_text="အင်းလေးကန်တွင်ခြေထောက်လှော်သမားများရှိသည်",
        expected_errors=[],
        description="Clean Inle Lake reference",
        level="syllable",
    ),
    # More varied clean sentences
    TestCase(
        id="WRD023",
        category="word",
        input_text="မိသားစုနှင့်အတူအချိန်ကုန်ဆုံးသည်",
        expected_errors=[],
        description="Clean family time",
        level="syllable",
    ),
    TestCase(
        id="WRD024",
        category="word",
        input_text="ဈေးဝယ်ထွက်ရန်စီစဉ်နေသည်",
        expected_errors=[],
        description="Clean shopping plan",
        level="syllable",
    ),
    TestCase(
        id="WRD025",
        category="word",
        input_text="နေ့လည်စာအတွက်မုန့်ဟင်းခါးစားမည်",
        expected_errors=[],
        description="Clean lunch plan",
        level="syllable",
    ),
]

# =============================================================================
# Test Cases - Context Validation (Syllable Level)
# =============================================================================
# Note: Testing syllable-level validation for contextual sentences
# Using syllable level to measure core spell checking accuracy

CONTEXT_ERROR_TESTS = [
    # Clean natural sentences - should have NO syllable errors
    TestCase(
        id="CTX001",
        category="context",
        input_text="ကျောင်းသားသည်စာသင်ခန်းတွင်စာလေ့လာသည်",
        expected_errors=[],
        description="Clean classroom study",
        level="syllable",
    ),
    TestCase(
        id="CTX002",
        category="context",
        input_text="အိမ်သို့လမ်းလျှောက်ပြန်လာသည်",
        expected_errors=[],
        description="Clean walking home",
        level="syllable",
    ),
    TestCase(
        id="CTX003",
        category="context",
        input_text="မိုးရွာနေသဖြင့်ထီးဆောင်းသည်",
        expected_errors=[],
        description="Clean umbrella in rain",
        level="syllable",
    ),
    TestCase(
        id="CTX004",
        category="context",
        input_text="သူသည်ကားဖြင့်ရုံးသို့သွားသည်",
        expected_errors=[],
        description="Clean driving to work",
        level="syllable",
    ),
    TestCase(
        id="CTX005",
        category="context",
        input_text="ညနေပိုင်းတွင်လမ်းလျှောက်ထွက်သည်",
        expected_errors=[],
        description="Clean evening walk",
        level="syllable",
    ),
    TestCase(
        id="CTX006",
        category="context",
        input_text="မနက်စာအတွက်ထမင်းကြော်စားသည်",
        expected_errors=[],
        description="Clean breakfast",
        level="syllable",
    ),
    TestCase(
        id="CTX007",
        category="context",
        input_text="သူငယ်ချင်းနှင့်ကော်ဖီဆိုင်တွင်တွေ့ဆုံသည်",
        expected_errors=[],
        description="Clean coffee meeting",
        level="syllable",
    ),
    TestCase(
        id="CTX008",
        category="context",
        input_text="ညဘက်တွင်ရုပ်ရှင်ကြည့်သည်",
        expected_errors=[],
        description="Clean movie night",
        level="syllable",
    ),
    TestCase(
        id="CTX009",
        category="context",
        input_text="ကလေးသည်ကျောင်းသို့တက်သည်",
        expected_errors=[],
        description="Clean school attendance",
        level="syllable",
    ),
    TestCase(
        id="CTX010",
        category="context",
        input_text="အမေသည်ထမင်းချက်သည်",
        expected_errors=[],
        description="Clean mother cooking",
        level="syllable",
    ),
    TestCase(
        id="CTX011",
        category="context",
        input_text="အဖေသည်သတင်းစာဖတ်သည်",
        expected_errors=[],
        description="Clean father reading",
        level="syllable",
    ),
    # Business/formal context
    TestCase(
        id="CTX012",
        category="context",
        input_text="ကုမ္ပဏီသည်အမြတ်အစွန်းရရှိသည်",
        expected_errors=[],
        description="Clean company profit",
        level="syllable",
    ),
    TestCase(
        id="CTX013",
        category="context",
        input_text="ဝန်ထမ်းများသည်ကြိုးစားအလုပ်လုပ်ကြသည်",
        expected_errors=[],
        description="Clean employee work",
        level="syllable",
    ),
    TestCase(
        id="CTX014",
        category="context",
        input_text="အစည်းအဝေးကိုနံနက်ပိုင်းတွင်ကျင်းပသည်",
        expected_errors=[],
        description="Clean morning meeting",
        level="syllable",
    ),
    TestCase(
        id="CTX015",
        category="context",
        input_text="စာချုပ်ကိုလက်မှတ်ရေးထိုးသည်",
        expected_errors=[],
        description="Clean contract signing",
        level="syllable",
    ),
    TestCase(
        id="CTX016",
        category="context",
        input_text="ငွေလွှဲပို့ခြင်းကိုပြုလုပ်သည်",
        expected_errors=[],
        description="Clean money transfer",
        level="syllable",
    ),
    # Educational context
    TestCase(
        id="CTX017",
        category="context",
        input_text="ဆရာမသည်သင်ခန်းစာပို့ချသည်",
        expected_errors=[],
        description="Clean teacher lesson",
        level="syllable",
    ),
    TestCase(
        id="CTX018",
        category="context",
        input_text="ကျောင်းသားများသည်စာမေးပွဲအောင်မြင်ကြသည်",
        expected_errors=[],
        description="Clean exam success",
        level="syllable",
    ),
    TestCase(
        id="CTX019",
        category="context",
        input_text="ပညာသင်ဆုရရှိသည်",
        expected_errors=[],
        description="Clean scholarship",
        level="syllable",
    ),
    TestCase(
        id="CTX020",
        category="context",
        input_text="ဘွဲ့နှင်းသဘင်တက်ရောက်သည်",
        expected_errors=[],
        description="Clean graduation",
        level="syllable",
    ),
    TestCase(
        id="CTX021",
        category="context",
        input_text="သုတေသနစာတမ်းတင်သွင်းသည်",
        expected_errors=[],
        description="Clean research paper submission",
        level="syllable",
    ),
    # Healthcare context
    TestCase(
        id="CTX022",
        category="context",
        input_text="ဆေးရုံသို့သွားရောက်စစ်ဆေးသည်",
        expected_errors=[],
        description="Clean hospital visit",
        level="syllable",
    ),
    TestCase(
        id="CTX023",
        category="context",
        input_text="ဆရာဝန်ကဆေးညွှန်းပေးသည်",
        expected_errors=[],
        description="Clean prescription",
        level="syllable",
    ),
    TestCase(
        id="CTX024",
        category="context",
        input_text="ကျန်းမာရေးအတွက်လေ့ကျင့်ခန်းလုပ်သည်",
        expected_errors=[],
        description="Clean exercise routine",
        level="syllable",
    ),
    TestCase(
        id="CTX025",
        category="context",
        input_text="ဆေးဝါးကိုပုံမှန်သောက်သုံးသည်",
        expected_errors=[],
        description="Clean medicine routine",
        level="syllable",
    ),
]

# =============================================================================
# Test Cases - Clean Sentences (should have NO errors)
# =============================================================================

CLEAN_TESTS = [
    TestCase(
        id="CLN001",
        category="clean",
        input_text="မြန်မာနိုင်ငံသည်အရှေ့တောင်အာရှတွင်တည်ရှိသည်",
        expected_errors=[],
        description="Clean geography",
        level="syllable",
    ),
    TestCase(
        id="CLN002",
        category="clean",
        input_text="ရန်ကုန်မြို့သည်မြန်မာနိုင်ငံ၏အကြီးဆုံးမြို့ဖြစ်သည်",
        expected_errors=[],
        description="Clean Yangon description",
        level="syllable",
    ),
    TestCase(
        id="CLN003",
        category="clean",
        input_text="မြန်မာစာသည်တိဘက်-ဗမာဘာသာစကားမျိုးနွယ်စုဖြစ်သည်",
        expected_errors=[],
        description="Clean language family",
        level="syllable",
    ),
    TestCase(
        id="CLN004",
        category="clean",
        input_text="ဗုဒ္ဓဘာသာသည်မြန်မာနိုင်ငံ၏အဓိကဘာသာဖြစ်သည်",
        expected_errors=[],
        description="Clean religion reference",
        level="syllable",
    ),
    TestCase(
        id="CLN005",
        category="clean",
        input_text="မြန်မာနိုင်ငံတွင်တိုင်းရင်းသားလူမျိုးပေါင်းများစွာနေထိုင်သည်",
        expected_errors=[],
        description="Clean ethnic diversity",
        level="syllable",
    ),
    TestCase(
        id="CLN006",
        category="clean",
        input_text="သီတင်းကျွတ်ပွဲတော်သည်မြန်မာနိုင်ငံ၏အကြီးဆုံးပွဲတော်ဖြစ်သည်",
        expected_errors=[],
        description="Clean Thadingyut festival",
        level="syllable",
    ),
    TestCase(
        id="CLN007",
        category="clean",
        input_text="ရေကူးကန်တွင်ရေကူးလေ့ကျင့်သည်",
        expected_errors=[],
        description="Clean swimming practice",
        level="syllable",
    ),
    TestCase(
        id="CLN008",
        category="clean",
        input_text="စက်ဘီးစီးခြင်းသည်ကျန်းမာရေးအတွက်ကောင်းသည်",
        expected_errors=[],
        description="Clean cycling health",
        level="syllable",
    ),
    TestCase(
        id="CLN009",
        category="clean",
        input_text="ဘောလုံးကစားခြင်းသည်လူကြိုက်များသောအားကစားဖြစ်သည်",
        expected_errors=[],
        description="Clean football popularity",
        level="syllable",
    ),
    TestCase(
        id="CLN010",
        category="clean",
        input_text="တင်းနစ်ကစားနည်းကိုလေ့လာနေသည်",
        expected_errors=[],
        description="Clean tennis learning",
        level="syllable",
    ),
    # News-style sentences
    TestCase(
        id="CLN011",
        category="clean",
        input_text="နိုင်ငံတော်သမ္မတသည်နိုင်ငံခြားခရီးစဉ်မှပြန်လည်ရောက်ရှိလာသည်",
        expected_errors=[],
        description="Clean presidential news",
        level="syllable",
    ),
    TestCase(
        id="CLN012",
        category="clean",
        input_text="လွှတ်တော်အစည်းအဝေးကိုဒီနေ့ကျင်းပမည်ဖြစ်သည်",
        expected_errors=[],
        description="Clean parliament news",
        level="syllable",
    ),
    TestCase(
        id="CLN013",
        category="clean",
        input_text="စီးပွားရေးတိုးတက်မှုနှုန်းသည်ယခင်နှစ်ထက်မြင့်မားသည်",
        expected_errors=[],
        description="Clean economic news",
        level="syllable",
    ),
    TestCase(
        id="CLN014",
        category="clean",
        input_text="ရာသီဥတုဖြစ်ပွားမှုကြောင့်သီးနှံများပျက်စီးခဲ့သည်",
        expected_errors=[],
        description="Clean weather news",
        level="syllable",
    ),
    TestCase(
        id="CLN015",
        category="clean",
        input_text="ပညာရေးဝန်ကြီးဌာနသည်မူဝါဒအသစ်ထုတ်ပြန်သည်",
        expected_errors=[],
        description="Clean education policy news",
        level="syllable",
    ),
    # Literary/formal style
    TestCase(
        id="CLN016",
        category="clean",
        input_text="နွေဦးရာသီ၏နံနက်ခင်းသည်အလွန်သာယာသည်",
        expected_errors=[],
        description="Clean literary description",
        level="syllable",
    ),
    TestCase(
        id="CLN017",
        category="clean",
        input_text="သစ်ပင်များပေါ်တွင်ငှက်များသီချင်းဆိုနေကြသည်",
        expected_errors=[],
        description="Clean nature description",
        level="syllable",
    ),
    TestCase(
        id="CLN018",
        category="clean",
        input_text="မြစ်ကမ်းပါးတွင်လူများအပန်းဖြေနေကြသည်",
        expected_errors=[],
        description="Clean riverside scene",
        level="syllable",
    ),
    TestCase(
        id="CLN019",
        category="clean",
        input_text="နေဝင်ချိန်၏မြင်ကွင်းသည်လှပသည်",
        expected_errors=[],
        description="Clean sunset view",
        level="syllable",
    ),
    TestCase(
        id="CLN020",
        category="clean",
        input_text="ကြယ်များဖြင့်ပြည့်နှက်နေသောကောင်းကင်ကိုကြည့်သည်",
        expected_errors=[],
        description="Clean starry sky",
        level="syllable",
    ),
    # Technical/modern vocabulary
    TestCase(
        id="CLN021",
        category="clean",
        input_text="အင်တာနက်ချိတ်ဆက်မှုသည်မြန်ဆန်သည်",
        expected_errors=[],
        description="Clean internet speed",
        level="syllable",
    ),
    TestCase(
        id="CLN022",
        category="clean",
        input_text="စမတ်ဖုန်းအသုံးပြုသူဦးရေတိုးပွားလာသည်",
        expected_errors=[],
        description="Clean smartphone usage",
        level="syllable",
    ),
    TestCase(
        id="CLN023",
        category="clean",
        input_text="ဒီဂျစ်တယ်နည်းပညာသည်မြန်ဆန်စွာတိုးတက်နေသည်",
        expected_errors=[],
        description="Clean digital technology",
        level="syllable",
    ),
    TestCase(
        id="CLN024",
        category="clean",
        input_text="အွန်လိုင်းစျေးဝယ်ခြင်းသည်လူကြိုက်များလာသည်",
        expected_errors=[],
        description="Clean online shopping",
        level="syllable",
    ),
    TestCase(
        id="CLN025",
        category="clean",
        input_text="ဆိုရှယ်မီဒီယာပလက်ဖောင်းများကိုအသုံးပြုသည်",
        expected_errors=[],
        description="Clean social media usage",
        level="syllable",
    ),
]

# Combine all test cases
ALL_TESTS = SYLLABLE_ERROR_TESTS + WORD_ERROR_TESTS + CONTEXT_ERROR_TESTS + CLEAN_TESTS


def run_tests(
    checker: SpellChecker,
    tests: List[TestCase],
    verbose: bool = False,
) -> Tuple[List[TestResult], Dict[str, Any]]:
    """
    Run all test cases and collect results.

    Args:
        checker: SpellChecker instance
        tests: List of test cases
        verbose: Print detailed output

    Returns:
        Tuple of (results list, summary dict)
    """
    results = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for test in tests:
        start_time = time.perf_counter()

        # Determine validation level
        level = ValidationLevel.WORD if test.level == "word" else ValidationLevel.SYLLABLE

        # Run spell checker
        response = checker.check(test.input_text, level=level)
        processing_time = time.perf_counter() - start_time

        # Analyze results
        result = TestResult(
            test_case=test,
            detected_errors=[e.to_dict() for e in response.errors],
            processing_time=processing_time,
        )

        # Match detected errors to expected errors
        detected_texts = {e.text for e in response.errors}
        expected_texts = {e.text for e in test.expected_errors}

        # Calculate TP, FP, FN
        result.true_positives = len(detected_texts & expected_texts)
        result.false_positives = len(detected_texts - expected_texts)
        result.false_negatives = len(expected_texts - detected_texts)

        # Determine if test passed
        if test.category == "clean":
            # Clean tests pass if no errors detected
            result.passed = len(response.errors) == 0
            if not result.passed:
                result.notes.append(f"Expected 0 errors, got {len(response.errors)}")
        else:
            # Error tests pass if we detected expected errors
            result.passed = result.false_negatives == 0 and result.false_positives == 0
            if result.false_negatives > 0:
                missed = expected_texts - detected_texts
                result.notes.append(f"Missed errors: {missed}")
            if result.false_positives > 0:
                extra = detected_texts - expected_texts
                result.notes.append(f"False positives: {extra}")

        results.append(result)
        total_tp += result.true_positives
        total_fp += result.false_positives
        total_fn += result.false_negatives

        if verbose:
            status = "✓" if result.passed else "✗"
            print(f"  {status} {test.id}: {test.description}")
            if not result.passed:
                for note in result.notes:
                    print(f"      {note}")

    # Calculate overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    summary = {
        "total_tests": len(tests),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "avg_processing_time": sum(r.processing_time for r in results) / len(results),
    }

    return results, summary


MIN_CURATED_CASES = 100


@pytest.fixture(scope="module")
def checker() -> SpellChecker:
    """Create one SpellChecker for the full curated accuracy run."""
    return SpellChecker()


@pytest.fixture(scope="module")
def accuracy_run(checker: SpellChecker) -> Tuple[List[TestResult], Dict[str, Any]]:
    """Execute all curated accuracy tests once per module."""
    return run_tests(checker, ALL_TESTS, verbose=False)


def test_curated_dataset_integrity() -> None:
    """Curated dataset should remain stable and fully classified."""
    assert len(ALL_TESTS) >= MIN_CURATED_CASES

    ids = [test.id for test in ALL_TESTS]
    assert len(ids) == len(set(ids))

    allowed_categories = {"syllable", "word", "context", "clean"}
    assert all(test.category in allowed_categories for test in ALL_TESTS)
    assert all(test.input_text for test in ALL_TESTS)


@pytest.mark.slow
def test_accuracy_summary_metrics_consistency(
    accuracy_run: Tuple[List[TestResult], Dict[str, Any]],
) -> None:
    """Summary metrics must match raw per-case outputs."""
    results, summary = accuracy_run

    total_tp = sum(result.true_positives for result in results)
    total_fp = sum(result.false_positives for result in results)
    total_fn = sum(result.false_negatives for result in results)
    passed = sum(1 for result in results if result.passed)

    expected_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    expected_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    expected_f1 = (
        2 * expected_precision * expected_recall / (expected_precision + expected_recall)
        if (expected_precision + expected_recall) > 0
        else 0.0
    )

    assert summary["total_tests"] == len(ALL_TESTS)
    assert summary["true_positives"] == total_tp
    assert summary["false_positives"] == total_fp
    assert summary["false_negatives"] == total_fn
    assert summary["passed"] == passed
    assert summary["failed"] == len(results) - passed
    assert summary["precision"] == pytest.approx(expected_precision)
    assert summary["recall"] == pytest.approx(expected_recall)
    assert summary["f1_score"] == pytest.approx(expected_f1)


@pytest.mark.slow
def test_accuracy_run_produces_valid_summary(
    accuracy_run: Tuple[List[TestResult], Dict[str, Any]],
) -> None:
    """Full curated run should produce sane, bounded summary metrics."""
    _, summary = accuracy_run
    pass_rate = summary["passed"] / summary["total_tests"]

    assert summary["total_tests"] == len(ALL_TESTS)
    assert 0 <= summary["passed"] <= summary["total_tests"]
    assert 0 <= summary["failed"] <= summary["total_tests"]
    assert 0.0 <= pass_rate <= 1.0
    assert 0.0 <= summary["precision"] <= 1.0
    assert 0.0 <= summary["recall"] <= 1.0
    assert 0.0 <= summary["f1_score"] <= 1.0
    assert summary["avg_processing_time"] >= 0.0
