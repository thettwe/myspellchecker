"""Colloquial-locative allowlist split regression tests.

Verifies that the ``DefaultSegmenter._COLLOQUIAL_LOCATIVE_MERGES`` allowlist
correctly splits known problematic tokens without affecting any other token.

The motivating case is BM-EXT-E010 where the segmenter retained ``ကုန်မာ`` as
a single token (it has dictionary freq=78), causing the homophone strategy
to fire at the wrong span. The allowlist forces ``ကုန်မာ → [ကုန်, မာ]`` at
segmentation time so downstream strategies see ``မာ`` as a standalone token
and can suggest the correct homophone ``မှာ``.
"""

from __future__ import annotations

import pytest

from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.constants import ValidationLevel
from myspellchecker.core.spellchecker import SpellChecker
from myspellchecker.providers.sqlite import SQLiteProvider


@pytest.fixture(scope="module")
def checker():
    """Build a SpellChecker with the production DB (if available).

    Skips tests if the production database is not present locally, since
    the allowlist fix depends on real dictionary frequencies.
    """
    from pathlib import Path

    db_path = Path("data/mySpellChecker_production.db")
    if not db_path.exists():
        pytest.skip(f"Production database not found at {db_path}; segmenter test skipped")

    provider = SQLiteProvider(database_path=str(db_path))
    config = SpellCheckerConfig()
    instance = SpellChecker(config=config, provider=provider)
    instance.check("ကျွန်တော် ကျန်းမာပါတယ်", level=ValidationLevel.WORD)  # warm up
    return instance


def test_kun_ma_allowlist_split(checker):
    """``ကုန်မာ`` must split to ``["ကုန်", "မာ"]`` via the allowlist."""
    text = "သူငယ်ချင်းက ရန်ကုန်မာ နေတယ်။"
    tokens = checker.segmenter.segment_words(text)
    assert "ကုန်မာ" not in tokens, f"Expected ကုန်မာ to be split via allowlist, got tokens={tokens}"
    assert "မာ" in tokens, f"Expected standalone မာ after allowlist split, got tokens={tokens}"
    assert "ကုန်" in tokens, f"Expected standalone ကုန် after allowlist split, got tokens={tokens}"


def test_allowlist_does_not_affect_non_allowlist_tokens():
    """The allowlist must only split tokens that are IN the allowlist.

    The allowlist has exactly one entry (``ကုန်မာ``). Any other common token
    must not be altered by the allowlist code path. This is a pure structural
    check on the class attribute — no fixture needed.
    """
    from myspellchecker.segmenters.default import DefaultSegmenter

    # Verify common tokens are NOT in the allowlist
    assert "ရန်ကုန်" not in DefaultSegmenter._COLLOQUIAL_LOCATIVE_MERGES
    assert "မန္တလေး" not in DefaultSegmenter._COLLOQUIAL_LOCATIVE_MERGES
    assert "အစိုးရ" not in DefaultSegmenter._COLLOQUIAL_LOCATIVE_MERGES
    assert "ကျောင်းသား" not in DefaultSegmenter._COLLOQUIAL_LOCATIVE_MERGES
    # The only entry is ကုန်မာ
    assert "ကုန်မာ" in DefaultSegmenter._COLLOQUIAL_LOCATIVE_MERGES
    assert len(DefaultSegmenter._COLLOQUIAL_LOCATIVE_MERGES) == 1


def test_allowlist_is_minimal():
    """Pin the allowlist to exactly the baseline entries.

    Pure structural check — guards against accidental expansion. If a new
    entry is deliberately added, update this list explicitly.
    """
    from myspellchecker.segmenters.default import DefaultSegmenter

    allowlist = DefaultSegmenter._COLLOQUIAL_LOCATIVE_MERGES
    assert allowlist == {
        "ကုန်မာ": ("ကုန်", "မာ"),
    }, f"Allowlist drifted from baseline. Actual: {allowlist}"
