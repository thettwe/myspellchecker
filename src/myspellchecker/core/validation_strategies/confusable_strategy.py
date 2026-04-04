"""
Confusable Variant Utilities (re-exports).

This module re-exports confusable variant generation functions from
``myspellchecker.core.myanmar_confusables`` so that existing consumers
that import from ``confusable_strategy`` continue to work.

The ``ConfusableVariantStrategy`` class that previously lived here has been
removed.  Its n-gram comparison logic is now handled by
``NgramContextChecker.check_word_in_context()`` which the homophone and
n-gram validation strategies delegate to.

**Why private names are re-exported:**
The ``_get_medial_variants``, ``_get_nasal_variants``, and
``_get_stop_coda_variants`` helpers are implementation details of variant
generation.  They are re-exported here because test modules
(``tests/test_confusable_*.py``) import them from this path, and
``confusable_semantic_strategy`` also references this module.  Keeping
these re-exports avoids breaking existing test imports while the canonical
source remains ``myanmar_confusables``.
"""

from __future__ import annotations

from myspellchecker.core.myanmar_confusables import (
    ALL_MEDIALS,
    ASPIRATION_PAIRS,
    MEDIAL_SWAP_PAIRS,
    _get_medial_variants,  # noqa: F401 — re-exported for tests
    _get_nasal_variants,  # noqa: F401 — re-exported for tests
    _get_stop_coda_variants,  # noqa: F401 — re-exported for tests
    generate_confusable_variants,  # noqa: F401 — re-exported for consumers
)
from myspellchecker.core.myanmar_confusables import (
    is_aspirated_confusable as _is_aspirated_confusable,  # noqa: F401 — re-exported
)
from myspellchecker.core.myanmar_confusables import (
    is_medial_confusable as _is_medial_confusable,  # noqa: F401 — re-exported
)

# Re-export pair constants under their original private names for backward
# compatibility with any code that imported them directly.
_ASPIRATION_PAIRS = ASPIRATION_PAIRS
_MEDIAL_SWAPS = MEDIAL_SWAP_PAIRS
_ALL_MEDIALS = ALL_MEDIALS
