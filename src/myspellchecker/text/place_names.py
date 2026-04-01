"""
Myanmar place-name dictionary for heuristic NER.

All data now lives in ``rules/named_entities.yaml`` and is loaded via
:func:`myspellchecker.text.ner.get_gazetteer_data`.  This module provides
backwards-compatible frozenset attributes that delegate to the YAML-backed
gazetteer singleton.
"""

from __future__ import annotations

from functools import lru_cache


def _build_compat_sets() -> dict[str, frozenset[str]]:
    """Build backwards-compatible frozenset mapping from the YAML gazetteer."""
    from myspellchecker.text.ner import get_gazetteer_data

    gaz = get_gazetteer_data()

    myanmar_places = gaz.townships | gaz.states_regions | gaz.major_cities
    # MYANMAR_STATE_SHORT = the 14 official state/region short names.
    # The YAML states_regions also includes "နေပြည်တော်" (Naypyidaw, a union
    # territory) which was never in the original 14-item set.
    _UNION_TERRITORIES = frozenset({"နေပြည်တော်"})
    myanmar_state_short = gaz.states_regions - _UNION_TERRITORIES
    myanmar_ethnic_groups = gaz.ethnic_groups
    myanmar_historical_places = gaz.historical_places
    international_cities = gaz.international_places
    international_countries = gaz.countries
    myanmar_geographic_features = gaz.geographic_features

    all_places = (
        myanmar_places
        | myanmar_state_short
        | myanmar_ethnic_groups
        | myanmar_historical_places
        | international_cities
        | international_countries
        | myanmar_geographic_features
    )

    return {
        "MYANMAR_PLACES": myanmar_places,
        "MYANMAR_STATE_SHORT": myanmar_state_short,
        "MYANMAR_ETHNIC_GROUPS": myanmar_ethnic_groups,
        "MYANMAR_HISTORICAL_PLACES": myanmar_historical_places,
        "INTERNATIONAL_CITIES": international_cities,
        "INTERNATIONAL_COUNTRIES": international_countries,
        "MYANMAR_GEOGRAPHIC_FEATURES": myanmar_geographic_features,
        "ALL_PLACES": all_places,
    }


@lru_cache(maxsize=1)
def _cached_sets() -> dict[str, frozenset[str]]:
    return _build_compat_sets()


_COMPAT_NAMES = {
    "MYANMAR_PLACES",
    "MYANMAR_STATE_SHORT",
    "MYANMAR_ETHNIC_GROUPS",
    "MYANMAR_HISTORICAL_PLACES",
    "INTERNATIONAL_CITIES",
    "INTERNATIONAL_COUNTRIES",
    "MYANMAR_GEOGRAPHIC_FEATURES",
    "ALL_PLACES",
}


def __getattr__(name: str) -> frozenset[str]:
    if name in _COMPAT_NAMES:
        return _cached_sets()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
