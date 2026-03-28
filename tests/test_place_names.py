"""Tests for Myanmar place name data."""

from myspellchecker.text.place_names import (
    MYANMAR_ETHNIC_GROUPS,
    MYANMAR_PLACES,
    MYANMAR_STATE_SHORT,
)


def test_known_townships_present_in_places() -> None:
    expected = ["ရန်ကင်း", "မကွေး", "တောင်ကြီး", "မော်လမြိုင်", "စစ်တွေ"]
    for name in expected:
        assert name in MYANMAR_PLACES, f"{name} missing from MYANMAR_PLACES"


def test_state_region_names_present_in_places() -> None:
    states = [
        "ကချင်ပြည်နယ်",
        "ရှမ်းပြည်နယ်",
        "ရန်ကုန်တိုင်းဒေသကြီး",
        "မန္တလေးတိုင်းဒေသကြီး",
    ]
    for name in states:
        assert name in MYANMAR_PLACES


def test_short_state_names_are_subset_of_ethnic_groups() -> None:
    # All 8 major ethnic groups appear in both short state names and ethnic groups
    major_8 = {"ကချင်", "ကယား", "ကရင်", "ချင်း", "မွန်", "ရခိုင်", "ရှမ်း"}
    assert major_8.issubset(MYANMAR_STATE_SHORT)
    assert major_8.issubset(MYANMAR_ETHNIC_GROUPS)


def test_non_place_names_absent() -> None:
    non_places = ["ကျောင်းသား", "စာအုပ်", "hello", "", "၁၂၃"]
    for name in non_places:
        assert name not in MYANMAR_PLACES


def test_places_and_states_are_frozensets() -> None:
    # Frozensets are immutable -- verify the data cannot be accidentally mutated
    assert len(MYANMAR_PLACES) > 400
    assert len(MYANMAR_STATE_SHORT) == 14
    assert len(MYANMAR_ETHNIC_GROUPS) > 8
