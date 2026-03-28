from myspellchecker.core.homophones import HomophoneChecker


def test_homophone_lookup():
    checker = HomophoneChecker()

    # Test known homophones (Ya-pin vs Ya-yit — kept in curated YAML)
    assert "ကြောင်း" in checker.get_homophones("ကျောင်း")
    assert "ကျောင်း" in checker.get_homophones("ကြောင်း")
    assert "ကြား" in checker.get_homophones("ကျား")

    # Test tone/final stop pairs (kept)
    assert "စား" in checker.get_homophones("စာ")
    assert "ကား" in checker.get_homophones("ကာ")

    # Test no homophones
    assert not checker.get_homophones("ကွန်ပြူတာ")  # Computer has no homophone in map


def test_removed_entries_not_present():
    """Entries removed during YAML curation should not appear."""
    checker = HomophoneChecker()

    # Ha-htoe section: မှာ↔မာ re-added as bidirectional homophone pair
    assert checker.get_homophones("မှာ") == {"မာ"}
    assert checker.get_homophones("မာ") == {"မှာ"}
    # ရှိ→ရိ and နှစ်→နစ် are valid homophones, kept in YAML
    assert checker.get_homophones("ရှိ") == {"ရိ"}
    assert checker.get_homophones("နှစ်") == {"နစ်"}

    # Non-homophones removed from vowel section
    assert not checker.get_homophones("နေ")  # was နေ→နဲ (different vowels)
    assert not checker.get_homophones("ပေး")  # was ပေး→ပဲ (different tones)

    # Non-words removed from wa-hswe section
    assert not checker.get_homophones("ထွက်")  # was ထွက်→ထုက် (ထုက် not a word)

    # Common pairs removed (overlaps with typo rules or not homophones)
    assert not checker.get_homophones("ပါ")  # was ပါ→ပေါ် (different vowels)
    assert not checker.get_homophones("လာ")  # was လာ→လား (question particle rule)


def test_symmetry_enforcement():
    """Asymmetric YAML entries should be made symmetric by _ensure_symmetry()."""
    # ပြည် has no reverse entry in the YAML (only forward: ပြည်→ပျည်)
    checker = HomophoneChecker()

    # Forward direction (from YAML)
    assert "ပျည်" in checker.get_homophones("ပြည်")

    # Reverse direction (added by _ensure_symmetry)
    assert "ပြည်" in checker.get_homophones("ပျည်")


def test_symmetry_with_custom_map():
    """Test _ensure_symmetry on a custom one-directional map."""
    # Only A→B is defined, not B→A
    custom_map = {"A": {"B", "C"}}
    checker = HomophoneChecker(homophone_map=custom_map)

    # Forward (explicit)
    assert checker.get_homophones("A") == {"B", "C"}

    # Reverse (added by symmetry)
    assert "A" in checker.get_homophones("B")
    assert "A" in checker.get_homophones("C")


def test_symmetry_merges_with_existing():
    """Test _ensure_symmetry merges into existing reverse entries."""
    # B already maps to D, symmetry should add A
    custom_map = {"A": {"B"}, "B": {"D"}}
    checker = HomophoneChecker(homophone_map=custom_map)

    # B should now map to both D and A
    b_homophones = checker.get_homophones("B")
    assert "D" in b_homophones
    assert "A" in b_homophones


def test_custom_map():
    custom_map = {"A": {"B"}, "B": {"A"}}
    checker = HomophoneChecker(homophone_map=custom_map)

    assert "B" in checker.get_homophones("A")
    assert not checker.get_homophones("ကျောင်း")  # Should not have default map
