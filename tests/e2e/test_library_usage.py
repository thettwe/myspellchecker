from myspellchecker import Response, SpellChecker, check_text


def test_basic_library_usage():
    """
    Test the basic instantiation and checking flow of the SpellChecker class.
    """
    checker = SpellChecker()
    # "မြန်မာ" is a correct word
    text = "မြန်မာ"
    res = checker.check(text)

    assert isinstance(res, Response)
    assert not res.has_errors
    assert len(res.errors) == 0
    assert res.text == text


def test_convenience_function():
    """
    Test the top-level check_text function.
    """
    text = "မြန်မာ"
    res = check_text(text)
    assert isinstance(res, Response)
    # Assuming "မြန်မာ" is correct (it is in the test DB)
    assert not res.has_errors
