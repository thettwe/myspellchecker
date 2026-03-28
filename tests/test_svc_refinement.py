from unittest.mock import MagicMock

from myspellchecker.grammar.engine import SyntacticRuleChecker


def get_mock_word_pos(word: str) -> str:
    if word in ["သွား", "လာ", "စား", "အိပ်", "လိုက်"]:
        return "V"
    return "UNK"


def test_svc_refinement():
    """
    Test that V-V sequences with auxiliary verbs (SVCs) are allowed.
    """
    mock_provider = MagicMock()
    mock_provider.get_word_pos.side_effect = get_mock_word_pos

    checker = SyntacticRuleChecker(mock_provider)

    # Force V-V rule to be "error" for testing logic
    for rule in checker.config.invalid_pos_sequences:
        if rule.get("pattern") == "V-V":
            rule["severity"] = "error"

    # 1. Test Ignored SVC (Exception)
    # "စား" (V) + "လိုက်" (V)
    # "လိုက်" is in exceptions list, so the V-V sequence error should NOT fire.
    # However, the sentence structure check may flag the bare verb at end
    # (suggesting "လိုက်တယ်" as a sentence-final particle addition).
    words_ok = ["စား", "လိုက်"]

    # We need to ensure V-V rule is loaded.
    # check_sequence returns list of errors.
    errors_ok = checker.check_sequence(words_ok)

    # Filter out sentence-structure errors (verb at end without particle).
    # These are separate from V-V sequence validation and are expected behavior.
    vv_errors = [e for e in errors_ok if not (e[2].endswith("တယ်") or e[2].endswith("ဖြစ်သည်"))]
    assert len(vv_errors) == 0, f"Expected no V-V errors for SVC 'စား လိုက်', got {vv_errors}"

    # 2. Test Flagged V-V (Not Exception)
    # "စား" (V) + "အိပ်" (V)
    # Neither is in exceptions.
    words_bad = ["စား", "အိပ်"]

    errors_bad = checker.check_sequence(words_bad)
    assert len(errors_bad) > 0, f"Expected error for 'စား အိပ်', got {errors_bad}"


if __name__ == "__main__":
    test_svc_refinement()
