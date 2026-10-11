"""The guard that decides whether an LLM rewrite of a card may replace the rule text.
No model needed: these are the properties a rewrite must keep."""

from swing_coach.llm import _check

ORIGINAL = "탑에서 골반이 오른쪽으로 밀렸어요 (기준 스윙보다 공 2개 정도)"


def test_natural_korean_rewrite_passes():
    assert _check(ORIGINAL, "탑에서 골반이 기준 스윙보다 공 2개 정도 오른쪽으로 밀렸어요.") is None


def test_changed_number_is_rejected():
    assert "numbers" in _check(ORIGINAL, "탑에서 골반이 기준 스윙보다 공 3개 정도 오른쪽으로 밀렸어요.")


def test_added_number_is_rejected():
    assert "numbers" in _check(ORIGINAL, "탑에서 골반이 기준 스윙보다 공 2개, 약 8cm 밀렸어요.")


def test_other_language_is_rejected_even_with_same_numbers():
    # Seen in practice: every number kept, sentence in Japanese.
    assert _check(ORIGINAL, "トップで骨盤が右にずれています (標準スイングよりボール2個分)") == "not Korean"


def test_dropping_the_comparison_baseline_is_rejected():
    assert "기준 스윙" in _check(ORIGINAL, "탑에서 골반이 오른쪽으로 공 2개 정도 밀렸어요.")


def test_runaway_length_is_rejected():
    assert _check("템포가 좋아요 (3.1:1)", "템포 3.1:1 " + "아주 좋아요 " * 20) == "too long"
