"""Textbook verdict: band checks, verdict levels and the card it writes (no model, no DB)."""

from swing_coach import rubric
from swing_coach.coach import Card

TEXTBOOK_DRIVER = {"tempo_ratio": 3.0, "pelvis_sway_top_sw": 0.0, "pelvis_shift_imp_sw": 0.2,
                   "pelvis_over_lead_foot_fin_sw": 0.0, "hands_ahead_imp_sw": 0.05, "stance_width_sw": 1.25,
                   "trail_shoulder_drop_deg": 10.0, "head_sway_top_sw": -0.1, "head_move_imp_sw": -0.1,
                   "lead_knee_imp_deg": 172.0}


def test_band_status_and_direction():
    rule = rubric.BY_RULE["pelvis_sway_top_sw"]
    assert rubric.check(rule, "driver", 0.0).status == "green"
    amber = rubric.check(rule, "driver", -0.12)   # swayed away from the target, within ok
    assert (amber.status, amber.direction) == ("amber", "low")
    assert rubric.check(rule, "driver", -0.30).status == "red"
    assert rubric.check(rule, "driver", 0.20).direction == "high"   # moved toward the target


def test_textbook_swing_is_good_and_two_core_faults_are_poor():
    assert rubric.judge(TEXTBOOK_DRIVER, "driver").level == "잘함"
    poor = dict(TEXTBOOK_DRIVER, tempo_ratio=4.6, pelvis_sway_top_sw=-0.4)
    assert rubric.judge(poor, "driver").level == "아쉬움"
    one = dict(TEXTBOOK_DRIVER, tempo_ratio=4.6)
    assert rubric.judge(one, "driver").level == "보통"


def test_soft_metrics_do_not_change_the_verdict():
    off = dict(TEXTBOOK_DRIVER, stance_width_sw=2.0, head_sway_top_sw=-0.8, trail_shoulder_drop_deg=30)
    assert rubric.judge(off, "driver").level == "잘함"


def test_withheld_when_unreliable():
    assert rubric.judge(TEXTBOOK_DRIVER, "driver", confidence="낮음").level == "판정 보류"
    assert rubric.judge(TEXTBOOK_DRIVER, None).level == "판정 보류"
    assert rubric.judge({"tempo_ratio": 3.0}, "driver").level == "판정 보류"   # too few core metrics


def test_card_quotes_textbook_range_for_worst_core_fault():
    row = dict(TEXTBOOK_DRIVER, tempo_ratio=4.6, pelvis_sway_top_sw=-0.4, ball_flight_code=2)
    card = Card(good="👍 x", fix="👉 y", cue="z", confidence="높음")
    links = {"pelvis_sway_top_sw": [{"flight": 2, "sign": -1, "n": 54}]}
    verdict = rubric.apply_to_card(card, row, "driver", links, [])
    assert verdict.level == "아쉬움"
    assert card.fix_key == "pelvis_sway_top_sw" and "정석 범위보다" in card.fix
    assert any("슬라이스" in e for e in card.extra)
    assert card.evidence["_verdict"]["level"] == "아쉬움"


def test_all_green_card_has_no_fix():
    card = Card(good="👍 x", fix="👉 y", cue="z", confidence="높음")
    rubric.apply_to_card(card, TEXTBOOK_DRIVER, "driver", {}, [])
    assert card.fix is None and "정석 범위 안" in card.cue
